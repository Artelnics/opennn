#!/usr/bin/env python3
"""Reproducible Qwen3-4B engine and runtime benchmark.

This module is dispatched by ``benchmarks/run.py --family qwen``.  Large
models, third-party sources, services, logs and generated prompts remain below
OPENNN_BENCH_DATA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
BENCHMARKS = HERE.parent
QWEN_TOOLS = BENCHMARKS / "tools"
sys.path.insert(0, str(BENCHMARKS))
sys.path.insert(0, str(QWEN_TOOLS))

from common import (BUSY_THRESHOLD, Monitor, clocks_locked, cpu_busy_fraction,  # noqa: E402
                    cpu_state, git_metadata, gpu_state, result_destination,
                    run_text, session_id)
from qwen_support import (common_prefix, last_json, llama_bench_samples,  # noqa: E402
                          raw_chatml, sample_statistics)

DATA_ROOT = Path(os.environ.get("OPENNN_BENCH_DATA",
                                str(Path.home() / "opennn-benchmark-data")))
QWEN_ROOT = DATA_ROOT / "qwen3"
MANIFEST_PATH = BENCHMARKS / "manifests" / "qwen_manifest.json"
TARGET_SM_CLOCK = 2505.0
TARGET_MEMORY_CLOCK = 11201.0
CLOCK_STEP = 15.0
MAX_TEMPERATURE = 45.0
MAX_GPU_UTILIZATION = 2.0
MAX_BASELINE_DRIFT_MIB = 64.0
MAX_CV = 0.03
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def binary(name: str, override: str, candidates: list[Path]) -> Path:
    explicit = os.environ.get(override)
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        raise SystemExit(f"{override} points to a missing file: {path}")
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit(f"missing {name}; run benchmarks/tools/qwen_benchmark.ps1 build")


def binaries() -> dict[str, Path]:
    llama_build = QWEN_ROOT / "tools" / "llama.cpp" / "build-windows-cuda"
    tag = manifest()["ollama"]["version"]
    return {
        "opennn": binary("qwen_opennn", "OPENNN_QWEN_BIN", [
            Path(os.environ.get("OPENNN_QWEN_BUILD", "")) / "qwen_opennn.exe",
            Path(os.environ.get("OPENNN_QWEN_BUILD", "")) / "bin" / "qwen_opennn.exe",
            Path("build") / "bin" / "qwen_opennn.exe",
            Path("build") / "benchmarks" / "qwen_opennn.exe",
        ]),
        "llama_bench": binary("llama-bench", "OPENNN_LLAMA_BENCH_BIN", [
            llama_build / "bin" / "llama-bench.exe",
            llama_build / "bin" / "Release" / "llama-bench.exe",
        ]),
        "llama_server": binary("llama-server", "OPENNN_LLAMA_SERVER_BIN", [
            llama_build / "bin" / "llama-server.exe",
            llama_build / "bin" / "Release" / "llama-server.exe",
        ]),
        "ollama": binary("ollama", "OPENNN_OLLAMA_BIN", [
            QWEN_ROOT / "tools" / "ollama" / tag / "ollama.exe",
        ]),
    }


def model_paths() -> dict[str, Path]:
    return {
        "opennn": QWEN_ROOT / "models" / "opennn",
        "gguf": QWEN_ROOT / "models" / "qwen3-4b-bf16.gguf",
        "validation": QWEN_ROOT / "validation.json",
    }


def pre_touch(path: Path) -> None:
    with path.open("rb") as handle:
        while handle.read(32 << 20):
            pass


def query_gpu_baseline() -> dict[str, float]:
    text = run_text([
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,temperature.gpu,clocks.current.sm,clocks.current.memory",
        "--format=csv,noheader,nounits",
    ], timeout=15)
    if not text:
        return {}
    try:
        values = [float(value.strip()) for value in text.splitlines()[0].split(",")]
        return dict(zip(("memory_mib", "utilization_percent", "temperature_c",
                         "sm_clock_mhz", "memory_clock_mhz"), values))
    except (ValueError, IndexError):
        return {}


def environment_gate(initial_memory: float | None) -> dict[str, Any]:
    busy = cpu_busy_fraction()
    baseline = query_gpu_baseline()
    state = gpu_state()
    reasons: list[str] = []
    if busy > BUSY_THRESHOLD:
        reasons.append(f"CPU busy {busy:.1%} > {BUSY_THRESHOLD:.1%}")
    if baseline.get("utilization_percent", 101.0) > MAX_GPU_UTILIZATION:
        reasons.append(f"GPU utilization {baseline.get('utilization_percent')}%")
    if baseline.get("temperature_c", 999.0) > MAX_TEMPERATURE:
        reasons.append(f"GPU temperature {baseline.get('temperature_c')} C")
    if initial_memory is not None and abs(baseline.get("memory_mib", initial_memory)
                                          - initial_memory) > MAX_BASELINE_DRIFT_MIB:
        reasons.append("GPU baseline memory drift exceeded 64 MiB")
    if not clocks_locked():
        reasons.append("GPU clocks were not locked by the benchmark wrapper")
    throttles = {key: value for key, value in state.items()
                 if key.endswith(("slowdown", "power_cap"))
                 and str(value).lower() == "active"}
    if throttles:
        reasons.append(f"active thermal/power throttle {sorted(throttles)}")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "cpu_busy_fraction": busy,
        "thresholds": {
            "cpu_busy_fraction": BUSY_THRESHOLD,
            "gpu_utilization_percent": MAX_GPU_UTILIZATION,
            "temperature_c": MAX_TEMPERATURE,
            "baseline_drift_mib": MAX_BASELINE_DRIFT_MIB,
            "sm_clock_mhz": TARGET_SM_CLOCK,
            "memory_clock_mhz": TARGET_MEMORY_CLOCK,
        },
        "baseline": baseline,
        "gpu_state": state,
    }


def wait_for_environment(initial_memory: float | None,
                         timeout: float = 120.0) -> dict[str, Any]:
    """Wait for transient idle gates; permanent clock-lock failure is diagnostic."""
    deadline = time.monotonic() + timeout
    last = environment_gate(initial_memory)
    while time.monotonic() < deadline:
        transient = [reason for reason in last["reasons"]
                     if "clocks were not locked" not in reason]
        if not transient:
            return last
        time.sleep(2.0)
        last = environment_gate(initial_memory)
    last["reasons"].append(f"environment did not settle within {timeout:.0f} seconds")
    last["valid"] = False
    return last


def monitor_window(monitor: Monitor, start: float | None,
                   end: float | None) -> dict[str, Any]:
    summary = monitor.summary(start, end)
    samples = [sample for sample in monitor.telemetry_samples
               if (start is None or sample["unix"] >= start)
               and (end is None or sample["unix"] <= end)]
    memory = [sample["memory_mib"] - monitor.idle_mib for sample in samples]
    before = [sample["memory_mib"] - monitor.idle_mib
              for sample in monitor.telemetry_samples
              if start is not None and start - 1.0 <= sample["unix"] < start]
    if not before and len(samples) >= 3:
        duration = samples[-1]["unix"] - samples[0]["unix"]
        margin = min(1.0, max(0.0, duration / 4.0))
        before = [sample["memory_mib"] - monitor.idle_mib for sample in samples
                  if samples[0]["unix"] + margin <= sample["unix"]
                  <= samples[-1]["unix"] - margin]
    summary["peak_mib"] = round(max(memory), 1) if memory else None
    summary["steady_mib"] = (round(sorted(before)[len(before) // 2], 1)
                              if before else None)
    summary["memory_metric"] = "whole_device_used_minus_launch_baseline"
    summary["steady_window"] = ("one_second_before_timed_region" if start is not None
                                  and any(start - 1.0 <= sample["unix"] < start
                                          for sample in monitor.telemetry_samples)
                                  else "middle_half_process_fallback")
    return summary


def combine_phase_instruments(phases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(phases.values())
    combined: dict[str, Any] = {"phases": phases, "memory_metric": "phase_union"}
    for key in ("peak_mib", "steady_mib"):
        observed = [float(value[key]) for value in values
                    if isinstance(value.get(key), (int, float))]
        combined[key] = max(observed) if observed else None
    energies = [float(value["energy_joules"]) for value in values
                if isinstance(value.get("energy_joules"), (int, float))]
    combined["energy_joules"] = sum(energies) if len(energies) == len(values) else None
    powers = [float(value["mean_watts"]) for value in values
              if isinstance(value.get("mean_watts"), (int, float))]
    combined["mean_watts"] = sum(powers) / len(powers) if powers else None
    combined["energy_measurable"] = all(value.get("energy_measurable") for value in values)
    telemetry = [value.get("telemetry", {}) for value in values]
    def extrema(key: str, function) -> float | None:
        observed = [float(value[key]) for value in telemetry
                    if isinstance(value.get(key), (int, float))]
        return function(observed) if observed else None
    combined["telemetry"] = {
        "max_temperature_c": extrema("max_temperature_c", max),
        "max_utilization_percent": extrema("max_utilization_percent", max),
        "min_sm_clock_mhz": extrema("min_sm_clock_mhz", min),
        "max_sm_clock_mhz": extrema("max_sm_clock_mhz", max),
        "min_memory_clock_mhz": extrema("min_memory_clock_mhz", min),
        "max_memory_clock_mhz": extrema("max_memory_clock_mhz", max),
        "power_throttled": any(value.get("power_throttled") for value in telemetry),
        "thermal_throttled": any(value.get("thermal_throttled") for value in telemetry),
    }
    return combined


def process_json(command: list[str], environment: dict[str, str] | None = None,
                 timeout: int = 14400) -> tuple[dict[str, Any], dict[str, Any], str]:
    with Monitor(device="cuda") as monitor:
        process = subprocess.run(command, capture_output=True, text=True,
                                 env=environment, timeout=timeout,
                                 creationflags=CREATE_NO_WINDOW)
    if process.returncode:
        raise RuntimeError(f"{' '.join(command)} failed ({process.returncode}):\n"
                           f"{process.stderr[-3000:]}")
    payload = last_json(process.stdout)
    start = payload.get("timed_start_unix") if isinstance(payload, dict) else None
    end = payload.get("timed_end_unix") if isinstance(payload, dict) else None
    return payload, monitor_window(monitor, start, end), process.stderr[-3000:]


def tokenize(opennn: Path, model_dir: Path, content_file: Path) -> dict[str, Any]:
    completed = subprocess.run([str(opennn), "tokens", str(model_dir), str(content_file)],
                               capture_output=True, text=True, timeout=180,
                               creationflags=CREATE_NO_WINDOW)
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    return last_json(completed.stdout)


def fixture(opennn: Path, model_dir: Path, target: int) -> tuple[Path, dict[str, Any]]:
    directory = QWEN_ROOT / "fixtures"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"prompt-{target}.txt"
    metadata_path = path.with_suffix(".json")
    if path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        observed = tokenize(opennn, model_dir, path)
        if observed["prompt_tokens"] == target and observed["token_hash"] == metadata["token_hash"]:
            return path, metadata

    instruction = (
        "Write a continuous, detailed technical explanation of GPU language-model "
        "inference. Do not conclude, summarize, or stop before the token limit."
    )
    scratch = directory / f"prompt-{target}.candidate"

    def count(content: str) -> dict[str, Any]:
        scratch.write_text(content, encoding="utf-8")
        return tokenize(opennn, model_dir, scratch)

    initial = count(instruction)
    if initial["prompt_tokens"] > target:
        raise RuntimeError(f"fixture instruction already needs {initial['prompt_tokens']} tokens")
    # A leading-space English word is one Qwen BPE token.  Verify rather than
    # assuming that remains true for a future tokenizer revision.
    content = instruction + " benchmark" * (target - initial["prompt_tokens"])
    observed = count(content)
    while observed["prompt_tokens"] > target:
        content = content.rsplit(" benchmark", 1)[0]
        observed = count(content)
    while observed["prompt_tokens"] < target:
        content += " x"
        observed = count(content)
        if observed["prompt_tokens"] > target:
            raise RuntimeError("could not construct an exact-token Qwen fixture")

    path.write_text(content, encoding="utf-8")
    scratch.unlink(missing_ok=True)
    metadata = {
        "schema_version": 1,
        "target_prompt_tokens": target,
        "prompt_tokens": observed["prompt_tokens"],
        "token_hash": observed["token_hash"],
        "token_ids": observed["token_ids"],
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path, metadata


def run_opennn_core(executable: Path, models: dict[str, Path], prompt: int,
                    generated: int, repeats: int) -> tuple[dict[str, Any], dict[str, Any]]:
    pre_touch(models["opennn"] / "qwen3_bf16.bin")
    payload, instruments, _ = process_json([
        str(executable), "core", str(models["opennn"]), str(prompt),
        str(generated), str(repeats), str(prompt + generated),
    ])
    return payload, instruments


def run_llama_core(executable: Path, models: dict[str, Path], prompt: int,
                   generated: int, repeats: int) -> tuple[dict[str, Any], dict[str, Any]]:
    pre_touch(models["gguf"])
    command = [
        str(executable), "-m", str(models["gguf"]),
        "-p", str(prompt), "-n", str(generated), "-b", "2048", "-ub", "512",
        "-ngl", "99", "-fa", "on", "-ctk", "bf16", "-ctv", "bf16",
        "-r", str(repeats), "-o", "json",
    ]
    environment = dict(os.environ)
    environment["OPENNN_BENCH_CTX_SIZE"] = str(prompt + generated)
    with Monitor(device="cuda") as monitor:
        started = time.time()
        completed = subprocess.run(command, capture_output=True, text=True,
                                   env=environment,
                                   timeout=14400, creationflags=CREATE_NO_WINDOW)
        ended = time.time()
    if completed.returncode:
        raise RuntimeError(f"llama-bench failed:\n{completed.stderr[-3000:]}")
    parsed = last_json(completed.stdout)
    samples, rows = llama_bench_samples(parsed, prompt, generated)
    full_gpu = all("CUDA" in str(row.get("backends", ""))
                   and int(row.get("n_gpu_layers", 0)) >= 37 for row in rows)
    exact_configuration = all(
        str(row.get("type_k", "")).lower() == "bf16"
        and str(row.get("type_v", "")).lower() == "bf16"
        and int(row.get("flash_attn", 0)) == 1
        and int(row.get("n_batch", 0)) == 2048
        and int(row.get("n_ubatch", 0)) == 512
        for row in rows)
    result = {
        "schema_version": 1,
        "engine": "llama_cpp",
        "track": "core",
        "precision": "bf16",
        "kv_precision": "bf16",
        "prompt_tokens": prompt,
        "generated_tokens": generated,
        "context_tokens": prompt + generated,
        "batch": 1,
        "logical_parameters": manifest()["logical_parameters"],
        "context_capacity": prompt + generated,
        "context_capacity_instrumentation": "OPENNN_BENCH_CTX_SIZE",
        "full_gpu_offload": full_gpu,
        "configuration_gate": exact_configuration,
        "samples": samples,
        "raw_rows": rows,
        "command": command,
        "timed_window_note": "throughput uses llama-bench kernel samples; telemetry covers its process",
        "stderr_tail": completed.stderr[-3000:],
    }
    prefill_row = next(row for row in rows if int(row.get("n_prompt", -1)) == prompt
                       and int(row.get("n_gen", 0)) == 0)
    decode_row = next(row for row in rows if int(row.get("n_prompt", 0)) == 0
                      and int(row.get("n_gen", -1)) == generated)
    phases = {
        "prefill": monitor_window(monitor, float(prefill_row["timed_start_unix"]),
                                  float(prefill_row["timed_end_unix"])),
        "decode": monitor_window(monitor, float(decode_row["timed_start_unix"]),
                                 float(decode_row["timed_end_unix"])),
    }
    instruments = combine_phase_instruments(phases)
    result["phase_instruments"] = phases
    result["process_window_unix"] = {"start": started, "end": ended}
    return result, instruments


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def json_request(url: str, payload: dict[str, Any] | None = None,
                 timeout: float = 600.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_server(url: str, process: subprocess.Popen, timeout: float = 180.0) -> float:
    begin = time.perf_counter()
    deadline = begin + timeout
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2):
                return (time.perf_counter() - begin) * 1000.0
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    raise RuntimeError(f"server did not become ready at {url}")


def stream_request(url: str, payload: dict[str, Any], engine: str) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    begin_unix = time.time()
    begin = time.perf_counter()
    first: float | None = None
    last: float | None = None
    pieces: list[str] = []
    final: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.strip()
            if line.startswith(b"data: "):
                line = line[6:]
            if not line or line == b"[DONE]":
                continue
            item = json.loads(line)
            piece = item.get("response", "") if engine == "ollama" else item.get("content", "")
            if piece:
                now = time.perf_counter()
                first = first or now
                last = now
                pieces.append(piece)
            final = item
    end = time.perf_counter()

    if engine == "ollama":
        prompt_tokens = int(final.get("prompt_eval_count", 0))
        generated_tokens = int(final.get("eval_count", 0))
        prefill_ms = float(final.get("prompt_eval_duration", 0)) / 1e6
        decode_ms = float(final.get("eval_duration", 0)) / 1e6
        load_ms = float(final.get("load_duration", 0)) / 1e6
        finish = final.get("done_reason", "unknown")
    else:
        timings = final.get("timings", {})
        prompt_tokens = int(timings.get("prompt_n", 0))
        generated_tokens = int(timings.get("predicted_n", 0))
        prefill_ms = float(timings.get("prompt_ms", 0))
        decode_ms = float(timings.get("predicted_ms", 0))
        load_ms = 0.0
        finish = str(final.get("stop_type", final.get("stop", "unknown")))

    text = "".join(pieces)
    total_ms = (end - begin) * 1000.0
    ttft_ms = ((first - begin) * 1000.0) if first is not None else total_ms
    decode_steps = max(0, generated_tokens - 1)
    client_decode_ms = ((last - first) * 1000.0
                        if first is not None and last is not None else 0.0)
    return {
        "timed_start_unix": begin_unix,
        "first_token_unix": (begin_unix + first - begin if first is not None else None),
        "last_token_unix": (begin_unix + last - begin if last is not None else None),
        "timed_end_unix": begin_unix + end - begin,
        "total_ms": total_ms,
        "ttft_ms": ttft_ms,
        "client_decode_ms": client_decode_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "model_load_ms": load_ms,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "finish_reason": finish,
        "output_text": text,
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "prefill_tokens_per_second": (1000.0 * prompt_tokens / prefill_ms
                                       if prefill_ms > 0 else None),
        "decode_tokens_per_second": (1000.0 * decode_steps / client_decode_ms
                                      if client_decode_ms > 0 else None),
        "native_decode_tokens_per_second": (1000.0 * generated_tokens / decode_ms
                                             if decode_ms > 0 else None),
        "end_to_end_tokens_per_second": (1000.0 * generated_tokens / total_ms
                                          if total_ms > 0 else None),
        "native_final": final,
    }


@contextmanager
def server_process(command: list[str], environment: dict[str, str], log_path: Path
                   ) -> Iterator[tuple[subprocess.Popen, Any]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w+", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               env=environment, creationflags=CREATE_NO_WINDOW)
    try:
        yield process, log
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        log.flush()
        log.close()


def server_runtime(engine: str, executable: Path, models: dict[str, Path],
                   content: str, metadata: dict[str, Any], prompt: int,
                   generated: int, repeats: int, round_index: int
                   ) -> tuple[dict[str, Any], dict[str, Any]]:
    port = free_port()
    context = prompt + generated
    environment = dict(os.environ)
    logs = QWEN_ROOT / "logs" / f"{engine}-{prompt}-r{round_index}.log"
    model_name = "opennn-qwen3-4b-bf16"

    if engine == "llama_cpp":
        command = [
            str(executable), "--model", str(models["gguf"]), "--host", "127.0.0.1",
            "--port", str(port), "--ctx-size", str(context), "--parallel", "1",
            "--n-gpu-layers", "99", "--flash-attn", "on",
            "--cache-type-k", "bf16", "--cache-type-v", "bf16",
            "--batch-size", "2048", "--ubatch-size", "512",
            "--kv-unified", "--no-context-shift",
        ]
        health = f"http://127.0.0.1:{port}/health"
        endpoint = f"http://127.0.0.1:{port}/completion"
    else:
        host = f"127.0.0.1:{port}"
        environment.update({
            "OLLAMA_HOST": host,
            "OLLAMA_MODELS": str(QWEN_ROOT / "models" / "ollama-store"),
            "OLLAMA_FLASH_ATTENTION": "1",
            "OLLAMA_KV_CACHE_TYPE": "f16",
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_NOPRUNE": "1",
        })
        command = [str(executable), "serve"]
        health = f"http://{host}/api/tags"
        endpoint = f"http://{host}/api/generate"

    pre_touch(models["gguf"])
    with Monitor(device="cuda") as monitor:
        with server_process(command, environment, logs) as (process, log):
            runtime_start_ms = wait_server(health, process)
            if engine == "ollama":
                listed = subprocess.run([str(executable), "list"], env=environment,
                                        capture_output=True, text=True,
                                        creationflags=CREATE_NO_WINDOW)
                if model_name not in listed.stdout:
                    modelfile = QWEN_ROOT / "models" / "Modelfile.qwen3-bf16"
                    modelfile.write_text(f"FROM {models['gguf']}\n", encoding="utf-8")
                    created = subprocess.run(
                        [str(executable), "create", model_name, "-f", str(modelfile)],
                        env=environment, capture_output=True, text=True, timeout=1800,
                        creationflags=CREATE_NO_WINDOW)
                    if created.returncode:
                        raise RuntimeError(f"ollama create failed: {created.stderr}")

            rendered = raw_chatml(content)
            if engine == "llama_cpp":
                tokenized = json_request(f"http://127.0.0.1:{port}/tokenize",
                                         {"content": rendered, "add_special": False})
                llama_ids = [int(item.get("id", item)) if isinstance(item, dict) else int(item)
                             for item in tokenized.get("tokens", tokenized)]
                expected = [int(token) - 1 for token in metadata["token_ids"]]
                token_gate = llama_ids == expected
                request_payload = {
                    "prompt": rendered, "n_predict": generated, "temperature": 0.0,
                    "top_k": 0, "top_p": 1.0, "repeat_penalty": 1.0,
                    "seed": 42, "stream": True, "cache_prompt": False,
                }
            else:
                llama_ids = []
                token_gate = True
                request_payload = {
                    "model": model_name, "prompt": rendered, "raw": True,
                    "stream": True, "keep_alive": "30m",
                    "options": {"num_ctx": context, "num_predict": generated,
                                "temperature": 0.0, "top_k": 0, "top_p": 1.0,
                                "repeat_penalty": 1.0, "seed": 42},
                }

            warm_payload = dict(request_payload)
            warm_payload["prompt"] = raw_chatml("Warm up this model without reusing the measured prompt.")
            if engine == "llama_cpp":
                warm_payload["n_predict"] = 1
            else:
                warm_payload["options"] = dict(request_payload["options"], num_predict=1)
            model_ready_begin = time.perf_counter()
            warm = stream_request(endpoint, warm_payload, engine)
            model_ready_ms = (time.perf_counter() - model_ready_begin) * 1000.0
            warmups = [(warm, model_ready_ms)]

            start_unix = time.time()
            samples = []
            for repeat in range(repeats):
                samples.append(stream_request(endpoint, request_payload, engine))
                if engine == "ollama" and repeat + 1 < repeats:
                    json_request(endpoint, {"model": model_name, "keep_alive": 0,
                                            "stream": False})
                    model_ready_begin = time.perf_counter()
                    next_warm = stream_request(endpoint, warm_payload, engine)
                    warmups.append((next_warm,
                                    (time.perf_counter() - model_ready_begin) * 1000.0))
            end_unix = time.time()

            full_gpu = True
            offload_evidence = "requested full offload"
            if engine == "ollama":
                ps = subprocess.run([str(executable), "ps"], env=environment,
                                    capture_output=True, text=True,
                                    creationflags=CREATE_NO_WINDOW)
                offload_evidence = ps.stdout.strip()
                full_gpu = "100% GPU" in ps.stdout.upper()
                try:
                    json_request(endpoint, {"model": model_name, "keep_alive": 0,
                                            "stream": False})
                except Exception:
                    pass

            log.flush()
            log.seek(0)
            log_content = log.read()
            log_tail = log_content[-6000:]

    request_windows = {
        f"request_{index + 1}": monitor_window(
            monitor, float(sample["timed_start_unix"]), float(sample["timed_end_unix"]))
        for index, sample in enumerate(samples)
    }
    instruments = combine_phase_instruments(request_windows)
    if engine == "llama_cpp":
        lowered = log_content.lower()
        full_gpu = ("offloaded 37/37 layers" in lowered
                    or "offloaded 37 repeating layers" in lowered
                    or "cuda0 model buffer" in lowered
                    or float(instruments.get("peak_mib") or 0) >= 7000.0)
        offload_evidence = "full CUDA model buffer/offload marker" if full_gpu else log_tail[-1500:]
    if engine == "ollama":
        token_gate = all(sample["prompt_tokens"] == prompt for sample in samples)
    allocated_contexts = sorted({int(value) for value in re.findall(
        r"(?:n_ctx|n_ctx_slot)\s*=\s*(\d+)", log_content)})
    context_gate = bool(allocated_contexts and min(allocated_contexts) >= context)
    context_exact = context in allocated_contexts
    native_load_values = [float(item.get("model_load_ms") or 0.0)
                          for item, _ in warmups]
    native_load_ms = float(sample_statistics(native_load_values)["median"] or 0.0)
    if engine == "llama_cpp":
        comparable_load_ms = runtime_start_ms
    else:
        inferred_load_values = [
            max(0.0, elapsed - float(item.get("prefill_ms") or 0.0)
                - float(item.get("decode_ms") or 0.0))
            for item, elapsed in warmups]
        inferred_load_ms = float(sample_statistics(inferred_load_values)["median"] or 0.0)
        comparable_load_ms = max(native_load_ms, inferred_load_ms)

    result = {
        "schema_version": 1,
        "engine": engine,
        "track": "runtime",
        "precision": "bf16",
        "kv_precision": "bf16" if engine == "llama_cpp" else "f16",
        "context_tokens": context,
        "requested_generated_tokens": generated,
        "batch": 1,
        "logical_parameters": manifest()["logical_parameters"],
        "runtime_start_ms": runtime_start_ms,
        "model_load_ms": comparable_load_ms,
        "native_model_load_ms": native_load_ms,
        "model_warmup_total_ms": model_ready_ms,
        "token_gate": token_gate,
        "context_capacity_gate": context_gate,
        "context_capacity_exact": context_exact,
        "allocated_context_tokens": allocated_contexts,
        "context_padding_note": (None if context_exact else
                                 "runtime allocated a larger aligned KV capacity; request limit stayed exact"),
        "external_token_hash": hashlib.sha256(json.dumps(
            llama_ids or [int(token) - 1 for token in metadata["token_ids"]]).encode()).hexdigest(),
        "token_ids_provenance": ("llama-server /tokenize" if llama_ids
                                 else "same pinned GGUF and raw bytes as llama-server"),
        "full_gpu_offload": full_gpu,
        "offload_evidence": offload_evidence,
        "samples": samples,
        "command": command,
        "log": str(logs),
        "log_tail": log_tail,
    }
    return result, instruments


def run_opennn_runtime(executable: Path, models: dict[str, Path], content_path: Path,
                       generated: int, repeats: int, context: int
                       ) -> tuple[dict[str, Any], dict[str, Any]]:
    pre_touch(models["opennn"] / "qwen3_bf16.bin")
    payload, instruments, _ = process_json([
        str(executable), "runtime", str(models["opennn"]), str(content_path),
        str(generated), str(repeats), str(context),
    ])
    payload["token_gate"] = all(sample["prompt_tokens"] == context - generated
                                for sample in payload["samples"])
    payload["full_gpu_offload"] = True
    return payload, instruments


def launch_valid(result: dict[str, Any], instruments: dict[str, Any],
                 gate: dict[str, Any], generated: int) -> tuple[bool, list[str]]:
    reasons = list(gate["reasons"])
    if result.get("logical_parameters") != manifest()["logical_parameters"]:
        reasons.append("logical parameter count mismatch")
    if result.get("track") == "runtime":
        if not result.get("token_gate"):
            reasons.append("prompt token gate failed")
        if result.get("engine") != "opennn" and not result.get("context_capacity_gate"):
            reasons.append("runtime did not report the exact requested context capacity")
        if not result.get("full_gpu_offload"):
            reasons.append("full GPU offload was not verified")
        for sample in result.get("samples", []):
            if sample.get("generated_tokens") != generated:
                reasons.append("runtime stopped before the requested token count")
                break
    if result.get("track") == "core" and result.get("engine") == "opennn":
        if not result.get("cuda_graph"):
            reasons.append("OpenNN CUDA graph was not captured")
    if result.get("track") == "core" and result.get("engine") == "llama_cpp":
        if not result.get("full_gpu_offload"):
            reasons.append("llama-bench full CUDA offload was not verified")
        if not result.get("configuration_gate"):
            reasons.append("llama-bench did not report the requested BF16 KV/Flash Attention configuration")
    telemetry = instruments.get("telemetry", {})
    if telemetry.get("max_temperature_c") is not None \
            and telemetry["max_temperature_c"] > MAX_TEMPERATURE:
        reasons.append("GPU exceeded the 45 C temperature gate during the cell")
    if telemetry.get("power_throttled"):
        reasons.append("GPU power-cap throttling was active during the cell")
    if telemetry.get("thermal_throttled"):
        reasons.append("GPU thermal throttling was active during the cell")
    for key, target in (("min_sm_clock_mhz", TARGET_SM_CLOCK),
                        ("max_sm_clock_mhz", TARGET_SM_CLOCK),
                        ("min_memory_clock_mhz", TARGET_MEMORY_CLOCK),
                        ("max_memory_clock_mhz", TARGET_MEMORY_CLOCK)):
        observed = telemetry.get(key)
        if observed is not None and abs(observed - target) > CLOCK_STEP:
            reasons.append(f"{key}={observed} differs from locked target {target}")
    return not reasons, reasons


def aggregate(launches: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metrics = ("prefill_tokens_per_second", "decode_tokens_per_second",
               "ttft_ms", "end_to_end_tokens_per_second", "model_load_ms")
    keys = sorted({(item["track"], item["engine"], item["prompt_tokens"])
                   for item in launches})
    for track, engine, prompt in keys:
        selected = [item for item in launches
                    if (item["track"], item["engine"], item["prompt_tokens"])
                    == (track, engine, prompt)]
        entry: dict[str, Any] = {}
        for metric in metrics:
            if metric == "model_load_ms":
                values = [launch["result"][metric] for launch in selected
                          if isinstance(launch["result"].get(metric), (int, float))]
            else:
                values = [sample[metric] for launch in selected
                          for sample in launch["result"].get("samples", [])
                          if isinstance(sample.get(metric), (int, float))]
            if values:
                entry[metric] = sample_statistics(values)
        for metric in ("peak_mib", "steady_mib", "mean_watts", "energy_joules"):
            values = [launch["instruments"][metric] for launch in selected
                      if isinstance(launch.get("instruments", {}).get(metric), (int, float))]
            if values:
                entry[metric] = sample_statistics(values)
        hashes = [sample.get("output_token_hash", sample.get("output_sha256"))
                  for launch in selected for sample in launch["result"].get("samples", [])]
        hashes = [value for value in hashes if value]
        entry["within_engine_deterministic"] = len(set(hashes)) <= 1
        unstable = [metric for metric in metrics
                    if isinstance(entry.get(metric, {}).get("cv"), (int, float))
                    and entry[metric]["count"] > 1 and entry[metric]["cv"] > MAX_CV]
        entry["unstable_metrics"] = unstable
        entry["valid"] = (all(launch["valid"] for launch in selected)
                          and not unstable and entry["within_engine_deterministic"])
        entry["rounds"] = len(selected)
        result[f"{track}:{engine}:{prompt}"] = entry
    return result


def compare_runtime_outputs(launches: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    prompts = sorted({launch["prompt_tokens"] for launch in launches
                      if launch["track"] == "runtime"})
    for prompt in prompts:
        outputs: dict[str, str] = {}
        hashes: dict[str, str] = {}
        engine_outputs: dict[str, list[str]] = {}
        for launch in launches:
            if launch["track"] != "runtime" or launch["prompt_tokens"] != prompt:
                continue
            samples = launch["result"].get("samples", [])
            if not samples:
                continue
            engine = launch["engine"]
            captured = [sample.get("output_text") for sample in samples
                        if isinstance(sample.get("output_text"), str)]
            engine_outputs.setdefault(engine, []).extend(captured)
            if engine not in outputs and captured:
                outputs[engine] = captured[0]
                hashes[engine] = str(
                    samples[0].get("output_token_hash", samples[0].get("output_sha256", "")))
        prefix = common_prefix(outputs.values())
        all_equal = len(set(outputs.values())) <= 1
        variants = {engine: len(set(captured))
                    for engine, captured in engine_outputs.items()}
        comparisons[str(prompt)] = {
            "engines": sorted(outputs),
            "output_hashes": hashes,
            "output_variants_within_engine": variants,
            "within_engine_deterministic": {
                engine: count <= 1 for engine, count in variants.items()
            },
            "common_greedy_prefix_characters": prefix,
            "first_divergence": {
                engine: text[prefix:prefix + 32] for engine, text in outputs.items()
            } if len(outputs) > 1 and not all_equal else {},
            "all_outputs_equal": all_equal,
            "performance_invalidated_by_divergence": False,
        }
    return comparisons


def write_csv(path: Path, launches: list[dict[str, Any]]) -> None:
    fields = ["track", "engine", "prompt_tokens", "generated_tokens", "round",
              "sample", "valid", "prefill_tokens_per_second",
              "decode_tokens_per_second", "ttft_ms", "end_to_end_tokens_per_second",
              "model_load_ms", "peak_mib", "steady_mib", "mean_watts", "energy_joules"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for launch in launches:
            for index, sample in enumerate(launch["result"].get("samples", []), 1):
                writer.writerow({
                    "track": launch["track"], "engine": launch["engine"],
                    "prompt_tokens": launch["prompt_tokens"],
                    "generated_tokens": launch["generated_tokens"],
                    "round": launch["round"], "sample": index,
                    "valid": launch["valid"],
                    **{name: sample.get(name) for name in fields if name in sample},
                    "model_load_ms": launch["result"].get("model_load_ms"),
                    **{name: launch["instruments"].get(name)
                       for name in ("peak_mib", "steady_mib", "mean_watts", "energy_joules")},
                })


def write_curves_svg(path: Path, summary: dict[str, Any]) -> None:
    prompts = sorted({int(key.rsplit(":", 1)[1]) for key in summary})
    engines = ("opennn", "llama_cpp", "ollama")
    colors = {"opennn": "#0b84f3", "llama_cpp": "#f59e0b", "ollama": "#10b981"}
    panels = [
        ("core", "prefill_tokens_per_second", "Core prefill tok/s"),
        ("core", "decode_tokens_per_second", "Core decode tok/s"),
        ("runtime", "prefill_tokens_per_second", "Runtime prefill tok/s"),
        ("runtime", "decode_tokens_per_second", "Runtime decode tok/s"),
    ]
    width, height = 1000, 660
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
              f'viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<style>text{font:13px Segoe UI,Arial;fill:#263238}.title{font-size:17px;font-weight:600}'
              '.axis{stroke:#90a4ae;stroke-width:1}.grid{stroke:#eceff1;stroke-width:1}</style>']
    for panel_index, (track, metric, title) in enumerate(panels):
        column, row = panel_index % 2, panel_index // 2
        left, top = 60 + column * 500, 55 + row * 310
        plot_width, plot_height = 400, 220
        values = [float(summary.get(f"{track}:{engine}:{prompt}", {})
                        .get(metric, {}).get("median"))
                  for engine in engines for prompt in prompts
                  if isinstance(summary.get(f"{track}:{engine}:{prompt}", {})
                                .get(metric, {}).get("median"), (int, float))]
        maximum = max(values, default=1.0) * 1.08
        chunks.append(f'<text class="title" x="{left}" y="{top - 18}">{title}</text>')
        for tick in range(5):
            y = top + plot_height - tick * plot_height / 4
            value = maximum * tick / 4
            chunks.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" '
                          f'x2="{left + plot_width}" y2="{y:.1f}"/>')
            chunks.append(f'<text text-anchor="end" x="{left - 7}" y="{y + 4:.1f}">{value:.0f}</text>')
        chunks.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>')
        chunks.append(f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
                      f'x2="{left + plot_width}" y2="{top + plot_height}"/>')
        for index, prompt in enumerate(prompts):
            x = left + (plot_width * index / max(1, len(prompts) - 1))
            chunks.append(f'<text text-anchor="middle" x="{x:.1f}" y="{top + plot_height + 20}">{prompt}</text>')
        for engine in engines:
            points = []
            for index, prompt in enumerate(prompts):
                observed = summary.get(f"{track}:{engine}:{prompt}", {}).get(metric, {}).get("median")
                if not isinstance(observed, (int, float)):
                    continue
                x = left + (plot_width * index / max(1, len(prompts) - 1))
                y = top + plot_height * (1.0 - float(observed) / maximum)
                points.append((x, y))
            if points:
                encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                chunks.append(f'<polyline fill="none" stroke="{colors[engine]}" stroke-width="2.5" points="{encoded}"/>')
                chunks.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{colors[engine]}"/>'
                              for x, y in points)
        if panel_index == 0:
            for offset, engine in enumerate(engines):
                chunks.append(f'<rect x="{left + 190 + offset * 85}" y="{top - 31}" width="10" height="10" fill="{colors[engine]}"/>')
                chunks.append(f'<text x="{left + 204 + offset * 85}" y="{top - 22}">{engine}</text>')
    chunks.append('</svg>')
    path.write_text("\n".join(chunks), encoding="utf-8")


def markdown_report(artifact: dict[str, Any]) -> str:
    summary = artifact["summary"]
    valid = artifact["valid"]

    def compact(value: dict[str, Any] | None, digits: int = 2) -> str:
        if not value or value.get("median") is None:
            return "—"
        median = float(value["median"])
        low = float(value["min"])
        high = float(value["max"])
        cv = float(value.get("cv") or 0.0) * 100.0
        return (f"{median:.{digits}f} [{low:.{digits}f}–{high:.{digits}f}], "
                f"CV {cv:.1f}%")

    headline_prompt = 2048
    headline_core_open = summary.get(f"core:opennn:{headline_prompt}", {})
    headline_core_llama = summary.get(f"core:llama_cpp:{headline_prompt}", {})
    def ratio(metric: str, left: dict[str, Any], right: dict[str, Any]) -> float | None:
        numerator = left.get(metric, {}).get("median")
        denominator = right.get(metric, {}).get("median")
        if isinstance(numerator, (int, float)) and isinstance(denominator, (int, float)) \
                and denominator:
            return float(numerator) / float(denominator)
        return None

    pp_ratio = ratio("prefill_tokens_per_second", headline_core_open, headline_core_llama)
    tg_ratio = ratio("decode_tokens_per_second", headline_core_open, headline_core_llama)
    lines = [
        "# Qwen3-4B benchmark — RTX 4080 / Windows",
        "",
        "> Internal engineering result. This is not an RTX 5070 Ti result and must not be presented as one.",
        "",
        f"Overall gate: **{'PASS' if valid else 'INVALID / DIAGNOSTIC ONLY'}**",
        "",
        "## Headline: 2048 + 256",
        "",
        (f"- OpenNN/llama.cpp engine prefill ratio: **{pp_ratio:.3f}×**."
         if pp_ratio is not None else "- Engine prefill ratio: unavailable."),
        (f"- OpenNN/llama.cpp engine decode ratio: **{tg_ratio:.3f}×**."
         if tg_ratio is not None else "- Engine decode ratio: unavailable."),
        "- A failed environmental or stability gate makes these diagnostic observations, not publishable headline figures.",
        "",
        "## Throughput curves",
        "",
        f"![Qwen3-4B throughput curves]({artifact.get('curves_file', '')})",
        "",
        "## Engine core",
        "",
        "| Prompt | Engine | Prefill tok/s (median [range], CV) | Decode tok/s (median [range], CV) | OpenNN/llama ratio | VRAM steady/peak MiB | Power W | Energy J | Valid |",
        "|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for key, value in summary.items():
        track, engine, prompt = key.split(":")
        if track != "core":
            continue
        counterpart = summary.get(f"core:llama_cpp:{prompt}", {})
        ratios = []
        if engine == "opennn":
            for metric in ("prefill_tokens_per_second", "decode_tokens_per_second"):
                observed = ratio(metric, value, counterpart)
                ratios.append(f"{observed:.3f}×" if observed is not None else "—")
        ratio_text = "/".join(ratios) if ratios else "reference"
        steady = value.get("steady_mib", {}).get("median")
        peak = value.get("peak_mib", {}).get("median")
        memory = (f"{steady:.1f}/{peak:.1f}" if isinstance(steady, (int, float))
                  and isinstance(peak, (int, float)) else "—")
        lines.append(f"| {prompt} | {engine} | {compact(value.get('prefill_tokens_per_second'))} | "
                     f"{compact(value.get('decode_tokens_per_second'))} | {ratio_text} | {memory} | "
                     f"{compact(value.get('mean_watts'), 1)} | {compact(value.get('energy_joules'))} | "
                     f"{'yes' if value['valid'] else 'no'} |")

    lines += ["", "## Runtime", "",
              "| Prompt | Runtime | Load ms | TTFT ms | Prefill tok/s | Decode tok/s | E2E tok/s | VRAM steady/peak MiB | Power W | Valid |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for key, value in summary.items():
        track, engine, prompt = key.split(":")
        if track != "runtime":
            continue
        steady = value.get("steady_mib", {}).get("median")
        peak = value.get("peak_mib", {}).get("median")
        memory = (f"{steady:.1f}/{peak:.1f}" if isinstance(steady, (int, float))
                  and isinstance(peak, (int, float)) else "—")
        lines.append(f"| {prompt} | {engine} | {compact(value.get('model_load_ms'))} | "
                     f"{compact(value.get('ttft_ms'))} | {compact(value.get('prefill_tokens_per_second'))} | "
                     f"{compact(value.get('decode_tokens_per_second'))} | "
                     f"{compact(value.get('end_to_end_tokens_per_second'))} | {memory} | "
                     f"{compact(value.get('mean_watts'), 1)} | "
                     f"{'yes' if value['valid'] else 'no'} |")

    lines += ["", "## Greedy output comparison", ""]
    for prompt, comparison in artifact.get("cross_engine_output", {}).items():
        divergence = ", ".join(
            f"{engine}={text!r}" for engine, text in comparison["first_divergence"].items())
        divergence = ("none; the first captured output from every engine is identical"
                      if comparison.get("all_outputs_equal") else divergence or "not available")
        internal = comparison.get("within_engine_deterministic", {})
        unstable_engines = sorted(engine for engine, stable in internal.items() if not stable)
        internal_note = (f" Internally non-deterministic engines: {', '.join(unstable_engines)}."
                         if unstable_engines else "")
        lines.append(
            f"- {prompt}: common text prefix {comparison['common_greedy_prefix_characters']} characters; "
            f"first divergence: {divergence}. Divergence does not invalidate performance."
            f"{internal_note}")

    reasons = sorted({reason for launch in artifact["launches"]
                      for reason in launch.get("invalid_reasons", [])})
    lines += ["", "## Gates and provenance", "",
              f"- OpenNN commit: `{artifact['git']['commit']}`; dirty: `{artifact['git']['dirty']}`.",
              f"- Model validation: `{artifact['model_validation'].get('valid')}`.",
              f"- Clocks locked by harness: `{artifact['clocks_locked']}`.",
              f"- Headline cell: 2048 prompt + {artifact['configuration']['generate_tokens']} generated tokens."]
    unstable = sorted({metric for value in summary.values()
                       for metric in value.get("unstable_metrics", [])})
    if unstable:
        lines.append(f"- Metrics above the 3% CV gate: `{', '.join(unstable)}`.")
    nondeterministic = sorted(key for key, value in summary.items()
                              if not value.get("within_engine_deterministic", True))
    if nondeterministic:
        lines.append("- Internally non-deterministic output cells: "
                     f"`{', '.join(nondeterministic)}`.")
    if reasons:
        lines += ["", "Invalidating observations:"] + [f"- {reason}" for reason in reasons]
    return "\n".join(lines) + "\n"


def parse_ints(specification: str) -> list[int]:
    values = [int(part) for part in specification.split(",") if part]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def rotated_order(values: list[str], round_index: int) -> list[str]:
    offset = round_index % len(values)
    return values[offset:] + values[:offset]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="qwen", choices=("qwen",), help=argparse.SUPPRESS)
    parser.add_argument("--track", default="all", choices=("all", "core", "runtime"))
    parser.add_argument("--prompt-tokens", default="128,512,2048,8192")
    parser.add_argument("--generate-tokens", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1, choices=(1,))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--engines", default="opennn,llama_cpp,ollama")
    parser.add_argument("--label", default="")
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args(argv)

    prompts = parse_ints(args.prompt_tokens)
    if args.generate_tokens <= 0 or args.rounds <= 0 or args.repeats <= 0:
        parser.error("generate-tokens, rounds and repeats must be positive")
    requested = [name.strip() for name in args.engines.split(",") if name.strip()]
    allowed = {"opennn", "llama_cpp", "ollama"}
    if not set(requested) <= allowed:
        parser.error(f"engines must be selected from {sorted(allowed)}")

    tools = binaries()
    models = model_paths()
    validation = json.loads(models["validation"].read_text(encoding="utf-8"))
    if not validation.get("valid"):
        raise SystemExit("Qwen model validation did not pass; run prepare again")

    fixture_data = {prompt: fixture(tools["opennn"], models["opennn"], prompt)
                    for prompt in prompts}
    git = git_metadata()
    initial = query_gpu_baseline()
    initial_memory = initial.get("memory_mib")
    launches: list[dict[str, Any]] = []
    tracks = ("core", "runtime") if args.track == "all" else (args.track,)

    for round_index in range(args.rounds):
        order = rotated_order(requested, round_index)
        print(f"round {round_index + 1}: {' -> '.join(order)}", flush=True)
        for engine in order:
            for prompt in prompts:
                for track in tracks:
                    if track == "core" and engine == "ollama":
                        continue
                    gate = (environment_gate(initial_memory) if args.no_wait or not clocks_locked()
                            else wait_for_environment(initial_memory))
                    print(f"  {track:<7} {engine:<10} {prompt}+{args.generate_tokens} ... ",
                          end="", flush=True)
                    try:
                        if track == "core" and engine == "opennn":
                            result, instruments = run_opennn_core(
                                tools["opennn"], models, prompt,
                                args.generate_tokens, args.repeats)
                        elif track == "core":
                            result, instruments = run_llama_core(
                                tools["llama_bench"], models, prompt,
                                args.generate_tokens, args.repeats)
                        elif engine == "opennn":
                            result, instruments = run_opennn_runtime(
                                tools["opennn"], models, fixture_data[prompt][0],
                                args.generate_tokens, args.repeats,
                                prompt + args.generate_tokens)
                        else:
                            result, instruments = server_runtime(
                                engine, tools["llama_server" if engine == "llama_cpp" else "ollama"],
                                models, fixture_data[prompt][0].read_text(encoding="utf-8"),
                                fixture_data[prompt][1], prompt, args.generate_tokens,
                                args.repeats, round_index + 1)
                        valid, reasons = launch_valid(result, instruments, gate,
                                                      args.generate_tokens)
                        status = "OK" if valid else "diagnostic"
                        print(status)
                    except Exception as error:
                        result = {"engine": engine, "track": track, "samples": [],
                                  "error": str(error)}
                        instruments = {}
                        valid = False
                        reasons = gate["reasons"] + [str(error)]
                        print(f"FAILED: {error}")
                    launches.append({
                        "track": track, "engine": engine,
                        "prompt_tokens": prompt,
                        "generated_tokens": args.generate_tokens,
                        "round": round_index + 1,
                        "environment_gate": gate,
                        "valid": valid, "invalid_reasons": reasons,
                        "result": result, "instruments": instruments,
                    })
                    if not args.no_wait:
                        time.sleep(2.0)

    summary = aggregate(launches)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact: dict[str, Any] = {
        "schema_version": 2,
        "benchmark_id": "cuda-qwen3-4b",
        "run_id": run_id,
        "session_id": session_id(),
        "label": args.label,
        "configuration": vars(args) | {"prompt_tokens": prompts,
                                        "data_root": str(DATA_ROOT)},
        "git": git,
        "manifest": manifest(),
        "model_validation": validation,
        "machine": gpu_state(),
        "cpu": cpu_state(),
        "platform_note": "RTX 4080 / Windows only; never an RTX 5070 Ti result",
        "clocks_locked": clocks_locked(),
        "launches": launches,
        "summary": summary,
        "cross_engine_output": compare_runtime_outputs(launches),
    }
    artifact["valid"] = bool(validation.get("valid") and summary
                             and all(value["valid"] for value in summary.values()))
    destination = result_destination(git.get("dirty"), "cuda", not artifact["valid"])
    stem = f"cuda-qwen3-4b{'-' + args.label if args.label else ''}-{run_id}"
    json_path = destination / f"{stem}.json"
    csv_path = destination / f"{stem}.csv"
    report_path = destination / f"{stem}.md"
    curves_path = destination / f"{stem}-curves.svg"
    artifact["curves_file"] = curves_path.name
    json_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    write_csv(csv_path, launches)
    write_curves_svg(curves_path, summary)
    report_path.write_text(markdown_report(artifact), encoding="utf-8")
    print(f"wrote {json_path}\nwrote {csv_path}\nwrote {report_path}\nwrote {curves_path}")
    return 0 if artifact["valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
