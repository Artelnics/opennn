#!/usr/bin/env python3
"""Publication-grade HIGGS dense GPU inference-speed harness.

Runs the OpenNN, PyTorch, and TensorFlow HIGGS dense forward-only benchmark
programs and writes an immutable result JSON under docs/benchmarks/results.

This is the inference-speed twin of run_higgs_dense.py: the same dataset
contract (28 -> hidden -> hidden -> 1, ReLU, sigmoid) and the same metadata /
versions / git capture, but each engine runs forward-only over the test split
and reports throughput (samples_per_sec) and per-batch latency (ms_per_batch)
instead of training epochs.

Example:

    python run_higgs_infer.py \
      --test "$OPENNN_BENCH_DATA/higgs/higgs_test.csv" \
      --batch 8192 --runs 5 --precision both
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RESULTS_DIR = (HERE.parent.parent / "results").resolve()
DEFAULT_BENCH_DATA = Path(
    os.environ.get("OPENNN_BENCH_DATA", str(Path.home() / "opennn-benchmark-data"))
)
DEFAULT_HIGGS_DIR = DEFAULT_BENCH_DATA / "higgs"
DEFAULT_TEST = DEFAULT_HIGGS_DIR / "higgs_test.csv"
PY = os.environ.get("BENCH_PYTHON", sys.executable)
KEY_VALUE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.+)$")


def run_text(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except Exception:
        return ""


def repo_root() -> Path:
    root = run_text(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"])
    return Path(root).resolve() if root else HERE.parents[3]


REPO_ROOT = repo_root()


def git_metadata() -> dict[str, Any]:
    commit = run_text(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    branch = run_text(["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"])
    status = run_text(["git", "-C", str(REPO_ROOT), "status", "--short"])
    return {
        "commit": commit or "unknown",
        "branch": branch or "unknown",
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def candidate_names(base: str) -> list[str]:
    return [base + ".exe", base] if os.name == "nt" else [base, base + ".exe"]


def find_opennn_higgs_infer() -> tuple[str, bool]:
    for env_name in ("OPENNN_HIGGS_INFER_BIN", "OPENNN_BIN"):
        override = os.environ.get(env_name)
        if override:
            return override, Path(override).exists()

    dirs = [
        HERE,
        REPO_ROOT / "build" / "bin",
        REPO_ROOT / "build" / "bin" / "Release",
        REPO_ROOT / "build" / "bin" / "RelWithDebInfo",
        REPO_ROOT / "build-gpu" / "bin",
        REPO_ROOT / "build-gpu" / "bin" / "Release",
        REPO_ROOT / "build-cuda" / "bin",
        REPO_ROOT / "build-cuda" / "bin" / "Release",
        REPO_ROOT / "build-benchmarks" / "bin",
        REPO_ROOT / "build-benchmarks" / "bin" / "Release",
        Path.home() / "opennn-wsl" / "build-gpu" / "bin",
    ]
    for directory in dirs:
        for name in candidate_names("opennn_higgs_infer"):
            candidate = directory / name
            if candidate.exists():
                return str(candidate), True

    fallback = REPO_ROOT / "build-benchmarks" / "bin" / candidate_names("opennn_higgs_infer")[0]
    return str(fallback), False


OPENNN_BIN, OPENNN_BIN_FOUND = find_opennn_higgs_infer()


def python_json(py: str, code: str) -> dict[str, Any]:
    try:
        out = subprocess.run([py, "-c", code], capture_output=True, text=True, check=False)
        lines = [line for line in out.stdout.splitlines() if line.strip()]
        return json.loads(lines[-1]) if lines else {}
    except Exception:
        return {}


def framework_versions() -> dict[str, Any]:
    code = r"""
import json
import platform
import sys

info = {
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
    "platform": platform.platform(),
}
try:
    import torch
    info["torch"] = torch.__version__
    info["torch_cuda"] = getattr(torch.version, "cuda", None)
    info["torch_cudnn"] = torch.backends.cudnn.version()
except Exception as exc:
    info["torch_error"] = str(exc)
try:
    import tensorflow as tf
    info["tensorflow"] = tf.__version__
    try:
        info["tensorflow_build_info"] = tf.sysconfig.get_build_info()
    except Exception:
        pass
except Exception as exc:
    info["tensorflow_error"] = str(exc)
print(json.dumps(info, default=str))
"""
    info = python_json(PY, code)
    if not info:
        info = {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "framework_version_error": f"could not query {PY}",
        }

    smi_query = (
        "name,driver_version,pci.bus_id,memory.total,power.limit,"
        "clocks.max.graphics,clocks.max.memory"
    )
    gpu = run_text(["nvidia-smi", f"--query-gpu={smi_query}", "--format=csv,noheader,nounits"])
    if gpu:
        info["gpu"] = gpu
    smi = run_text(["nvidia-smi"])
    if smi:
        info["nvidia_smi"] = smi
    return info


def tensorflow_library_dirs(py: str) -> list[str]:
    override = os.environ.get("TF_NV_LIBS")
    if override:
        return [part for part in override.split(os.pathsep) if part]

    code = r"""
import json
import site
from pathlib import Path

roots = []
for base in list(site.getsitepackages()) + [site.getusersitepackages()]:
    nvidia = Path(base) / "nvidia"
    if nvidia.exists():
        roots.extend(str(path) for path in nvidia.rglob("lib") if path.is_dir())
print(json.dumps(roots))
"""
    try:
        out = subprocess.run([py, "-c", code], capture_output=True, text=True, check=False)
        lines = [line for line in out.stdout.splitlines() if line.strip()]
        return json.loads(lines[-1]) if lines else []
    except Exception:
        return []


def prepend_env_path(env: dict[str, str], key: str, values: list[str]) -> None:
    existing = os.environ.get(key, "")
    pieces = [value for value in values if value]
    if existing:
        pieces.append(existing)
    if pieces:
        env[key] = os.pathsep.join(pieces)


def parse_scalar(text: str) -> Any:
    value = text.strip()
    try:
        number = float(value)
        if not math.isfinite(number):
            return value
        if number.is_integer() and "." not in value and "e" not in value.lower():
            return int(number)
        return number
    except ValueError:
        return value


def parse_metrics(raw: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for line in raw.splitlines():
        match = KEY_VALUE.match(line.strip())
        if match:
            metrics[match.group(1)] = parse_scalar(match.group(2))
    return metrics


def numeric_values(runs: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for run in runs:
        value = run.get("metrics", {}).get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def add_stat(summary: dict[str, Any], runs: list[dict[str, Any]], key: str) -> None:
    values = numeric_values(runs, key)
    if not values:
        return
    summary[f"{key}_median"] = round(statistics.median(values), 6)
    summary[f"{key}_stdev"] = round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ok_runs = [
        run
        for run in runs
        if run.get("returncode") == 0
        and run.get("metrics", {}).get("RESULT") == "OK"
        and isinstance(run.get("metrics", {}).get("samples_per_sec"), (int, float))
    ]
    summary: dict[str, Any] = {
        "n_runs": len(runs),
        "n_ok": len(ok_runs),
        "last_metrics": runs[-1].get("metrics", {}) if runs else {},
    }

    for key in (
        "samples_per_sec",
        "ms_per_batch",
        "median_pass_s",
        "peak_vram_mb",
    ):
        add_stat(summary, ok_runs, key)

    if not ok_runs:
        summary["error"] = "no successful RESULT=OK run with samples_per_sec"
        if runs:
            raw = runs[-1].get("raw_output", "")
            summary["last_output_tail"] = raw[-1000:]
    return summary


def file_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path)}
    if path.exists():
        stat = path.stat()
        info.update({"exists": True, "bytes": stat.st_size, "mtime": stat.st_mtime})
    else:
        info["exists"] = False
    return info


def load_higgs_metadata(test_path: Path) -> dict[str, Any] | None:
    for candidate in (test_path.parent / "higgs_metadata.json", HERE.parent / "higgs" / "data" / "higgs_metadata.json"):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text())
            except Exception:
                return {"metadata_error": f"could not read {candidate}"}
    return None


def engine_cmd(
    engine: str,
    precision: str,
    args: argparse.Namespace,
    test_path: Path,
) -> tuple[list[str], dict[str, str]]:
    env: dict[str, str] = {}

    if engine == "opennn":
        cmd = [
            OPENNN_BIN,
            str(test_path),
            str(args.batch),
            str(args.runs),
            precision,
            str(args.hidden),
            str(args.hidden_layers),
            args.activation,
        ]
        wsl_cuda = os.environ.get("WSL_CUDA_LIB", "/usr/lib/wsl/lib")
        prepend_env_path(env, "LD_LIBRARY_PATH", [wsl_cuda if Path(wsl_cuda).exists() else ""])
    elif engine == "pytorch":
        cmd = [
            PY,
            str(HERE / "pytorch_higgs_infer.py"),
            str(test_path),
            str(args.batch),
            str(args.runs),
            precision,
            str(args.hidden),
            str(args.hidden_layers),
            args.activation,
        ]
    elif engine == "tensorflow":
        cmd = [
            PY,
            str(HERE / "tensorflow_higgs_infer.py"),
            str(test_path),
            str(args.batch),
            str(args.runs),
            precision,
            str(args.hidden),
            str(args.hidden_layers),
            args.activation,
        ]
        wsl_cuda = os.environ.get("WSL_CUDA_LIB", "/usr/lib/wsl/lib")
        libs = tensorflow_library_dirs(PY)
        if Path(wsl_cuda).exists():
            libs.append(wsl_cuda)
        prepend_env_path(env, "LD_LIBRARY_PATH", libs)
    else:
        raise ValueError(engine)
    return cmd, env


def display_command(cmd: list[str], env_over: dict[str, str]) -> str:
    env_bits = [f"{key}={value}" for key, value in sorted(env_over.items())]
    return " ".join(env_bits + [shlex.join(cmd)])


def run_once(cmd: list[str], env_over: dict[str, str], index: int) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(env_over)
    try:
        out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        raw = out.stdout + out.stderr
        return {
            "run_index": index,
            "returncode": out.returncode,
            "metrics": parse_metrics(raw),
            "stdout": out.stdout,
            "stderr": out.stderr,
            "raw_output": raw,
        }
    except Exception as exc:
        return {
            "run_index": index,
            "returncode": None,
            "error": str(exc),
            "metrics": {},
            "stdout": "",
            "stderr": str(exc),
            "raw_output": str(exc),
        }


def metrics_summary(results: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "samples_per_sec": {},
        "ms_per_batch": {},
        "speedup": {},
    }
    for precision, per_precision in results.items():
        for engine, summary in per_precision.items():
            label = f"{engine}_{precision}"
            if "samples_per_sec_median" in summary:
                metrics["samples_per_sec"][label] = summary["samples_per_sec_median"]
            if "ms_per_batch_median" in summary:
                metrics["ms_per_batch"][label] = summary["ms_per_batch_median"]

        opennn = per_precision.get("opennn", {}).get("samples_per_sec_median")
        if isinstance(opennn, (int, float)) and opennn > 0:
            for competitor in ("pytorch", "tensorflow"):
                value = per_precision.get(competitor, {}).get("samples_per_sec_median")
                if isinstance(value, (int, float)) and value > 0:
                    metrics["speedup"][f"{precision}_opennn_vs_{competitor}"] = round(opennn / value, 3)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test",
        default=str(DEFAULT_TEST),
        help="prepared HIGGS test CSV; defaults under $OPENNN_BENCH_DATA/higgs",
    )
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--activation", default="relu", choices=["relu", "tanh"])
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--precision", default="both", choices=["bf16", "fp32", "both"])
    parser.add_argument("--engines", default="opennn,pytorch,tensorflow")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output", default=None, help="optional explicit result JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_path = Path(args.test).resolve()

    if not test_path.exists():
        raise SystemExit(f"HIGGS test file not found: {test_path}")
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    valid_engines = {"opennn", "pytorch", "tensorflow"}
    engines = [engine.strip() for engine in args.engines.split(",") if engine.strip()]
    unknown = [engine for engine in engines if engine not in valid_engines]
    if unknown:
        raise SystemExit(f"unknown engine(s): {', '.join(unknown)}")
    if not engines:
        raise SystemExit("--engines must include at least one of: opennn,pytorch,tensorflow")

    precisions = ["bf16", "fp32"] if args.precision == "both" else [args.precision]
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output).resolve() if args.output else RESULTS_DIR / f"gpu-higgs-dense-inference-speed-{run_id}.json"

    git = git_metadata()
    result: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": "gpu-higgs-dense-inference-speed",
        "run_id": run_id,
        "git_commit": git["commit"],
        "git": git,
        "dataset": "HIGGS",
        "dataset_files": {
            "test": file_info(test_path),
            "metadata": load_higgs_metadata(test_path),
        },
        "configuration": {
            "mode": "infer",
            "device": "cuda",
            "model": "28 -> hidden layers -> 1 dense binary classifier",
            "batch": args.batch,
            "runs": args.runs,
            "hidden": args.hidden,
            "hidden_layers": args.hidden_layers,
            "activation": args.activation,
            "precision": args.precision,
            "precisions": precisions,
            "engines": engines,
            "metric": "samples_per_sec, ms_per_batch (median of runs)",
        },
        "machine": framework_versions(),
        "runner": {
            "path": os.path.relpath(__file__, REPO_ROOT),
            "cwd": os.getcwd(),
            "argv": sys.argv,
            "python": PY,
            "opennn_binary": OPENNN_BIN,
            "opennn_binary_found": OPENNN_BIN_FOUND,
        },
        "commands": {},
        "results": {},
    }

    for precision in precisions:
        print(f"\n=== HIGGS dense inference {precision} ===")
        result["results"][precision] = {}
        for engine in engines:
            cmd, env_over = engine_cmd(engine, precision, args, test_path)
            command_text = display_command(cmd, env_over)
            result["commands"][f"{engine}_{precision}"] = command_text
            print(f"  {engine}: {command_text}")

            runs = []
            for index in range(1, args.runs + 1):
                run = run_once(cmd, env_over, index)
                runs.append(run)
                sps = run.get("metrics", {}).get("samples_per_sec")
                ms = run.get("metrics", {}).get("ms_per_batch")
                status = run.get("metrics", {}).get("RESULT", "NO_RESULT")
                if isinstance(sps, (int, float)):
                    print(f"    run {index}: {sps:.0f} samples/s, {ms} ms/batch, result={status}")
                else:
                    print(f"    run {index}: failed, result={status}, returncode={run.get('returncode')}")

            summary = summarize_runs(runs)
            summary.update({
                "command": command_text,
                "argv": cmd,
                "env": env_over,
                "runs": runs,
            })
            result["results"][precision][engine] = summary

    result["metrics"] = metrics_summary(result["results"])
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
