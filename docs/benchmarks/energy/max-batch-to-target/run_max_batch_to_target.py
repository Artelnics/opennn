#!/usr/bin/env python3
"""Measure GPU energy and time from maximum training batch to a loss target.

The capacity phase is deliberately separate and immutable: pass the JSON emitted
by the HIGGS, ResNet-50, or Transformer max-batch runner.  This runner selects
each engine's own largest successful training batch, executes the matching
training path until a common training-loss target, samples board power at 20 Hz,
and keeps both the full engine log and raw power trace.
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCHMARKS = HERE.parents[1]
REPO = HERE.parents[3]
RESULTS = BENCHMARKS / "results"
CAPACITY = BENCHMARKS / "capacity"
TRANSFORMER_ENERGY = BENCHMARKS / "energy" / "transformer-energy"

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def transformer_tokens_bin(corpus, shape):
    """Return the exact OpenNN token cache shared by all three engines."""
    cache_dir = Path(str(corpus) + ".cache")
    legacy = cache_dir / "tokens.bin"
    if legacy.exists():
        return legacy

    record_bytes = (int(shape["input_seq"]) + int(shape["decoder_seq"])) * 4
    expected_bytes = (
        int(shape["samples"]) * record_bytes
        if shape.get("samples") is not None else None
    )
    candidates = [
        path for path in cache_dir.glob("*.bin")
        if path.is_file()
        and (
            path.stat().st_size == expected_bytes
            if expected_bytes is not None
            else path.stat().st_size > 0 and path.stat().st_size % record_bytes == 0
        )
    ]
    if len(candidates) != 1:
        detail = ", ".join(
            f"{path.name}:{path.stat().st_size}" for path in cache_dir.glob("*.bin")
        )
        raise ValueError(
            f"expected exactly one compatible token cache in "
            f"{cache_dir}; found [{detail}]"
        )
    return candidates[0]

def capture(command, timeout=15):
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout, check=False)
        return result.stdout.strip() or result.stderr.strip() or None
    except Exception:
        return None

def git_metadata():
    status = capture(["git", "-C", str(REPO), "status", "--short"]) or ""
    return {
        "commit": capture(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "branch": capture(["git", "-C", str(REPO), "rev-parse",
                           "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }

def gpu_metadata():
    query = ("name,driver_version,memory.total,power.limit,"
             "clocks.max.sm,compute_cap")
    raw = capture(["nvidia-smi", f"--query-gpu={query}",
                   "--format=csv,noheader,nounits"])
    return {"nvidia_smi": raw}

def framework_versions(python):
    versions = {"python": capture([python, "--version"])}
    for module in ("torch", "tensorflow", "numpy"):
        value = capture([
            python, "-c",
            f"import {module}; print({module}.__version__)"
        ], timeout=120)
        if value:
            versions[module] = value.splitlines()[-1]
    return versions

def tensorflow_library_dirs(python):
    code = (
        "import json,site\n"
        "from pathlib import Path\n"
        "roots=[]\n"
        "for base in list(site.getsitepackages())+[site.getusersitepackages()]:\n"
        " p=Path(base)/'nvidia'\n"
        " if p.exists(): roots += [str(x) for x in p.rglob('lib') if x.is_dir()]\n"
        "print(json.dumps(roots))"
    )
    raw = capture([python, "-c", code], timeout=120)
    try:
        return json.loads(raw.splitlines()[-1]) if raw else []
    except (ValueError, IndexError):
        return []

def measure_idle(seconds=5.0):
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw",
             "--format=csv,noheader,nounits", "-lms", "100"],
            capture_output=True, text=True, timeout=seconds, check=False)
        raw = proc.stdout
    except subprocess.TimeoutExpired as exc:
        raw = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    values = [float(value) for value in re.findall(r"(?m)^([0-9.]+)", raw)]
    return statistics.median(values) if values else 30.0

def cooldown(idle_w, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        raw = capture([
            "nvidia-smi", "--query-gpu=memory.used,power.draw",
            "--format=csv,noheader,nounits"
        ])
        try:
            used, watts = [float(value.strip()) for value in raw.split(",")]
            if used <= 1200 and watts <= idle_w + 12:
                return
        except (AttributeError, ValueError):
            return
        time.sleep(1)

def parse_trace(path):
    samples = []
    previous = None
    offset = 0.0
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 3:
                continue
            try:
                hms = fields[0].split(" ")[1].split(":")
                seconds = int(hms[0]) * 3600 + int(hms[1]) * 60 + float(hms[2])
                watts = float(fields[1])
                clock = float(fields[2])
            except (IndexError, ValueError):
                continue
            if previous is not None and seconds < previous - 1:
                offset += 86400
            previous = seconds
            samples.append((seconds + offset, watts, clock))
    return samples

def unix_to_trace_time(timestamp, samples):
    stamp = datetime.fromtimestamp(timestamp)
    seconds = (stamp.hour * 3600 + stamp.minute * 60 + stamp.second
               + stamp.microsecond / 1e6)
    if samples and seconds < samples[0][0] - 43200:
        seconds += 86400
    return seconds

def integrate(samples, idle_w, low, high):
    window = [sample for sample in samples if low <= sample[0] <= high]
    total_j = active_j = 0.0
    for previous, current in zip(window, window[1:]):
        dt = current[0] - previous[0]
        if 0 < dt < 2:
            total_j += 0.5 * (previous[1] + current[1]) * dt
            active_j += 0.5 * ((previous[1] - idle_w)
                              + (current[1] - idle_w)) * dt
    watts = [sample[1] for sample in window]
    clocks = [sample[2] for sample in window]
    return {
        "window_power_samples": len(window),
        "energy_total_j": round(total_j, 3),
        "energy_active_j": round(active_j, 3),
        "energy_total_wh": round(total_j / 3600, 6),
        "energy_active_wh": round(active_j / 3600, 6),
        "avg_power_w": round(statistics.mean(watts), 3) if watts else None,
        "median_sm_clock_mhz": statistics.median(clocks) if clocks else None,
    }

def marker(pattern, text, cast=float):
    match = re.search(pattern, text, re.MULTILINE)
    return cast(match.group(1)) if match else None

def capacity_batches(path, workload, precision, engines):
    with open(path, encoding="utf-8") as stream:
        artifact = json.load(stream)
    selected = {}
    if isinstance(artifact.get("results"), list):
        for engine in engines:
            rows = [
                row for row in artifact["results"]
                if row.get("engine") == engine
                and row.get("precision") == precision
                and row.get("mode") == "train"
            ]
            if not rows:
                raise ValueError(f"no {engine}/{precision}/train result in {path}")
            selected[engine] = {
                "batch": int(rows[0]["max_batch"]),
                "capacity_row": rows[0],
                "path": engine,
            }
        return selected, artifact

    if workload != "resnet50":
        raise ValueError("nested capacity JSON is only supported for ResNet-50")
    precision_rows = artifact["results"].get(precision, artifact["results"])
    aliases = {
        "opennn": ["opennn_pool1"],
        "pytorch": ["pytorch_compile", "pytorch_eager"],
        "tensorflow": ["tensorflow"],
    }
    for engine in engines:
        candidates = []
        for name in aliases[engine]:
            if name in precision_rows:
                row = precision_rows[name]
                candidates.append((int(row["max_batch"]), name, row))
        if not candidates:
            raise ValueError(f"no {engine}/{precision} result in {path}")
        batch, name, row = max(candidates, key=lambda item: item[0])
        selected[engine] = {
            "batch": batch,
            "capacity_row": row,
            "path": name,
        }
    return selected, artifact

def common_environment(args, engine):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    wsl_cuda = "/usr/lib/wsl/lib"
    libs = []
    if engine == "tensorflow":
        libs.extend(tensorflow_library_dirs(args.bench_python))
    if Path(wsl_cuda).exists():
        libs.append(wsl_cuda)
    if libs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            libs + [env.get("LD_LIBRARY_PATH", "")])
    return env

def command_for(args, engine, batch, capacity_artifact, selected_path, seed):
    env = common_environment(args, engine)
    if args.workload == "higgs":
        source = CAPACITY / "higgs-max-batch"
        env["OPENNN_BENCH_SEED"] = str(seed)
        if args.higgs_bin:
            env["OPENNN_HIGGS_BIN"] = args.higgs_bin
            env["HIGGS_BIN"] = args.higgs_bin
        if args.precision == "bf16":
            env["OPENNN_BF16"] = "1"
            env["PT_BF16"] = "1"
            env["TF_BF16"] = "1"
        if engine == "opennn":
            env["OPENNN_TARGET_LOSS"] = str(args.target)
            command = [
                args.opennn_bin, "train", str(batch), str(args.hidden),
                str(args.layers), str(args.max_steps), "cuda"
            ]
        else:
            env["TF_XLA"] = args.tf_xla
            command = [
                args.bench_python,
                str(source / f"{engine}_higgs_maxbatch.py"),
                "--mode", "train", "--batch", str(batch),
                "--hidden", str(args.hidden), "--layers", str(args.layers),
                "--steps", str(args.max_steps), "--warmup", "0",
                "--device", "cuda", "--target", str(args.target),
                "--seed", str(seed),
            ]
        return command, env

    if args.workload == "resnet50":
        source = CAPACITY / "resnet50-max-batch"
        env["OPENNN_BENCH_SEED"] = str(seed)
        total_mib = int(capacity_artifact["machine"]["gpu"]["memory_total_mib"])
        reserve_mib = int(capacity_artifact["configuration"]["vram_reserve_mib"])
        cap_mib = total_mib - reserve_mib
        if engine == "opennn":
            env["OPENNN_TARGET_LOSS"] = str(args.target)
            env["OPENNN_MAX_STEPS"] = str(args.max_steps)
            command = [
                args.opennn_bin, args.data_dir, str(batch), args.precision,
                "1", args.workspace_mib, "1",
            ]
        elif engine == "pytorch":
            path = "compile" if selected_path == "pytorch_compile" else "eager"
            command = [
                args.bench_python,
                str(source / "pytorch_resnet50_maxbatch.py"),
                "--data", args.data_dir, "--batch", str(batch),
                "--path", path, "--precision", args.precision,
                "--memory-fraction", str(cap_mib / total_mib),
                "--target", str(args.target),
                "--max-steps", str(args.max_steps),
                "--seed", str(seed),
            ]
        else:
            env["TF_XLA"] = args.tf_xla
            command = [
                args.bench_python,
                str(source / "tensorflow_resnet50_maxbatch.py"),
                "--data", args.data_dir, "--batch", str(batch),
                "--precision", args.precision,
                "--memory-limit-mb", str(cap_mib),
                "--target", str(args.target),
                "--max-steps", str(args.max_steps),
                "--seed", str(seed),
            ]
        return command, env

    shape = capacity_artifact["model"]
    tokens_bin = transformer_tokens_bin(args.corpus, shape)
    common = [
        "--tokens-bin", str(tokens_bin),
        "--in-seq", str(shape["input_seq"]),
        "--dec-seq", str(shape["decoder_seq"]),
        "--in-vocab", str(shape["input_vocab"]),
        "--out-vocab", str(shape["output_vocab"]),
        "--target", str(args.target), "--batch", str(batch),
        "--max-epochs", str(args.max_steps), "--lr", str(args.lr),
        "--d", str(shape["d_model"]), "--h", str(shape["heads"]),
        "--ff", str(shape["ff"]), "--layers", str(shape["layers"]),
        "--seed", str(seed),
    ]
    if args.precision == "bf16":
        env["OPENNN_BF16"] = "1"
        env["PT_BF16"] = "1"
        env["TF_BF16"] = "1"
    if engine == "opennn":
        env["OPENNN_GRAPH"] = "0"
        command = [
            args.opennn_bin, args.corpus, str(args.target), str(batch),
            str(args.max_steps), str(args.lr), str(shape["d_model"]),
            str(shape["heads"]), str(shape["ff"]), str(shape["layers"]),
            str(seed),
        ]
    else:
        if engine == "tensorflow":
            env["TF_XLA"] = args.tf_xla
        command = [
            args.bench_python,
            str(TRANSFORMER_ENERGY / f"{engine}_transformer_energy.py"),
            *common,
        ]
    return command, env

def run_one(args, engine, batch, capacity_artifact, selected_path,
            seed, idle_w, log_path, trace_path):
    command, env = command_for(
        args, engine, batch, capacity_artifact, selected_path, seed)
    with open(trace_path, "w", encoding="utf-8") as trace:
        logger = subprocess.Popen([
            "nvidia-smi",
            "--query-gpu=timestamp,power.draw,clocks.current.sm",
            "--format=csv,noheader,nounits", "-lms", "50",
        ], stdout=trace, stderr=subprocess.DEVNULL)
        time.sleep(0.3)
        started = time.perf_counter()
        try:
            proc = subprocess.run(command, env=env, capture_output=True,
                                  text=True, timeout=args.timeout_s,
                                  check=False)
            output = proc.stdout + proc.stderr
            return_code = proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            output = stdout + stderr
            return_code = "timeout"
        process_wall_s = time.perf_counter() - started
        time.sleep(0.3)
        logger.terminate()
        logger.wait(timeout=10)

    with open(log_path, "w", encoding="utf-8") as stream:
        stream.write("$ " + subprocess.list2cmdline(command) + "\n\n")
        stream.write(output)

    samples = parse_trace(trace_path)
    train_start = marker(r"^TRAIN_START_UNIX=([0-9.]+)", output)
    train_end = marker(r"^TRAIN_END_UNIX=([0-9.]+)", output)
    metrics = {
        "return_code": return_code,
        "batch": batch,
        "seed": seed,
        "process_wall_s": round(process_wall_s, 3),
        "train_start_unix": train_start,
        "train_end_unix": train_end,
        "train_window_s": round(train_end - train_start, 3)
                          if train_start and train_end else None,
        "steps_run": marker(r"^(?:steps_run|epochs_run|epochs)=(\d+)",
                            output, int),
        "final_error": marker(r"^final_error=([0-9.eE+-]+)", output),
        "reached_goal": marker(r"^reached_goal=(\d+)", output, int),
        "samples_per_sec": marker(r"^samples_per_sec=([0-9.eE+-]+)", output),
        "peak_vram_mib_reported": marker(
            r"^(?:peak_vram_mib|peak_vram_mb)=([0-9.eE+-]+)", output),
        "command": command,
        "log": str(log_path.relative_to(REPO)).replace("\\", "/"),
        "power_trace": str(trace_path.relative_to(REPO)).replace("\\", "/"),
    }
    history = re.search(r"^loss_history=([0-9.,eE+-]+)", output, re.MULTILINE)
    metrics["loss_history"] = (
        [float(value) for value in history.group(1).split(",")]
        if history else None
    )
    if train_start and train_end and samples:
        metrics.update(integrate(
            samples, idle_w,
            unix_to_trace_time(train_start, samples),
            unix_to_trace_time(train_end, samples),
        ))
    metrics["ok"] = (
        return_code == 0
        and "RESULT=OK" in output
        and metrics["reached_goal"] == 1
        and metrics.get("energy_total_j") is not None
    )
    if not metrics["ok"]:
        metrics["output_tail"] = output[-3000:]
    return metrics

def aggregate(per_run):
    successful = [run for run in per_run if run["ok"]]
    result = {
        "n_ok": len(successful),
        "n_total": len(per_run),
        "per_run": per_run,
    }
    for key in ("energy_total_j", "energy_active_j", "energy_total_wh",
                "energy_active_wh", "train_window_s", "avg_power_w",
                "steps_run", "samples_per_sec"):
        values = [run[key] for run in successful if run.get(key) is not None]
        if values:
            result[f"{key}_median"] = round(statistics.median(values), 6)
            result[f"{key}_stdev"] = round(
                statistics.pstdev(values), 6) if len(values) > 1 else 0.0
    return result

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--workload", required=True,
                        choices=["higgs", "resnet50", "transformer"])
    parser.add_argument("--capacity-json", required=True)
    parser.add_argument("--target", required=True, type=float)
    parser.add_argument("--max-steps", type=int, default=20,
                        help="maximum optimizer steps (HIGGS/ResNet) or "
                             "epochs (Transformer)")
    parser.add_argument("--precision", default="fp32",
                        choices=["fp32", "bf16"])
    parser.add_argument("--engines", default="opennn,pytorch,tensorflow")
    parser.add_argument("--batches", default="",
                        help="optional stable-batch overrides, e.g. "
                             "tensorflow=1562")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--bench-python",
                        default=os.environ.get("BENCH_PYTHON", sys.executable))
    parser.add_argument("--opennn-bin", required=True)
    parser.add_argument("--higgs-bin", default="")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--corpus", default="")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--workspace-mib", default="16")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tf-xla", choices=["0", "1"], default="0",
                        help="ResNet/Transformer TensorFlow XLA switch")
    parser.add_argument("--idle", type=float, default=None)
    parser.add_argument("--timeout-s", type=int, default=7200)
    parser.add_argument("--run-id", default="")
    return parser.parse_args()

def main():
    args = parse_args()
    capacity_path = Path(args.capacity_json).resolve()
    if not capacity_path.exists():
        raise SystemExit(f"capacity JSON not found: {capacity_path}")
    if not Path(args.opennn_bin).exists():
        raise SystemExit(f"OpenNN binary not found: {args.opennn_bin}")
    if args.workload == "higgs" and (
            not args.higgs_bin or not Path(args.higgs_bin).is_file()):
        raise SystemExit(f"HIGGS binary not found: {args.higgs_bin}")
    if args.workload == "resnet50" and not Path(args.data_dir).exists():
        raise SystemExit(f"CIFAR data directory not found: {args.data_dir}")
    if args.workload == "transformer" and not Path(args.corpus).exists():
        raise SystemExit(f"Transformer corpus not found: {args.corpus}")

    engines = [engine.strip() for engine in args.engines.split(",")
               if engine.strip()]
    batches, capacity_artifact = capacity_batches(
        capacity_path, args.workload, args.precision, engines)
    batch_overrides = {}
    if args.batches:
        for item in args.batches.split(","):
            engine, value = item.split("=", 1)
            batch_overrides[engine.strip()] = int(value)
        for engine, value in batch_overrides.items():
            if engine not in batches:
                raise SystemExit(f"batch override for unselected engine: {engine}")
            batches[engine]["capacity_batch"] = batches[engine]["batch"]
            batches[engine]["batch"] = value
            batches[engine]["batch_override_reason"] = (
                "explicit protocol override; use the capacity_batch and "
                "source artifact fields to retain the original candidate"
            )
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    evidence_dir = RESULTS / "evidence" / f"max-batch-to-target-{run_id}"
    evidence_dir.mkdir(parents=True, exist_ok=False)

    idle_w = args.idle if args.idle is not None else measure_idle()
    print(f"idle_baseline_w={idle_w:.3f}")
    for engine in engines:
        print(f"{engine}_max_batch={batches[engine]['batch']}")

    artifact = {
        "schema_version": 1,
        "benchmark_id": f"gpu-{args.workload}-max-batch-to-target",
        "run_id": run_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "question": "Does each engine's maximum training batch reduce "
                        "end-to-end energy and time to a common training-loss target?",
            "target_training_loss": args.target,
            "maximum_steps_or_epochs": args.max_steps,
            "precision": args.precision,
            "runs": args.runs,
            "energy": "GPU board power.draw sampled at 20 Hz and integrated "
                      "between engine TRAIN_START_UNIX/TRAIN_END_UNIX markers",
            "setup_excluded": "process/import/data/model setup before TRAIN_START",
            "compile_included": "first-step graph/XLA/compile work after TRAIN_START",
            "tensorflow_xla": args.tf_xla == "1",
            "caveat": "HIGGS and ResNet repeat one maximum-size resident batch; "
                      "Transformer trains the corpus in epochs. The gate is "
                      "training loss, not held-out quality.",
        },
        "capacity_artifact": {
            "path": str(capacity_path),
            "sha256": sha256(capacity_path),
        },
        "batches": batches,
        "git": git_metadata(),
        "machine": gpu_metadata(),
        "frameworks": framework_versions(args.bench_python),
        "idle_baseline_w": idle_w,
        "results": {},
    }
    if args.workload == "resnet50":
        artifact["protocol"]["opennn_cudnn_workspace_mib"] = args.workspace_mib
    if args.workload == "transformer":
        tokens_bin = transformer_tokens_bin(args.corpus, capacity_artifact["model"])
        artifact["dataset"] = {
            "corpus": str(Path(args.corpus).resolve()),
            "corpus_sha256": sha256(args.corpus),
            "tokens_bin": str(tokens_bin.resolve()),
            "tokens_bin_sha256": sha256(tokens_bin),
            "token_records": (
                tokens_bin.stat().st_size
                // ((capacity_artifact["model"]["input_seq"]
                     + capacity_artifact["model"]["decoder_seq"]) * 4)
            ),
        }
    elif args.workload == "higgs":
        higgs_path = Path(args.higgs_bin).resolve()
        artifact["dataset"] = {
            "higgs_bin": str(higgs_path),
            "higgs_bin_sha256": sha256(higgs_path),
            "rows": higgs_path.stat().st_size // (29 * 4),
            "format": "float32 rows x 29 (28 features, label)",
        }
    else:
        data_dir = Path(args.data_dir).resolve()
        dataset = {"data_dir": str(data_dir)}
        for name in ("cifar_images.npy", "cifar_labels.npy", "metadata.json"):
            path = data_dir / name
            if path.exists():
                dataset[f"{name}_sha256"] = sha256(path)
        artifact["dataset"] = dataset

    for engine in engines:
        per_run = []
        print(f"\n=== {engine}: batch={batches[engine]['batch']} ===")
        for index in range(args.runs):
            cooldown(idle_w)
            seed = args.seed_base + index
            stem = f"{args.workload}-{engine}-{args.precision}-run{index}"
            log_path = evidence_dir / f"{stem}.log"
            trace_path = evidence_dir / f"{stem}-power.csv"
            metrics = run_one(
                args, engine, batches[engine]["batch"], capacity_artifact,
                batches[engine]["path"], seed, idle_w, log_path, trace_path)
            per_run.append(metrics)
            print(f"run={index} ok={metrics['ok']} "
                  f"time={metrics.get('train_window_s')}s "
                  f"energy={metrics.get('energy_total_j')}J "
                  f"final_error={metrics.get('final_error')}")
        artifact["results"][engine] = aggregate(per_run)

    base = artifact["results"].get("opennn", {})
    if base.get("n_ok"):
        for engine, result in artifact["results"].items():
            if engine == "opennn" or not result.get("n_ok"):
                continue
            result["energy_ratio_vs_opennn"] = round(
                result["energy_total_j_median"]
                / base["energy_total_j_median"], 6)
            result["time_ratio_vs_opennn"] = round(
                result["train_window_s_median"]
                / base["train_window_s_median"], 6)

    output_path = RESULTS / (
        f"gpu-{args.workload}-max-batch-to-target-{run_id}.json")
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(artifact, stream, indent=2)
    print(f"\nwrote {output_path}")
    print(f"evidence {evidence_dir}")
    return 0 if all(value["n_ok"] for value in artifact["results"].values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
