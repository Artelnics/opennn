#!/usr/bin/env python3
"""Full ImageNet ResNet-50 training-speed harness: OpenNN vs PyTorch.

Runs the real ImageNet class-folder tree through the lazy 224x224 path and
writes an immutable result JSON under docs/benchmarks/results.

  usage: run_imagenet_resnet50.py --data /path/to/imagenet
         [--batches 64,32,16] [--epochs 1] [--runs 1]
         [--precision fp32|bf16|both]
         [--engines opennn,pytorch_fast,pytorch_eager]
         [--gpu-index 0] [--opennn-cache-dir /local_nvme/opennn-image-cache]
"""

import argparse
import json
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
RESULTS_DIR = HERE.parent.parent / "results"
PY = os.environ.get("BENCH_PYTHON", sys.executable)
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg"}


def env_gpu_index():
    value = os.environ.get("BENCH_GPU_INDEX")
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def default_opennn_bin():
    env_bin = os.environ.get("OPENNN_RESNET_BIN")
    if env_bin:
        return env_bin

    candidates = [
        HERE / "opennn_resnet50_speed",
        REPO_ROOT / "build-ninja" / "bin" / "opennn_resnet50_speed",
        REPO_ROOT / "build-ninja" / "bin" / "opennn_resnet50_speed.exe",
        REPO_ROOT / "build" / "bin" / "opennn_resnet50_speed",
        REPO_ROOT / "build" / "bin" / "opennn_resnet50_speed.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def command_output(command):
    try:
        out = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


def git_commit():
    return command_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]) or "unknown"


def versions():
    return versions_for_gpu(None)


def versions_for_gpu(gpu_index):
    gpu_command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ]
    if gpu_index is not None:
        gpu_command[1:1] = ["-i", str(gpu_index)]

    data = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": command_output(gpu_command),
    }
    try:
        import torch
        data["torch"] = torch.__version__
    except Exception:
        pass
    return data


def train_dir_for(data_path):
    path = Path(data_path).expanduser()
    return path / "train" if (path / "train").is_dir() else path


def validate_imagenet_tree(data_path):
    train_dir = train_dir_for(data_path)
    if not train_dir.is_dir():
        raise SystemExit(f"ImageNet train directory not found: {train_dir}")

    classes = sorted(p for p in train_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    if len(classes) < 2:
        raise SystemExit(f"Need at least two class folders under: {train_dir}")

    sample_images = []
    for class_dir in classes[: min(8, len(classes))]:
        sample_images.extend(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if sample_images:
            break

    if not sample_images:
        raise SystemExit(f"No supported images found under: {train_dir}")

    return train_dir, len(classes)


def parse_scalar(raw, name, cast=float):
    match = re.search(rf"(?:^|\s){re.escape(name)}=([0-9.]+)", raw, re.MULTILINE)
    if not match:
        return None
    try:
        return cast(match.group(1))
    except ValueError:
        return None


def quote_command(cmd):
    return " ".join(shlex.quote(str(part)) for part in cmd)


def current_gpu_memory_mib(gpu_index):
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    if gpu_index is not None:
        command[1:1] = ["-i", str(gpu_index)]
    raw = command_output(command)
    if not raw:
        return None
    first = raw.splitlines()[0].strip().split()[0]
    try:
        return int(float(first))
    except ValueError:
        return None


def gpu_compute_processes(gpu_index):
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    if gpu_index is not None:
        command[1:1] = ["-i", str(gpu_index)]
    raw = command_output(command)
    return raw or ""


def monitor_gpu(stop_event, samples, interval_s, gpu_index):
    while not stop_event.is_set():
        value = current_gpu_memory_mib(gpu_index)
        if value is not None:
            samples.append(value)
        stop_event.wait(interval_s)


def run_process(cmd, env_overrides, timeout_s, poll_s, gpu_index):
    env = dict(os.environ)
    env.update(env_overrides)

    samples = []
    stop_event = threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(stop_event, samples, poll_s, gpu_index), daemon=True)
    monitor.start()

    started = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate()
    finally:
        stop_event.set()
        monitor.join(timeout=2.0)

    elapsed = time.perf_counter() - started
    raw = (stdout or "") + (stderr or "")
    return {
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "elapsed_wall_s": round(elapsed, 3),
        "peak_gpu_memory_mib": max(samples) if samples else None,
        "raw_output": raw,
    }


def engine_command(engine, train_dir, data_arg, epochs, batch, precision, workers,
                   image_size, opennn_bin, cuda_graph, gpu_index, opennn_cache_dir):
    env = {}
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    if engine == "opennn":
        # CUDA graph is driven by the benchmark's own positional arg (1/0), not by
        # an env var. The 224px ImageNet data is too large to stay GPU-resident, so
        # opennn_resnet50_speed.cpp leaves residency off for image_size>0 anyway.
        cmd = [opennn_bin, str(train_dir), str(epochs), str(batch), precision,
               str(image_size), "1" if cuda_graph else "0"]
        # Image-cache dir is passed as a positional arg (set in code via
        # set_image_cache_dir), not as an environment variable.
        if opennn_cache_dir:
            cmd.append(str(opennn_cache_dir))
        if Path("/usr/lib/wsl/lib").exists():
            env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    elif engine in ("pytorch_fast", "pytorch_eager"):
        cmd = [
            PY,
            str(HERE / "pytorch_resnet50_lazy.py"),
            str(epochs),
            str(batch),
            str(data_arg),
            str(workers),
            str(image_size),
        ]
        if engine == "pytorch_fast":
            env["PT_FAST"] = "1"
        if precision == "bf16":
            env["PT_BF16"] = "1"
    else:
        raise ValueError(engine)
    return cmd, env


def summarize_runs(runs):
    ok = [r for r in runs if r.get("samples_per_sec") is not None]
    summary = {
        "n_ok": len(ok),
        "n_runs": len(runs),
        "runs": runs,
    }
    if ok:
        values = [r["samples_per_sec"] for r in ok]
        summary["samples_per_sec_median"] = round(statistics.median(values), 1)
        summary["samples_per_sec_stdev"] = round(statistics.pstdev(values), 1) if len(values) > 1 else 0.0
        peaks = [r["peak_gpu_memory_mib"] for r in ok if r["peak_gpu_memory_mib"] is not None]
        if peaks:
            summary["peak_gpu_memory_mib_max"] = max(peaks)
        for key in ("epoch_s", "samples", "classes", "parameters", "peak_vram_mib", "peak_reserved_mib"):
            vals = [r[key] for r in ok if r.get(key) is not None]
            if vals:
                summary[key] = vals[-1]
    else:
        tail = runs[-1]["raw_output"][-1200:] if runs else ""
        summary["error_tail"] = tail
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="ImageNet root or train directory")
    parser.add_argument("--batches", default="64,32,16")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "both"])
    parser.add_argument("--engines", default="opennn,pytorch_fast,pytorch_eager")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--timeout-s", type=int, default=0, help="0 means no timeout")
    parser.add_argument("--poll-s", type=float, default=0.5)
    parser.add_argument("--gpu-index", type=int, default=env_gpu_index(),
                        help="Physical GPU index for CUDA_VISIBLE_DEVICES and nvidia-smi sampling")
    parser.add_argument("--opennn-bin", default=default_opennn_bin())
    parser.add_argument("--opennn-cache-dir", default=None,
                        help="Directory for OpenNN's uint8 image cache "
                             "(passed to the benchmark as a positional arg, set in code)")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--require-gpu-idle", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_arg = Path(args.data).expanduser()
    train_dir, class_count = validate_imagenet_tree(data_arg)
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    engines = [x.strip() for x in args.engines.split(",") if x.strip()]
    precisions = ["fp32", "bf16"] if args.precision == "both" else [args.precision]
    timeout_s = args.timeout_s if args.timeout_s > 0 else None

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "schema_version": 1,
        "benchmark_id": "gpu-resnet50-training",
        "run_id": run_id,
        "git_commit": git_commit(),
        "dataset": "imagenet",
        "configuration": {
            "model": "ResNet-50 v1.5 bottleneck, ImageNet 224x224",
            "data": str(data_arg),
            "train_dir": str(train_dir),
            "classes_detected": class_count,
            "batches": batches,
            "epochs": args.epochs,
            "runs": args.runs,
            "image_size": args.image_size,
            "workers": args.workers,
            "opennn_cache_dir": args.opennn_cache_dir,
            "engines": engines,
            "precisions": precisions,
            "metric": "samples_per_sec (median of runs)",
        },
        "machine": versions_for_gpu(args.gpu_index),
        "results": {},
    }

    for precision in precisions:
        result["results"][precision] = {}
        for batch in batches:
            print(f"\n=== precision={precision} batch={batch} ===")
            per_batch = {}
            for engine in engines:
                cmd, env_over = engine_command(
                    engine, train_dir, data_arg, args.epochs, batch, precision,
                    args.workers, args.image_size, args.opennn_bin,
                    not args.no_cuda_graph, args.gpu_index, args.opennn_cache_dir,
                )
                print(f"  {engine}: {quote_command(cmd)}")
                if args.dry_run:
                    per_batch[engine] = {"command": quote_command(cmd), "env": env_over}
                    continue

                runs = []
                for run_index in range(args.runs):
                    if args.require_gpu_idle:
                        busy = gpu_compute_processes(args.gpu_index).strip()
                        if busy:
                            raise SystemExit(f"GPU is not idle before {engine} batch {batch}: {busy}")

                    run = run_process(cmd, env_over, timeout_s, args.poll_s, args.gpu_index)
                    raw = run["raw_output"]
                    run.update({
                        "command": quote_command(cmd),
                        "env": env_over,
                        "samples_per_sec": parse_scalar(raw, "samples_per_sec", float),
                        "epoch_s": parse_scalar(raw, "epoch_s", float),
                        "samples": parse_scalar(raw, "samples", int),
                        "classes": parse_scalar(raw, "classes", int),
                        "parameters": parse_scalar(raw, "parameters", int),
                        "peak_vram_mib": parse_scalar(raw, "peak_vram_mib", float),
                        "peak_reserved_mib": parse_scalar(raw, "peak_reserved_mib", float),
                        "result_ok": "RESULT=OK" in raw and run["returncode"] == 0,
                    })
                    runs.append(run)
                    label = run["samples_per_sec"] if run["samples_per_sec"] is not None else "FAIL"
                    print(f"    run {run_index + 1}/{args.runs}: {label} samples/s")
                per_batch[engine] = summarize_runs(runs)

            if "opennn" in per_batch and "samples_per_sec_median" in per_batch["opennn"]:
                base = per_batch["opennn"]["samples_per_sec_median"]
                for engine in ("pytorch_fast", "pytorch_eager"):
                    if engine in per_batch and "samples_per_sec_median" in per_batch[engine]:
                        per_batch[f"opennn_vs_{engine}"] = round(
                            base / per_batch[engine]["samples_per_sec_median"], 3)

            result["results"][precision][str(batch)] = per_batch

    output = RESULTS_DIR / f"gpu-resnet50-training-imagenet-{run_id}.json"
    if not args.dry_run:
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
