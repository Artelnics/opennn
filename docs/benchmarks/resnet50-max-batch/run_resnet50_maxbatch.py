#!/usr/bin/env python3
"""Max training batch search for ResNet-50 on CIFAR-10.

The result is a capacity benchmark, not a throughput benchmark: the largest
batch that completes warmup plus a real training step (forward, backward, Adam)
inside the physical VRAM budget.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
RESULTS_DIR = os.path.normpath(os.path.join(HERE, "..", "results"))
RESNET_SPEED_DIR = os.path.normpath(os.path.join(HERE, "..", "resnet50-training-speed"))
PY = os.environ.get("BENCH_PYTHON", sys.executable)


def existing_python_sites():
    candidates = [
        Path.home() / "benchenv" / "lib" / "python3.12" / "site-packages",
        Path.home() / ".venvs" / "ml" / "lib" / "python3.12" / "site-packages",
    ]
    return [str(path) for path in candidates if path.exists()]


def nvidia_library_paths(site_packages):
    paths = []
    for site in site_packages:
        nvidia_root = Path(site) / "nvidia"
        if not nvidia_root.exists():
            continue
        paths.extend(str(path) for path in nvidia_root.glob("*/lib") if path.is_dir())
    if Path("/usr/lib/wsl/lib").exists():
        paths.insert(0, "/usr/lib/wsl/lib")
    return paths


BENCH_SITE_PACKAGES = existing_python_sites()
for site_path in reversed(BENCH_SITE_PACKAGES):
    if site_path not in sys.path:
        sys.path.insert(0, site_path)

BENCH_LD_LIBRARY_PATHS = nvidia_library_paths(BENCH_SITE_PACKAGES)


def run_text(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def git_commit():
    try:
        out = run_text(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], check=False)
        return (out.stdout.strip() or "unknown")[:12]
    except Exception:
        return "unknown"


def default_opennn_bin():
    env_bin = os.environ.get("OPENNN_RESNET50_MAXBATCH_BIN")
    if env_bin:
        return env_bin
    candidates = [
        os.path.join(HERE, "opennn_resnet50_maxbatch_trial"),
        os.path.join(REPO_ROOT, "build-gpu", "bin", "opennn_resnet50_maxbatch_trial"),
        os.path.join(REPO_ROOT, "build", "bin", "opennn_resnet50_maxbatch_trial"),
        os.path.join(REPO_ROOT, "build-gpu", "bin", "opennn_resnet50_maxbatch_trial.exe"),
        os.path.join(REPO_ROOT, "build", "bin", "opennn_resnet50_maxbatch_trial.exe"),
        os.path.join(REPO_ROOT, "build-ninja", "bin", "opennn_resnet50_maxbatch_trial"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[2]


def prepare_dataset(dataset):
    if dataset != "cifar10":
        raise ValueError("Only cifar10 is supported for this benchmark.")
    data_dir = os.path.join(RESNET_SPEED_DIR, dataset)
    needed = [
        os.path.join(data_dir, "cifar_images.npy"),
        os.path.join(data_dir, "cifar_labels.npy"),
        os.path.join(data_dir, "train"),
    ]
    if all(os.path.exists(path) for path in needed):
        return data_dir
    subprocess.run([PY, os.path.join(RESNET_SPEED_DIR, "prepare_cifar10.py"), data_dir],
                   check=True)
    return data_dir


def parse_gpu_info(gpu_index):
    query = "name,driver_version,memory.total,memory.used"
    cmd = [
        "nvidia-smi",
        f"--id={gpu_index}",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    out = run_text(cmd, check=True).stdout.strip().splitlines()[0]
    name, driver, total, used = [part.strip() for part in out.split(",")]
    return {
        "name": name,
        "driver_version": driver,
        "memory_total_mib": int(float(total)),
        "memory_used_mib": int(float(used)),
    }


def current_gpu_used_mib(gpu_index):
    out = run_text([
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ], check=True).stdout.strip().splitlines()[0]
    return int(float(out.strip()))


class PeakMonitor:
    def __init__(self, gpu_index, interval_s):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.peak_mib = 0
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                used = current_gpu_used_mib(self.gpu_index)
                self.samples.append(used)
                self.peak_mib = max(self.peak_mib, used)
            except Exception:
                pass
            self._stop.wait(self.interval_s)


def versions():
    result = {"python": sys.version.split()[0]}
    try:
        import torch
        result["torch"] = torch.__version__
        result["torch_cuda"] = torch.version.cuda
        result["torch_cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    try:
        import tensorflow as tf
        result["tensorflow"] = tf.__version__
        build_info = tf.sysconfig.get_build_info()
        result["tensorflow_built_cuda"] = build_info.get("cuda_version")
        result["tensorflow_built_cudnn"] = build_info.get("cudnn_version")
    except Exception:
        pass
    try:
        out = run_text(["nvcc", "--version"], check=False).stdout
        match = re.search(r"release\s+([0-9.]+)", out)
        if match:
            result["cuda_nvcc"] = match.group(1)
    except Exception:
        pass
    return result


def expanded_engines(value):
    engines = []
    for item in [x.strip() for x in value.split(",") if x.strip()]:
        if item == "pytorch":
            engines.extend(["pytorch_compile", "pytorch_eager"])
        elif item == "opennn":
            engines.append("opennn_pool1")
        elif item in {"opennn_pool1", "opennn_default", "pytorch_compile",
                      "pytorch_eager", "tensorflow"}:
            engines.append(item)
        else:
            raise ValueError(f"Unknown engine: {item}")
    return engines


def command_for(engine, data_dir, batch, opennn_bin, memory_fraction, memory_limit_mb):
    env = {}
    if engine in {"opennn_pool1", "opennn_default"}:
        cmd = [opennn_bin, data_dir, str(batch), "fp32"]
        env.update({
            "OPENNN_CUDA_GRAPH": "1",
            "OPENNN_NO_SHUFFLE": "1",
            "OPENNN_CONV_AUTOTUNE": "0",
        })
        if engine == "opennn_pool1":
            env["OPENNN_BATCH_POOL"] = "1"
    elif engine in {"pytorch_compile", "pytorch_eager"}:
        path = "compile" if engine == "pytorch_compile" else "eager"
        cmd = [
            PY,
            os.path.join(HERE, "pytorch_resnet50_maxbatch.py"),
            "--data", data_dir,
            "--batch", str(batch),
            "--path", path,
        ]
        if memory_fraction:
            cmd += ["--memory-fraction", f"{memory_fraction:.6f}"]
    elif engine == "tensorflow":
        cmd = [
            PY,
            os.path.join(HERE, "tensorflow_resnet50_maxbatch.py"),
            "--data", data_dir,
            "--batch", str(batch),
        ]
        if memory_limit_mb:
            cmd += ["--memory-limit-mb", str(memory_limit_mb)]
    else:
        raise ValueError(engine)
    return cmd, env


def run_trial(engine, batch, data_dir, args, gpu_info):
    cap_mib = max(1, gpu_info["memory_total_mib"] - args.reserve_mib)
    memory_fraction = cap_mib / gpu_info["memory_total_mib"]
    cmd, env_over = command_for(
        engine,
        data_dir,
        batch,
        args.opennn_bin,
        memory_fraction,
        cap_mib,
    )
    env = dict(os.environ)
    env.update(env_over)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    if BENCH_SITE_PACKAGES:
        env["PYTHONPATH"] = os.pathsep.join(BENCH_SITE_PACKAGES + [env.get("PYTHONPATH", "")])
    if BENCH_LD_LIBRARY_PATHS:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(BENCH_LD_LIBRARY_PATHS + [env.get("LD_LIBRARY_PATH", "")])

    t0 = time.perf_counter()
    try:
        with PeakMonitor(args.gpu_index, args.poll_s) as mon:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                  timeout=args.timeout_s)
            peak_mib = mon.peak_mib
    except subprocess.TimeoutExpired as exc:
        raw = (exc.stdout or "") + (exc.stderr or "")
        return {
            "batch": batch,
            "ok": False,
            "reason": "timeout",
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "peak_vram_mib": None,
            "command": " ".join(cmd),
            "raw_output": raw[-4000:],
        }

    raw = proc.stdout + proc.stderr
    ok = proc.returncode == 0 and "RESULT=OK" in raw and "RESULT=ERROR" not in raw
    reason = "ok" if ok else f"exit_{proc.returncode}"
    if ok and peak_mib > cap_mib:
        ok = False
        reason = "vram_cap_exceeded"

    return {
        "batch": batch,
        "ok": ok,
        "reason": reason,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "peak_vram_mib": peak_mib,
        "vram_cap_mib": cap_mib,
        "command": " ".join(cmd),
        "raw_output": raw[-4000:],
    }


def wait_for_cooldown(gpu_index, threshold_mib, timeout_s=30):
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            if current_gpu_used_mib(gpu_index) <= threshold_mib:
                return True
        except Exception:
            return False
        time.sleep(0.5)
    return False


def search_engine(engine, data_dir, args, gpu_info):
    cache = {}

    def trial(batch):
        if batch not in cache:
            cache[batch] = run_trial(engine, batch, data_dir, args, gpu_info)
            print(f"{engine:16s} batch={batch:<7d} "
                  f"{'OK' if cache[batch]['ok'] else 'FAIL'} "
                  f"peak={cache[batch].get('peak_vram_mib')} "
                  f"reason={cache[batch]['reason']}")
            wait_for_cooldown(args.gpu_index, args.idle_threshold_mib)
        return cache[batch]

    lo = 0
    hi = max(1, args.start_batch)

    while hi <= args.max_batch_limit:
        result = trial(hi)
        if not result["ok"]:
            break
        lo = hi
        hi *= 2
    else:
        hi = args.max_batch_limit + 1

    if lo == 0:
        hi = max(hi, 1)

    upper = min(hi - 1, args.max_batch_limit) if hi > args.max_batch_limit else hi - 1
    if hi <= args.max_batch_limit:
        upper = hi - 1

    left = lo + 1
    right = upper
    while left <= right:
        mid = (left + right) // 2
        result = trial(mid)
        if result["ok"]:
            lo = mid
            left = mid + 1
        else:
            right = mid - 1

    fail_next = None
    if lo + 1 <= args.max_batch_limit:
        fail_next = trial(lo + 1)

    ok_trials = [t for t in cache.values() if t["ok"]]
    max_trial = cache.get(lo) if lo else None
    return {
        "max_batch": lo,
        "max_trial": max_trial,
        "next_batch_trial": fail_next,
        "peak_vram_mib_at_max": max_trial.get("peak_vram_mib") if max_trial else None,
        "all_trials": [cache[k] for k in sorted(cache)],
        "ok_trial_count": len(ok_trials),
        "median_elapsed_s_ok": round(statistics.median([t["elapsed_s"] for t in ok_trials]), 3)
        if ok_trials else None,
    }


def summarize(results):
    metrics = {}
    if "opennn_pool1" in results:
        metrics["opennn_batch_pool_1"] = results["opennn_pool1"].get("max_batch", 0)
    pytorch_candidates = [
        results.get("pytorch_compile", {}).get("max_batch", 0),
        results.get("pytorch_eager", {}).get("max_batch", 0),
    ]
    if any(pytorch_candidates):
        metrics["pytorch_best"] = max(pytorch_candidates)
    if "tensorflow" in results:
        metrics["tensorflow_xla"] = results["tensorflow"].get("max_batch", 0)
    ratios = {}
    opennn = metrics.get("opennn_batch_pool_1")
    if opennn:
        for key in ("pytorch_best", "tensorflow_xla"):
            value = metrics.get(key)
            if value:
                ratios[f"opennn_vs_{key}"] = round(opennn / value, 3)
    return metrics, ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    ap.add_argument("--precision", default="fp32", choices=["fp32"])
    ap.add_argument("--engines", default="opennn,pytorch,tensorflow")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--require-gpu-idle", action="store_true")
    ap.add_argument("--idle-threshold-mib", type=int, default=512)
    ap.add_argument("--reserve-mib", type=int, default=256)
    ap.add_argument("--start-batch", type=int, default=8)
    ap.add_argument("--max-batch-limit", type=int, default=65536)
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--poll-s", type=float, default=0.05)
    ap.add_argument("--opennn-bin", default=default_opennn_bin())
    args = ap.parse_args()

    data_dir = prepare_dataset(args.dataset)
    gpu_info = parse_gpu_info(args.gpu_index)

    if args.require_gpu_idle and gpu_info["memory_used_mib"] > args.idle_threshold_mib:
        raise SystemExit(
            f"GPU is not idle: {gpu_info['memory_used_mib']} MiB used "
            f"(threshold {args.idle_threshold_mib} MiB).")

    engines = expanded_engines(args.engines)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"GPU: {gpu_info['name']} ({gpu_info['memory_total_mib']} MiB)")
    print(f"OpenNN binary: {args.opennn_bin}")
    print(f"Engines: {', '.join(engines)}")

    results = {}
    for engine in engines:
        results[engine] = search_engine(engine, data_dir, args, gpu_info)

    max_batches, ratios = summarize(results)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact = {
        "schema_version": 1,
        "benchmark_id": "gpu-resnet50-max-batch",
        "run_id": run_id,
        "git_commit": git_commit(),
        "protocol": {
            "style": "mlperf_inspired",
            "official_mlperf": False,
            "benchmark_class": "training_capacity",
            "division": "closed",
            "quality_rule": {
                "metric": "finite_cross_entropy_after_training_step",
                "target": "finite",
                "status": "gated",
            },
            "measurement_rule": {
                "warmup": "one warmup/capture step at the tested batch",
                "runs": "fresh process per batch candidate",
                "aggregation": "largest successful batch by exponential growth plus binary search",
            },
        },
        "dataset": args.dataset,
        "configuration": {
            "model": "ResNet-50 v1.5 bottleneck, CIFAR-10 geometry",
            "input_shape": [32, 32, 3],
            "classes": 10,
            "precision": args.precision,
            "optimizer": "Adam lr=0.001",
            "loss": "cross-entropy",
            "vram_reserve_mib": args.reserve_mib,
            "max_batch_limit": args.max_batch_limit,
            "engines": engines,
        },
        "machine": {
            "gpu": gpu_info,
            "versions": versions(),
        },
        "metrics": {
            "max_train_batch": max_batches,
            "ratio": ratios,
        },
        "commands": {
            "build": "cmake --build build-gpu --target opennn_resnet50_maxbatch_trial",
            "run": "python run_resnet50_maxbatch.py --dataset cifar10 --precision fp32 --engines opennn,pytorch,tensorflow --gpu-index 0 --require-gpu-idle --start-batch 128",
        },
        "results": results,
    }

    out_path = os.path.join(RESULTS_DIR, f"gpu-resnet50-max-batch-{args.dataset}-{run_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(json.dumps({"max_train_batch": max_batches, "ratio": ratios}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
