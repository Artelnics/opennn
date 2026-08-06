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
import shlex
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
RESULTS_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "results"))
RESNET_SPEED_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "throughput", "resnet50"))
# Datasets live under $OPENNN_BENCH_DATA (see ../../DATA_POLICY.md), never inside
# a benchmark folder. The CIFAR-10 tree is prepared under $OPENNN_BENCH_DATA/cifar10.
BENCH_DATA_ROOT = os.environ.get(
    "OPENNN_BENCH_DATA", os.path.expanduser("~/opennn-benchmark-data"))
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

CAPACITY_FAILURE_REASONS = {"oom", "vram_cap_exceeded"}
OPENNN_ENGINES = {"opennn_pool1", "opennn_default"}
OOM_PATTERNS = (
    "out of memory",
    "cuda_error_out_of_memory",
    "cuda out of memory",
    "cudnn_status_alloc_failed",
    "failed to allocate",
    "allocation failed",
)


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
    data_dir = os.path.join(BENCH_DATA_ROOT, dataset)
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
    def __init__(self, gpu_index, interval_s, cap_mib=None):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.cap_mib = cap_mib
        self.peak_mib = 0
        self.cap_exceeded = threading.Event()
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
                self.peak_mib = max(self.peak_mib, used)
                if self.cap_mib is not None and used > self.cap_mib:
                    self.cap_exceeded.set()
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


def command_for(engine, precision, data_dir, batch, opennn_bin, memory_fraction,
                memory_limit_mb, opennn_workspace_mode=None):
    env = {}
    if engine in OPENNN_ENGINES:
        # CUDA graph, shuffle and conv autotune are all set in the benchmark code.
        # The prefetch-pool depth is the 5th positional arg: pool1 -> 1 (fewest
        # device batch copies, largest reachable batch), default -> 0 (library auto).
        # Precision (fp32|bf16) selects Configuration::set(Device::CUDA, ...).
        batch_pool = "1" if engine == "opennn_pool1" else "0"
        workspace_mode = opennn_workspace_mode or "16"
        cmd = [opennn_bin, data_dir, str(batch), precision, batch_pool,
               workspace_mode]
    elif engine in {"pytorch_compile", "pytorch_eager"}:
        path = "compile" if engine == "pytorch_compile" else "eager"
        cmd = [
            PY,
            os.path.join(HERE, "pytorch_resnet50_maxbatch.py"),
            "--data", data_dir,
            "--batch", str(batch),
            "--path", path,
            "--precision", precision,
        ]
        if memory_fraction:
            cmd += ["--memory-fraction", f"{memory_fraction:.6f}"]
    elif engine == "tensorflow":
        cmd = [
            PY,
            os.path.join(HERE, "tensorflow_resnet50_maxbatch.py"),
            "--data", data_dir,
            "--batch", str(batch),
            "--precision", precision,
        ]
        if memory_limit_mb:
            cmd += ["--memory-limit-mb", str(memory_limit_mb)]
    else:
        raise ValueError(engine)
    return cmd, env


def run_trial(engine, precision, batch, data_dir, args, gpu_info,
              opennn_workspace_mode=None):
    cap_mib = max(1, gpu_info["memory_total_mib"] - args.reserve_mib)
    memory_fraction = cap_mib / gpu_info["memory_total_mib"]
    cmd, env_over = command_for(
        engine,
        precision,
        data_dir,
        batch,
        args.opennn_bin,
        memory_fraction,
        cap_mib,
        opennn_workspace_mode,
    )
    env = dict(os.environ)
    env.update(env_over)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    if BENCH_SITE_PACKAGES:
        env["PYTHONPATH"] = os.pathsep.join(BENCH_SITE_PACKAGES + [env.get("PYTHONPATH", "")])
    if BENCH_LD_LIBRARY_PATHS:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(BENCH_LD_LIBRARY_PATHS + [env.get("LD_LIBRARY_PATH", "")])

    # nvidia-smi reports GLOBAL device memory: desktop/compositor VRAM counts
    # too. Give the trial its cap as a delta over the idle level sampled
    # immediately before it, so external usage cannot fabricate a capacity
    # boundary or kill a healthy trial.
    idle_before = 0
    try:
        idle_before = current_gpu_used_mib(args.gpu_index)
    except Exception:
        pass

    t0 = time.perf_counter()
    termination_reason = None
    with PeakMonitor(args.gpu_index, args.poll_s,
                     cap_mib=cap_mib + idle_before) as mon:
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True)
        deadline = t0 + args.timeout_s
        while proc.poll() is None:
            if mon.cap_exceeded.is_set():
                termination_reason = "vram_cap_exceeded"
                proc.terminate()
                break
            if time.perf_counter() >= deadline:
                termination_reason = "timeout"
                proc.terminate()
                break
            time.sleep(min(0.1, args.poll_s))

        try:
            stdout, stderr = proc.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        peak_mib = mon.peak_mib

    raw = stdout + stderr
    peak_delta_mib = max(0, peak_mib - idle_before)
    if termination_reason:
        result = {
            "batch": batch,
            "ok": False,
            "reason": termination_reason,
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "peak_vram_mib": peak_mib,
            "idle_before_mib": idle_before,
            "peak_delta_mib": peak_delta_mib,
            "vram_cap_mib": cap_mib,
            "command": shlex.join(cmd),
            "raw_output": raw[-4000:],
        }
        if opennn_workspace_mode is not None:
            result["workspace_mode"] = opennn_workspace_mode
        return result

    ok = proc.returncode == 0 and "RESULT=OK" in raw and "RESULT=ERROR" not in raw
    raw_lower = raw.lower()
    if ok:
        reason = "ok"
    elif any(pattern in raw_lower for pattern in OOM_PATTERNS):
        reason = "oom"
    else:
        reason = f"exit_{proc.returncode}"
    if ok and peak_delta_mib > cap_mib:
        ok = False
        reason = "vram_cap_exceeded"

    result = {
        "batch": batch,
        "ok": ok,
        "reason": reason,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "peak_vram_mib": peak_mib,
        "idle_before_mib": idle_before,
        "peak_delta_mib": peak_delta_mib,
        "vram_cap_mib": cap_mib,
        "command": shlex.join(cmd),
        "raw_output": raw[-4000:],
    }
    if opennn_workspace_mode is not None:
        result["workspace_mode"] = opennn_workspace_mode
    return result


def run_opennn_capacity_trial(engine, precision, batch, data_dir, args, gpu_info):
    """Try bounded cuDNN workspace policies in fresh processes."""
    attempts = []
    final = None
    for workspace_mode in args.opennn_workspace_modes:
        trial = run_trial(
            engine,
            precision,
            batch,
            data_dir,
            args,
            gpu_info,
            opennn_workspace_mode=workspace_mode,
        )
        attempts.append({
            "workspace_mode": workspace_mode,
            "ok": trial["ok"],
            "reason": trial["reason"],
            "peak_vram_mib": trial.get("peak_vram_mib"),
            "elapsed_s": trial["elapsed_s"],
        })
        final = trial
        if trial["ok"]:
            break

    assert final is not None
    final["workspace_attempts"] = attempts
    return final


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


def search_engine(engine, precision, data_dir, args, gpu_info):
    cache = {}

    def trial(batch):
        if batch not in cache:
            if engine in OPENNN_ENGINES:
                cache[batch] = run_opennn_capacity_trial(
                    engine, precision, batch, data_dir, args, gpu_info)
            else:
                cache[batch] = run_trial(
                    engine, precision, batch, data_dir, args, gpu_info)
            workspace = cache[batch].get("workspace_mode")
            print(f"{engine:16s} {precision:4s} batch={batch:<7d} "
                  f"{'OK' if cache[batch]['ok'] else 'FAIL'} "
                  f"peak={cache[batch].get('peak_vram_mib')} "
                  f"reason={cache[batch]['reason']}"
                  + (f" workspace={workspace}" if workspace else ""))
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

    boundary_found = (
        fail_next is not None
        and not fail_next["ok"]
        and fail_next["reason"] in CAPACITY_FAILURE_REASONS
    )
    censored_by_limit = lo >= args.max_batch_limit and fail_next is None
    if boundary_found:
        search_status = "bounded_by_memory_failure"
    elif censored_by_limit:
        search_status = "censored_at_max_batch_limit"
    elif lo == 0:
        search_status = "no_passing_batch"
    else:
        search_status = "invalid_non_memory_failure"

    ok_trials = [t for t in cache.values() if t["ok"]]
    max_trial = cache.get(lo) if lo else None
    return {
        "max_batch": lo,
        "max_batch_is_lower_bound": not boundary_found,
        "boundary_found": boundary_found,
        "search_status": search_status,
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


def expanded_precisions(value):
    if value == "both":
        return ["fp32", "bf16"]
    return [value]


def parse_workspace_modes(value):
    modes = []
    for item in (part.strip().lower() for part in value.split(",")):
        if not item:
            continue
        if item in {"auto", "heur"}:
            modes.append(item)
            continue
        try:
            mib = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid OpenNN workspace mode: {item}") from exc
        if mib <= 0:
            raise argparse.ArgumentTypeError(
                "OpenNN capacity workspace caps must be positive MiB values")
        modes.append(str(mib))
    if not modes:
        raise argparse.ArgumentTypeError("at least one workspace mode is required")
    return modes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    ap.add_argument(
        "--data-dir",
        default=None,
        help="prepared CIFAR-10 directory; bypasses automatic preparation",
    )
    ap.add_argument("--precision", default="both", choices=["fp32", "bf16", "both"])
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
    ap.add_argument(
        "--opennn-workspace-modes",
        type=parse_workspace_modes,
        default=parse_workspace_modes("16,32,64,128,256,auto"),
        help="bounded cuDNN workspace policies tried per OpenNN candidate",
    )
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir) if args.data_dir else prepare_dataset(args.dataset)
    gpu_info = parse_gpu_info(args.gpu_index)

    if args.require_gpu_idle and gpu_info["memory_used_mib"] > args.idle_threshold_mib:
        raise SystemExit(
            f"GPU is not idle: {gpu_info['memory_used_mib']} MiB used "
            f"(threshold {args.idle_threshold_mib} MiB).")

    engines = expanded_engines(args.engines)
    precisions = expanded_precisions(args.precision)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"GPU: {gpu_info['name']} ({gpu_info['memory_total_mib']} MiB)")
    print(f"OpenNN binary: {args.opennn_bin}")
    print(f"Engines: {', '.join(engines)}")
    print(f"Precisions: {', '.join(precisions)}")
    print(f"OpenNN workspace modes: {', '.join(args.opennn_workspace_modes)}")

    results = {}
    max_batches = {}
    ratios = {}
    for precision in precisions:
        results[precision] = {}
        for engine in engines:
            results[precision][engine] = search_engine(
                engine, precision, data_dir, args, gpu_info)
        max_batches[precision], ratios[precision] = summarize(results[precision])

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
            "dataset_path": data_dir,
            "model": "ResNet-50 v1.5 bottleneck, CIFAR-10 geometry",
            "input_shape": [32, 32, 3],
            "classes": 10,
            "precisions": precisions,
            "optimizer": "Adam lr=0.001",
            "loss": "cross-entropy",
            "vram_reserve_mib": args.reserve_mib,
            "max_batch_limit": args.max_batch_limit,
            "engines": engines,
            "opennn_workspace_modes": args.opennn_workspace_modes,
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
            "run": shlex.join([sys.executable, os.path.abspath(__file__), *sys.argv[1:]]),
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
