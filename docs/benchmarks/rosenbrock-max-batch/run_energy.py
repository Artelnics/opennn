#!/usr/bin/env python3
"""Fair, repeated-run GPU energy benchmark: OpenNN vs PyTorch vs TensorFlow.

Each engine runs the SAME workload (same network, same batch, same iteration
count -> same number of samples processed) so energy/sample is an apples-to-apples
comparison. For every engine we:

  * run a fixed-batch steady-state loop (warmup excluded by each program),
  * sample GPU power.draw at 20 Hz while it runs and trapezoid-integrate it,
  * repeat N times and report median +/- stdev of energy/sample,
  * report BOTH total energy (includes the ~idle floor) and active energy
    (idle baseline subtracted).

The idle baseline is measured fresh at startup. Results (per-run + aggregate,
versions, git commit, raw power traces) are written to ../results/ as immutable
JSON per results/README.md.

  usage: run_energy.py [--mode train|inference|both] [--batch N] [--iters N]
                       [--inputs N] [--hidden N] [--runs N] [--engines list]

This script is GPU-energy only (the board power sensor; CPU/system not included)
and uses sampled power, not a hardware joule counter -- both documented in the
benchmark note. It is designed to be run on a quiet GPU.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(HERE, "..", "results"))
PY = os.environ.get("BENCH_PYTHON", sys.executable)

# Engine command builders. Each returns (cmd_list, env_overrides). All three run
# the identical (mode, batch, iters, inputs, hidden) workload.
def opennn_cmd(mode, batch, iters, inputs, hidden):
    env = {}
    if mode == "train":
        env = {"OPENNN_GPU_RESIDENT_DATA": "1", "OPENNN_CUDA_GRAPH": "1"}
    # OpenNN train takes a dataset size; use batch*1 so one epoch == `iters` steps
    # would need iters batches. To match a fixed (batch x iters) sample count with
    # the others, drive `iters` steps of `batch` at samples=batch (1 batch/epoch),
    # iters epochs -> batch*iters samples, same as the others.
    if mode == "train":
        cmd = [os.path.join(HERE, "opennn_rosenbrock_throughput"),
               "train", str(batch), str(batch), str(iters), str(inputs), str(hidden)]
    else:
        cmd = [os.path.join(HERE, "opennn_rosenbrock_resident_infer"),
               str(batch), str(iters), str(inputs), str(hidden)]
    return cmd, env


def pytorch_cmd(mode, batch, iters, inputs, hidden):
    cmd = [PY, os.path.join(HERE, "pytorch_rosenbrock_throughput.py"),
           mode, str(batch), str(iters), str(inputs), str(hidden)]
    return cmd, {}


def tensorflow_cmd(mode, batch, iters, inputs, hidden):
    cmd = [PY, os.path.join(HERE, "tensorflow_rosenbrock_throughput.py"),
           mode, str(batch), str(iters), str(inputs), str(hidden)]
    return cmd, {}


ENGINES = {"opennn": opennn_cmd, "pytorch": pytorch_cmd, "tensorflow": tensorflow_cmd}


def parse_power_csv(path):
    """Return list of (epoch_seconds, watts) from an nvidia-smi timestamp,power log."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            ts, w = parts[0], parts[1]
            # ts like 2026/06/14 02:10:33.456 -> seconds of day is enough for dt
            try:
                hms = ts.split(" ")[1].split(":")
                sec = int(hms[0]) * 3600 + int(hms[1]) * 60 + float(hms[2])
                samples.append((sec, float(w)))
            except (IndexError, ValueError):
                continue
    return samples


def integrate(samples, idle_w):
    """Trapezoidal ∫power dt over consecutive samples; return (total_J, active_J, avg_w, n)."""
    e_total = e_active = span = sumw = 0.0
    n = 0
    prev = None
    for t, w in samples:
        sumw += w
        if prev is not None:
            pt, pw = prev
            dt = t - pt
            if dt < 0:
                dt += 86400  # midnight wrap
            if 0 < dt < 2:   # ignore startup/teardown gaps
                e_total += 0.5 * (w + pw) * dt
                e_active += 0.5 * ((w - idle_w) + (pw - idle_w)) * dt
                span += dt
                n += 1
        prev = (t, w)
    avg_w = sumw / len(samples) if samples else 0.0
    return e_total, e_active, avg_w, span


def measure_idle(seconds=5.0):
    """Mean idle GPU power over `seconds` on a quiet GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
             "-lms", "100"],
            capture_output=True, text=True, timeout=seconds).stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    vals = []
    for x in out.split("\n"):
        x = x.strip()
        if not x:
            continue
        try:
            vals.append(float(x))
        except ValueError:
            continue
    return sum(vals) / len(vals) if vals else 26.8


def run_one(cmd, env_over, idle_w, trace_path):
    """Run cmd while logging power; return (metrics dict, stdout)."""
    env = dict(os.environ)
    env.update(env_over)
    logf = open(trace_path, "w")
    logger = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,power.draw",
         "--format=csv,noheader,nounits", "-lms", "50"],
        stdout=logf, stderr=subprocess.DEVNULL)
    time.sleep(0.3)  # let the logger spin up
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    time.sleep(0.3)
    logger.terminate()
    logger.wait()
    logf.close()

    samples = parse_power_csv(trace_path)
    e_total, e_active, avg_w, span = integrate(samples, idle_w)
    return {
        "wall_s": round(wall, 3),
        "integ_span_s": round(span, 3),
        "power_samples": len(samples),
        "avg_power_w": round(avg_w, 2),
        "active_power_w": round(avg_w - idle_w, 2),
        "energy_total_j": round(e_total, 3),
        "energy_active_j": round(e_active, 3),
        "rc": proc.returncode,
    }, proc.stdout + proc.stderr


def git_commit():
    try:
        r = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        c = r.stdout.strip()[:12]
        return c if c else "unknown"
    except Exception:
        return "unknown"


def versions():
    v = {}
    try:
        import torch
        v["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import tensorflow as tf
        v["tensorflow"] = tf.__version__
    except Exception:
        pass
    v["python"] = sys.version.split()[0]
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        v["gpu"] = gpu
    except Exception:
        pass
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["train", "inference", "both"])
    ap.add_argument("--batch", type=int, default=8000)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--inputs", type=int, default=1000)
    ap.add_argument("--hidden", type=int, default=1000)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--engines", default="opennn,pytorch,tensorflow")
    ap.add_argument("--idle", type=float, default=None, help="override idle W (else measured)")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    engines = [e for e in args.engines.split(",") if e in ENGINES]
    modes = ["inference", "train"] if args.mode == "both" else [args.mode]

    idle_w = args.idle if args.idle is not None else measure_idle()
    print(f"idle_baseline_W={idle_w:.2f}")

    samples_per_run = args.batch * args.iters
    result = {
        "schema_version": 1,
        "benchmark_id": "gpu-dense-rosenbrock-energy",
        "run_id": run_id,
        "git_commit": git_commit(),
        "configuration": {
            "network": f"{args.inputs}->{args.hidden}(tanh)->1 dense, MSE, Adam, fp32",
            "batch": args.batch, "iters_per_run": args.iters,
            "samples_per_run": samples_per_run, "runs": args.runs,
            "idle_baseline_w": round(idle_w, 2),
            "power_source": "nvidia-smi power.draw 20Hz, trapezoidal integration",
            "note": "GPU energy only (board sensor); sampled power, not a HW joule counter",
        },
        "machine": versions(),
        "results": {},
    }

    for mode in modes:
        result["results"][mode] = {}
        for eng in engines:
            print(f"\n=== {mode} / {eng} ({args.runs} runs) ===")
            per_run = []
            raw_outputs = []
            for r in range(args.runs):
                cmd, env_over = ENGINES[eng](mode, args.batch, args.iters,
                                             args.inputs, args.hidden)
                trace = os.path.join(RESULTS_DIR, f".trace-{eng}-{mode}-{run_id}-{r}.csv")
                m, out = run_one(cmd, env_over, idle_w, trace)
                os.remove(trace)
                if m["rc"] != 0 or m["power_samples"] < 10:
                    print(f"  run {r}: FAILED rc={m['rc']} samples={m['power_samples']}")
                    raw_outputs.append(out[-500:])
                    continue
                m["j_per_sample_total"] = round(m["energy_total_j"] / samples_per_run * 1e6, 4)  # µJ
                m["j_per_sample_active"] = round(m["energy_active_j"] / samples_per_run * 1e6, 4)
                per_run.append(m)
                raw_outputs.append(out.strip().split("\n")[-3:])
                print(f"  run {r}: total={m['j_per_sample_total']:.2f} active={m['j_per_sample_active']:.2f} "
                      f"µJ/sample  avg_W={m['avg_power_w']}")

            if per_run:
                tot = [m["j_per_sample_total"] for m in per_run]
                act = [m["j_per_sample_active"] for m in per_run]
                pw = [m["avg_power_w"] for m in per_run]
                agg = {
                    "n_ok": len(per_run),
                    "uj_per_sample_total_median": round(statistics.median(tot), 4),
                    "uj_per_sample_total_stdev": round(statistics.pstdev(tot), 4) if len(tot) > 1 else 0.0,
                    "uj_per_sample_active_median": round(statistics.median(act), 4),
                    "uj_per_sample_active_stdev": round(statistics.pstdev(act), 4) if len(act) > 1 else 0.0,
                    "avg_power_w_median": round(statistics.median(pw), 2),
                    "per_run": per_run,
                    "raw_tail": raw_outputs,
                }
                result["results"][mode][eng] = agg
                print(f"  => median {agg['uj_per_sample_total_median']:.2f} µJ/sample (total), "
                      f"{agg['uj_per_sample_active_median']:.2f} (active), ±{agg['uj_per_sample_total_stdev']:.2f}")

    # Energy ratios vs OpenNN (lower energy is better -> ratio = other/opennn).
    for mode in result["results"]:
        r = result["results"][mode]
        if "opennn" in r:
            base_t = r["opennn"]["uj_per_sample_total_median"]
            base_a = r["opennn"]["uj_per_sample_active_median"]
            for eng in r:
                if eng == "opennn":
                    continue
                r[eng]["energy_ratio_vs_opennn_total"] = round(r[eng]["uj_per_sample_total_median"] / base_t, 3)
                r[eng]["energy_ratio_vs_opennn_active"] = round(r[eng]["uj_per_sample_active_median"] / base_a, 3)

    out_path = os.path.join(RESULTS_DIR, f"gpu-dense-rosenbrock-energy-{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
