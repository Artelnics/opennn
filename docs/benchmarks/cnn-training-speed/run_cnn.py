#!/usr/bin/env python3
"""3-way MNIST CNN training-speed harness: OpenNN vs PyTorch vs TensorFlow.

Drives the three programs on the IDENTICAL model (28x28x1 -> Conv 16@3x3 ReLU
-> MaxPool 2x2 -> Flatten -> Dense 10, cross-entropy, Adam, same batch),
repeats each N times, and reports median +/- stdev of samples/sec. Each engine
runs its FAIR fast path so the comparison survives a skeptical reviewer:

  OpenNN     -> CUDA, GPU-resident data (its existing fast path).
  PyTorch    -> PT_FAST: channels_last (NHWC, the layout OpenNN's cuDNN convs
                use) + torch.compile + TF32; PT_BF16 adds bf16 autocast.
  TensorFlow -> TF_FAST: jit_compile XLA (TF convs are already NHWC); TF_BF16
                adds the mixed_bfloat16 policy.

Running the fair path is the whole point: the published article previously
compared OpenNN's best path against PyTorch eager-NCHW and un-XLA'd Keras,
which inflated OpenNN's lead. This harness removes that asymmetry.

Emits an immutable results JSON per results/README.md.

  usage: run_cnn.py [--epochs 10] [--batch 128] [--runs 5]
                    [--precision bf16|fp32|both] [--engines opennn,pytorch,tensorflow]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(HERE, "..", "results"))
PY = os.environ.get("BENCH_PYTHON", sys.executable)
OPENNN_BIN = os.environ.get("OPENNN_CNN_BIN", os.path.join(HERE, "opennn_cnn_speed"))
MNIST_DIR = os.environ.get("MNIST_DIR", os.path.normpath(
    os.path.join(HERE, "..", "..", "..", "examples", "mnist", "data")))


def engine_cmd(engine, epochs, batch, bf16):
    """Return (cmd, env_overrides) for one engine, fair fast path enabled."""
    args = [str(epochs), str(batch)]
    env = {}
    if engine == "opennn":
        # OpenNN takes precision as a positional arg; GPU-resident data is enabled
        # in the benchmark code (opennn_cnn_speed.cpp), not via an env var.
        cmd = [OPENNN_BIN, MNIST_DIR, str(epochs), str(batch),
               "bf16" if bf16 else "fp32"]
    elif engine == "pytorch":
        cmd = [PY, os.path.join(HERE, "pytorch_cnn_speed.py")] + args
        env["PT_FAST"] = "1"
        if bf16:
            env["PT_BF16"] = "1"
    elif engine == "tensorflow":
        cmd = [PY, os.path.join(HERE, "tensorflow_cnn_speed.py")] + args
        env["TF_FAST"] = "1"
        if bf16:
            env["TF_BF16"] = "1"
    else:
        raise ValueError(engine)
    return cmd, env


def run_once(cmd, env_over):
    env = dict(os.environ)
    env.update(env_over)
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sps = None
    for line in (out.stdout + out.stderr).splitlines():
        if line.startswith("samples_per_sec="):
            try:
                sps = float(line.split("=", 1)[1])
            except ValueError:
                pass
    return sps, out.stdout + out.stderr


def versions():
    v = {"python": sys.version.split()[0]}
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
    try:
        v["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return v


def git_commit():
    try:
        c = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"],
                           capture_output=True, text=True).stdout.strip()[:12]
        return c or "unknown"
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--precision", default="both", choices=["bf16", "fp32", "both"])
    ap.add_argument("--engines", default="opennn,pytorch,tensorflow")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    engines = [e for e in args.engines.split(",")
               if e in ("opennn", "pytorch", "tensorflow")]
    precisions = ["bf16", "fp32"] if args.precision == "both" else [args.precision]

    result = {
        "schema_version": 1,
        "benchmark_id": "gpu-cnn-training-speed",
        "run_id": run_id,
        "git_commit": git_commit(),
        "configuration": {
            "model": "MNIST: Conv16@3x3(ReLU) -> MaxPool2 -> Flatten -> Dense10",
            "batch": args.batch, "epochs": args.epochs, "runs": args.runs,
            "metric": "samples_per_sec (median of runs)",
            "fair_paths": "PyTorch channels_last+compile+TF32; TF XLA; OpenNN GPU-resident",
        },
        "machine": versions(),
        "results": {},
    }

    for prec in precisions:
        bf16 = (prec == "bf16")
        print(f"\n=== {prec} ===")
        per_prec = {}
        for eng in engines:
            cmd, env_over = engine_cmd(eng, args.epochs, args.batch, bf16)
            vals = []
            last_raw = ""
            for _ in range(args.runs):
                sps, last_raw = run_once(cmd, env_over)
                if sps:
                    vals.append(sps)
            if vals:
                med = statistics.median(vals)
                sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                per_prec[eng] = {"samples_per_sec_median": round(med, 1),
                                 "samples_per_sec_stdev": round(sd, 1),
                                 "n_ok": len(vals)}
                print(f"  {eng:11s} {med:>12.0f} +/- {sd:.0f} samples/s  (n={len(vals)})")
            else:
                per_prec[eng] = {"error": "no samples_per_sec parsed",
                                 "tail": last_raw[-400:]}
                print(f"  {eng:11s} FAILED")
        if "opennn" in per_prec and "samples_per_sec_median" in per_prec["opennn"]:
            base = per_prec["opennn"]["samples_per_sec_median"]
            for eng in ("pytorch", "tensorflow"):
                if eng in per_prec and "samples_per_sec_median" in per_prec[eng]:
                    per_prec[f"opennn_vs_{eng}"] = round(
                        base / per_prec[eng]["samples_per_sec_median"], 3)
        result["results"][prec] = per_prec

    out_path = os.path.join(RESULTS_DIR, f"gpu-cnn-training-speed-{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
