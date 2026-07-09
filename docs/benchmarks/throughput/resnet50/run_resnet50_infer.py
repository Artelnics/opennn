#!/usr/bin/env python3
"""3-way CIFAR ResNet-50 inference-speed harness: OpenNN vs PyTorch vs TensorFlow.

Forward-only twin of run_resnet50.py. Drives the three inference programs on the
IDENTICAL v1.5 bottleneck ResNet-50 (CIFAR geometry, same batch), repeats each N
times, and reports median +/- stdev of samples/sec and ms/batch. No optimizer, no
loss, no backward pass -- just the forward pass, each engine on its FAIR fast
path so the comparison survives a skeptical reviewer:

  OpenNN     -> CUDA, GPU-resident batch, device-resident forward
                (calculate_outputs_resident): parameters uploaded once, activation
                buffers built once, output left on the GPU.
  PyTorch    -> .eval() + torch.no_grad(); PT_FAST: channels_last (NHWC, the
                layout OpenNN's cuDNN convs use) + torch.compile + TF32; PT_BF16
                adds bf16 autocast.
  TensorFlow -> model(x, training=False) under @tf.function XLA (TF convs are
                already NHWC); TF_BF16 adds the mixed_bfloat16 policy.

Data layout: prepare_<dataset>.py writes <dataset>/cifar_images.npy (read by
the Python engines) and <dataset>/train/ BMPs (read by OpenNN).

Emits an immutable results JSON per results/README.md.

  usage: run_resnet50_infer.py [--dataset cifar10|cifar100] [--batch 128]
                               [--runs 5] [--precision bf16|fp32|both]
                               [--engines opennn,pytorch,tensorflow]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "results"))
PY = os.environ.get("BENCH_PYTHON", sys.executable)


def default_data_root():
    return os.environ.get(
        "OPENNN_BENCH_DATA", os.path.expanduser("~/opennn-benchmark-data"))


def default_opennn_bin():
    env_bin = os.environ.get("OPENNN_RESNET_INFER_BIN")
    if env_bin:
        return env_bin
    repo_root = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
    candidates = [
        os.path.join(HERE, "opennn_resnet50_infer"),
        os.path.join(repo_root, "build-ninja", "bin", "opennn_resnet50_infer"),
        os.path.join(repo_root, "build-ninja", "bin", "opennn_resnet50_infer.exe"),
        os.path.join(repo_root, "build", "bin", "opennn_resnet50_infer"),
        os.path.join(repo_root, "build", "bin", "opennn_resnet50_infer.exe"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


OPENNN_BIN = default_opennn_bin()


def engine_cmd(engine, data_dir, batch, runs, bf16):
    """Return (cmd, env_overrides). OpenNN reads <data_dir>/train (BMPs); the
    Python engines read <data_dir>/ (npy)."""
    env = {}
    if engine == "opennn":
        # GPU-resident batch and device-resident forward are enabled in the
        # benchmark code (opennn_resnet50_infer.cpp); no env vars needed.
        cmd = [OPENNN_BIN, os.path.join(data_dir, "train"),
               str(batch), str(runs), "bf16" if bf16 else "fp32"]
    elif engine == "pytorch":
        cmd = [PY, os.path.join(HERE, "pytorch_resnet50_infer.py"),
               str(batch), str(runs), data_dir]
        env["PT_FAST"] = "1"
        if bf16:
            env["PT_BF16"] = "1"
    elif engine == "tensorflow":
        cmd = [PY, os.path.join(HERE, "tensorflow_resnet50_infer.py"),
               str(batch), str(runs), data_dir]
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
    mpb = None
    for line in (out.stdout + out.stderr).splitlines():
        if line.startswith("samples_per_sec="):
            try:
                sps = float(line.split("=", 1)[1])
            except ValueError:
                pass
        elif line.startswith("ms_per_batch="):
            try:
                mpb = float(line.split("=", 1)[1])
            except ValueError:
                pass
    return sps, mpb, out.stdout + out.stderr


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
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--precision", default="both", choices=["bf16", "fp32", "both"])
    ap.add_argument("--engines", default="opennn,pytorch,tensorflow")
    ap.add_argument("--data-root", default=default_data_root(),
                    help="Root for benchmark datasets "
                         "(default $OPENNN_BENCH_DATA or ~/opennn-benchmark-data).")
    ap.add_argument("--data-dir", default=None,
                    help="Dataset directory (default <data-root>/<dataset>).")
    args = ap.parse_args()

    data_dir = args.data_dir or os.path.join(args.data_root, args.dataset)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    engines = [e for e in args.engines.split(",")
               if e in ("opennn", "pytorch", "tensorflow")]
    precisions = ["bf16", "fp32"] if args.precision == "both" else [args.precision]

    result = {
        "schema_version": 1,
        "benchmark_id": "gpu-resnet50-inference-speed",
        "run_id": run_id,
        "git_commit": git_commit(),
        "configuration": {
            "model": "ResNet-50 v1.5 bottleneck, CIFAR geometry",
            "dataset": args.dataset,
            "batch": args.batch, "runs": args.runs,
            "mode": "inference (forward-only)",
            "metric": "samples_per_sec / ms_per_batch (median of runs)",
            "fair_paths": "PyTorch eval+no_grad channels_last+compile+TF32; TF training=False XLA; OpenNN GPU-resident+calculate_outputs_resident",
        },
        "machine": versions(),
        "results": {},
    }

    for prec in precisions:
        bf16 = (prec == "bf16")
        print(f"\n=== {prec} ===")
        per_prec = {}
        for eng in engines:
            cmd, env_over = engine_cmd(eng, data_dir, args.batch, args.runs, bf16)
            sps_vals = []
            mpb_vals = []
            last_raw = ""
            for _ in range(args.runs):
                sps, mpb, last_raw = run_once(cmd, env_over)
                if sps:
                    sps_vals.append(sps)
                if mpb:
                    mpb_vals.append(mpb)
            if sps_vals:
                med = statistics.median(sps_vals)
                sd = statistics.pstdev(sps_vals) if len(sps_vals) > 1 else 0.0
                entry = {"samples_per_sec_median": round(med, 1),
                         "samples_per_sec_stdev": round(sd, 1),
                         "n_ok": len(sps_vals)}
                if mpb_vals:
                    entry["ms_per_batch_median"] = round(statistics.median(mpb_vals), 3)
                per_prec[eng] = entry
                print(f"  {eng:11s} {med:>12.0f} +/- {sd:.0f} samples/s  (n={len(sps_vals)})")
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

    out_path = os.path.join(
        RESULTS_DIR, f"gpu-resnet50-inference-speed-{args.dataset}-{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
