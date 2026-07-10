#!/usr/bin/env python3
"""3-way CIFAR ResNet-50 training-speed harness: OpenNN vs PyTorch vs TensorFlow.

Drives the three programs on the IDENTICAL v1.5 bottleneck ResNet-50 (CIFAR
geometry, cross-entropy, Adam, same batch), repeats each N times, and reports
median +/- stdev of samples/sec. Each engine runs its FAIR fast path so the
comparison survives a skeptical reviewer:

  OpenNN     -> CUDA, GPU-resident data, CUDA-graph capture (its fast path).
  PyTorch    -> PT_FAST: channels_last (NHWC, the layout OpenNN's cuDNN convs
                use) + torch.compile + TF32; PT_BF16 adds bf16 autocast.
  TensorFlow -> @tf.function XLA (TF convs are already NHWC); TF_BF16 adds the
                mixed_bfloat16 policy.

This replaces the OpenNN-best-vs-PyTorch-eager-NCHW comparison in
run_resnet50.sh, which understated the frameworks (audit P1/P2/P5). CIFAR is
small-input geometry; the full ImageNet run is tracked separately
(IMAGENET_CONTINUE.md) for the real-scale credibility claim.

Data layout: prepare_<dataset>.py writes <dataset>/cifar_images.npy (read by
the Python engines) and <dataset>/train/ BMPs (read by OpenNN).

Emits an immutable results JSON per results/README.md.

  usage: run_resnet50.py [--dataset cifar10|cifar100] [--epochs 5] [--batch 128]
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
    env_bin = os.environ.get("OPENNN_RESNET_BIN")
    if env_bin:
        return env_bin
    repo_root = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
    candidates = [
        os.path.join(HERE, "opennn_resnet50_speed"),
        os.path.join(repo_root, "build-ninja", "bin", "opennn_resnet50_speed"),
        os.path.join(repo_root, "build-ninja", "bin", "opennn_resnet50_speed.exe"),
        os.path.join(repo_root, "build", "bin", "opennn_resnet50_speed"),
        os.path.join(repo_root, "build", "bin", "opennn_resnet50_speed.exe"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


OPENNN_BIN = default_opennn_bin()


def tensorflow_library_dirs(py):
    """TF loads its CUDA runtime from the nvidia-*-cu1x pip wheels; their lib/
    dirs must be on LD_LIBRARY_PATH or TF sees no GPU. Resolve them in the
    target interpreter (PY may be a venv distinct from this process)."""
    override = os.environ.get("TF_NV_LIBS")
    if override:
        return [part for part in override.split(os.pathsep) if part]
    code = (
        "import json, site\n"
        "from pathlib import Path\n"
        "roots = []\n"
        "for base in list(site.getsitepackages()) + [site.getusersitepackages()]:\n"
        "    nvidia = Path(base) / 'nvidia'\n"
        "    if nvidia.exists():\n"
        "        roots.extend(str(p) for p in nvidia.rglob('lib') if p.is_dir())\n"
        "print(json.dumps(roots))\n"
    )
    try:
        out = subprocess.run([py, "-c", code], capture_output=True, text=True)
        lines = [line for line in out.stdout.splitlines() if line.strip()]
        return json.loads(lines[-1]) if lines else []
    except Exception:
        return []


def engine_cmd(engine, data_dir, epochs, batch, bf16):
    """Return (cmd, env_overrides). OpenNN reads <data_dir>/train (BMPs); the
    Python engines read <data_dir>/ (npy)."""
    env = {}
    if engine == "opennn":
        # GPU-resident data (CIFAR fits in VRAM) and the CUDA graph are enabled in
        # the benchmark code (opennn_resnet50_speed.cpp); no env vars needed.
        cmd = [OPENNN_BIN, os.path.join(data_dir, "train"),
               str(epochs), str(batch), "bf16" if bf16 else "fp32"]
    elif engine == "pytorch":
        cmd = [PY, os.path.join(HERE, "pytorch_resnet50_speed.py"),
               str(epochs), str(batch), data_dir]
        env["PT_FAST"] = "1"
        if bf16:
            env["PT_BF16"] = "1"
    elif engine == "tensorflow":
        cmd = [PY, os.path.join(HERE, "tensorflow_resnet50_speed.py"),
               str(epochs), str(batch), data_dir]
        if bf16:
            env["TF_BF16"] = "1"
        libs = tensorflow_library_dirs(PY)
        if libs:
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                libs + ([existing] if existing else []))
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
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--epochs", type=int, default=5)
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
        "benchmark_id": "gpu-resnet50-training-speed",
        "run_id": run_id,
        "git_commit": git_commit(),
        "configuration": {
            "model": "ResNet-50 v1.5 bottleneck, CIFAR geometry",
            "dataset": args.dataset,
            "batch": args.batch, "epochs": args.epochs, "runs": args.runs,
            "metric": "samples_per_sec (median of runs)",
            "fair_paths": "PyTorch channels_last+compile+TF32; TF XLA; OpenNN GPU-resident+CUDA-graph",
        },
        "machine": versions(),
        "results": {},
    }

    for prec in precisions:
        bf16 = (prec == "bf16")
        print(f"\n=== {prec} ===")
        per_prec = {}
        for eng in engines:
            cmd, env_over = engine_cmd(eng, data_dir, args.epochs, args.batch, bf16)
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

    out_path = os.path.join(
        RESULTS_DIR, f"gpu-resnet50-training-speed-{args.dataset}-{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
