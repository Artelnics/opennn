#!/usr/bin/env python3
"""3-way Transformer TRAINING throughput harness: OpenNN vs PyTorch vs TensorFlow.

Drives the three training programs on the IDENTICAL config (same corpus, i.e. the
same #samples/seq-lengths/vocab, plus the same d_model/heads/ff/layers/batch/
epochs), repeats each N times, and reports median +/- stdev of samples/sec and
tokens/sec (forward + backward + Adam update). Each engine runs its fair fast
path (OpenNN device-resident + fused flash-attention + CUDA graph; PyTorch
torch.autocast + fused Adam; TF @tf.function XLA + mixed precision).

The corpus is a synthetic tab-separated file (make_synthetic_corpus.py); it is
generated once if missing and all engines read the SAME file so the FLOPs match
token-for-token. Emits an immutable results JSON per results/README.md.

  usage: run_transformer_train.py [--d-model 256] [--heads 8] [--ff 1024]
                                  [--layers 2] [--vocab 256] [--seq-len 256]
                                  [--samples 4096] [--batch 32] [--epochs 30]
                                  [--runs 5] [--precision bf16|fp32|both]
                                  [--engines opennn,pytorch,tensorflow]
                                  [--corpus PATH]
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
OPENNN_BIN = os.path.join(HERE, "opennn_transformer_train")


# TF needs the venv's bundled CUDA libs on LD_LIBRARY_PATH to see the GPU
# (same mechanism as capacity/transformer-max-batch/run_transformer_maxbatch.py).
def tf_ld_path():
    site = os.path.join(os.path.dirname(os.path.dirname(PY)),
                        "lib", "python3.12", "site-packages", "nvidia")
    libs = []
    if os.path.isdir(site):
        for d in sorted(os.listdir(site)):
            p = os.path.join(site, d, "lib")
            if os.path.isdir(p):
                libs.append(p)
    return os.pathsep.join(libs)


TF_LD = tf_ld_path()


def ensure_corpus(path, vocab, seq_len, samples):
    """Generate the synthetic training corpus once if it is missing."""
    if os.path.exists(path):
        return
    gen = os.path.join(HERE, "make_synthetic_corpus.py")
    subprocess.run([PY, gen, path, str(vocab), str(seq_len), str(samples)], check=True)


def engine_cmd(engine, corpus, cfg, bf16):
    d, h, ff, L, batch, epochs = cfg
    args = [corpus] + [str(x) for x in (d, h, ff, L, batch, epochs)]
    env = {}
    if engine == "opennn":
        cmd = [OPENNN_BIN] + args
        if bf16:
            env["OPENNN_BF16"] = "1"
    elif engine == "pytorch":
        cmd = [PY, os.path.join(HERE, "pytorch_transformer_train.py")] + args
        if bf16:
            env["PT_BF16"] = "1"
    elif engine == "tensorflow":
        cmd = [PY, os.path.join(HERE, "tensorflow_transformer_train.py")] + args
        if bf16:
            env["TF_BF16"] = "1"
        if TF_LD:
            env["LD_LIBRARY_PATH"] = TF_LD + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    else:
        raise ValueError(engine)
    return cmd, env


def run_once(cmd, env_over):
    env = dict(os.environ)
    env.update(env_over)
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    metrics = {}
    for line in (out.stdout + out.stderr).splitlines():
        for key in ("samples_per_sec", "tokens_per_sec"):
            if line.startswith(key + "="):
                try:
                    metrics[key] = float(line.split("=", 1)[1])
                except ValueError:
                    pass
    return metrics, out.stdout + out.stderr


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
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32", "both"])
    ap.add_argument("--engines", default="opennn,pytorch,tensorflow")
    ap.add_argument("--corpus", default=os.path.join(HERE, "synthetic_corpus.txt"))
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    engines = [e for e in args.engines.split(",") if e in ("opennn", "pytorch", "tensorflow")]
    precisions = ["bf16", "fp32"] if args.precision == "both" else [args.precision]

    ensure_corpus(args.corpus, args.vocab, args.seq_len, args.samples)
    cfg = (args.d_model, args.heads, args.ff, args.layers, args.batch, args.epochs)

    result = {
        "schema_version": 1,
        "benchmark_id": "gpu-transformer-training-speed",
        "run_id": run_id,
        "git_commit": git_commit(),
        "protocol": {
            "style": "mlperf_inspired",
            "official_mlperf": False,
            "benchmark_class": "training_throughput_with_quality",
            "division": "closed",
            "quality_rule": {
                "metric": "final_loss",
                "target": None,
                "status": "reported_not_gated",
            },
            "measurement_rule": {
                "warmup": "one untimed warmup epoch per engine",
                "runs": args.runs,
                "aggregation": "median",
            },
        },
        "dataset": "synthetic_corpus",
        "configuration": {
            "shape": f"d{args.d_model}/h{args.heads}/ff{args.ff}/{args.layers}L vocab{args.vocab}",
            "corpus": os.path.basename(args.corpus),
            "seq_len": args.seq_len, "samples": args.samples,
            "batch": args.batch, "epochs": args.epochs, "runs": args.runs,
            "metric": "samples_per_sec / tokens_per_sec (median of runs)",
        },
        "machine": versions(),
        "results": {},
    }

    for prec in precisions:
        bf16 = (prec == "bf16")
        print(f"\n=== {prec} ===")
        per_prec = {}
        for eng in engines:
            cmd, env_over = engine_cmd(eng, args.corpus, cfg, bf16)
            sps, tps = [], []
            for r in range(args.runs):
                metrics, raw = run_once(cmd, env_over)
                if "samples_per_sec" in metrics:
                    sps.append(metrics["samples_per_sec"])
                if "tokens_per_sec" in metrics:
                    tps.append(metrics["tokens_per_sec"])
            if sps and tps:
                per_prec[eng] = {
                    "samples_per_sec_median": round(statistics.median(sps), 1),
                    "samples_per_sec_stdev": round(statistics.pstdev(sps) if len(sps) > 1 else 0.0, 1),
                    "tokens_per_sec_median": round(statistics.median(tps), 1),
                    "tokens_per_sec_stdev": round(statistics.pstdev(tps) if len(tps) > 1 else 0.0, 1),
                    "n_ok": len(tps),
                }
                print(f"  {eng:11s} {per_prec[eng]['tokens_per_sec_median']:>14.0f} "
                      f"± {per_prec[eng]['tokens_per_sec_stdev']:.0f} tok/s  (n={len(tps)})")
            else:
                per_prec[eng] = {"error": "no samples_per_sec/tokens_per_sec parsed"}
                print(f"  {eng:11s} FAILED")
        # ratios vs each competitor (OpenNN / competitor) on tokens/sec
        if "opennn" in per_prec and "tokens_per_sec_median" in per_prec["opennn"]:
            base = per_prec["opennn"]["tokens_per_sec_median"]
            for eng in ("pytorch", "tensorflow"):
                if eng in per_prec and "tokens_per_sec_median" in per_prec[eng]:
                    per_prec[f"opennn_vs_{eng}"] = round(base / per_prec[eng]["tokens_per_sec_median"], 3)
        result["results"][prec] = per_prec

    out_path = os.path.join(RESULTS_DIR, f"gpu-transformer-training-speed-{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
