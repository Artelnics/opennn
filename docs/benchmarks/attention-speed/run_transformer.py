#!/usr/bin/env python3
"""3-way Transformer inference throughput harness: OpenNN vs PyTorch vs TensorFlow.

Drives the three programs on the IDENTICAL config (same seq/d_model/heads/ff/
layers/vocab/batch), repeats each N times, and reports median +/- stdev of
tokens/sec. Each engine runs its fair fast path (OpenNN device-resident + fused
flash-attention; PyTorch torch.autocast; TF @tf.function XLA + mixed precision).
Emits an immutable results JSON per results/README.md.

The headline is bf16 (the precision transformers actually run in) at the paper
base shape across sequence lengths -- OpenNN's fused flash-attention is strongest
at long sequences (the LLM / long-context regime).

  usage: run_transformer.py [--seqs 128,256,512] [--d-model 512] [--heads 8]
                            [--ff 2048] [--layers 6] [--vocab 10000]
                            [--batch 32] [--iters 30] [--runs 5]
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
OPENNN_BIN = os.path.join(HERE, "opennn_transformer_resident")


def engine_cmd(engine, cfg, bf16):
    seq, d, h, ff, L, vocab, batch, iters = cfg
    args = [str(x) for x in (seq, d, h, ff, L, vocab, batch, iters)]
    env = {}
    if engine == "opennn":
        cmd = [OPENNN_BIN] + args
        if bf16:
            env["OPENNN_BF16"] = "1"
    elif engine == "pytorch":
        cmd = [PY, os.path.join(HERE, "pytorch_transformer_infer.py")] + args
        if bf16:
            env["PT_BF16"] = "1"
    elif engine == "tensorflow":
        cmd = [PY, os.path.join(HERE, "tensorflow_transformer_infer.py")] + args
        if bf16:
            env["TF_BF16"] = "1"
    else:
        raise ValueError(engine)
    return cmd, env


def run_once(cmd, env_over):
    env = dict(os.environ)
    env.update(env_over)
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    tok = None
    for line in (out.stdout + out.stderr).splitlines():
        if line.startswith("tokens_per_sec="):
            try:
                tok = float(line.split("=", 1)[1])
            except ValueError:
                pass
    return tok, out.stdout + out.stderr


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
    ap.add_argument("--seqs", default="128,256,512")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--vocab", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32", "both"])
    ap.add_argument("--engines", default="opennn,pytorch,tensorflow")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seqs = [int(s) for s in args.seqs.split(",")]
    engines = [e for e in args.engines.split(",") if e in ("opennn", "pytorch", "tensorflow")]
    precisions = ["bf16", "fp32"] if args.precision == "both" else [args.precision]

    result = {
        "schema_version": 1,
        "benchmark_id": "gpu-transformer-inference",
        "run_id": run_id,
        "git_commit": git_commit(),
        "configuration": {
            "shape": f"d{args.d_model}/h{args.heads}/ff{args.ff}/{args.layers}L vocab{args.vocab}",
            "batch": args.batch, "iters": args.iters, "runs": args.runs,
            "seqs": seqs, "metric": "tokens_per_sec (median of runs)",
        },
        "machine": versions(),
        "results": {},
    }

    for prec in precisions:
        bf16 = (prec == "bf16")
        result["results"][prec] = {}
        for seq in seqs:
            cfg = (seq, args.d_model, args.heads, args.ff, args.layers,
                   args.vocab, args.batch, args.iters)
            print(f"\n=== {prec} seq={seq} ===")
            per_seq = {}
            for eng in engines:
                cmd, env_over = engine_cmd(eng, cfg, bf16)
                vals = []
                for r in range(args.runs):
                    tok, raw = run_once(cmd, env_over)
                    if tok:
                        vals.append(tok)
                if vals:
                    med = statistics.median(vals)
                    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                    per_seq[eng] = {"tokens_per_sec_median": round(med, 1),
                                    "tokens_per_sec_stdev": round(sd, 1),
                                    "n_ok": len(vals)}
                    print(f"  {eng:11s} {med:>12.0f} ± {sd:.0f} tok/s  (n={len(vals)})")
                else:
                    per_seq[eng] = {"error": "no tokens_per_sec parsed"}
                    print(f"  {eng:11s} FAILED")
            # ratios vs each competitor (OpenNN / competitor)
            if "opennn" in per_seq and "tokens_per_sec_median" in per_seq["opennn"]:
                base = per_seq["opennn"]["tokens_per_sec_median"]
                for eng in ("pytorch", "tensorflow"):
                    if eng in per_seq and "tokens_per_sec_median" in per_seq[eng]:
                        per_seq[f"opennn_vs_{eng}"] = round(base / per_seq[eng]["tokens_per_sec_median"], 3)
            result["results"][prec][str(seq)] = per_seq

    out_path = os.path.join(RESULTS_DIR, f"gpu-transformer-inference-{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
