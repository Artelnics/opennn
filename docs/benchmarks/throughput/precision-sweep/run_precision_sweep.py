#!/usr/bin/env python3
"""fp32-vs-bf16 precision sweep for OpenNN GPU workloads (inference + training).

Answers a single question with matched workloads: how much does bf16 buy over
fp32 for each OpenNN workload, in inference AND in training? Runs each
(workload, mode, precision) cell N times and reports median +/- stdev of the
throughput plus the bf16/fp32 speedup ratio. Emits an immutable results JSON.

Why it matters: bf16 is OpenNN's tensor-core path. Under Type::FP32 the dense /
FFN / projection GEMMs run TF32 (correct, but slower) and attention runs via the
fp32-via-bf16 cast path; under Type::BF16 everything runs bf16 tensor cores. This
quantifies that gap so the precision choice is data-driven, not assumed.

Drivers:
  transformer  -> ../attention-speed/opennn_transformer_resident (infer, build.sh)
                  ../attention-speed/opennn_transformer_train    (train, needs corpus)
  dense        -> ../../capacity/higgs-max-batch/opennn_higgs_maxbatch_trial
                  (HIGGS dense 28->hidden->hidden->1, ReLU, BCE; a CMake target;
                  infer and train modes; synthetic contract-shaped data by
                  default, so no CSV is needed). Override its path with the
                  OPENNN_HIGGS_MAXBATCH_BIN env var if it is not in build-benchmarks/bin.

Precision is selected by the OPENNN_BF16 env var (set => bf16, unset => fp32),
uniformly across every driver.

  usage: run_precision_sweep.py [--runs 3] [--workloads transformer,dense]
                                [--modes inference,training]

NOTE: run on native Windows for trustworthy absolute numbers -- WSL2 degrades
OpenNN's bf16 tensor-core path (esp. small dense shapes), biasing the ratio
against bf16. The relative train-vs-infer pattern still holds under WSL.
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
ATTN = os.path.normpath(os.path.join(HERE, "..", "attention-speed"))
CORPUS = os.path.join(ATTN, "corpus_sweep.txt")

# Dense workload driver: the HIGGS dense trial (a CMake target). It runs the
# canonical 28 -> hidden -> hidden -> 1 ReLU/BCE net in infer and train modes on
# synthetic contract-shaped data, so the sweep needs no dataset file. Override
# the path with OPENNN_HIGGS_MAXBATCH_BIN.
HIGGS_BIN = os.environ.get(
    "OPENNN_HIGGS_MAXBATCH_BIN",
    os.path.normpath(os.path.join(
        HERE, "..", "..", "..", "..",
        "build-benchmarks", "bin", "opennn_higgs_maxbatch_trial")))

# Each cell: (cmd builder, metric token). cmd builder takes bf16:bool -> argv list.
# Workloads use a fixed representative shape; tweak here to match the headline.
CELLS = {
    ("transformer", "inference"): (
        lambda bf16: [os.path.join(ATTN, "opennn_transformer_resident"),
                      "256", "512", "8", "2048", "6", "10000", "32", "80"],
        "gpu_bound_tokens_per_sec"),
    ("transformer", "training"): (
        lambda bf16: [os.path.join(ATTN, "opennn_transformer_train"),
                      CORPUS, "512", "8", "2048", "6", "32", "12"],
        "samples_per_sec"),
    # Dense HIGGS trial args: <mode> <batch> <hidden> <hidden_layers> <iters>
    # <device> <tile>. tile == batch forces the single-tile (untiled) resident
    # path, so both precisions run the same plain dense GEMMs.
    ("dense", "inference"): (
        lambda bf16: [HIGGS_BIN,
                      "infer", "8000", "4096", "2", "100", "cuda", "8000"],
        "samples_per_sec"),
    ("dense", "training"): (
        lambda bf16: [HIGGS_BIN,
                      "train", "2000", "4096", "2", "8", "cuda", "2000"],
        "samples_per_sec"),
}


def run_once(cmd, metric, bf16):
    env = dict(os.environ)
    if bf16:
        env["OPENNN_BF16"] = "1"
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    val = None
    for line in (out.stdout + out.stderr).splitlines():
        if line.startswith(metric + "="):
            try:
                val = float(line.split("=", 1)[1])
            except ValueError:
                pass
    return val, out.stdout + out.stderr


def versions():
    v = {"python": sys.version.split()[0]}
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
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workloads", default="transformer,dense")
    ap.add_argument("--modes", default="inference,training")
    args = ap.parse_args()

    workloads = args.workloads.split(",")
    modes = args.modes.split(",")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    result = {
        "schema_version": 1,
        "benchmark_id": "gpu-precision-sweep",
        "run_id": run_id,
        "git_commit": git_commit(),
        "configuration": {
            "runs": args.runs,
            "metric": "throughput median of runs (tokens/sec for transformer, samples/sec for dense)",
            "precision_selector": "OPENNN_BF16 env",
            "note": "run on native Windows for trustworthy bf16 absolute numbers",
        },
        "machine": versions(),
        "results": {},
    }

    for wl in workloads:
        for mode in modes:
            cell = CELLS.get((wl, mode))
            if cell is None:
                continue
            cmd_fn, metric = cell
            print(f"\n=== {wl} {mode} ===")
            entry = {}
            for prec, bf16 in (("fp32", False), ("bf16", True)):
                vals = []
                last = ""
                for _ in range(args.runs):
                    v, last = run_once(cmd_fn(bf16), metric, bf16)
                    if v:
                        vals.append(v)
                if vals:
                    med = statistics.median(vals)
                    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                    entry[prec] = {"median": round(med, 1), "stdev": round(sd, 1),
                                   "n_ok": len(vals)}
                    print(f"  {prec}: {med:>14.0f} +/- {sd:.0f}  (n={len(vals)})")
                else:
                    entry[prec] = {"error": "no metric parsed", "tail": last[-300:]}
                    print(f"  {prec}: FAILED")
            if "median" in entry.get("fp32", {}) and "median" in entry.get("bf16", {}):
                entry["bf16_speedup"] = round(entry["bf16"]["median"] / entry["fp32"]["median"], 3)
                print(f"  bf16 speedup: {entry['bf16_speedup']}x")
            result["results"][f"{wl}_{mode}"] = entry

    out_path = os.path.join(RESULTS_DIR, f"gpu-precision-sweep-{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
