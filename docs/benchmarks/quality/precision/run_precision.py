#!/usr/bin/env python3
"""Precision benchmark: lowest error floor each engine reaches on Rosenbrock.

DOCUMENTED REGRESSION EXCEPTION. Every other dense benchmark uses HIGGS (BCE
classification). This one intentionally stays on the Rosenbrock REGRESSION task
because it exists to document OpenNN's SECOND-ORDER optimizers -- Levenberg-
Marquardt and Quasi-Newton -- which are least-squares methods that do not apply
to HIGGS/BCE classification. See ../DATA_POLICY.md and README.md.

Each engine trains the identical MLP (10 -> 10 tanh -> 1 linear, U(-1,1) init,
MSE, full dataset, no split) with each optimizer it ships, over N seeds. The
neutral scorer (score.py) computes the full-dataset MSE the same way for every
engine, and this runner reports per-engine/optimizer best MSE, mean MSE, and
mean training time.

  usage: run_precision.py [--runs 10] [--seeds 0,1,2]
                          [--engines opennn,pytorch,tensorflow]

Binary discovery (NO hardcoded machine paths):
  OpenNN engine  -> $OPENNN_PRECISION_BIN, else the CMake target opennn_precision
                    found in the benchmark build dir (build/bin, ...).
  Python engines -> $BENCH_PYTHON (default python3).
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone

import score

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "results"))
PY = os.environ.get("BENCH_PYTHON", "python3")
GEN = os.path.join(HERE, "generate_rosenbrock.py")
CSV = os.path.join(HERE, "rosenbrock.csv")


def repo_root():
    try:
        root = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        root = ""
    return root or os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))


REPO_ROOT = repo_root()


def find_opennn_bin():
    """Locate the opennn_precision binary without hardcoding a machine path."""
    override = os.environ.get("OPENNN_PRECISION_BIN")
    if override:
        return override, os.path.exists(override)
    names = ["opennn_precision", "opennn_precision.exe"]
    dirs = [
        os.path.join(REPO_ROOT, "build", "bin"),
        os.path.join(REPO_ROOT, "build", "bin", "Release"),
        os.path.join(REPO_ROOT, "build-benchmarks", "bin"),
        os.path.join(REPO_ROOT, "build-benchmarks", "bin", "Release"),
        HERE,
    ]
    for directory in dirs:
        for name in names:
            candidate = os.path.join(directory, name)
            if os.path.exists(candidate):
                return candidate, True
    return os.path.join(REPO_ROOT, "build", "bin", names[0]), False


# Each engine reaches its own error floor with the optimizers it actually ships.
# (config label, optimizer argv token, epochs, prediction file)
ENGINE_CONFIGS = {
    "opennn": [
        ("OpenNN-LM", "LevenbergMarquardt", 1000, "pred_opennn.txt"),
        ("OpenNN-QNM", "QuasiNewtonMethod", 1000, "pred_opennn.txt"),
        ("OpenNN-Adam", "AdaptiveMomentEstimation", 10000, "pred_opennn.txt"),
    ],
    # PyTorch's only built-in second-order optimizer is LBFGS; plus Adam.
    "pytorch": [
        ("PyTorch-LBFGS", "LBFGS", 1000, "pred_pytorch.txt"),
        ("PyTorch-Adam", "Adam", 10000, "pred_pytorch.txt"),
    ],
    # TensorFlow core keras.optimizers has no second-order option: Adam only.
    "tensorflow": [
        ("TensorFlow-Adam", "Adam", 10000, "pred_tensorflow.txt"),
    ],
}


def ensure_data():
    if not os.path.exists(CSV):
        subprocess.run([PY, GEN], cwd=HERE, check=True)


def engine_cmd(engine, opennn_bin, optimizer, seed, epochs):
    if engine == "opennn":
        return [opennn_bin, str(seed), optimizer, str(epochs)]
    if engine == "pytorch":
        return [PY, os.path.join(HERE, "pytorch_precision.py"), str(seed), optimizer, str(epochs)]
    if engine == "tensorflow":
        return [PY, os.path.join(HERE, "tensorflow_precision.py"), str(seed)]
    raise ValueError(engine)


def run_once(cmd):
    # CPU-only protocol (the blog benchmark is CPU): hide any GPU from PyTorch/TF.
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    out = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, env=env)
    fields = {}
    for line in (out.stdout + out.stderr).splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            fields[k.strip()] = v.strip()
    return fields, out.returncode, out.stdout + out.stderr


def score_mse(pred_file):
    """Full-dataset MSE via score.py's logic, computed identically per engine."""
    pred_path = os.path.join(HERE, pred_file)
    if not os.path.exists(pred_path):
        return None
    return score.mse(CSV, pred_path)


def versions():
    v = {"python": sys.version.split()[0]}
    code = ("import json;out={}\n"
            "for m in ('torch','tensorflow'):\n"
            "    try:\n"
            "        out[m]=__import__(m).__version__\n"
            "    except Exception:\n"
            "        pass\n"
            "print(json.dumps(out))")
    try:
        proc = subprocess.run([PY, "-c", code], capture_output=True, text=True)
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if lines:
            v.update(json.loads(lines[-1]))
    except Exception:
        pass
    return v


def git_commit():
    try:
        c = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"],
                           capture_output=True, text=True).stdout.strip()
        return c or "unknown"
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--engines", default="opennn,pytorch,tensorflow")
    args = ap.parse_args()

    ensure_data()
    engines = [e for e in args.engines.split(",") if e in ENGINE_CONFIGS]
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(range(args.runs)))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    opennn_bin, opennn_found = find_opennn_bin()
    if "opennn" in engines and not opennn_found:
        print(f"warning: opennn_precision binary not found ({opennn_bin}); "
              f"set OPENNN_PRECISION_BIN or build the opennn_precision CMake target")

    result = {
        "schema_version": 1,
        "benchmark_id": "precision-rosenbrock",
        "run_id": run_id,
        "git_commit": git_commit(),
        "configuration": {
            "task": "Rosenbrock 10-input regression (DOCUMENTED HIGGS EXCEPTION)",
            "exception_reason": ("this benchmark documents OpenNN's second-order "
                                 "optimizers (Levenberg-Marquardt, Quasi-Newton), "
                                 "which are least-squares methods that do not apply "
                                 "to HIGGS/BCE classification"),
            "model": "MLP 10 -> 10 tanh -> 1 linear, U(-1,1) init, MSE, full dataset (no split)",
            "seeds": seeds,
            "metric": "full-dataset MSE via the neutral scorer (score.py)",
            "engines": engines,
        },
        "machine": versions(),
        "runner": {
            "python": PY,
            "opennn_binary": opennn_bin,
            "opennn_binary_found": opennn_found,
        },
        "results": {},
    }

    for engine in engines:
        for label, optimizer, epochs, pred_file in ENGINE_CONFIGS[engine]:
            print(f"\n=== {label} ({args.runs} runs) ===")
            mses, times = [], []
            for seed in seeds:
                cmd = engine_cmd(engine, opennn_bin, optimizer, seed, epochs)
                fields, code, raw = run_once(cmd)
                if code != 0:
                    print(f"  seed {seed}: FAILED (rc={code}) {raw[-150:]}")
                    continue
                mse = score_mse(pred_file)
                try:
                    t = float(fields.get("train_time"))
                except (TypeError, ValueError):
                    t = None
                if mse is None:
                    print(f"  seed {seed}: NO PREDICTIONS ({pred_file})")
                    continue
                mses.append(mse)
                if t is not None:
                    times.append(t)
                tstr = f"{t:.2f}s" if t is not None else "n/a"
                print(f"  seed {seed}: mse={mse:.4f} time={tstr}")
            entry = {
                "engine": engine,
                "optimizer": optimizer,
                "epochs": epochs,
                "n_runs": len(seeds),
                "n_ok": len(mses),
            }
            if mses:
                entry["best_mse"] = round(min(mses), 6)
                entry["mean_mse"] = round(statistics.mean(mses), 6)
                entry["mean_time_s"] = round(statistics.mean(times), 4) if times else None
                print(f"  -> best_mse={entry['best_mse']} mean_mse={entry['mean_mse']} "
                      f"mean_time={entry['mean_time_s']}s")
            else:
                entry["error"] = "no successful run produced predictions"
            result["results"][label] = entry

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"precision-rosenbrock-{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
