#!/usr/bin/env python3
"""3-way convergence-gate harness: OpenNN vs PyTorch vs TensorFlow.

MLPerf-style metric: WALL-CLOCK TIME TO REACH A FIXED QUALITY TARGET, not
throughput at a fixed epoch count. Each engine trains the identical MLP on the
shared normalized Rosenbrock split until its epoch training-MSE reaches the same
target, then reports time-to-target, epochs taken, and the held-out TEST MSE.

Why it matters (audit S4.2): timing fixed epochs is gameable -- an engine can be
"fast" by not learning. A quality gate forces every engine to reach the same
held-out MSE and times how long that takes. The held-out test MSE is reported so
a low training loss that did not generalize cannot pass.

Each cell runs N times; reports median +/- stdev of time-to-target. A run that
fails to converge within max_epochs is recorded (reached_goal=0) and excluded
from the timing median.

  usage: run_convergence.py [--target 0.05] [--max-epochs 5000] [--lr 1e-3]
                            [--runs 5] [--seeds 0,1,2,3,4]
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
OPENNN_BIN = os.environ.get("OPENNN_CONVERGENCE_BIN", os.path.join(HERE, "opennn_convergence"))
GEN = os.path.join(HERE, "generate_rosenbrock.py")


def ensure_data():
    if not (os.path.exists(os.path.join(HERE, "rosenbrock_train.csv"))
            and os.path.exists(os.path.join(HERE, "rosenbrock_test.csv"))):
        subprocess.run([PY, GEN], cwd=HERE, check=True)


def engine_cmd(engine, seed, target, max_epochs, lr):
    args = [str(seed), str(target), str(max_epochs), str(lr)]
    if engine == "opennn":
        return [OPENNN_BIN] + args
    if engine == "pytorch":
        return [PY, os.path.join(HERE, "pytorch_convergence.py")] + args
    if engine == "tensorflow":
        return [PY, os.path.join(HERE, "tensorflow_convergence.py")] + args
    raise ValueError(engine)


def run_once(cmd):
    out = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    fields = {}
    for line in (out.stdout + out.stderr).splitlines():
        if "=" in line and not line.startswith("RESULT="):
            k, _, v = line.partition("=")
            fields[k.strip()] = v.strip()
        elif line.startswith("RESULT="):
            fields["RESULT"] = line.split("=", 1)[1].strip()
    return fields, out.stdout + out.stderr


def versions():
    v = {"python": sys.version.split()[0]}
    for mod in ("torch", "tensorflow"):
        try:
            v[mod] = __import__(mod).__version__
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
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--max-epochs", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--engines", default="opennn,pytorch,tensorflow")
    args = ap.parse_args()

    ensure_data()
    engines = [e for e in args.engines.split(",")
               if e in ("opennn", "pytorch", "tensorflow")]
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(range(args.runs)))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    result = {
        "schema_version": 1,
        "benchmark_id": "convergence-gate-rosenbrock",
        "run_id": run_id,
        "git_commit": git_commit(),
        "configuration": {
            "task": "Rosenbrock 10-input regression, MLP 10->50->50->1 tanh, Adam, MSE",
            "target_train_mse": args.target, "max_epochs": args.max_epochs,
            "lr": args.lr, "seeds": seeds,
            "metric": "wall-clock seconds to reach target training MSE (median over seeds)",
            "gate": "held-out test MSE reported; a non-generalizing fit cannot pass",
        },
        "machine": versions(),
        "results": {},
    }

    for eng in engines:
        print(f"\n=== {eng} ===")
        times, test_mses, epochs_list, n_reached = [], [], [], 0
        for seed in seeds:
            fields, raw = run_once(engine_cmd(eng, seed, args.target, args.max_epochs, args.lr))
            reached = fields.get("reached_goal") == "1"
            try:
                t = float(fields.get("time_to_target_s"))
                tm = float(fields.get("test_mse"))
                ep = int(fields.get("epochs_to_target"))
            except (TypeError, ValueError):
                print(f"  seed {seed}: PARSE FAIL ({raw[-150:]})")
                continue
            if reached:
                n_reached += 1
                times.append(t)
                test_mses.append(tm)
                epochs_list.append(ep)
                print(f"  seed {seed}: {t:7.2f}s  {ep:5d} ep  test_mse={tm:.4f}")
            else:
                print(f"  seed {seed}: DID NOT CONVERGE ({ep} ep, train_mse={fields.get('final_train_mse')})")
        entry = {"n_reached": n_reached, "n_runs": len(seeds)}
        if times:
            entry["time_to_target_s_median"] = round(statistics.median(times), 4)
            entry["time_to_target_s_stdev"] = round(statistics.pstdev(times) if len(times) > 1 else 0.0, 4)
            entry["test_mse_median"] = round(statistics.median(test_mses), 6)
            entry["epochs_to_target_median"] = int(statistics.median(epochs_list))
            print(f"  -> median {entry['time_to_target_s_median']}s, "
                  f"test_mse {entry['test_mse_median']}, {n_reached}/{len(seeds)} converged")
        result["results"][eng] = entry

    # OpenNN/competitor speedup on time-to-target (lower time is better).
    base = result["results"].get("opennn", {}).get("time_to_target_s_median")
    if base:
        for eng in ("pytorch", "tensorflow"):
            other = result["results"].get(eng, {}).get("time_to_target_s_median")
            if other:
                result["results"][f"opennn_speedup_vs_{eng}"] = round(other / base, 3)

    out_path = os.path.join(RESULTS_DIR, f"convergence-gate-rosenbrock-{run_id}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
