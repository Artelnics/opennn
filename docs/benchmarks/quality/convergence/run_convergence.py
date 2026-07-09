#!/usr/bin/env python3
"""3-way HIGGS convergence-gate harness: OpenNN vs PyTorch vs TensorFlow.

MLPerf-style metric: WALL-CLOCK TIME TO REACH A FIXED QUALITY TARGET, not
throughput at a fixed epoch count. Each engine trains the identical canonical
HIGGS dense classifier (28 -> 1024 -> 1024 -> 1, ReLU, sigmoid, BCE, Adam) on
the shared HIGGS split until the HELD-OUT (test) log-loss reaches the same
target, then reports time-to-target, epochs taken, and the final held-out
log-loss.

Why it matters (audit S4.2): timing fixed epochs is gameable -- an engine can
look "fast" by not learning. A quality gate forces every engine to reach the
same held-out log-loss and times how long that takes. The clock counts training
time only; per-epoch (per-chunk for OpenNN) evaluation is excluded.

Data comes from $OPENNN_BENCH_DATA/higgs/{higgs_train.csv,higgs_test.csv}; see
../DATA_POLICY.md and ../../throughput/higgs/README.md.

Each cell runs N times; reports median +/- stdev of time-to-target over the runs
that reached the target. A run that fails to converge within --max-epochs is
recorded (reached_goal=0) and excluded from the timing median.

  usage: run_convergence.py [--target 0.60] [--max-epochs 50] [--runs 5]
                            [--batch 1024] [--hidden 1024] [--hidden-layers 2]
                            [--engines opennn,pytorch,tensorflow]
                            [--train TRAIN.csv] [--test TEST.csv]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RESULTS_DIR = (HERE.parent.parent / "results").resolve()
DEFAULT_BENCH_DATA = Path(
    os.environ.get("OPENNN_BENCH_DATA", str(Path.home() / "opennn-benchmark-data"))
)
DEFAULT_HIGGS_DIR = DEFAULT_BENCH_DATA / "higgs"
PY = os.environ.get("BENCH_PYTHON", sys.executable)


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except Exception:
        return ""


def repo_root() -> Path:
    root = run_text(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"])
    return Path(root).resolve() if root else HERE.parents[3]


REPO_ROOT = repo_root()


def candidate_names(base: str) -> list[str]:
    return [base, base + ".exe"] if os.name != "nt" else [base + ".exe", base]


def find_opennn_bin() -> tuple[str, bool]:
    override = os.environ.get("OPENNN_CONVERGENCE_BIN")
    if override:
        return override, Path(override).exists()
    dirs = [
        REPO_ROOT / "build-benchmarks" / "bin",
        REPO_ROOT / "build-benchmarks" / "bin" / "Release",
        REPO_ROOT / "build" / "bin",
        REPO_ROOT / "build" / "bin" / "Release",
    ]
    for directory in dirs:
        for name in candidate_names("opennn_convergence"):
            candidate = directory / name
            if candidate.exists():
                return str(candidate), True
    fallback = REPO_ROOT / "build-benchmarks" / "bin" / candidate_names("opennn_convergence")[0]
    return str(fallback), False


def engine_cmd(engine: str, args: argparse.Namespace) -> list[str]:
    if engine == "opennn":
        binary, _ = find_opennn_bin()
        return [
            binary,
            str(args.train),
            str(args.test),
            str(args.target),
            str(args.max_epochs),
            str(args.batch),
            str(args.hidden),
            str(args.hidden_layers),
        ]
    script = "pytorch_convergence.py" if engine == "pytorch" else "tensorflow_convergence.py"
    cmd = [
        PY,
        str(HERE / script),
        "--train", str(args.train),
        "--test", str(args.test),
        "--target", str(args.target),
        "--max-epochs", str(args.max_epochs),
        "--batch", str(args.batch),
        "--hidden", str(args.hidden),
        "--hidden-layers", str(args.hidden_layers),
    ]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    return cmd


def run_once(cmd: list[str]) -> tuple[dict[str, str], str]:
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    raw = out.stdout + out.stderr
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip()
    return fields, raw


def file_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path)}
    if path.exists():
        stat = path.stat()
        info.update({"exists": True, "bytes": stat.st_size, "mtime": stat.st_mtime})
    else:
        info["exists"] = False
    return info


def versions() -> dict[str, Any]:
    v: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    code = (
        "import json\n"
        "info={}\n"
        "try:\n import torch; info['torch']=torch.__version__\n"
        "except Exception as e: info['torch_error']=str(e)\n"
        "try:\n import tensorflow as tf; info['tensorflow']=tf.__version__\n"
        "except Exception as e: info['tensorflow_error']=str(e)\n"
        "print(json.dumps(info))\n"
    )
    try:
        out = subprocess.run([PY, "-c", code], capture_output=True, text=True, check=False)
        lines = [line for line in out.stdout.splitlines() if line.strip()]
        if lines:
            v.update(json.loads(lines[-1]))
    except Exception as exc:
        v["version_error"] = str(exc)
    return v


def git_metadata() -> dict[str, Any]:
    commit = run_text(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    branch = run_text(["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"])
    status = run_text(["git", "-C", str(REPO_ROOT), "status", "--short", "--untracked-files=no"])
    status_lines = status.splitlines()
    return {
        "commit": commit or "unknown",
        "branch": branch or "unknown",
        "dirty": bool(status_lines),
        "status_short_count": len(status_lines),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train", type=Path, default=DEFAULT_HIGGS_DIR / "higgs_train.csv")
    parser.add_argument("--test", type=Path, default=DEFAULT_HIGGS_DIR / "higgs_test.csv")
    parser.add_argument("--target", type=float, default=0.60,
                        help="held-out test log-loss target (lower is better)")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--engines", default="opennn,pytorch,tensorflow")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train.exists():
        raise SystemExit(f"HIGGS train file not found: {args.train}")
    if not args.test.exists():
        raise SystemExit(f"HIGGS test file not found: {args.test}")

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    valid = {"opennn", "pytorch", "tensorflow"}
    unknown = [e for e in engines if e not in valid]
    if unknown:
        raise SystemExit(f"unknown engine(s): {', '.join(unknown)}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or RESULTS_DIR / f"convergence-higgs-{run_id}.json"
    opennn_bin, opennn_found = find_opennn_bin()
    git = git_metadata()

    result: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": "convergence-higgs",
        "run_id": run_id,
        "git_commit": git["commit"],
        "git": git,
        "dataset": "HIGGS",
        "dataset_files": {
            "train": file_info(args.train),
            "test": file_info(args.test),
            "metadata": file_info(args.test.parent / "higgs_metadata.json"),
        },
        "protocol": {
            "style": "mlperf_inspired",
            "official_mlperf": False,
            "benchmark_class": "time_to_target_quality",
            "quality_rule": {
                "metric": "test_log_loss",
                "direction": "<=",
                "target": args.target,
                "evaluated_on": "held-out test split, after each epoch "
                                "(OpenNN: after each training chunk)",
                "excluded_from_clock": "per-epoch/per-chunk held-out evaluation",
            },
            "measurement_rule": {
                "runs": args.runs,
                "aggregation": "median over runs that reached the target",
                "timed_region": "training epochs until the held-out target is reached",
            },
        },
        "configuration": {
            "task": "HIGGS binary classification, dense MLP 28 -> hidden -> hidden -> 1, "
                    "ReLU, sigmoid, BCE, Adam",
            "device": "cpu",
            "precision": "fp32",
            "target_test_log_loss": args.target,
            "max_epochs": args.max_epochs,
            "batch": args.batch,
            "hidden": args.hidden,
            "hidden_layers": args.hidden_layers,
            "runs": args.runs,
            "threads_argument": args.threads or None,
            "engines": engines,
            "metric": "wall-clock seconds to reach target held-out log-loss "
                      "(median over runs)",
            "gate": "held-out test log-loss; a non-generalizing fit cannot pass",
        },
        "machine": versions(),
        "runner": {
            "path": os.path.relpath(__file__, REPO_ROOT),
            "cwd": os.getcwd(),
            "argv": sys.argv,
            "python": PY,
            "opennn_binary": opennn_bin,
            "opennn_binary_found": opennn_found,
        },
        "commands": {},
        "results": {},
    }

    for engine in engines:
        print(f"\n=== {engine} ===")
        cmd = engine_cmd(engine, args)
        result["commands"][engine] = " ".join(cmd)
        times: list[float] = []
        log_losses: list[float] = []
        epochs_list: list[int] = []
        runs: list[dict[str, Any]] = []
        n_reached = 0
        for index in range(1, args.runs + 1):
            fields, raw = run_once(cmd)
            reached = fields.get("reached_goal") == "1"
            try:
                t = float(fields["time_to_target_s"])
                ll = float(fields["test_log_loss"])
                ep = int(fields["epochs_to_target"])
            except (KeyError, TypeError, ValueError):
                print(f"  run {index}: PARSE FAIL ({raw[-200:]})")
                runs.append({"run_index": index, "reached_goal": False,
                             "metrics": fields, "raw_output_tail": raw[-2000:]})
                continue
            runs.append({
                "run_index": index,
                "reached_goal": reached,
                "time_to_target_s": t,
                "test_log_loss": ll,
                "epochs_to_target": ep,
                "result": fields.get("RESULT", "NO_RESULT"),
            })
            if reached:
                n_reached += 1
                times.append(t)
                log_losses.append(ll)
                epochs_list.append(ep)
                print(f"  run {index}: {t:8.2f}s  {ep:4d} ep  test_log_loss={ll:.4f}")
            else:
                print(f"  run {index}: DID NOT CONVERGE ({ep} ep, "
                      f"test_log_loss={fields.get('test_log_loss')})")

        entry: dict[str, Any] = {"n_reached": n_reached, "n_runs": args.runs, "runs": runs}
        if times:
            entry["time_to_target_s_median"] = round(statistics.median(times), 4)
            entry["time_to_target_s_stdev"] = round(
                statistics.pstdev(times) if len(times) > 1 else 0.0, 4)
            entry["test_log_loss_median"] = round(statistics.median(log_losses), 6)
            entry["epochs_to_target_median"] = int(statistics.median(epochs_list))
            print(f"  -> median {entry['time_to_target_s_median']}s, "
                  f"test_log_loss {entry['test_log_loss_median']}, "
                  f"{n_reached}/{args.runs} converged")
        else:
            print(f"  -> 0/{args.runs} converged")
        result["results"][engine] = entry

    # OpenNN/competitor speedup on time-to-target (lower time is better).
    base = result["results"].get("opennn", {}).get("time_to_target_s_median")
    if base:
        for engine in ("pytorch", "tensorflow"):
            other = result["results"].get(engine, {}).get("time_to_target_s_median")
            if other:
                result["results"][f"opennn_speedup_vs_{engine}"] = round(other / base, 3)

    out_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
