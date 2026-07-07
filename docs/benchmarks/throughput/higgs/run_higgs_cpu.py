#!/usr/bin/env python3
"""Run CPU HIGGS dense training or inference across OpenNN, PyTorch, TensorFlow."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shlex
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
KEY_VALUE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.+)$")
PY = os.environ.get("BENCH_PYTHON", str(Path.home() / "benchenv" / "bin" / "python"))


def run_text(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except Exception:
        return ""


def repo_root() -> Path:
    root = run_text(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"])
    return Path(root).resolve() if root else HERE.parents[2]


REPO_ROOT = repo_root()


def parse_scalar(text: str) -> Any:
    value = text.strip()
    try:
        number = float(value)
        if not math.isfinite(number):
            return value
        if number.is_integer() and "." not in value and "e" not in value.lower():
            return int(number)
        return number
    except ValueError:
        return value


def parse_metrics(raw: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for line in raw.splitlines():
        match = KEY_VALUE.match(line.strip())
        if match:
            metrics[match.group(1)] = parse_scalar(match.group(2))
    return metrics


def file_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path)}
    if path.exists():
        stat = path.stat()
        info.update({"exists": True, "bytes": stat.st_size, "mtime": stat.st_mtime})
    else:
        info["exists"] = False
    return info


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
        "status_short_sample": status_lines[:50],
        "status_short_truncated": len(status_lines) > 50,
    }


def framework_versions() -> dict[str, Any]:
    code = r"""
import json, platform, sys
info = {"python": sys.version.split()[0], "python_executable": sys.executable,
        "platform": platform.platform()}
try:
    import torch
    info["torch"] = torch.__version__
except Exception as exc:
    info["torch_error"] = str(exc)
try:
    import tensorflow as tf
    info["tensorflow"] = tf.__version__
except Exception as exc:
    info["tensorflow_error"] = str(exc)
print(json.dumps(info))
"""
    try:
        out = subprocess.run([PY, "-c", code], capture_output=True, text=True, check=False)
        lines = [line for line in out.stdout.splitlines() if line.strip()]
        return json.loads(lines[-1]) if lines else {}
    except Exception as exc:
        return {"version_error": str(exc), "python": PY, "platform": platform.platform()}


def candidate_names(base: str) -> list[str]:
    return [base, base + ".exe"] if os.name != "nt" else [base + ".exe", base]


def find_opennn_higgs_cpu() -> tuple[str, bool]:
    override = os.environ.get("OPENNN_HIGGS_CPU_BIN")
    if override:
        return override, Path(override).exists()
    dirs = [
        REPO_ROOT / "build-benchmarks" / "bin",
        REPO_ROOT / "build-benchmarks" / "bin" / "Release",
        REPO_ROOT / "build" / "bin",
        REPO_ROOT / "build" / "bin" / "Release",
        Path.home() / "opennn-wsl" / "build-gpu" / "bin",
    ]
    for directory in dirs:
        for name in candidate_names("opennn_higgs_cpu"):
            candidate = directory / name
            if candidate.exists():
                return str(candidate), True
    fallback = REPO_ROOT / "build-benchmarks" / "bin" / candidate_names("opennn_higgs_cpu")[0]
    return str(fallback), False


def binary_dependencies(path: str) -> str:
    binary = Path(path)
    if not binary.exists():
        return ""
    if sys.platform.startswith("linux"):
        return run_text(["ldd", str(binary)])
    if sys.platform == "darwin":
        return run_text(["otool", "-L", str(binary)])
    return ""


def opennn_binary_info(path: str, found: bool) -> dict[str, Any]:
    info = file_info(Path(path))
    info["found"] = found
    deps = binary_dependencies(path)
    if deps:
        dep_lines = deps.splitlines()
        info["dependency_lines"] = dep_lines
        info["mkl_linked"] = any("mkl" in line.lower() for line in dep_lines)
        info["tbb_linked"] = any("tbb" in line.lower() for line in dep_lines)
    return info


def benchmark_environment(args: argparse.Namespace) -> dict[str, Any]:
    keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_DYNAMIC",
        "MKL_DYNAMIC",
        "KMP_BLOCKTIME",
        "CUDA_VISIBLE_DEVICES",
        "TF_CPP_MIN_LOG_LEVEL",
        "OPENNN_HIGGS_CPU_BIN",
        "BENCH_PYTHON",
    ]
    return {
        "threads_argument": args.threads or None,
        "env": {key: os.environ.get(key) for key in keys if key in os.environ},
    }


def protocol_metadata(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "train":
        benchmark_class = "training_throughput_with_quality"
        quality_rule = {
            "metrics": ["test_accuracy", "test_log_loss", "test_roc_auc"],
            "status": "reported_not_gated",
        }
        timed_region = "one or more full training epochs after optional warmup epochs"
    else:
        benchmark_class = "inference_offline"
        quality_rule = {"status": "not_applicable"}
        timed_region = "repeated full passes over the test split after two warmup passes"

    return {
        "style": "mlperf_inspired",
        "official_mlperf": False,
        "benchmark_class": benchmark_class,
        "scope": "internal_subset",
        "quality_rule": quality_rule,
        "measurement_rule": {
            "runs": args.runs,
            "aggregation": "median",
            "timed_region": timed_region,
        },
    }


def display_command(cmd: list[str], env_over: dict[str, str]) -> str:
    env_bits = [f"{key}={value}" for key, value in sorted(env_over.items())]
    return " ".join(env_bits + [shlex.join(cmd)])


def engine_cmd(engine: str, args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    env = {"CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "2"}
    if engine == "opennn":
        binary, _ = find_opennn_higgs_cpu()
        if args.mode == "train":
            return [
                binary,
                "train",
                str(args.train),
                str(args.test),
                str(args.epochs),
                str(args.batch),
                str(args.hidden),
                str(args.hidden_layers),
                args.activation,
                str(args.warmup_epochs),
            ], env
        return [
            binary,
            "infer",
            str(args.test),
            str(args.reps),
            str(args.batch),
            str(args.hidden),
            str(args.hidden_layers),
            args.activation,
        ], env

    cmd = [
        PY,
        str(HERE / "higgs_framework_cpu.py"),
        engine,
        args.mode,
        "--test",
        str(args.test),
        "--batch",
        str(args.batch),
        "--hidden",
        str(args.hidden),
        "--hidden-layers",
        str(args.hidden_layers),
        "--activation",
        args.activation,
    ]
    if args.mode == "train":
        cmd += [
            "--train",
            str(args.train),
            "--epochs",
            str(args.epochs),
            "--warmup-epochs",
            str(args.warmup_epochs),
        ]
    else:
        cmd += ["--reps", str(args.reps)]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    return cmd, env


def run_once(cmd: list[str], env_over: dict[str, str], index: int) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(env_over)
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    raw = out.stdout + out.stderr
    return {
        "run_index": index,
        "returncode": out.returncode,
        "metrics": parse_metrics(raw),
        "stdout": out.stdout,
        "stderr": out.stderr,
        "raw_output": raw,
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [
        r for r in runs
        if r.get("returncode") == 0
        and r.get("metrics", {}).get("RESULT") == "OK"
        and isinstance(r.get("metrics", {}).get("samples_per_sec"), (int, float))
    ]
    summary: dict[str, Any] = {
        "n_runs": len(runs),
        "n_ok": len(ok),
        "last_metrics": runs[-1].get("metrics", {}) if runs else {},
    }
    for key in (
        "samples_per_sec",
        "median_epoch_s",
        "median_pass_s",
        "test_accuracy",
        "test_log_loss",
        "test_roc_auc",
    ):
        values = [
            float(r["metrics"][key])
            for r in ok
            if isinstance(r.get("metrics", {}).get(key), (int, float))
        ]
        if values:
            summary[f"{key}_median"] = round(statistics.median(values), 6)
            summary[f"{key}_stdev"] = round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0
    if not ok:
        summary["error"] = "no successful RESULT=OK run with samples_per_sec"
        if runs:
            summary["last_output_tail"] = runs[-1].get("raw_output", "")[-2000:]
    return summary


def metrics_summary(results: dict[str, Any]) -> dict[str, Any]:
    speed = {
        engine: data.get("samples_per_sec_median")
        for engine, data in results.items()
        if isinstance(data.get("samples_per_sec_median"), (int, float))
    }
    metrics: dict[str, Any] = {"samples_per_sec": speed, "speedup": {}, "energy_proxy": {}}
    base = speed.get("opennn")
    if isinstance(base, (int, float)) and base > 0:
        metrics["energy_proxy"]["opennn"] = 1.0
        for engine, value in speed.items():
            if engine == "opennn":
                continue
            if isinstance(value, (int, float)) and value > 0:
                metrics["speedup"][f"opennn_vs_{engine}"] = round(base / value, 3)
                metrics["energy_proxy"][engine] = round(base / value, 3)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=["train", "infer"])
    parser.add_argument("--train", type=Path, default=DEFAULT_HIGGS_DIR / "higgs_train.csv")
    parser.add_argument("--test", type=Path, default=DEFAULT_HIGGS_DIR / "higgs_test.csv")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--engines", default="opennn,pytorch,tensorflow")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train" and not args.train.exists():
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
    out_path = args.output or RESULTS_DIR / f"cpu-dense-higgs-{args.mode}-{run_id}.json"
    opennn_bin, opennn_found = find_opennn_higgs_cpu()
    git = git_metadata()

    result: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": f"cpu-dense-higgs-{args.mode}",
        "run_id": run_id,
        "git_commit": git["commit"],
        "protocol": protocol_metadata(args),
        "git": git,
        "dataset": "HIGGS",
        "dataset_files": {
            "train": file_info(args.train) if args.mode == "train" else None,
            "test": file_info(args.test),
            "metadata": file_info(args.test.parent / "higgs_metadata.json"),
        },
        "configuration": {
            "mode": args.mode,
            "device": "cpu",
            "model": "28 -> hidden -> hidden -> 1 dense binary classifier",
            "epochs": args.epochs if args.mode == "train" else None,
            "warmup_epochs": args.warmup_epochs if args.mode == "train" else None,
            "reps": args.reps if args.mode == "infer" else None,
            "batch": args.batch,
            "hidden": args.hidden,
            "hidden_layers": args.hidden_layers,
            "activation": args.activation,
            "runs": args.runs,
            "threads_argument": args.threads or None,
            "engines": engines,
            "energy_note": "CPU joules not measured in WSL; energy_proxy is normalized inverse throughput.",
        },
        "machine": framework_versions(),
        "environment": benchmark_environment(args),
        "runner": {
            "path": os.path.relpath(__file__, REPO_ROOT),
            "cwd": os.getcwd(),
            "argv": sys.argv,
            "python": PY,
            "opennn_binary": opennn_bin,
            "opennn_binary_found": opennn_found,
            "opennn_binary_info": opennn_binary_info(opennn_bin, opennn_found),
        },
        "commands": {},
        "results": {},
    }

    for engine in engines:
        cmd, env_over = engine_cmd(engine, args)
        command_text = display_command(cmd, env_over)
        result["commands"][engine] = command_text
        print(f"{engine}: {command_text}")
        runs = []
        for index in range(1, args.runs + 1):
            run = run_once(cmd, env_over, index)
            runs.append(run)
            metrics = run.get("metrics", {})
            sps = metrics.get("samples_per_sec")
            status = metrics.get("RESULT", "NO_RESULT")
            if isinstance(sps, (int, float)):
                print(f"  run {index}: {sps:.0f} samples/s, result={status}")
            else:
                print(f"  run {index}: failed, result={status}, returncode={run.get('returncode')}")
        summary = summarize(runs)
        summary.update({"command": command_text, "runs": runs})
        result["results"][engine] = summary

    result["metrics"] = metrics_summary(result["results"])
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
