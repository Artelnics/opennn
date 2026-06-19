#!/usr/bin/env python3
"""OpenNN recurrent-vs-LSTM forecasting benchmark harness.

Prepares UCI Beijing PM2.5 data, runs the C++ forecasting driver, parses its
machine-readable METRIC and SPEEDUP lines, and writes an immutable JSON artifact.

  usage: run_forecasting.py [--bin /path/to/no2_forecasting]
                            [--data-dir ./data] [--gpu-index 0]
                            [--timeout-s 0] [--dry-run]
"""

import argparse
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RESULTS_DIR = HERE.parent / "results"
PREP_SCRIPT = HERE / "prepare_beijing_pm25.py"
DATASET_PROFILE = "beijing_pm25"
DATA_FILE = "beijing_pm25_forecasting.csv"
UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data"
PROTOCOL = {
    "style": "mlperf_inspired",
    "official_mlperf": False,
    "benchmark_class": "training_throughput_with_quality",
    "division": "closed",
    "quality_rule": {
        "metric": "test_rmse_mean",
        "target": None,
        "status": "reported_not_gated_until_reference_linux_calibration",
    },
    "measurement_rule": {
        "timed_region": "C++ training loop per network/scenario",
        "warmup": "none; full training wall time is measured",
        "runs": 1,
        "aggregation": "mean over C++ seed count; repeat JSON artifacts before headline use",
    },
}


def env_gpu_index():
    value = os.environ.get("BENCH_GPU_INDEX")
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def default_binary():
    env_bin = os.environ.get("OPENNN_FORECASTING_BIN") or os.environ.get("OPENNN_NO2_FORECASTING_BIN")
    if env_bin:
        return env_bin

    candidates = [
        REPO_ROOT / "build-gpu" / "bin" / "no2_forecasting",
        REPO_ROOT / "build-ninja" / "bin" / "no2_forecasting",
        REPO_ROOT / "build-ninja" / "bin" / "no2_forecasting.exe",
        REPO_ROOT / "build" / "bin" / "no2_forecasting",
        REPO_ROOT / "build" / "bin" / "no2_forecasting.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_commit():
    return command_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]) or "unknown"


def versions(gpu_index):
    gpu_command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ]
    if gpu_index is not None:
        gpu_command[1:1] = ["-i", str(gpu_index)]

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": command_output(gpu_command),
    }


def quote_command(command):
    return " ".join(shlex.quote(str(part)) for part in command)


def count_csv_rows(path):
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return None


def prepare_command(data_dir, raw_path):
    command = [sys.executable, str(PREP_SCRIPT), "--output-dir", str(data_dir)]
    if raw_path:
        command.extend(["--raw-path", str(raw_path)])
    return command


def ensure_prepared_data(data_dir, raw_path, no_prepare):
    data_dir.mkdir(parents=True, exist_ok=True)
    prepared_csv = data_dir / DATA_FILE
    if prepared_csv.exists():
        return prepared_csv, False

    if no_prepare:
        raise SystemExit(
            f"Prepared dataset not found: {prepared_csv}. "
            "Run prepare_beijing_pm25.py or omit --no-prepare."
        )

    command = prepare_command(data_dir, raw_path)
    print(f"prepare_command={quote_command(command)}")
    subprocess.run(command, cwd=HERE, check=True)

    if not prepared_csv.exists():
        raise SystemExit(f"Preparation did not create {prepared_csv}")
    return prepared_csv, True


def parse_value(value):
    try:
        number = float(value)
    except ValueError:
        return value

    if not math.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return number


def parse_key_values(line):
    fields = {}
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = parse_value(value)
    return fields


def parse_output(raw):
    metrics = {}
    speedups = {}

    for line in raw.splitlines():
        if line.startswith("METRIC "):
            fields = parse_key_values(line)
            phase = fields.pop("phase", None)
            scenario = fields.pop("scenario", None)
            net = fields.pop("net", None)
            if not phase or not scenario or not net:
                continue
            phase_entry = metrics.setdefault(phase, {})
            scenario_entry = phase_entry.setdefault(scenario, {})
            scenario_entry[net] = fields
            if "winner" in fields:
                scenario_entry["winner"] = fields["winner"]
        elif line.startswith("SPEEDUP "):
            fields = parse_key_values(line)
            scenario = fields.pop("scenario", None)
            net = fields.pop("net", None)
            if not scenario or not net:
                continue
            speedups.setdefault(scenario, {})[net] = fields

    return metrics, speedups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", default=default_binary(), help="Path to no2_forecasting executable")
    parser.add_argument("--data-dir", default=HERE / "data", type=Path,
                        help="Directory containing/preparing beijing_pm25_forecasting.csv")
    parser.add_argument("--raw-path", type=Path,
                        help="Optional local UCI raw CSV or ZIP passed to prepare_beijing_pm25.py")
    parser.add_argument("--no-prepare", action="store_true",
                        help="Require the prepared CSV to already exist")
    parser.add_argument("--gpu-index", type=int, default=env_gpu_index())
    parser.add_argument("--timeout-s", type=int, default=0, help="0 means no timeout")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    binary = Path(args.bin).expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    prepared_csv = data_dir / DATA_FILE
    command = [str(binary)]
    cwd = binary.parent
    env = dict(os.environ)
    env["OPENNN_FORECASTING_DATASET"] = DATASET_PROFILE
    env["OPENNN_FORECASTING_DATA_DIR"] = str(data_dir)
    if args.gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    print(f"command={quote_command(command)}")
    print(f"cwd={cwd}")
    print(f"data_dir={data_dir}")
    print(f"prepared_csv={prepared_csv}")
    print(f"dataset_profile={DATASET_PROFILE}")
    if args.dry_run:
        if not prepared_csv.exists() and not args.no_prepare:
            print(f"prepare_command={quote_command(prepare_command(data_dir, args.raw_path))}")
        return

    prepared_csv, prepared_by_runner = ensure_prepared_data(data_dir, args.raw_path, args.no_prepare)

    timeout = args.timeout_s if args.timeout_s > 0 else None
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=timeout)
    elapsed = time.perf_counter() - started
    raw = proc.stdout + proc.stderr
    metrics, speedups = parse_output(raw)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "schema_version": 1,
        "benchmark_id": "recurrent-lstm-forecasting",
        "run_id": run_id,
        "git_commit": git_commit(),
        "protocol": PROTOCOL,
        "dataset": {
            "name": "UCI Beijing PM2.5",
            "profile": DATASET_PROFILE,
            "source": UCI_DATASET_URL,
            "prepared_csv": str(prepared_csv),
            "rows": count_csv_rows(prepared_csv),
            "prepared_by_runner": prepared_by_runner,
        },
        "configuration": {
            "task": "Time-series forecasting: OpenNN Recurrent vs OpenNN LSTM",
            "driver": str(binary),
            "working_directory": str(cwd),
            "data_dir": str(data_dir),
            "metric": "test RMSE, training time, CPU/GPU speedup by scenario",
            "precision": "fp32",
            "seed_count": 1,
            "dataset_profile_env": DATASET_PROFILE,
        },
        "machine": versions(args.gpu_index),
        "command": quote_command(command),
        "returncode": proc.returncode,
        "elapsed_wall_s": round(elapsed, 3),
        "results": {
            "metrics": metrics,
            "speedups": speedups,
        },
        "raw_output": raw,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = RESULTS_DIR / f"recurrent-lstm-forecasting-{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    print(f"wrote {out_path}")
    if proc.returncode != 0:
        print(raw[-2000:], file=sys.stderr)
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
