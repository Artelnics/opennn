#!/usr/bin/env python3
"""OpenNN recurrent-vs-LSTM forecasting benchmark harness.

Prepares UCI Beijing PM2.5 data, runs the C++ forecasting driver, parses its
machine-readable METRIC and SPEEDUP lines, and writes an immutable JSON artifact.

  usage: run_forecasting.py [--bin /path/to/recurrent_lstm_forecasting_benchmark]
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
REPO_ROOT = HERE.parents[3]
RESULTS_DIR = HERE.parent.parent / "results"
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
        "timed_region": "training loop per network/scenario/seed (each engine)",
        "warmup": "none; full training wall time is measured",
        "runs": 5,
        "aggregation": "mean +/- sample std over 5 seeds (0..4); best = min test RMSE",
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


BINARY_NAME = "recurrent_lstm_forecasting_benchmark"


def default_binary():
    env_bin = os.environ.get("OPENNN_FORECASTING_BIN") or os.environ.get("OPENNN_NO2_FORECASTING_BIN")
    if env_bin:
        return env_bin

    candidates = [
        REPO_ROOT / "build-gpu" / "bin" / BINARY_NAME,
        REPO_ROOT / "build-ninja" / "bin" / BINARY_NAME,
        REPO_ROOT / "build-ninja" / "bin" / f"{BINARY_NAME}.exe",
        REPO_ROOT / "build" / "bin" / BINARY_NAME,
        REPO_ROOT / "build" / "bin" / f"{BINARY_NAME}.exe",
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


def parse_metrics(raw, metrics, speedups, default_engine="opennn"):
    """Merge METRIC/SPEEDUP lines from one engine's output into metrics/speedups.

    metrics is keyed engine -> phase -> scenario -> net. OpenNN lines carry no
    'engine'/'seed' token (default_engine, single aggregated value); PyTorch and
    TensorFlow emit per-seed lines plus one 'seed=aggregate' line, and only the
    aggregate is kept as the headline value.
    """
    for line in raw.splitlines():
        if line.startswith("METRIC "):
            fields = parse_key_values(line)
            engine = fields.pop("engine", default_engine)
            phase = fields.pop("phase", None)
            scenario = fields.pop("scenario", None)
            net = fields.pop("net", None)
            seed = fields.get("seed", None)
            if seed not in (None, "aggregate"):
                continue  # per-seed line; the aggregate row is the headline
            if not phase or not scenario or not net:
                continue
            scenario_entry = (metrics.setdefault(engine, {})
                                     .setdefault(phase, {})
                                     .setdefault(scenario, {}))
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


PYTHON_ENGINES = {
    "pytorch": {"script": HERE / "pt_forecasting.py",
                "version": "import torch; print(torch.__version__)"},
    "tensorflow": {"script": HERE / "tf_forecasting.py",
                   "version": "import tensorflow as tf; print(tf.__version__)"},
}


def python_engine_version(snippet):
    return command_output([sys.executable, "-c", snippet])


def run_python_engine(name, gpu_index, force_cpu, timeout):
    """Run a PyTorch/TensorFlow engine script from HERE (so xf_common finds
    data/). force_cpu hides the GPU via CUDA_VISIBLE_DEVICES="" and passes
    --allow-cpu; without it the scripts abort on a silent CPU fallback.
    Returns the combined stdout+stderr, or None on failure."""
    script = PYTHON_ENGINES[name]["script"]
    env = dict(os.environ)
    if force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    command = [sys.executable, str(script)]
    if force_cpu:
        command.append("--allow-cpu")
    print(f"{name}_command={quote_command(command)}"
          f"{' (forced CPU)' if force_cpu else ''}")
    try:
        proc = subprocess.run(command, cwd=HERE, env=env, capture_output=True,
                              text=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"{name} failed to launch: {exc}", file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(f"{name} returncode={proc.returncode}", file=sys.stderr)
        print((proc.stdout + proc.stderr)[-2000:], file=sys.stderr)
        if name == "tensorflow" and proc.returncode == 2:
            print("hint: install the TF CUDA stack in this interpreter's env:\n"
                  f"  {sys.executable} -m pip install -r "
                  f"{HERE / 'requirements-gpu.txt'}", file=sys.stderr)
    return proc.stdout + proc.stderr


def device_check(metrics):
    """Per engine/phase device summary parsed from METRIC lines, flagging any
    GPU phase that actually ran on cpu (device mismatch)."""
    summary = {}
    for engine, phases in metrics.items():
        for phase, scenarios in phases.items():
            devices = set()
            for nets in scenarios.values():
                for net, fields in nets.items():
                    if isinstance(fields, dict) and "device" in fields:
                        devices.add(str(fields["device"]))
            if not devices:
                continue
            entry = summary.setdefault(engine, {})
            entry[phase] = sorted(devices)
            if phase == "GPU" and "cpu" in devices:
                entry["mismatch"] = True
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", default=default_binary(),
                        help=f"Path to the {BINARY_NAME} executable")
    parser.add_argument("--data-dir", default=HERE / "data", type=Path,
                        help="Directory containing/preparing beijing_pm25_forecasting.csv")
    parser.add_argument("--raw-path", type=Path,
                        help="Optional local UCI raw CSV or ZIP passed to prepare_beijing_pm25.py")
    parser.add_argument("--no-prepare", action="store_true",
                        help="Require the prepared CSV to already exist")
    parser.add_argument("--gpu-index", type=int, default=env_gpu_index())
    parser.add_argument("--timeout-s", type=int, default=0, help="0 means no timeout")
    parser.add_argument("--frameworks", default="opennn",
                        help="Comma list of engines to run: opennn,pytorch,tensorflow "
                             "(pytorch/tensorflow read data/ prepared by this runner). "
                             "The OpenNN binary only runs when 'opennn' is listed.")
    parser.add_argument("--python-cpu", action="store_true",
                        help="Also run a forced-CPU pass of each Python engine "
                             "(CUDA_VISIBLE_DEVICES=\"\") so their CPU phase is recorded")
    parser.add_argument("--opennn-phase", default="", choices=["", "gpu", "cpu", "both"],
                        help="Restrict the OpenNN driver to one phase via "
                             "OPENNN_FORECASTING_PHASE (default: both)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    frameworks = [f.strip() for f in args.frameworks.split(",") if f.strip()]
    python_frameworks = [f for f in frameworks if f in PYTHON_ENGINES]
    run_opennn = "opennn" in frameworks
    unknown = [f for f in frameworks if f not in ("opennn",) and f not in PYTHON_ENGINES]
    if unknown:
        raise SystemExit(f"Unknown framework(s): {', '.join(unknown)}")

    binary = Path(args.bin).expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    prepared_csv = data_dir / DATA_FILE
    command = [str(binary)]
    cwd = binary.parent
    env = dict(os.environ)
    env["OPENNN_FORECASTING_DATASET"] = DATASET_PROFILE
    env["OPENNN_FORECASTING_DATA_DIR"] = str(data_dir)
    if args.opennn_phase in ("gpu", "cpu"):
        env["OPENNN_FORECASTING_PHASE"] = args.opennn_phase
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
    metrics = {}
    speedups = {}
    raw_outputs = {}
    framework_versions = {}

    started = time.perf_counter()
    opennn_returncode = 0
    raw = ""
    if run_opennn:
        proc = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                              text=True, timeout=timeout)
        raw = proc.stdout + proc.stderr
        opennn_returncode = proc.returncode
        parse_metrics(raw, metrics, speedups, default_engine="opennn")
        raw_outputs["opennn"] = raw
    else:
        print("skipping OpenNN binary ('opennn' not in --frameworks)")

    for name in python_frameworks:
        framework_versions[name] = python_engine_version(PYTHON_ENGINES[name]["version"])
        passes = [False, True] if args.python_cpu else [False]
        for force_cpu in passes:
            out = run_python_engine(name, args.gpu_index, force_cpu, timeout)
            if out is None:
                continue
            parse_metrics(out, metrics, speedups, default_engine=name)
            raw_outputs[f"{name}{'_cpu' if force_cpu else ''}"] = out
    elapsed = time.perf_counter() - started

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
            "task": "Time-series forecasting: Recurrent vs LSTM",
            "engines": ["opennn"] + python_frameworks,
            "driver": str(binary),
            "working_directory": str(cwd),
            "data_dir": str(data_dir),
            "metric": "test RMSE (standard sqrt(mean sq), original units), "
                      "training time, CPU/GPU speedup by scenario",
            "rmse_convention": "standard sqrt(mean((pred-true)^2)); OpenNN's native "
                               "errs(2)=sqrt(sum/2N) is reported as "
                               "test_rmse_native_halfconv_mean",
            "precision": "fp32",
            "seed_count": 5,
            "seeds": [0, 1, 2, 3, 4],
            "dataset_profile_env": DATASET_PROFILE,
            "opennn_phase": args.opennn_phase or "both",
            "device_check": device_check(metrics),
        },
        "machine": versions(args.gpu_index),
        "framework_versions": framework_versions,
        "command": quote_command(command) if run_opennn else None,
        "returncode": opennn_returncode,
        "elapsed_wall_s": round(elapsed, 3),
        "results": {
            "metrics": metrics,
            "speedups": speedups,
        },
        "raw_outputs": raw_outputs,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = RESULTS_DIR / f"recurrent-lstm-forecasting-{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    print(f"wrote {out_path}")
    checks = device_check(metrics)
    for engine, entry in checks.items():
        if entry.get("mismatch"):
            print(f"WARNING device_mismatch engine={engine}: GPU phase ran on cpu",
                  file=sys.stderr)
    if opennn_returncode != 0:
        print(raw[-2000:], file=sys.stderr)
        raise SystemExit(opennn_returncode)


if __name__ == "__main__":
    main()
