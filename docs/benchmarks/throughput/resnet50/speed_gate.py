#!/usr/bin/env python3
"""ResNet-50 GPU speed gate: catch throughput regressions before they ship.

Runs the OpenNN ResNet-50 training harness at a few (batch, precision) points
and fails if any of the invariants that past regressions violated no longer
hold, or if throughput drops below this machine's recorded baseline:

  * RESULT=OK, whole batches only (processed_samples == floor(N/batch)*batch),
    or every sample when the point keeps the tail
  * convolution workspace bounded (workspace_cap_mib > 0) and autotune on
  * CUDA graphs requested and never abandoned ("continuing without graphs")
  * bf16: no batch-norm backward shape falls back to FP32 staging
  * samples/s >= baseline * (1 - tolerance) for this (GPU, cuDNN)

Baselines live next to this script in speed_gate_baselines.json, keyed by the
"device=" and "cudnn=" lines the harness prints, so a laptop and the
benchmarks machine each keep their own. `--record` writes/refreshes them.

Each point is run --repeats times and the best samples/s is kept, on both
sides: a laptop GPU throttles under back-to-back load and single runs scatter
by ~8%, so best-of-N against best-of-N is the honest comparison. The tolerance
is stored with the baseline so each machine keeps its own (10% for a laptop,
5% for the desktop benchmarks machine).

A point written as 128:bf16:tail keeps the remainder batch
(OPENNN_RESNET50_KEEP_TAIL=1) - the library's tail path - and additionally
asserts that the whole batches kept their CUDA graph, which a 2026-08-14
change had silently switched off for every dataset the batch does not divide.

  usage: speed_gate.py [--record] [--points 128:bf16,128:bf16:tail,1024:bf16,1024:fp32]
                       [--repeats 2] [--tolerance 0.10] [--epochs 3]
                       [--bin PATH] [--data-dir DIR]

Exit status 0 = pass, 1 = regression or invariant broken, 2 = could not run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINES = HERE / "speed_gate_baselines.json"
DEFAULT_BENCH_DATA = Path(os.environ.get("OPENNN_BENCH_DATA",
                                         str(Path.home() / "opennn-benchmark-data")))


def find_binary(explicit: str | None) -> str:
    if explicit:
        return explicit
    for env in ("OPENNN_RESNET50_BIN",):
        if os.environ.get(env):
            return os.environ[env]
    repo = HERE.parents[3]
    for directory in ("build", "build-benchmarks"):
        candidate = repo / directory / "bin" / "opennn_resnet50_speed"
        if candidate.exists():
            return str(candidate)
    return "opennn_resnet50_speed"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=False).stdout.strip()
    except Exception:
        return "unknown"


def run_point(binary: str, data: Path, epochs: int, batch: int, precision: str,
              keep_tail: bool = False) -> dict:
    cmd = [binary, str(data / "train"), str(epochs), str(batch), precision]
    env = dict(os.environ)
    env.setdefault("LD_LIBRARY_PATH", "/usr/local/cuda/lib64:")
    if keep_tail:
        env["OPENNN_RESNET50_KEEP_TAIL"] = "1"
    out = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    raw = out.stdout + out.stderr
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        if "=" in line and " " not in line.split("=", 1)[0]:
            key, _, value = line.partition("=")
            fields.setdefault(key.strip(), value.strip())
    config = re.search(r"samples=(\d+) batch=(\d+) .*cuda_graph=(\d)", raw)
    return {
        "batch": batch,
        "precision": precision,
        "keep_tail": keep_tail,
        "returncode": out.returncode,
        "fields": fields,
        "samples": int(config.group(1)) if config else None,
        "cuda_graph_requested": (config.group(3) == "1") if config else None,
        "graphs_abandoned": "continuing without graphs" in raw or "graph capture failed" in raw,
        "bn_staged": len(re.findall(r"BatchNormalizationOperator backward .*FP32-staged", raw)),
        "bn_degraded": len(re.findall(r"BatchNormalizationOperator backward ", raw)),
        "wgrad_no_fp32_store": len(re.findall(r"no FP32-store engine", raw)),
        "raw_tail": raw[-1500:],
    }


def check_point(result: dict, baseline: float | None, tolerance: float) -> list[str]:
    f = result["fields"]
    failures: list[str] = []
    if result["returncode"] != 0 or f.get("RESULT") != "OK":
        failures.append(f"run failed (exit {result['returncode']}, RESULT={f.get('RESULT')})")
        return failures

    batch = result["batch"]
    samples = result["samples"]
    processed = int(f.get("processed_samples", "0") or 0)
    if samples is not None:
        expected = samples if result["keep_tail"] else (samples // batch) * batch
        if processed != expected:
            failures.append(f"processed_samples={processed}, expected {expected}")
        if result["keep_tail"] and samples % batch == 0:
            failures.append(f"tail point but {samples} % {batch} == 0 - pick a batch that leaves a remainder")

    if int(f.get("workspace_cap_mib", "0") or 0) <= 0:
        failures.append("convolution workspace is unbounded (workspace_cap_mib=0) - the 2026-08-15 cliff")
    if f.get("conv_autotune") != "1":
        failures.append("conv_autotune is off")

    if result["cuda_graph_requested"] is False:
        failures.append("CUDA graph not requested by the harness")
    if result["graphs_abandoned"]:
        failures.append("CUDA graph capture was abandoned (continuing without graphs)")

    if result["precision"] == "bf16" and result["bn_staged"] > 0:
        failures.append(f"{result['bn_staged']} batch-norm backward shape(s) staged through FP32")

    sps = float(f.get("samples_per_sec", "0") or 0)
    if sps <= 0:
        failures.append("no samples_per_sec")
    elif baseline is not None and sps < baseline * (1.0 - tolerance):
        failures.append(f"throughput {sps:.0f} < baseline {baseline:.0f} - {tolerance:.0%}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--record", action="store_true", help="write this run as the baseline")
    parser.add_argument("--points", default="128:bf16,128:bf16:tail,1024:bf16,1024:fp32")
    parser.add_argument("--repeats", type=int, default=2,
                        help="runs per point; the best samples/s counts")
    parser.add_argument("--tolerance", type=float, default=0.10,
                        help="allowed drop below baseline; stored with --record, "
                             "the stored value wins on later checks")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bin")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_BENCH_DATA / "cifar10")
    args = parser.parse_args()

    binary = find_binary(args.bin)
    if not args.data_dir.exists():
        print(f"data dir not found: {args.data_dir}", file=sys.stderr)
        return 2

    points = []
    for item in args.points.split(","):
        parts = item.split(":")
        batch = int(parts[0])
        precision = parts[1] if len(parts) > 1 and parts[1] else "bf16"
        keep_tail = len(parts) > 2 and parts[2] == "tail"
        points.append((batch, precision, keep_tail))

    baselines = json.loads(BASELINES.read_text()) if BASELINES.exists() else {}

    machine_key = None
    all_failures: dict[str, list[str]] = {}
    measured: dict[str, float] = {}
    rows = []
    for batch, precision, keep_tail in points:
        # Best of --repeats: keep the run with the highest samples/s, but any
        # invariant failure in any run counts.
        result = None
        for _ in range(max(1, args.repeats)):
            attempt = run_point(binary, args.data_dir, args.epochs, batch, precision, keep_tail)
            attempt_sps = float(attempt["fields"].get("samples_per_sec", "0") or 0)
            if result is None or attempt_sps > float(result["fields"].get("samples_per_sec", "0") or 0):
                if result is not None:
                    attempt["graphs_abandoned"] |= result["graphs_abandoned"]
                    attempt["bn_staged"] = max(attempt["bn_staged"], result["bn_staged"])
                result = attempt
        f = result["fields"]
        if machine_key is None and f.get("device"):
            machine_key = f"{f.get('device')} | cudnn {f.get('cudnn', '?')}"
        point_key = f"{precision}@{batch}" + ("+tail" if keep_tail else "")
        baseline = None
        tolerance = args.tolerance
        if machine_key and machine_key in baselines:
            entry = baselines[machine_key]
            baseline = entry.get("points", {}).get(point_key, {}).get("samples_per_sec")
            tolerance = float(entry.get("tolerance", tolerance))
        failures = check_point(result, baseline, tolerance)
        sps = float(f.get("samples_per_sec", "0") or 0)
        measured[point_key] = sps
        rows.append((point_key, sps, baseline, result["bn_staged"], result["bn_degraded"],
                     result["wgrad_no_fp32_store"], failures))
        if failures:
            all_failures[point_key] = failures
            if result["returncode"] != 0:
                print(result["raw_tail"], file=sys.stderr)

    print(f"machine: {machine_key or 'unknown'}   binary: {binary}   commit: {git_commit()}")
    print(f"{'point':<12}{'samples/s':>11}{'baseline':>11}{'staged':>8}{'degraded':>10}{'wgrad-bf16':>12}  status")
    for point_key, sps, baseline, staged, degraded, wgrad, failures in rows:
        status = "PASS" if not failures else "FAIL: " + "; ".join(failures)
        print(f"{point_key:<12}{sps:>11.0f}{(baseline or 0):>11.0f}{staged:>8}{degraded:>10}{wgrad:>12}  {status}")

    if args.record and machine_key:
        entry = baselines.setdefault(machine_key, {})
        entry["recorded"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        entry["commit"] = git_commit()
        entry["epochs"] = args.epochs
        entry["repeats"] = args.repeats
        entry["tolerance"] = args.tolerance
        entry.setdefault("points", {})
        for point_key, sps in measured.items():
            if sps > 0:
                entry["points"][point_key] = {"samples_per_sec": round(sps)}
        BASELINES.write_text(json.dumps(baselines, indent=2, sort_keys=True) + "\n")
        print(f"baseline recorded for '{machine_key}' in {BASELINES.name}")

    return 1 if all_failures else 0


if __name__ == "__main__":
    sys.exit(main())
