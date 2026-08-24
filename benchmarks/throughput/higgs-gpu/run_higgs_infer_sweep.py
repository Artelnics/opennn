#!/usr/bin/env python3
"""Rotated batch sweep for the GPU HIGGS dense inference benchmark.

`run_higgs_infer.py` measures one batch size and writes the publication
artifact. This runner is the instrument the CPU campaign converged on, ported to
the GPU family: it sweeps a ladder of batch sizes and it rotates.

Why rotation. On the CPU side a three-arm A/B measured 20% apart on rungs where
all three arms ran *identical code*, ordered purely by slot: whatever runs first
after an idle gap runs in the boost window. The GPU here has its clock pinned
(benchmarks/tools/gpu_clocks.sh), which removes most of that, but the same
discipline is what makes a 3% difference decidable at all, and it costs nothing:

* every arm sweeps the whole batch ladder inside ONE process, so the rungs of a
  row share one load and one thermal window;
* the arm order rotates every round, and the batch order rotates with it;
* the first round is a soak and is discarded;
* medians are taken over the remaining rounds;
* the per-pass times each driver prints in temporal order are kept in the
  artifact, so drift stays visible in the data instead of being averaged away.

An "arm" is an engine plus an environment. With no --arm the arms are the three
engines at their best configuration, which is the head-to-head. With --arm the
same engine can appear more than once under different environments, which is the
A/B: two arms of OpenNN differing in one OPENNN_* switch are then alternated
against each other under the same protocol as the head-to-head.

    # head-to-head
    python run_higgs_infer_sweep.py --batches 256,1024,8192,65536 --rounds 6

    # A/B of one lever, alternated
    python run_higgs_infer_sweep.py --batches 1024,8192 --rounds 6 \
        --arm opennn:on:OPENNN_SOME_LEVER=1 \
        --arm opennn:off:OPENNN_SOME_LEVER=0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS_DIR = (HERE.parent.parent / "results").resolve()

def load_base():
    spec = importlib.util.spec_from_file_location("run_higgs_infer", HERE / "run_higgs_infer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

BASE = load_base()

BATCH_METRIC = re.compile(r"^batch_(\d+)_samples_per_sec=(\S+)(?:\s+median_pass_s=(\S+))?(?:\s+ms_per_batch=(\S+))?")
BATCH_TIMES = re.compile(r"^batch_(\d+)_pass_times=(.+)$")
BATCH_TF_PATH = re.compile(r"^batch_(\d+)_tf_path=(\S+)")

def rotate(items: list, by: int) -> list:
    if not items:
        return items
    shift = by % len(items)
    return items[shift:] + items[:shift]

def engine_env(engine: str, precision: str, plain: bool) -> dict[str, str]:
    """The best-configuration environment each engine is measured under."""
    env: dict[str, str] = {}

    if engine == "pytorch" and not plain:
        # The driver times its candidate paths at every rung and reports the
        # faster, because PyTorch's best is not the same configuration at 256 as
        # at 65,536 - see the table in pytorch_higgs_infer.py. What is left to
        # set here is the weight dtype: bf16 weights spare the autocast a cast
        # on every replay, which is what a deployment would do.
        if precision == "bf16":
            env["PT_BF16_WEIGHTS"] = "1"

    if engine in ("tensorflow", "opennn"):
        libs: list[str] = []
        if engine == "tensorflow":
            libs = BASE.tensorflow_library_dirs(BASE.PY)
        wsl_cuda = os.environ.get("WSL_CUDA_LIB", "/usr/lib/wsl/lib")
        if Path(wsl_cuda).exists():
            libs.append(wsl_cuda)
        if libs:
            BASE.prepend_env_path(env, "LD_LIBRARY_PATH", libs)

    return env

def arm_command(engine: str, batches: list[int], args: argparse.Namespace,
                precision: str) -> list[str]:
    batch_arg = ",".join(str(b) for b in batches)
    common = [batch_arg, str(args.runs), precision,
              str(args.hidden), str(args.hidden_layers), args.activation]

    if engine == "opennn":
        return [BASE.OPENNN_BIN, str(args.test), *common]
    script = {"pytorch": "pytorch_higgs_infer.py",
              "tensorflow": "tensorflow_higgs_infer.py"}[engine]
    return [BASE.PY, str(HERE / script), str(args.test), *common]

def parse_sweep(text: str) -> dict[str, Any]:
    per_batch: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        line = line.strip()

        match = BATCH_METRIC.match(line)
        if match:
            entry = per_batch.setdefault(match.group(1), {})
            entry["samples_per_sec"] = float(match.group(2))
            if match.group(3):
                entry["median_pass_s"] = float(match.group(3))
            if match.group(4):
                entry["ms_per_batch"] = float(match.group(4))
            continue

        match = BATCH_TIMES.match(line)
        if match:
            entry = per_batch.setdefault(match.group(1), {})
            entry["pass_times"] = [float(v) for v in match.group(2).split(",") if v]
            continue

        match = BATCH_TF_PATH.match(line)
        if match:
            per_batch.setdefault(match.group(1), {})["tf_path"] = match.group(2)

    return per_batch

def run_arm(arm: dict[str, Any], batches: list[int], args: argparse.Namespace,
            precision: str) -> dict[str, Any]:
    cmd = arm_command(arm["engine"], batches, args, precision)
    env = dict(os.environ)
    env.update(engine_env(arm["engine"], precision, args.plain))
    env.update(arm["env"])

    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    raw = out.stdout + out.stderr
    return {
        "returncode": out.returncode,
        "ok": "RESULT=OK" in out.stdout,
        "batches": parse_sweep(raw),
        "tail": raw[-2000:] if out.returncode != 0 or "RESULT=OK" not in out.stdout else "",
    }

def parse_arm_spec(spec: str) -> dict[str, Any]:
    # engine:label[:KEY=VALUE[,KEY=VALUE...]]
    parts = spec.split(":", 2)
    engine = parts[0]
    label = parts[1] if len(parts) > 1 and parts[1] else engine
    env: dict[str, str] = {}
    if len(parts) > 2 and parts[2]:
        for item in parts[2].split(","):
            if not item.strip():
                continue
            key, _, value = item.partition("=")
            env[key.strip()] = value.strip()
    return {"engine": engine, "label": label, "env": env}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", default=str(BASE.DEFAULT_TEST))
    parser.add_argument("--batches", default="256,1024,4096,8192,16384,65536")
    parser.add_argument("--runs", type=int, default=5,
                        help="timed passes per batch per round, inside the driver")
    parser.add_argument("--rounds", type=int, default=6, help="rounds kept after the soak")
    parser.add_argument("--soak", type=int, default=1, help="rounds run and discarded first")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--activation", default="relu", choices=["relu", "tanh"])
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32", "both"])
    parser.add_argument("--engines", default="opennn,pytorch,tensorflow")
    parser.add_argument("--arm", action="append", default=[],
                        help="engine:label[:KEY=VAL,...]; repeatable. Overrides --engines.")
    parser.add_argument("--plain", action="store_true",
                        help="do not apply each engine's best-configuration environment")
    parser.add_argument("--label", default="sweep")
    parser.add_argument("--output", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    arms = ([parse_arm_spec(spec) for spec in args.arm]
            if args.arm
            else [{"engine": e, "label": e, "env": {}}
                  for e in args.engines.split(",") if e.strip()])
    precisions = ["bf16", "fp32"] if args.precision == "both" else [args.precision]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record: dict[str, Any] = {
        "benchmark": "gpu-higgs-dense-inference-sweep",
        "label": args.label,
        "run_id": run_id,
        "protocol": {
            "batches": batches,
            "arms": arms,
            "rounds_kept": args.rounds,
            "soak_rounds": args.soak,
            "passes_per_batch": args.runs,
            "rotation": "arm order and batch order both rotate by round index",
        },
        "git": BASE.git_metadata(),
        "rounds": [],
    }

    # arm -> precision -> batch -> [samples/s per kept round]
    kept: dict[str, dict[str, dict[int, list[float]]]] = {}

    for round_index in range(args.soak + args.rounds):
        soaking = round_index < args.soak
        round_arms = rotate(arms, round_index)
        round_batches = rotate(batches, round_index)
        round_record: dict[str, Any] = {"index": round_index, "soak": soaking,
                                        "arm_order": [a["label"] for a in round_arms],
                                        "batch_order": round_batches, "results": {}}

        for precision in precisions:
            for arm in round_arms:
                result = run_arm(arm, round_batches, args, precision)
                key = f"{arm['label']}/{precision}"
                round_record["results"][key] = result

                if not result["ok"]:
                    print(f"  !! {key} failed: {result['tail'][-400:]}", file=sys.stderr)
                    continue
                if soaking:
                    continue

                for batch_text, entry in result["batches"].items():
                    if "samples_per_sec" not in entry:
                        continue
                    (kept.setdefault(arm["label"], {})
                         .setdefault(precision, {})
                         .setdefault(int(batch_text), [])
                         .append(entry["samples_per_sec"]))

        record["rounds"].append(round_record)
        if not args.quiet:
            tag = "soak" if soaking else f"round {round_index - args.soak + 1}"
            order = " ".join(round_record["arm_order"])
            print(f"[{tag}] arms: {order} | batches: {round_batches}", flush=True)

    summary: dict[str, Any] = {}
    for label, per_precision in kept.items():
        for precision, per_batch in per_precision.items():
            for batch, values in per_batch.items():
                cell = (summary.setdefault(precision, {})
                                .setdefault(str(batch), {}))
                cell[label] = {
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "rounds": values,
                }
    record["summary"] = summary

    output = Path(args.output) if args.output else RESULTS_DIR / f"gpu-higgs-infer-sweep-{args.label}-{run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2))

    print()
    for precision in precisions:
        cells = summary.get(precision, {})
        if not cells:
            continue
        labels = [arm["label"] for arm in arms]
        print(f"== {precision} == medians of {args.rounds} rotated rounds, samples/s")
        header = "batch".rjust(8) + "".join(l.rjust(16) for l in labels)
        if len(labels) > 1:
            header += "".join(f"{labels[0]}/{l}".rjust(18) for l in labels[1:])
        print(header)
        for batch in batches:
            row = str(batch).rjust(8)
            medians = {}
            for label in labels:
                value = cells.get(str(batch), {}).get(label, {}).get("median")
                medians[label] = value
                row += (f"{value:,.0f}" if value else "-").rjust(16)
            if len(labels) > 1 and medians.get(labels[0]):
                for label in labels[1:]:
                    other = medians.get(label)
                    ratio = f"{medians[labels[0]] / other:.3f}x" if other else "-"
                    row += ratio.rjust(18)
            print(row)
        print()

    print(f"artifact: {output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
