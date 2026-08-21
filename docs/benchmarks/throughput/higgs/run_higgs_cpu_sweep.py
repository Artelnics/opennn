#!/usr/bin/env python3
"""Rotated sweep for the CPU HIGGS dense benchmark, training or inference.

`run_higgs_cpu.py` measures one configuration and writes the publication
artifact, but it runs the engines in BLOCKS - all N runs of one, then the next -
which is the instrument flaw the CPU inference note documents at length: a
three-arm A/B measured 20% apart on rungs where all three arms ran *identical
code*, ordered purely by slot, because whatever runs first after an idle gap
runs inside the processor's boost window. Every CPU table taken that way
flattered whichever engine held slot one, and that was always OpenNN.

This runner is the protocol that replaces it, the same one the GPU family uses:

* every arm sweeps the whole batch ladder inside ONE process, so the rungs of a
  row share one load and one thermal window;
* the arm order rotates every round, and the batch order rotates with it;
* the first round is a soak and is discarded;
* medians are taken over the remaining rounds;
* the ratio reported is the median of the PAIRED per-round ratios, not the ratio
  of the medians - the two differ at tight cells and only the paired one is a
  statement about rounds that actually ran against each other;
* the per-epoch and per-pass times each driver prints in temporal order are kept
  in the artifact, so drift stays visible in the data instead of averaged away.

An "arm" is an engine plus an environment. With no --arm the arms are the three
engines at their best configuration, which is the head-to-head. With --arm the
same engine can appear more than once under different environments, which is the
A/B.

    # training head-to-head
    python run_higgs_cpu_sweep.py train --train higgs_train_1m.csv \\
        --batches 1024 --epochs 3 --warmup-epochs 1 --rounds 6

    # inference head-to-head across a ladder
    python run_higgs_cpu_sweep.py infer --batches 256,1024,4096,16384 --rounds 6

    # A/B of one lever, alternated under the same protocol
    python run_higgs_cpu_sweep.py train --batches 1024 --rounds 6 \\
        --arm opennn:blocks: --arm opennn:mkl:OPENNN_GEMM_MODE=mkl
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
    spec = importlib.util.spec_from_file_location("run_higgs_cpu", HERE / "run_higgs_cpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base()

BATCH_METRIC = re.compile(
    r"^batch_(\d+)_samples_per_sec=(\S+)"
    r"(?:\s+median_epoch_s=(\S+))?"
    r"(?:\s+median_pass_s=(\S+))?"
    r"(?:\s+samples_per_epoch=(\S+))?")
BATCH_TIMES = re.compile(r"^batch_(\d+)_(?:epoch|pass)_times=(.+)$")
BATCH_QUALITY = re.compile(
    r"^batch_(\d+)_test_accuracy=(\S+)\s+test_log_loss=(\S+)\s+test_roc_auc=(\S+)")


def rotate(items: list, by: int) -> list:
    if not items:
        return items
    shift = by % len(items)
    return items[shift:] + items[:shift]


def engine_env(engine: str, threads: int, plain: bool) -> dict[str, str]:
    """The best-configuration environment each engine is measured under.

    Every engine gets the same pinning. It used to skip TensorFlow, which on a
    hybrid CPU is not a neutral omission: unpinned threads land on efficiency
    cores.
    """
    env: dict[str, str] = {"CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "2"}

    if threads:
        env["OMP_NUM_THREADS"] = str(threads)
        env["OMP_PLACES"] = "cores"
        env["OMP_PROC_BIND"] = "close"
        if engine == "opennn":
            env["OPENNN_THREADS"] = str(threads)
            env["MKL_NUM_THREADS"] = str(threads)

    if plain:
        # The A/B against each engine's own default path: XLA off for
        # TensorFlow, no torch.compile for PyTorch.
        if engine == "tensorflow":
            env["TF_PLAIN"] = "1"
        if engine == "pytorch":
            env["PYTORCH_PLAIN"] = "1"

    return env


def arm_command(engine: str, batches: list[int], args: argparse.Namespace) -> list[str]:
    batch_arg = ",".join(str(b) for b in batches)

    if engine == "opennn":
        binary, _ = BASE.find_opennn_higgs_cpu()
        if args.mode == "train":
            return [binary, "train", str(args.train), str(args.test),
                    str(args.epochs), batch_arg, str(args.hidden),
                    str(args.hidden_layers), args.activation, str(args.warmup_epochs)]
        return [binary, "infer", str(args.test), str(args.reps), batch_arg,
                str(args.hidden), str(args.hidden_layers), args.activation]

    script = {"pytorch": "pytorch_higgs_cpu.py",
              "tensorflow": "tensorflow_higgs_cpu.py"}[engine]
    cmd = [BASE.PY, str(HERE / script), args.mode,
           "--test", str(args.test),
           "--batches", batch_arg,
           "--hidden", str(args.hidden),
           "--hidden-layers", str(args.hidden_layers),
           "--activation", args.activation]
    if args.mode == "train":
        cmd += ["--train", str(args.train), "--epochs", str(args.epochs),
                "--warmup-epochs", str(args.warmup_epochs)]
    else:
        cmd += ["--reps", str(args.reps)]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    return cmd


def parse_sweep(text: str) -> dict[str, dict[str, Any]]:
    per_batch: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        line = line.strip()

        match = BATCH_METRIC.match(line)
        if match:
            entry = per_batch.setdefault(match.group(1), {})
            entry["samples_per_sec"] = float(match.group(2))
            if match.group(3):
                entry["median_epoch_s"] = float(match.group(3))
            if match.group(4):
                entry["median_pass_s"] = float(match.group(4))
            if match.group(5):
                entry["samples_per_epoch"] = int(float(match.group(5)))
            continue

        match = BATCH_TIMES.match(line)
        if match:
            entry = per_batch.setdefault(match.group(1), {})
            entry["times"] = [float(v) for v in match.group(2).split(",") if v]
            continue

        match = BATCH_QUALITY.match(line)
        if match:
            entry = per_batch.setdefault(match.group(1), {})
            entry["test_accuracy"] = float(match.group(2))
            entry["test_log_loss"] = float(match.group(3))
            entry["test_roc_auc"] = float(match.group(4))

    return per_batch


def run_arm(arm: dict[str, Any], batches: list[int],
            args: argparse.Namespace) -> dict[str, Any]:
    cmd = arm_command(arm["engine"], batches, args)
    env = dict(os.environ)
    env.update(engine_env(arm["engine"], args.threads, args.plain))
    env.update(arm["env"])

    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    raw = out.stdout + out.stderr
    ok = "RESULT=OK" in out.stdout
    return {
        "command": " ".join(cmd),
        "returncode": out.returncode,
        "ok": ok,
        "batches": parse_sweep(raw),
        "tail": "" if ok else raw[-2000:],
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


def paired_ratio(first: list[float], second: list[float]) -> dict[str, Any] | None:
    """The median of the per-round ratios, and how many rounds the first won.

    Not the ratio of the medians. The two differ wherever a cell is tight, and
    only this one is a statement about rounds that ran against each other in the
    same thermal state.
    """
    pairs = [a / b for a, b in zip(first, second) if b]
    if not pairs:
        return None
    return {
        "median": statistics.median(pairs),
        "ahead": sum(1 for r in pairs if r > 1.0),
        "rounds": len(pairs),
        "worst": min(pairs),
        "best": max(pairs),
        "per_round": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["train", "infer"])
    parser.add_argument("--train", type=Path, default=BASE.DEFAULT_HIGGS_DIR / "higgs_train.csv")
    parser.add_argument("--test", type=Path, default=BASE.DEFAULT_HIGGS_DIR / "higgs_test.csv")
    parser.add_argument("--batches", default="1024")
    parser.add_argument("--epochs", type=int, default=3, help="timed epochs per rung (train)")
    parser.add_argument("--warmup-epochs", type=int, default=1,
                        help="epochs run and discarded first, inside the same call (train)")
    parser.add_argument("--reps", type=int, default=10, help="timed passes per rung (infer)")
    parser.add_argument("--rounds", type=int, default=6, help="rounds kept after the soak")
    parser.add_argument("--soak", type=int, default=1, help="rounds run and discarded first")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--activation", default="relu", choices=["relu", "tanh"])
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--engines", default="opennn,pytorch,tensorflow")
    parser.add_argument("--arm", action="append", default=[],
                        help="engine:label[:KEY=VAL,...]; repeatable. Overrides --engines.")
    parser.add_argument("--plain", action="store_true",
                        help="do not apply each engine's best-configuration environment")
    parser.add_argument("--label", default="sweep")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.mode == "train" and not args.train.exists():
        raise SystemExit(f"HIGGS train file not found: {args.train}")
    if not args.test.exists():
        raise SystemExit(f"HIGGS test file not found: {args.test}")
    if args.mode == "train" and args.warmup_epochs < 1:
        raise SystemExit("train mode needs at least one warmup epoch: the first epoch "
                         "of a call carries that call's setup")

    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    arms = ([parse_arm_spec(spec) for spec in args.arm]
            if args.arm
            else [{"engine": e, "label": e, "env": {}}
                  for e in args.engines.split(",") if e.strip()])

    binary, found = BASE.find_opennn_higgs_cpu()
    binary_info = BASE.opennn_binary_info(binary, found)
    if any(a["engine"] == "opennn" for a in arms) and not binary_info.get("mkl_linked"):
        print(f"WARNING: OpenNN binary is NOT MKL-linked ({binary}); the CPU protocol "
              "requires the MKL build. Plain-Eigen numbers under-report OpenNN.",
              file=sys.stderr)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": f"cpu-higgs-dense-{'training' if args.mode == 'train' else 'inference'}-sweep",
        "label": args.label,
        "run_id": run_id,
        "protocol": {
            "mode": args.mode,
            "batches": batches,
            "arms": arms,
            "rounds_kept": args.rounds,
            "soak_rounds": args.soak,
            "epochs_per_rung": args.epochs if args.mode == "train" else None,
            "warmup_epochs": args.warmup_epochs if args.mode == "train" else None,
            "passes_per_rung": args.reps if args.mode == "infer" else None,
            "rotation": "arm order and batch order both rotate by round index",
            "ratio_statistic": "median of the paired per-round ratios",
        },
        "configuration": {
            "model": "28 -> hidden -> hidden -> 1 dense binary classifier",
            "hidden": args.hidden,
            "hidden_layers": args.hidden_layers,
            "activation": args.activation,
            "threads": args.threads or None,
            "plain": args.plain,
        },
        "dataset": "HIGGS",
        "dataset_files": {
            "train": BASE.file_info(args.train) if args.mode == "train" else None,
            "test": BASE.file_info(args.test),
        },
        "git": BASE.git_metadata(),
        "machine": BASE.framework_versions(),
        "runner": {
            "path": os.path.relpath(__file__, BASE.REPO_ROOT),
            "cwd": os.getcwd(),
            "argv": sys.argv,
            "opennn_binary": binary,
            "opennn_binary_info": binary_info,
        },
        "rounds": [],
    }

    # label -> batch -> [samples/s per kept round]
    kept: dict[str, dict[int, list[float]]] = {}
    quality: dict[str, dict[int, dict[str, float]]] = {}

    for round_index in range(args.soak + args.rounds):
        soaking = round_index < args.soak
        round_arms = rotate(arms, round_index)
        round_batches = rotate(batches, round_index)
        round_record: dict[str, Any] = {
            "index": round_index,
            "soak": soaking,
            "arm_order": [a["label"] for a in round_arms],
            "batch_order": round_batches,
            "results": {},
        }

        for arm in round_arms:
            result = run_arm(arm, round_batches, args)
            round_record["results"][arm["label"]] = result

            if not result["ok"]:
                print(f"  !! {arm['label']} failed: {result['tail'][-400:]}", file=sys.stderr)
                continue
            if soaking:
                continue

            for batch_text, entry in result["batches"].items():
                if "samples_per_sec" not in entry:
                    continue
                kept.setdefault(arm["label"], {}).setdefault(int(batch_text), []).append(
                    entry["samples_per_sec"])
                if "test_accuracy" in entry:
                    quality.setdefault(arm["label"], {})[int(batch_text)] = {
                        "test_accuracy": entry["test_accuracy"],
                        "test_log_loss": entry["test_log_loss"],
                        "test_roc_auc": entry["test_roc_auc"],
                    }

        record["rounds"].append(round_record)
        if not args.quiet:
            tag = "soak" if soaking else f"round {round_index - args.soak + 1}"
            print(f"[{tag}] arms: {' '.join(round_record['arm_order'])}"
                  f" | batches: {round_batches}", flush=True)

    labels = [arm["label"] for arm in arms]
    summary: dict[str, Any] = {}
    for label, per_batch in kept.items():
        for batch, values in per_batch.items():
            cell = summary.setdefault(str(batch), {})
            cell[label] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "rounds": values,
            }
            if label in quality and batch in quality[label]:
                cell[label]["quality"] = quality[label][batch]

    ratios: dict[str, Any] = {}
    for batch in batches:
        cell = summary.get(str(batch), {})
        for other in labels[1:]:
            pair = paired_ratio(kept.get(labels[0], {}).get(batch, []),
                                kept.get(other, {}).get(batch, []))
            if pair:
                ratios.setdefault(str(batch), {})[f"{labels[0]}/{other}"] = pair
        if not cell:
            continue

    record["summary"] = summary
    record["paired_ratios"] = ratios

    stem = f"cpu-higgs-{args.mode}-sweep-{args.label}-{run_id}.json"
    output = args.output or RESULTS_DIR / stem
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")

    unit = "samples/s"
    print()
    print(f"== {args.mode} == medians of {args.rounds} rotated rounds, {unit}")
    header = "batch".rjust(8) + "".join(l.rjust(15) for l in labels)
    for other in labels[1:]:
        header += f"vs {other}".rjust(12) + "ahead".rjust(7) + "worst".rjust(8)
    print(header)
    for batch in batches:
        cell = summary.get(str(batch), {})
        row = str(batch).rjust(8)
        for label in labels:
            value = cell.get(label, {}).get("median")
            row += (f"{value:,.0f}" if value else "-").rjust(15)
        for other in labels[1:]:
            pair = ratios.get(str(batch), {}).get(f"{labels[0]}/{other}")
            if pair:
                row += f"{pair['median']:.3f}x".rjust(12)
                row += f"{pair['ahead']}/{pair['rounds']}".rjust(7)
                row += f"{pair['worst']:.3f}".rjust(8)
            else:
                row += "-".rjust(12) + "-".rjust(7) + "-".rjust(8)
        print(row)

    print()
    print(f"artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
