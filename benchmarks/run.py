#!/usr/bin/env python3
"""The benchmark suite. One entry point, one execution, every observation.

PLAN.md step 2.

    run.py --family dense --mode train  --batch 8192
    run.py --family dense --mode infer  --batch 8192 --precision fp32
    run.py --family dense --mode train  --batch 1024:OOM        # capacity sweep
    run.py --family dense --mode train  --batch 1024,8192,65536 # explicit rungs

Every run reports throughput, peak memory, energy and quality **from the same
execution**. They were four benchmarks in four directories; they are four
readings of one run. The suite this replaces launched the identical binary
twice -- once timed, once with a power meter -- in two thermal states, and
filed two results that could not be cross-referenced.

`--batch` is the only sweep axis, and it is what the old protocol directories
actually differed in:

    8192            one rung: the speed cell
    1024,8192       explicit rungs: the peak-batch curve
    1024:OOM        double until a launch fails: the capacity frontier

A sweep re-launches per rung because it must. A CUDA out-of-memory fault leaves
the context unusable, so a second attempt in the same process would measure the
wreck of the first. Exit code 0 fits, 1 does not.

Engines are launched as the contract requires -- OpenNN with captured graphs
and a device-resident split, PyTorch compiled. The model files own that; this
only chooses which to run, and never learns which one it is talking to.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BENCHMARKS,
    Monitor,
    agrees,
    file_info,
    find_binary,
    framework_versions,
    git_metadata,
    gpu_state,
    result_destination,
    session_id,
    wait_for_idle,
)

KEY_VALUE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")

BENCH_DATA = Path(os.environ.get("OPENNN_BENCH_DATA",
                                 str(Path.home() / "opennn-benchmark-data")))

# The only per-family knowledge here. Everything else -- which binary, which
# script -- follows from the family name, so adding a family adds one entry.
# The only per-family knowledge here: where its data is, and the model
# options only it understands. Everything else -- which binary, which script --
# follows from the family name.
FAMILIES = {
    "dense": {
        "data": lambda root: {"train": root / "higgs/higgs_train_250k.csv",
                              "test": root / "higgs/higgs_test.csv"},
        "options": lambda a: [str(a.hidden), str(a.layers), a.activation],
    },
    "cnn": {
        "data": lambda root: {"train": root / "imagenet_subset/train"},
        "options": lambda a: [str(a.image_size)],
    },
    "transformer": {
        "data": lambda root: {"train": root / "wmt14/wmt14_pairs.txt"},
        "options": lambda a: [str(a.hidden), str(a.layers)],
    },
    # footprint has no dataset and no batch: it measures what the framework
    # costs before any of that exists. Its "modes" are its three questions.
    "footprint": {
        "data": lambda root: {},
        "options": lambda a: [],
        "modes": ("memory", "startup", "export"),
    },
    "lstm": {
        "data": lambda root: {"train": root / "beijing_pm25/beijing_pm25_forecasting.csv"},
        "options": lambda a: [str(a.lstm_hidden), str(a.past)],
    },
}

def engine_command(family: str, engine: str) -> list[str]:
    """OpenNN is a compiled program, PyTorch a script; both take the same tail."""
    if engine == "opennn":
        path, found = find_binary(f"{family}_opennn")
        if not found:
            raise SystemExit(f"{family}_opennn not built (looked at {path})")
        return [path]

    script = BENCHMARKS / "families" / f"{family}.py"
    if not script.exists():
        raise SystemExit(f"{script} does not exist")
    return [sys.executable, str(script)]

def engine_arguments(mode: str, data: dict, batch: int, args) -> list[str]:
    """The positional tail both engines share, so neither is special-cased."""
    options = [*FAMILIES[args.family]["options"](args), args.device, args.precision]

    if mode == "infer":
        return [mode, str(data.get("test", data["train"])), str(args.repeats),
                str(batch), *options]
    return [mode, str(data["train"]), str(data.get("test", data["train"])),
            str(args.epochs), str(batch), *options]

def rungs(spec: str) -> tuple[list[int], bool]:
    """`--batch` in its three forms. Returns the rungs and whether to sweep to OOM."""
    if spec.endswith(":OOM"):
        return [int(spec[:-4])], True
    return [int(part) for part in spec.split(",") if part], False

def launch(command: list[str], quiet_wait: bool, device: str = "cuda") -> dict:
    """One execution, fully instrumented.

    The monitor samples for the whole process; energy is integrated only
    between the marks the engine prints around its timed region, so warmup and
    data loading are excluded from the energy figure as they are from the
    throughput one.
    """
    if quiet_wait and device == "cuda":
        wait_for_idle(seconds=30.0)

    with Monitor(device=device) as monitor:
        started = time.time()

        # Popen rather than run(), so a CPU launch can be watched for its peak
        # resident set while it is alive -- there is nothing to read once it
        # has exited.
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        if device != "cuda":
            while process.poll() is None:
                monitor.watch_rss(process.pid)
                time.sleep(0.02)
            monitor.watch_rss(process.pid)

        stdout, stderr = process.communicate(timeout=14400)
        wall = time.time() - started

    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    fields = dict(KEY_VALUE.findall(completed.stdout))

    def mark(name: str) -> float | None:
        try:
            return float(fields[name])
        except (KeyError, ValueError):
            return None

    start, end = mark("TIMED_START_UNIX"), mark("TIMED_END_UNIX")

    throughput = next((int(v) for k, v in fields.items()
                       if k.endswith("_samples_per_sec")), 0)

    return {
        "command": command,
        "returncode": completed.returncode,
        "wall_seconds": round(wall, 3),
        "samples_per_sec": throughput,
        "fits": fields.get("fits") != "0" and completed.returncode == 0,
        "quality": {k: float(v) for k, v in fields.items()
                    if k.endswith(("_test_accuracy", "test_roc_auc", "test_log_loss"))
                    and _is_number(v)},
        "instruments": monitor.summary(start, end),
        "timed_window": {"start_unix": start, "end_unix": end},
        "fields": fields,
        "stderr_tail": completed.stderr[-1500:] if completed.returncode else "",
    }

def format_wh(value: float | None) -> str:
    """Energy, or why there isn't one. A run whose timed window was too short
    to sample has no energy figure, and printing 0.0000 Wh would assert one."""
    return f"{value:.5f} Wh" if value is not None else "-- Wh"

def watt_hours(instruments: dict) -> str:
    return format_wh(instruments.get("energy_wh"))

def median_energy(launches: list[dict]) -> float | None:
    values = sorted(l["instruments"]["energy_wh"] for l in launches
                    if l["instruments"].get("energy_wh") is not None)
    return values[len(values) // 2] if values else None

def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", default="dense", choices=sorted(FAMILIES))
    parser.add_argument("--mode", default="train", choices=("train", "infer"))
    parser.add_argument("--engines", default="opennn,pytorch")
    parser.add_argument("--batch", default="8192",
                        help="8192 | 1024,8192 | 1024:OOM")
    parser.add_argument("--precision", default="bf16", choices=("fp32", "bf16", "strict"))
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--epochs", type=int, default=5, help="timed epochs per launch")
    parser.add_argument("--repeats", type=int, default=5, help="timed passes, infer")
    parser.add_argument("--rounds", type=int, default=3, help="launches per engine, order rotated")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--activation", default="relu", choices=("relu", "tanh"))
    parser.add_argument("--image-size", type=int, default=224, help="cnn")
    parser.add_argument("--lstm-hidden", type=int, default=128, help="lstm")
    parser.add_argument("--past", type=int, default=24, help="lstm window")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="cross-engine quality agreement band")
    parser.add_argument("--label", default="")
    parser.add_argument("--no-wait", action="store_true",
                        help="skip the cooldown between launches")
    args = parser.parse_args()

    data = FAMILIES[args.family]["data"](BENCH_DATA)
    missing = [str(p) for p in data.values() if not Path(p).exists()]
    if missing:
        raise SystemExit("missing dataset:\n  " + "\n  ".join(missing)
                         + "\n\nprepare it with: python prepare.py " + args.family)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    start_batch, to_oom = rungs(args.batch)
    git = git_metadata()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"=== {args.family} {args.mode} {args.precision} {args.device} ===")
    launches: list[dict] = []

    if "modes" in FAMILIES[args.family]:
        # One process per question, because a startup cost is already paid by
        # anything sharing a process with it.
        for question in FAMILIES[args.family]["modes"]:
            for engine in engines:
                outcome = launch(engine_command(args.family, engine) + [question],
                                 not args.no_wait, "cpu")
                outcome.update(engine=engine, batch=0, round=1, question=question)
                launches.append(outcome)
                reported = {k: v for k, v in outcome["fields"].items()
                            if k not in ("engine", "mode", "RESULT")}
                print(f"  {question:<8} {engine:<8} {reported}")

        artifact = {
            "schema_version": 1,
            "benchmark_id": f"{args.device}-{args.family}",
            "run_id": run_id,
            "session_id": session_id(),
            "label": args.label,
            "configuration": vars(args) | {"data_root": str(BENCH_DATA)},
            "git": git,
            "machine": gpu_state(),
            "frameworks": framework_versions(),
            "launches": launches,
        }
        name = (f"{artifact['benchmark_id']}"
                f"{'-' + args.label if args.label else ''}-{run_id}.json")
        path = result_destination(git.get("dirty")) / name
        path.write_text(json.dumps(artifact, indent=2, default=str))
        if git.get("dirty"):
            print("\n  dirty tree -> results/scratch/, not the evidence store")
        print(f"\nwrote {path}")
        return 0

    if to_oom:
        # Capacity: double until a launch fails, per engine. The last rung that
        # fits is the frontier, and its run carries speed, memory and energy
        # like any other.
        for engine in engines:
            batch = start_batch[0]
            while True:
                print(f"  {engine:<8} batch {batch:>9,} ... ", end="", flush=True)
                outcome = launch(engine_command(args.family, engine)
                                 + engine_arguments(args.mode, data, batch, args),
                                 not args.no_wait, args.device)
                outcome.update(engine=engine, batch=batch, round=1)
                launches.append(outcome)

                # A crash is not a capacity limit. An engine that dies on a
                # signal has not told us the batch was too large -- it has told
                # us it is broken -- and reporting that batch as the frontier
                # would publish a bug as a measurement.
                crashed = outcome["returncode"] < 0
                print("fits" if outcome["fits"]
                      else f"CRASHED (signal {-outcome['returncode']})" if crashed
                      else "does not fit")
                if crashed:
                    outcome["crashed"] = True
                if not outcome["fits"]:
                    break
                batch *= 2
    else:
        for index in range(args.rounds):
            order = engines[index % len(engines):] + engines[: index % len(engines)]
            print(f"  round {index + 1} ({' -> '.join(order)})")

            for engine in order:
                for batch in start_batch:
                    outcome = launch(engine_command(args.family, engine)
                                     + engine_arguments(args.mode, data, batch, args),
                                     not args.no_wait, args.device)
                    outcome.update(engine=engine, batch=batch, round=index + 1)
                    launches.append(outcome)

                    instruments = outcome["instruments"]
                    status = "OK" if outcome["returncode"] == 0 else f"rc={outcome['returncode']}"
                    print(f"    {engine:<8} b{batch:<7} "
                          f"{outcome['samples_per_sec']:>12,}/s  "
                          f"{instruments['peak_mib']:>7.0f} MiB  "
                          f"{watt_hours(instruments):>9}  {status}")

    summary: dict = {}
    for engine in engines:
        ok = [l for l in launches if l["engine"] == engine and l["returncode"] == 0]
        if not ok:
            continue

        rates = sorted(l["samples_per_sec"] for l in ok)
        entry = {
            "median_samples_per_sec": rates[len(rates) // 2],
            "min_samples_per_sec": rates[0],
            "max_samples_per_sec": rates[-1],
            "peak_mib": max(l["instruments"]["peak_mib"] for l in ok),
            "energy_wh": median_energy(ok),
            "launches": len(ok),
        }
        if to_oom:
            entry["max_batch"] = max(l["batch"] for l in ok if l["fits"])
            # Only a genuine out-of-memory bounds the frontier; a crash leaves
            # it unknown, and the artifact must say so rather than imply the
            # last surviving rung was a limit.
            failure = next((l for l in launches
                            if l["engine"] == engine and not l["fits"]), None)
            entry["frontier_valid"] = bool(failure and failure["returncode"] > 0)
            if failure and failure["returncode"] < 0:
                entry["frontier_note"] = (
                    f"engine crashed with signal {-failure['returncode']} at batch "
                    f"{failure['batch']}; max_batch is where it stopped working, "
                    f"not where it ran out of memory")
        summary[engine] = entry

    # The quality gate: a speed win bought by computing something different is
    # not a speed win, so agreement is checked before any number is published.
    #
    # Compared per batch, across engines. Across batches it is not a gate at
    # all -- training the same model at 8,192 and 16,384 legitimately reaches
    # different accuracies, and lumping those together makes a single-engine
    # sweep disagree with itself.
    def accuracy_of(launch: dict) -> float:
        return next((v for k, v in launch["quality"].items()
                     if k.endswith("test_accuracy")), float("nan"))

    per_batch: dict[int, list[float]] = {}
    for launch_result in launches:
        if launch_result["returncode"] == 0:
            per_batch.setdefault(launch_result["batch"], []).append(accuracy_of(launch_result))

    accuracies = {str(batch): values for batch, values in sorted(per_batch.items())}
    gate = all(agrees(values, args.tolerance) for values in per_batch.values())

    # A shape gate, alongside the quality gate. Two engines can only be
    # compared on the same tensor shape: transformer.cpp derives sequence
    # length from OpenNN's tokeniser and transformer.py from whitespace, and
    # on WMT14 those give 158 against 128 -- 23% more positions per sequence
    # for one engine. That is invisible in a throughput number and fatal to
    # what it means, so it is checked rather than assumed.
    shapes: dict[str, dict[str, str]] = {}
    for launch_result in launches:
        if launch_result["returncode"] != 0:
            continue
        reported = {k: v for k, v in launch_result["fields"].items()
                    if k in ("sequence", "input_vocab", "target_vocab", "samples",
                             "parameters", "hidden", "inputs", "past")}
        if reported:
            shapes.setdefault(launch_result["engine"], reported)

    shape_agrees = len({tuple(sorted(v.items())) for v in shapes.values()}) <= 1

    artifact = {
        "schema_version": 1,
        "benchmark_id": f"{args.device}-{args.family}-{args.mode}",
        "run_id": run_id,
        "session_id": session_id(),
        "label": args.label,
        "configuration": vars(args) | {"data_root": str(BENCH_DATA)},
        "git": git,
        "machine": gpu_state(),
        "frameworks": framework_versions(),
        "datasets": {name: file_info(Path(path)) for name, path in data.items()},
        "quality_gate": {"agrees": gate, "tolerance": args.tolerance,
                         "accuracies": accuracies},
        "shape_gate": {"agrees": shape_agrees, "reported": shapes},
        "summary": summary,
        "launches": launches,
    }

    name = f"{artifact['benchmark_id']}{'-' + args.label if args.label else ''}-{run_id}.json"
    path = result_destination(git.get("dirty")) / name
    path.write_text(json.dumps(artifact, indent=2, default=str))

    print()
    for engine, stats in summary.items():
        line = (f"  {engine:<8} {stats['median_samples_per_sec']:>12,}/s  "
                f"{stats['peak_mib']:>7.0f} MiB  "
                f"{format_wh(stats['energy_wh']):>9}")
        if "max_batch" in stats:
            line += f"  max batch {stats['max_batch']:,}"
        print(line)

    if len(summary) == 2:
        names = list(summary)
        ratio = (summary[names[0]]["median_samples_per_sec"]
                 / max(summary[names[1]]["median_samples_per_sec"], 1))
        print(f"  {names[0]} / {names[1]} = {ratio:.3f}x")

    if not shape_agrees:
        print("\n  SHAPE GATE FAILED: engines report different tensor shapes --")
        for engine, reported in shapes.items():
            print(f"    {engine:<8} {reported}")
        print("    the throughput numbers above are not measuring the same work")

    if not gate:
        print(f"\n  QUALITY GATE FAILED: accuracies disagree beyond {args.tolerance:.0%}"
              f" -- the speed numbers above are not a like-for-like comparison")
    if git.get("dirty"):
        print("\n  dirty tree -> results/scratch/, not the evidence store")

    print(f"\nwrote {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
