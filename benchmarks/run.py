#!/usr/bin/env python3
"""Run benchmark families with shared provenance, memory and energy monitoring. Use --batch
N, a comma-separated sweep, or N:OOM. See PROTOCOL.md.
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

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
from common import (  # noqa: E402
    BENCHMARKS,
    clocks_locked,
    core_layout,
    cpu_state,
    physical_cores,
    Monitor,
    agrees,
    file_info,
    find_binary,
    framework_versions,
    git_metadata,
    gpu_state,
    HOST_BASELINE_FIELD,
    HOST_MEMORY_METRIC,
    result_destination,
    cpu_busy_fraction,
    ForeignActivity,
    BUSY_THRESHOLD,
    session_id,
    wait_for_idle,
)

KEY_VALUE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")

BENCH_DATA = Path(os.environ.get("OPENNN_BENCH_DATA",
                                 str(Path.home() / "opennn-benchmark-data")))

# Per-family data paths and model options; executable names follow the family.
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
        "options": lambda a: [str(a.d_model), str(a.transformer_layers)],
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

def cpu_pinning(threads: int | None) -> tuple[list[str], dict[str, str], dict]:
    """Pin CPU runs to detected performance cores. Apply an explicit thread count to both
    engines, or retain each engine's default.
    """
    environment: dict[str, str] = {}
    if threads:
        environment = {name: str(threads) for name in
                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "OPENNN_THREADS", "TORCH_NUM_THREADS")}
        environment.update({
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "OPENNN_OMP_DYNAMIC": "0",
        })

    layout = core_layout()
    cores = layout["performance"]

    if not cores:
        return [], environment, {
            "pinned": False,
            "reason": "no per-core frequency data",
            "threads": threads or "engine default",
        }

    span = f"{cores[0]}-{cores[-1]}" if cores == list(range(cores[0], cores[-1] + 1)) \
        else ",".join(str(c) for c in cores)

    count = threads

    environment.setdefault("GOMP_SPINCOUNT", "300000")

    return (["taskset", "-c", span], environment,
            {"pinned": True, "cores": span,
             "threads": count or "engine default",
             "omp_wait": "GOMP_SPINCOUNT=" + environment["GOMP_SPINCOUNT"],
             "excluded_efficiency_cores": layout["efficiency"]})

# What a launch wrote to stderr is kept whether or not it failed. The text
# worth having is printed by runs that succeed: both graph-capture paths report
# why capture was refused and then carry on eagerly (network.cpp,
# optimizer.cpp), so gating this on a nonzero return code threw away the reason
# behind every `cuda_graph="failed"` the artifacts record.
#
# Head *and* tail, not the tail alone. Those messages are printed at the first
# capture attempt, inside warmup, so on a run that goes on to say anything else
# -- a torch warning per epoch -- a tail-only excerpt would push the one line
# that matters out. Both ends are bounded, so a chatty engine cannot grow the
# artifact: the cost is at most ~4 kB per launch either way.
STDERR_HEAD_BYTES = 2000
STDERR_TAIL_BYTES = 2000

def stderr_excerpt(text: str) -> str:
    """Bounded excerpt of a launch's stderr, keeping both ends."""
    if len(text) <= STDERR_HEAD_BYTES + STDERR_TAIL_BYTES:
        return text
    elided = len(text) - STDERR_HEAD_BYTES - STDERR_TAIL_BYTES
    return (text[:STDERR_HEAD_BYTES]
            + f"\n... [{elided:,} bytes elided] ...\n"
            + text[-STDERR_TAIL_BYTES:])

def failure_kind(returncode: int, stdout: str, stderr: str) -> str | None:
    """Only allocation errors bound capacity; signals and Windows crashes do not."""
    fields = dict(KEY_VALUE.findall(stdout))
    if returncode < 0 or returncode >= 0x80000000:
        return "crash"
    if returncode == 0:
        failed = fields.get("fits") == "0" or fields.get("RESULT") in ("ERROR", "OOM")
        return "error" if failed else None
    evidence = stdout + "\n" + stderr
    allocation_error = re.search(
        r"CUDA out of memory|CUDA error: out of memory|cudaErrorMemoryAllocation|"
        r"CUDA_ERROR_OUT_OF_MEMORY|out of memory[^\n]*cudaMalloc|"
        r"std::bad_alloc|bad allocation|DefaultCPUAllocator[^\n]*can't allocate memory|"
        r"(?:^|\n)MemoryError(?::|\s*$)", evidence, re.IGNORECASE)
    return "oom" if fields.get("RESULT") == "OOM" or allocation_error else "error"


def capacity_summary(launches: list[dict]) -> dict:
    successful = [item["batch"] for item in launches if item["fits"]]
    failure = next((item for item in launches if not item["fits"]), None)
    valid = bool(successful and failure and failure.get("failure_kind") == "oom")
    return {
        "max_batch": max(successful) if successful else None,
        "frontier_valid": valid,
        "frontier_note": ("largest tested batch before a confirmed allocation failure" if valid
                          else "capacity unknown: no successful batch or no confirmed OOM"),
    }


def footprint_metrics(outcome: dict) -> dict:
    fields = outcome["fields"]

    def number(name: str) -> float | None:
        try:
            return float(fields[name])
        except (KeyError, ValueError):
            return None
    return {
        "baseline_ram_mib": number("baseline_ram_mb"),
        "baseline_ram_metric": fields.get("baseline_ram_metric"),
        "baseline_ram_note": fields.get("baseline_ram_note"),
        "internal_first_prediction_seconds": number("first_prediction_s"),
        "internal_first_prediction_scope": fields.get("first_prediction_scope"),
        "process_lifetime_seconds": outcome["process_lifetime_seconds"],
        "process_time_scope": outcome["process_time_scope"],
    }


def launch(command: list[str], quiet_wait: bool, device: str = "cuda",
           threads: int | None = None,
           watched_cores: list[int] | None = None) -> dict:
    """One execution, fully instrumented.

    The monitor samples for the whole process; energy is integrated only
    between the marks the engine prints around its timed region, so warmup and
    data loading are excluded from the energy figure as they are from the
    throughput one. Foreign CPU activity is watched the same way -- every
    second of the process, judged over the timed window -- and reported in
    `foreign_activity` for the caller's quiet gate.
    """
    if quiet_wait:
        if device == "cuda":
            wait_for_idle(seconds=30.0)
        else:
            time.sleep(float(os.environ.get("OPENNN_BENCH_CPU_SETTLE", "8")))

    prefix: list[str] = []
    environment = dict(os.environ)
    pinning: dict = {"pinned": False}

    if device != "cuda":
        prefix, extra, pinning = cpu_pinning(threads)
        environment.update(extra)

    with Monitor(device=device) as monitor:
        started = time.perf_counter()

        # Popen rather than run(), so a CPU launch can be watched for its peak
        # resident set while it is alive -- there is nothing to read once it
        # has exited.
        process = subprocess.Popen(prefix + command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True,
                                   env=environment)
        with ForeignActivity(process.pid, watched_cores) as foreign:
            try:
                while True:
                    if device != "cuda":
                        monitor.watch_rss(process.pid)
                    remaining = 14400 - (time.perf_counter() - started)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, 14400)
                    try:
                        stdout, stderr = process.communicate(timeout=min(0.02, remaining)
                                                             if device != "cuda" else remaining)
                        break
                    except subprocess.TimeoutExpired:
                        if device == "cuda":
                            raise
                wall = time.perf_counter() - started
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise

    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    fields = dict(KEY_VALUE.findall(completed.stdout))

    def mark(name: str) -> float | None:
        try:
            return float(fields[name])
        except (KeyError, ValueError):
            return None

    start, end = mark("TIMED_START_UNIX"), mark("TIMED_END_UNIX")

    instruments = monitor.summary(start, end)
    activity = foreign.worst(start, end)

    if instruments.get("memory_metric") == HOST_MEMORY_METRIC:
        baseline = mark(HOST_BASELINE_FIELD)
        if baseline is not None:
            instruments["baseline_mib"] = round(baseline, 1)
            instruments["workload_mib"] = round(max(instruments["peak_mib"] - baseline, 0.0), 1)
        else:
            instruments["workload_note"] = (
                f"no {HOST_BASELINE_FIELD}: peak_mib is the whole process, "
                "framework baseline included. The baseline the drivers do "
                "print is a total-RSS reading, which is not commensurable "
                f"with {HOST_MEMORY_METRIC}")

    throughput = next((int(v) for k, v in fields.items()
                       if k.endswith("_samples_per_sec")), 0)
    failure = failure_kind(completed.returncode, stdout, stderr)

    return {
        "command": prefix + command,
        "pinning": pinning,
        "returncode": completed.returncode,
        "wall_seconds": round(wall, 3),
        "process_lifetime_seconds": wall,
        "process_time_scope": "before_Popen_to_exit_and_output_collection_includes_teardown",
        "failure_kind": failure,
        "samples_per_sec": throughput,
        "fits": failure is None,
        "quality": {k: float(v) for k, v in fields.items()
                    if k.endswith(("_test_accuracy", "test_roc_auc", "test_log_loss"))
                    and _is_number(v)},
        "instruments": instruments,
        "foreign_activity": activity,
        "timed_window": {"start_unix": start, "end_unix": end},
        # Which BLAS the engine dispatched to. OpenNN defaults to Eigen and its
        # driver opts into MKL, so this is a property of the run rather than of
        # the binary, and comparing an Eigen number against an MKL one measures
        # the BLAS instead of the engine.
        "blas": fields.get("blas"),
        "fields": fields,
        "stderr_excerpt": stderr_excerpt(completed.stderr),
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

def note_activity(outcome: dict) -> None:
    """Say so at once when a launch was disturbed; the artifact records it
    either way and the gate below files the cell."""
    activity = outcome["foreign_activity"]
    if activity["max"] > BUSY_THRESHOLD:
        print(f"    foreign activity {activity['max']:.1%} during the "
              f"{activity['window']} window of {outcome.get('engine', '?')}")

def busiest_second(launches: list[dict]) -> tuple[float, float | None]:
    """The worst foreign second over every launch's timed window."""
    peak, at = 0.0, None
    for outcome in launches:
        activity = outcome.get("foreign_activity") or {}
        if activity.get("max", 0.0) > peak:
            peak, at = activity["max"], activity.get("at")
    return peak, at

def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False

def main() -> int:
    # Qwen is a token-length/runtime matrix rather than the batch/epoch matrix
    # used by the supervised families.  Keep one public entry point while
    # letting the family own its materially different command-line contract.
    if any(argument == "qwen" or argument == "--family=qwen"
           for argument in sys.argv[1:]):
        from families.qwen import main as qwen_main
        return qwen_main(sys.argv[1:])

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
    parser.add_argument("--hidden", type=int, default=1024, help="dense")
    parser.add_argument("--layers", type=int, default=2, help="dense")
    parser.add_argument("--activation", default="relu", choices=("relu", "tanh"))
    parser.add_argument("--d-model", type=int, default=512,
                        help="transformer; heads and feed-forward follow it")
    parser.add_argument("--transformer-layers", type=int, default=6, help="transformer")
    parser.add_argument("--image-size", type=int, default=224, help="cnn")
    parser.add_argument("--lstm-hidden", type=int, default=128, help="lstm")
    parser.add_argument("--past", type=int, default=24, help="lstm window")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="cross-engine quality agreement band")
    parser.add_argument("--label", default="")
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads; default is one per physical P-core")
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

    watched_cores = core_layout()["performance"] if args.device != "cuda" else None
    busy_before = cpu_busy_fraction(cores=watched_cores)
    machine_busy = busy_before > BUSY_THRESHOLD

    if machine_busy:
        print(f"  machine not quiet: {busy_before:.1%} busy before launch "
              f"(threshold {BUSY_THRESHOLD:.1%}) -> results/scratch/")

    launches: list[dict] = []

    if "modes" in FAMILIES[args.family]:
        # One process per question, because a startup cost is already paid by
        # anything sharing a process with it.
        for question in FAMILIES[args.family]["modes"]:
            for engine in engines:
                outcome = launch(engine_command(args.family, engine) + [question],
                                 not args.no_wait, "cpu", watched_cores=watched_cores)
                outcome.update(engine=engine, batch=0, round=1, question=question)
                outcome["footprint"] = footprint_metrics(outcome)
                if not sys.platform.startswith("linux"):
                    # The common anonymous-RSS peak sampler uses Linux procfs.
                    outcome["instruments"].update(peak_mib=None, workload_mib=None,
                        peak_file_mib=None, memory_metric=None,
                        workload_note="anonymous_peak_sampler_unavailable; use footprint baseline_ram_mib")
                launches.append(outcome)
                reported = {k: v for k, v in outcome["fields"].items()
                            if k not in ("engine", "mode", "RESULT")}
                print(f"  {question:<8} {engine:<8} {reported}")
                if question == "startup":
                    print(f"    process lifetime (including teardown): {outcome['process_lifetime_seconds']:.6f} s")
                note_activity(outcome)

        busy_after = cpu_busy_fraction(cores=watched_cores)
        busy_during, busy_during_at = busiest_second(launches)

        if busy_after > BUSY_THRESHOLD:
            print(f"\n  machine became busy during the run: {busy_after:.1%}")

        machine_busy = (machine_busy or busy_after > BUSY_THRESHOLD
                        or busy_during > BUSY_THRESHOLD)

        artifact = {
            "schema_version": 1,
            "benchmark_id": f"cpu-{args.family}",
            "run_id": run_id,
            "session_id": session_id(),
            "label": args.label,
            "configuration": vars(args) | {"data_root": str(BENCH_DATA),
                                            "device": "cpu", "precision": "fp32"},
            "git": git,
            "machine": gpu_state(),
        "cpu": cpu_state(),
            "frameworks": framework_versions(),
            "machine_quiet": {"busy_before": round(busy_before, 4),
                              "busy_after": round(busy_after, 4),
                              "busy_during_max": round(busy_during, 4),
                              "busy_during_at": busy_during_at,
                              "threshold": BUSY_THRESHOLD,
                              "quiet": not machine_busy},
            "launches": launches,
        }
        name = (f"{artifact['benchmark_id']}"
                f"{'-' + args.label if args.label else ''}-{run_id}.json")
        path = result_destination(git.get("dirty"), "cpu", machine_busy) / name
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
                                 not args.no_wait, args.device, args.threads,
                                 watched_cores)
                outcome.update(engine=engine, batch=batch, round=1)
                launches.append(outcome)

                # A crash is not a capacity limit. An engine that dies on a
                # signal has not told us the batch was too large -- it has told
                # us it is broken -- and reporting that batch as the frontier
                # would publish a bug as a measurement.
                crashed = outcome["failure_kind"] == "crash"
                print("fits" if outcome["fits"]
                      else f"failed: {outcome['failure_kind']} (rc={outcome['returncode']})")
                if crashed:
                    outcome["crashed"] = True
                note_activity(outcome)
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
                                     not args.no_wait, args.device, args.threads,
                                     watched_cores)
                    outcome.update(engine=engine, batch=batch, round=index + 1)
                    launches.append(outcome)

                    instruments = outcome["instruments"]
                    status = "OK" if outcome["returncode"] == 0 else f"rc={outcome['returncode']}"
                    print(f"    {engine:<8} b{batch:<7} "
                          f"{outcome['samples_per_sec']:>12,}/s  "
                          f"{instruments.get('workload_mib', instruments['peak_mib']):>7.0f} MiB  "
                          f"{watt_hours(instruments):>9}  {status}")
                    note_activity(outcome)

    summary: dict = {}
    for engine in engines:
        ok = [l for l in launches if l["engine"] == engine and l["fits"]]
        if not ok:
            if to_oom:
                summary[engine] = capacity_summary([l for l in launches if l["engine"] == engine])
            continue

        rates = sorted(l["samples_per_sec"] for l in ok)
        entry = {
            "median_samples_per_sec": rates[len(rates) // 2],
            "min_samples_per_sec": rates[0],
            "max_samples_per_sec": rates[-1],
            "peak_mib": max(l["instruments"]["peak_mib"] for l in ok),
            "workload_mib": max(l["instruments"].get("workload_mib",
                                                     l["instruments"]["peak_mib"]) for l in ok),
            "energy_wh": median_energy(ok),
            "launches": len(ok),
        }
        if to_oom:
            entry.update(capacity_summary([l for l in launches if l["engine"] == engine]))
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
        # Dataset identity is recorded once from the runner's authoritative
        # paths in artifact["datasets"].  It is not a tensor shape, and the
        # key/value log parser intentionally stops at whitespace, so putting a
        # Windows path here made two launches of the same file disagree at the
        # first space in its resolved name.
        reported = {k: v for k, v in launch_result["fields"].items()
                    if k in ("sequence", "input_vocab", "target_vocab", "samples",
                             "parameters", "hidden", "inputs", "past")}
        if reported:
            shapes.setdefault(launch_result["engine"], reported)

    shape_agrees = len({tuple(sorted(v.items())) for v in shapes.values()}) <= 1

    busy_after = cpu_busy_fraction(cores=watched_cores)
    busy_during, busy_during_at = busiest_second(launches)

    if busy_after > BUSY_THRESHOLD:
        print(f"\n  machine became busy during the run: {busy_after:.1%}")

    machine_busy = (machine_busy or busy_after > BUSY_THRESHOLD
                    or busy_during > BUSY_THRESHOLD)

    artifact = {
        "schema_version": 1,
        "benchmark_id": f"{args.device}-{args.family}-{args.mode}",
        "run_id": run_id,
        "session_id": session_id(),
        "label": args.label,
        "configuration": vars(args) | {"data_root": str(BENCH_DATA)},
        "git": git,
        "machine": gpu_state(),
        "cpu": cpu_state(),
        "frameworks": framework_versions(),
        "datasets": {name: file_info(Path(path)) for name, path in data.items()},
        "clocks_locked": clocks_locked(),
        "machine_quiet": {"busy_before": round(busy_before, 4),
                          "busy_after": round(busy_after, 4),
                          "busy_during_max": round(busy_during, 4),
                          "busy_during_at": busy_during_at,
                          "threshold": BUSY_THRESHOLD,
                          "quiet": not machine_busy},
        "quality_gate": {"agrees": gate, "tolerance": args.tolerance,
                         "accuracies": accuracies},
        "shape_gate": {"agrees": shape_agrees, "reported": shapes},
        "summary": summary,
        "launches": launches,
    }

    name = f"{artifact['benchmark_id']}{'-' + args.label if args.label else ''}-{run_id}.json"
    capacity_valid = not to_oom or all(stats["frontier_valid"] for stats in summary.values())
    path = result_destination(git.get("dirty"), args.device, machine_busy or not capacity_valid) / name
    path.write_text(json.dumps(artifact, indent=2, default=str))

    print()
    for engine, stats in summary.items():
        if "median_samples_per_sec" not in stats:
            print(f"  {engine}: {stats['frontier_note']}")
            continue
        line = (f"  {engine:<8} {stats['median_samples_per_sec']:>12,}/s  "
                f"{stats.get('workload_mib', stats['peak_mib']):>7.0f} MiB  "
                f"{format_wh(stats['energy_wh']):>9}")
        if "max_batch" in stats:
            line += f"  max batch {stats['max_batch']:,}"
            if not stats["frontier_valid"]:
                line += " (capacity unknown; diagnostic only)"
        print(line)

    if len(summary) == 2 and all("median_samples_per_sec" in stats for stats in summary.values()):
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
    elif machine_busy:
        # The largest of the three readings, because any can be what tripped
        # the threshold: `busy_before` catches a machine that was already
        # working, `busy_after` a sync client that woke mid-cell, and the
        # per-second watch anything in between. Naming none was a NameError
        # on the one path where the warning matters, so a busy clean-tree run
        # printed a traceback instead of its reason.
        print(f"\n  machine was {max(busy_before, busy_after, busy_during):.1%} "
              "busy -> results/scratch/, not the evidence store")
    elif args.device == "cuda" and not clocks_locked():
        print("\n  clocks unlocked -> results/scratch/. Provisional: margins under"
              "\n  ~2% are not resolvable while the clock floats.")

    print(f"\nwrote {path}")
    return 0 if capacity_valid else 3

if __name__ == "__main__":
    raise SystemExit(main())
