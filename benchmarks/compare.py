#!/usr/bin/env python3
"""Fail when an OpenNN benchmark regresses beyond reviewed tolerances."""

import argparse
import json
from pathlib import Path


def engine_summary(artifact, engine):
    for name, summary in artifact["summary"].items():
        if name.casefold() == engine.casefold():
            return summary
    raise ValueError(f"engine {engine!r} is absent")


def compare(baseline, candidate, engine="opennn", throughput_tolerance=0.05,
            memory_tolerance=0.05):
    failures = []
    if baseline["benchmark_id"] != candidate["benchmark_id"]:
        failures.append("benchmark identifiers differ")

    for name, artifact in (("baseline", baseline), ("candidate", candidate)):
        if not artifact.get("shape_gate", {}).get("agrees"):
            failures.append(f"{name} shape gate failed")
        if not artifact.get("quality_gate", {}).get("agrees"):
            failures.append(f"{name} quality gate failed")
        if not artifact.get("machine_quiet", {}).get("quiet"):
            failures.append(f"{name} machine was busy")

    old = engine_summary(baseline, engine)
    new = engine_summary(candidate, engine)
    old_rate = old["median_samples_per_sec"]
    new_rate = new["median_samples_per_sec"]
    if new_rate < old_rate * (1.0 - throughput_tolerance):
        failures.append(
            f"throughput {new_rate:g} is more than {throughput_tolerance:.1%} below {old_rate:g}"
        )

    memory_key = "workload_mib" if "workload_mib" in old and "workload_mib" in new else "peak_mib"
    if old.get(memory_key) is not None and new.get(memory_key) is not None:
        if new[memory_key] > old[memory_key] * (1.0 + memory_tolerance):
            failures.append(
                f"memory {new[memory_key]:g} MiB is more than {memory_tolerance:.1%} above "
                f"{old[memory_key]:g} MiB"
            )

    if old.get("max_batch") is not None and new.get("max_batch") is not None:
        if new["max_batch"] < old["max_batch"]:
            failures.append(f"capacity fell from {old['max_batch']} to {new['max_batch']}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--engine", default="opennn")
    parser.add_argument("--throughput-tolerance", type=float, default=0.05)
    parser.add_argument("--memory-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    failures = compare(baseline, candidate, args.engine,
                       args.throughput_tolerance, args.memory_tolerance)
    if failures:
        print("Benchmark regression:\n" + "\n".join(failures))
        return 1
    print("Benchmark regression check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
