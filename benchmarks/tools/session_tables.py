#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

"""Throughput, memory and energy tables for one benchmark session.

Usage: session_tables.py <label> [results_dir]

Reads every run.py artifact whose label is <label>, in the results folder and
its scratch/ folder, keeps the latest artifact per cell, and prints Markdown
tables with each engine's median and OpenNN's ratio to every other engine.
Memory is each engine's peak (device memory over idle on CUDA, peak anonymous
RSS on CPU); CPU cells add the workload figure, peak minus the runtime's own
baseline. A cell that went to scratch/ is marked, with the reason.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ENGINES = ("opennn", "pytorch", "tensorflow", "onnxruntime")
NAMES = {"opennn": "OpenNN", "pytorch": "PyTorch", "tensorflow": "TensorFlow",
         "onnxruntime": "ONNX Runtime"}
ORDER = ("dense", "lstm", "cnn", "transformer")

def load(label: str, root: Path) -> list[dict]:
    cells: dict[tuple, dict] = {}
    for folder in (root, root / "scratch"):
        for path in sorted(folder.glob(f"*-{label}-*.json")):
            artifact = json.loads(path.read_text())
            if artifact.get("label") != label:
                continue
            artifact["_scratch"] = folder.name == "scratch"
            c = artifact["configuration"]
            key = (c["device"], c["family"], c["mode"], c["precision"], str(c["batch"]))
            if key not in cells or artifact["run_id"] > cells[key]["run_id"]:
                cells[key] = artifact
    return sorted(cells.values(), key=lambda a: (
        a["configuration"]["device"] != "cuda",
        ORDER.index(a["configuration"]["family"]) if a["configuration"]["family"] in ORDER else 9,
        a["configuration"]["mode"] != "train", a["configuration"]["precision"] != "fp32"))

def cell_name(artifact: dict) -> str:
    c = artifact["configuration"]
    name = (f"{'GPU' if c['device'] == 'cuda' else 'CPU'} {c['family']} {c['mode']} "
            f"{c['precision']} b{c['batch']}")
    return name + (" †" if artifact["_scratch"] else "")

def number(value: float, digits: int = 0) -> str:
    if value >= 1e6:
        return f"{value / 1e6:,.2f} M"
    return f"{value:,.{digits}f}"

def table(artifacts: list[dict], title: str, metric, digits: int, higher_better: bool,
          engines: list[str]) -> list[str]:
    others = [e for e in engines if e != "opennn"]
    lines = [f"### {title}", "",
             "| Cell | " + " | ".join(NAMES[e] for e in engines) + " | "
             + " | ".join(f"OpenNN / {NAMES[e]}" for e in others) + " |",
             "|---|" + "---:|" * (len(engines) + len(others))]
    for artifact in artifacts:
        values = {e: metric(artifact["summary"][e]) for e in engines if e in artifact["summary"]}
        row = [number(values[e], digits) if values.get(e) else "—" for e in engines]
        mine = values.get("opennn")
        for e in others:
            if not mine or not values.get(e):
                row.append("—")
            elif higher_better:
                row.append(f"{mine / values[e]:.3f}x")
            else:
                row.append(f"{100 * mine / values[e]:.0f} %")
        lines.append(f"| {cell_name(artifact)} | " + " | ".join(row) + " |")
    return lines + [""]

def scratch_reasons(artifact: dict) -> str:
    reasons = []
    if artifact.get("git", {}).get("dirty"):
        reasons.append("uncommitted changes")
    if artifact["configuration"]["device"] == "cuda" and not artifact.get("clocks_locked"):
        reasons.append("GPU clocks not locked")
    quiet = artifact.get("machine_quiet", {})
    if not quiet.get("quiet", True):
        busiest = max(quiet.get(k) or 0 for k in ("busy_before", "busy_during_max", "busy_after"))
        reasons.append(f"machine busy (up to {busiest:.1%}, threshold {quiet.get('threshold', 0):.0%})")
    if not artifact.get("shape_gate", {}).get("agrees", True):
        reasons.append("shape gate failed")
    if not artifact.get("quality_gate", {}).get("agrees", True):
        reasons.append("quality gate failed")
    return ", ".join(reasons) or "see the artifact"

def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    label = sys.argv[1]
    root = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ.get(
        "OPENNN_BENCH_RESULTS", Path(__file__).resolve().parents[3] / "opennn-benchmark-results"))
    artifacts = load(label, root)
    if not artifacts:
        print(f"no artifacts labelled {label!r} in {root}")
        return 1

    engines = [e for e in ENGINES if any(e in a["summary"] for a in artifacts)]
    cpu = [a for a in artifacts if a["configuration"]["device"] == "cpu"]
    lines = [f"## Session {label}", ""]
    lines += table(artifacts, "Throughput (samples/s, higher is better)",
                   lambda s: s.get("median_samples_per_sec"), 0, True, engines)
    lines += table(artifacts, "Memory (MiB, peak; lower is better)",
                   lambda s: s.get("peak_mib"), 0, False, engines)
    if cpu:
        lines += table(cpu, "CPU memory (MiB, workload = peak minus runtime baseline)",
                       lambda s: s.get("workload_mib"), 0, False, engines)
    lines += table(artifacts, "Energy (Wh, lower is better)",
                   lambda s: s.get("energy_wh"), 3, False, engines)

    scratch = [a for a in artifacts if a["_scratch"]]
    if scratch:
        lines.append("† In scratch/, not the evidence store:")
        lines += [f"- {cell_name(a)[:-2]}: {scratch_reasons(a)}" for a in scratch]
    print("\n".join(lines))
    return 0

if __name__ == "__main__":
    sys.exit(main())
