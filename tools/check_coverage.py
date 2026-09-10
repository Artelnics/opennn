#!/usr/bin/env python3
"""Enforce coverage floors per OpenNN library module."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


FLOORS = {
    "core": (80.0, 45.0),
    "dataset": (60.0, 30.0),
    "evaluation": (60.0, 25.0),
    "model_selection": (80.0, 35.0),
    "models": (60.0, 25.0),
    "network": (80.0, 45.0),
    "response_optimization": (85.0, 45.0),
    "training": (70.0, 35.0),
}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_coverage.py coverage-summary.json")

    report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    totals = defaultdict(lambda: [0, 0, 0, 0])
    for file in report["files"]:
        parts = Path(file["filename"]).parts
        if len(parts) < 3 or parts[0] != "opennn":
            continue
        module = parts[1]
        values = totals[module]
        values[0] += file["line_covered"]
        values[1] += file["line_total"]
        values[2] += file["branch_covered"]
        values[3] += file["branch_total"]

    failures = []
    for module, (line_floor, branch_floor) in FLOORS.items():
        line_covered, line_total, branch_covered, branch_total = totals[module]
        line_percent = 100.0 * line_covered / line_total
        branch_percent = 100.0 * branch_covered / branch_total
        print(f"{module}: lines {line_percent:.1f}%, branches {branch_percent:.1f}%")
        if line_percent < line_floor:
            failures.append(f"{module} line coverage {line_percent:.1f}% < {line_floor:.1f}%")
        if branch_percent < branch_floor:
            failures.append(f"{module} branch coverage {branch_percent:.1f}% < {branch_floor:.1f}%")

    if failures:
        print("Coverage floors failed:\n" + "\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
