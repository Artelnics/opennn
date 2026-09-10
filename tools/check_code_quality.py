#!/usr/bin/env python3
"""Ratchet OpenNN source size, duplication, and function complexity."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "opennn"
BASELINE = ROOT / "CODE_QUALITY.json"
EXCLUDE = "*/core/cuda/flash_attention_shim/*"


def analyze() -> dict[str, int | float]:
    try:
        import lizard
    except ImportError as error:
        raise SystemExit(
            "lizard is required; install tools/code-quality-requirements.txt"
        ) from error

    files = list(lizard.analyze(
        [str(SOURCE)], exclude_pattern=[EXCLUDE], lans=["cpp"]
    ))
    functions = [function for file in files for function in file.function_list]

    duplicate = subprocess.run(
        [sys.executable, "-m", "lizard", str(SOURCE), "-l", "cpp",
         "-x", EXCLUDE, "-Eduplicate", "-w"],
        check=False, capture_output=True, text=True,
    )
    match = re.search(
        r"Total duplicate rate:\s*([0-9.]+)%",
        duplicate.stdout + duplicate.stderr,
    )
    if not match:
        raise SystemExit("lizard did not report a duplicate rate")

    source_files = [
        path for path in SOURCE.rglob("*")
        if path.suffix in {".cpp", ".h"}
        and "flash_attention_shim" not in path.parts
    ]
    missing_spdx = [
        path.relative_to(ROOT).as_posix() for path in source_files
        if not path.read_text(encoding="utf-8-sig").startswith(
            "// SPDX-License-Identifier: LGPL-2.1-or-later"
        )
    ]
    if missing_spdx:
        raise SystemExit("Missing standard source header:\n" + "\n".join(missing_spdx))

    return {
        "files": len(files),
        "nloc": sum(file.nloc for file in files),
        "functions": len(functions),
        "functions_nloc_100": sum(function.nloc >= 100 for function in functions),
        "functions_ccn_25": sum(function.cyclomatic_complexity >= 25 for function in functions),
        "maximum_function_nloc": max(function.nloc for function in functions),
        "maximum_function_ccn": max(function.cyclomatic_complexity for function in functions),
        "duplicate_percent": float(match.group(1)),
        "physical_lines": sum(
            len(path.read_text(encoding="utf-8-sig").splitlines())
            for path in source_files
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                        help="replace the reviewed baseline with current metrics")
    args = parser.parse_args()
    metrics = analyze()

    if args.update:
        BASELINE.write_text(
            json.dumps({"limits": metrics}, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Updated {BASELINE.name}")
        return 0

    limits = json.loads(BASELINE.read_text(encoding="utf-8"))["limits"]
    failures = []
    for name, limit in limits.items():
        value = metrics[name]
        if value > limit:
            failures.append(f"{name}: {value} exceeds reviewed limit {limit}")

    print(json.dumps(metrics, indent=2))
    if failures:
        print("Code-quality ratchet failed:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        print("Reduce the regression or review it and run --update.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
