#!/usr/bin/env python3
"""Ratchet OpenNN source size, duplication, and function complexity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "opennn"
BASELINE = ROOT / "CODE_QUALITY.json"
CPP_SUFFIXES = {".cpp", ".h"}
CUDA_SUFFIXES = {".cu", ".cuh"}
VENDOR_DIRECTORY = ("core", "cuda", "flash_attention_shim")


def source_files(source: Path = SOURCE) -> list[Path]:
    """The same first-party inventory drives metrics, headers and dependencies."""
    return sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix in CPP_SUFFIXES | CUDA_SUFFIXES
        and path.relative_to(source).parts[:3] != VENDOR_DIRECTORY
    )


def duplicate_percentage(files) -> float:
    from lizard_ext.lizardduplicate import LizardExtension

    duplicate = LizardExtension()
    # Reuse the token hashes already produced by the pinned duplicate extension.
    list(duplicate.cross_file_process(files))
    for _ in duplicate.get_duplicates():
        pass
    return round(100 * duplicate.duplicate_rate(), 2)


def scope_metrics(files, paths: list[Path]) -> dict[str, int | float]:
    functions = [function for file in files for function in file.function_list]
    return {
        "files": len(files),
        "nloc": sum(file.nloc for file in files),
        "functions": len(functions),
        "functions_nloc_100": sum(function.nloc >= 100 for function in functions),
        "functions_ccn_25": sum(function.cyclomatic_complexity >= 25 for function in functions),
        "maximum_function_nloc": max((function.nloc for function in functions), default=0),
        "maximum_function_ccn": max((function.cyclomatic_complexity for function in functions), default=0),
        "duplicate_percent": duplicate_percentage(files),
        "physical_lines": sum(
            len(path.read_text(encoding="utf-8-sig").splitlines()) for path in paths
        ),
    }


def measure(paths: list[Path]) -> dict[str, int | float]:
    try:
        import lizard
    except ImportError as error:
        raise SystemExit(
            "lizard is required; install tools/code-quality-requirements.txt"
        ) from error

    # Lizard 1.24's directory discovery does not recognize .cu/.cuh. Passing
    # explicit files uses its C++ reader for CUDA, including duplicate tokens.
    # This is lexical analysis, not NVCC preprocessing or device validation.
    files = list(lizard.analyze_files(
        [str(path) for path in paths], exts=lizard.get_extensions(["duplicate"])
    ))
    cpp_files = [file for file in files if Path(file.filename).suffix in CPP_SUFFIXES]
    cuda_files = [file for file in files if Path(file.filename).suffix in CUDA_SUFFIXES]
    metrics = scope_metrics(cpp_files, [path for path in paths if path.suffix in CPP_SUFFIXES])
    cuda_metrics = scope_metrics(cuda_files, [path for path in paths if path.suffix in CUDA_SUFFIXES])
    # Keep the historical C++ ceilings independent of the newly measured CUDA.
    metrics.update({f"cuda_{key}": value for key, value in cuda_metrics.items()})
    metrics["combined_duplicate_percent"] = duplicate_percentage(files)
    return metrics


def analyze(source: Path = SOURCE) -> dict[str, int | float]:
    paths = source_files(source)
    missing_spdx = [
        path.relative_to(source.parent).as_posix() for path in paths
        if not path.read_text(encoding="utf-8-sig").startswith(
            "// SPDX-License-Identifier: LGPL-2.1-or-later"
        )
    ]
    if missing_spdx:
        raise SystemExit("Missing standard source header:\n" + "\n".join(missing_spdx))
    return measure(paths)


def limit_failures(metrics: dict, limits: dict) -> list[str]:
    failures = [f"{name}: missing reviewed limit" for name in metrics.keys() - limits.keys()]
    failures.extend(f"{name}: unknown metric" for name in limits.keys() - metrics.keys())
    failures.extend(
        f"{name}: {value} exceeds reviewed limit {limits[name]}"
        for name, value in metrics.items() if name in limits and value > limits[name]
    )
    return sorted(failures)


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
    failures = limit_failures(metrics, limits)

    print(json.dumps(metrics, indent=2))
    if failures:
        print("Code-quality ratchet failed:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        print("Reduce the regression or review it and run --update.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
