#!/usr/bin/env python3
"""Validate the benchmark inventory.

This checks that the benchmark manifest, the per-benchmark README guides, and the
runner folders stay consistent, and that no benchmark results or build artifacts
are committed to git. It intentionally does not run any benchmark.

The suite ships guides and code to run each benchmark, not results: a clean
checkout must contain no measured numbers, result JSON, generated data, or
compiled binaries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmark_manifest.json"

# Admin/guide docs that live at the top level and are not per-benchmark folders.
ADMIN_DOCS = {
    "README.md",
    "DATA_POLICY.md",
}

REQUIRED_FIELDS = {
    "id",
    "folder",
    "readme",
    "category",
    "comparison",
    "metrics",
    "runner",
}


def load_manifest() -> dict[str, Any]:
    try:
        return json.loads(MANIFEST.read_text())
    except Exception as exc:
        raise SystemExit(f"ERROR: cannot read {MANIFEST}: {exc}") from exc


def add_error(errors: list[str], message: str) -> None:
    errors.append(f"ERROR: {message}")


def add_warning(warnings: list[str], message: str) -> None:
    warnings.append(f"WARNING: {message}")


def validate_manifest(data: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    benchmarks = data.get("benchmarks")
    if not isinstance(benchmarks, list):
        add_error(errors, "manifest field 'benchmarks' must be a list")
        return errors, warnings

    ids: set[str] = set()
    for index, bench in enumerate(benchmarks):
        if not isinstance(bench, dict):
            add_error(errors, f"benchmark #{index} is not an object")
            continue

        missing = sorted(REQUIRED_FIELDS - set(bench))
        if missing:
            add_error(errors, f"{bench.get('id', '<missing id>')} missing fields: {', '.join(missing)}")

        bench_id = bench.get("id")
        if not isinstance(bench_id, str) or not bench_id:
            add_error(errors, f"benchmark #{index} has invalid id")
        elif bench_id in ids:
            add_error(errors, f"duplicate benchmark id: {bench_id}")
        else:
            ids.add(bench_id)

        folder = bench.get("folder")
        if isinstance(folder, str) and folder:
            if not (ROOT / folder).is_dir():
                add_error(errors, f"{bench_id}: folder does not exist: {folder}")
        else:
            add_error(errors, f"{bench_id}: folder must be a non-empty string")

        readme = bench.get("readme")
        if isinstance(readme, str) and readme:
            if not (ROOT / readme).exists():
                add_error(errors, f"{bench_id}: readme does not exist: {readme}")
        else:
            add_error(errors, f"{bench_id}: readme must be a non-empty string")

        metrics = bench.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            add_error(errors, f"{bench_id}: metrics must be a non-empty list")

        runner = bench.get("runner")
        if not isinstance(runner, list) or not runner:
            add_error(errors, f"{bench_id}: runner must be a non-empty list of commands")

    return errors, warnings


def validate_runner_readmes(strict: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    runner_patterns = ("run_*.py", "run_*.sh", "*.ps1")
    skip = {"results", "tools", "__pycache__"}

    runner_folders: set[Path] = set()
    for pattern in runner_patterns:
        for runner in ROOT.rglob(pattern):
            if any(part in skip for part in runner.relative_to(ROOT).parts):
                continue
            runner_folders.add(runner.parent)

    for folder in sorted(runner_folders):
        if not (folder / "README.md").exists():
            rel = folder.relative_to(ROOT)
            message = f"{rel}: contains runner(s), but no README.md"
            if strict:
                add_error(errors, message)
            else:
                add_warning(warnings, message)

    return errors, warnings


def committed_artifact_reason(rel: str) -> str | None:
    """Classify a git-tracked path that should never be committed."""
    name = rel.rsplit("/", 1)[-1]
    lower = name.lower()
    if rel.startswith("results/") and lower.endswith(".json"):
        return "committed result JSON"
    if lower.endswith((".csv", ".onnx", ".npy", ".nsys-rep", ".sqlite", ".log")):
        return "generated data/result artifact"
    if lower.endswith(".onnx.data"):
        return "generated ONNX artifact"
    if lower.startswith("pred_") and lower.endswith(".txt"):
        return "generated prediction output"
    if "." not in name and name.startswith("opennn_"):
        return "compiled benchmark executable"
    return None


def validate_no_committed_artifacts() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        add_warning(warnings, f"could not list git-tracked files (skipping artifact check): {exc}")
        return errors, warnings

    for rel in filter(None, out.split("\0")):
        reason = committed_artifact_reason(rel)
        if reason:
            add_error(errors, f"{reason} committed (results must stay out of git): {rel}")

    return errors, warnings


import re

BENCH_BUCKETS = {"quality", "throughput", "capacity", "energy", "footprint"}
_MACHINE_PATH = re.compile(r"/home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/|[A-Za-z]:\\\\|/Documents/datasets")
_DATA_SUFFIXES = (".csv", ".npy", ".bmp", ".png", ".jpg", ".jpeg", ".onnx", ".zip", ".gz", ".tar")


def validate_benchmark_ids(data: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Every run_*.py's emitted benchmark_id must equal a manifest id."""
    errors: list[str] = []
    warnings: list[str] = []
    ids = {b["id"] for b in data.get("benchmarks", []) if isinstance(b, dict) and "id" in b}
    for runner in ROOT.rglob("run_*.py"):
        rel = runner.relative_to(ROOT)
        if any(part in {"archive", "tools", "__pycache__"} for part in rel.parts):
            continue
        text = runner.read_text(encoding="utf-8", errors="ignore")
        if re.search(r'"benchmark"\s*:\s*"', text):
            add_error(errors, f'{rel}: result JSON uses key "benchmark" (must be "benchmark_id")')
        for match in re.finditer(r'"benchmark_id"\s*:\s*f?"([^"]*)"', text):
            value = match.group(1)
            if "{" in value:  # f-string with a mode interpolation: check the prefix
                prefix = value.split("{", 1)[0]
                if not any(i.startswith(prefix) for i in ids):
                    add_error(errors, f"{rel}: benchmark_id prefix {prefix!r} matches no manifest id")
            elif value not in ids:
                add_error(errors, f"{rel}: benchmark_id {value!r} is not a manifest id")
    return errors, warnings


def validate_no_data_in_benchmark_folders() -> tuple[list[str], list[str]]:
    """Datasets must live under $OPENNN_BENCH_DATA, never inside a benchmark folder."""
    errors: list[str] = []
    warnings: list[str] = []
    for bucket in sorted(BENCH_BUCKETS):
        bdir = ROOT / bucket
        if not bdir.is_dir():
            continue
        for path in bdir.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            name = path.name.lower()
            if name.endswith(_DATA_SUFFIXES) or name.startswith("cifar_") or name.endswith("_pairs.txt"):
                add_warning(warnings, f"data file inside a benchmark folder (datasets belong under $OPENNN_BENCH_DATA): {path.relative_to(ROOT)}")
    return errors, warnings


def validate_no_hardcoded_paths() -> tuple[list[str], list[str]]:
    """No absolute machine-specific paths in runnable sources."""
    errors: list[str] = []
    warnings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in {"archive", "tools", "__pycache__"} for part in rel.parts):
            continue
        if path.suffix not in {".py", ".sh", ".ps1", ".cpp", ".c", ".h", ".hpp"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hit = _MACHINE_PATH.search(text)
        if hit:
            add_error(errors, f"{rel}: hardcoded machine path {hit.group(0)!r}")
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-readmes",
        action="store_true",
        help="fail when a benchmark runner folder has no README.md",
    )
    args = parser.parse_args()

    data = load_manifest()
    errors, warnings = validate_manifest(data)
    for check in (
        lambda: validate_runner_readmes(args.strict_readmes),
        validate_no_committed_artifacts,
        lambda: validate_benchmark_ids(data),
        validate_no_data_in_benchmark_folders,
        validate_no_hardcoded_paths,
    ):
        e, w = check()
        errors.extend(e)
        warnings.extend(w)

    for warning in warnings:
        print(warning)
    for error in errors:
        print(error)

    if errors:
        print(f"FAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1

    print(f"OK: benchmark inventory valid ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
