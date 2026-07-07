#!/usr/bin/env python3
"""Validate the consolidated benchmark inventory.

This checks that the top-level benchmark notes, benchmark manifest, and result
artifact references stay aligned. It intentionally does not run benchmarks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmark_manifest.json"

ADMIN_DOCS = {
    "README.md",
    "BENCHMARK_RUNBOOK.md",
    "PRESENTATION_CLAIMS.md",
    "TRACKS.md",
    "DATA_POLICY.md",
    "DENSE_HIGGS_MIGRATION.md",
}

REQUIRED_FIELDS = {
    "id",
    "doc",
    "category",
    "status",
    "lifecycle",
    "publication_readiness",
    "comparison",
    "metrics",
    "runner",
    "needs",
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

    lifecycle_values = data.get("lifecycle_values", {})
    allowed_lifecycles = set(lifecycle_values)
    if not allowed_lifecycles:
        add_error(errors, "manifest must define lifecycle_values")

    ids: set[str] = set()
    docs: set[str] = set()
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

        doc = bench.get("doc")
        if isinstance(doc, str) and doc:
            docs.add(doc)
            if not (ROOT / doc).exists():
                add_error(errors, f"{bench_id}: doc does not exist: {doc}")
        else:
            add_error(errors, f"{bench_id}: doc must be a non-empty string")

        lifecycle = bench.get("lifecycle")
        if lifecycle not in allowed_lifecycles:
            add_error(errors, f"{bench_id}: unknown lifecycle '{lifecycle}'")

        metrics = bench.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            add_error(errors, f"{bench_id}: metrics must be a non-empty list")

        runner = bench.get("runner")
        if not isinstance(runner, dict) or "status" not in runner:
            add_error(errors, f"{bench_id}: runner must contain a status")

        for key in ("latest_result", "comparison_result"):
            ref = bench.get(key)
            if isinstance(ref, str) and not (ROOT / ref).exists():
                add_error(errors, f"{bench_id}: {key} missing: {ref}")

        local_ref = bench.get("latest_local_result")
        if isinstance(local_ref, str) and not (ROOT / local_ref).exists():
            add_warning(warnings, f"{bench_id}: local result not present in this checkout: {local_ref}")

    top_level_notes = {
        path.name for path in ROOT.glob("*.md")
        if path.name not in ADMIN_DOCS
    }

    for note in sorted(top_level_notes - docs):
        add_error(errors, f"top-level benchmark note is not in manifest: {note}")

    for doc in sorted(docs - top_level_notes):
        if doc not in ADMIN_DOCS and (ROOT / doc).parent == ROOT:
            add_error(errors, f"manifest doc is not a top-level benchmark note: {doc}")

    return errors, warnings


def validate_track_readmes(strict: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    runner_patterns = ("run_*.py", "run_*.sh", "*.ps1")

    for folder in sorted(path for path in ROOT.iterdir() if path.is_dir()):
        if folder.name in {"results", "tools", "__pycache__"}:
            continue
        runner_count = 0
        for pattern in runner_patterns:
            runner_count += sum(1 for _ in folder.rglob(pattern))
        if runner_count and not (folder / "README.md").exists():
            message = f"{folder.name}: {runner_count} runner(s), but no README.md"
            if strict:
                add_error(errors, message)
            else:
                add_warning(warnings, message)

    return errors, warnings


def generated_artifact_reason(path: Path) -> str | None:
    name = path.name
    if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
        return "Python bytecode/cache"
    if name.endswith(".onnx.data") or path.suffix == ".onnx":
        return "generated ONNX artifact"
    if path.suffix == ".csv":
        return "generated CSV artifact"
    if path.suffix == ".txt" and name.startswith("pred_"):
        return "generated prediction output"
    if not path.suffix and name.startswith("opennn_"):
        return "compiled benchmark executable"
    return None


def validate_generated_artifacts(data: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    legacy_artifacts = data.get("legacy_artifacts", [])
    if not isinstance(legacy_artifacts, list):
        add_error(errors, "manifest field 'legacy_artifacts' must be a list")
        legacy_artifacts = []

    allowed: set[str] = set()
    for index, artifact in enumerate(legacy_artifacts):
        if not isinstance(artifact, dict):
            add_error(errors, f"legacy_artifacts #{index} is not an object")
            continue
        rel = artifact.get("path")
        if not isinstance(rel, str) or not rel:
            add_error(errors, f"legacy_artifacts #{index} has invalid path")
            continue
        if "\\" in rel:
            add_error(errors, f"legacy artifact path must use forward slashes: {rel}")
        allowed.add(rel)
        full_path = ROOT / rel
        if not full_path.exists():
            add_error(errors, f"legacy artifact missing from checkout: {rel}")
        elif generated_artifact_reason(full_path) is None:
            add_error(errors, f"legacy artifact is not recognized as generated: {rel}")

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        reason = generated_artifact_reason(path)
        if reason and rel not in allowed:
            add_error(errors, f"unlisted generated artifact ({reason}): {rel}")

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
    readme_errors, readme_warnings = validate_track_readmes(args.strict_readmes)
    errors.extend(readme_errors)
    warnings.extend(readme_warnings)
    artifact_errors, artifact_warnings = validate_generated_artifacts(data)
    errors.extend(artifact_errors)
    warnings.extend(artifact_warnings)

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
