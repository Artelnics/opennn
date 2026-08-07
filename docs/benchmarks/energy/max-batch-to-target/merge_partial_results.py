#!/usr/bin/env python3
"""Merge single-engine max-batch-to-target artifacts with provenance."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument(
        "--capacity-verification", action="append", default=[],
        help="additional capacity artifact whose hash should be retained")
    parser.add_argument(
        "artifacts", nargs="+",
        help="artifact path, optionally followed by ::engine to select one result")
    args = parser.parse_args()

    selections = []
    paths = []
    for value in args.artifacts:
        if "::" in value:
            path_value, engine = value.rsplit("::", 1)
            selections.append(engine)
        else:
            path_value = value
            selections.append(None)
        paths.append(Path(path_value).resolve())
    documents = [json.loads(path.read_text(encoding="utf-8"))
                 for path in paths]
    first = documents[0]
    capacity_hash = first["capacity_artifact"]["sha256"]
    target = first["protocol"]["target_training_loss"]
    precision = first["protocol"]["precision"]
    benchmark_id = first["benchmark_id"]
    for path, document in zip(paths[1:], documents[1:]):
        checks = (
            (document["benchmark_id"], benchmark_id, "benchmark_id"),
            (document["capacity_artifact"]["sha256"], capacity_hash,
             "capacity artifact"),
            (document["protocol"]["target_training_loss"], target, "target"),
            (document["protocol"]["precision"], precision, "precision"),
        )
        for actual, expected, label in checks:
            if actual != expected:
                raise SystemExit(
                    f"{path}: incompatible {label}: {actual!r} != {expected!r}")

    merged = dict(first)
    merged["generated_utc"] = datetime.now(timezone.utc).isoformat()
    merged["run_id"] = Path(args.output).stem
    merged["batches"] = {}
    merged["results"] = {}
    merged["source_artifacts"] = []
    if args.capacity_verification:
        merged["capacity_verification_artifacts"] = [
            {
                "path": str(Path(value).resolve()),
                "sha256": sha256(Path(value).resolve()),
            }
            for value in args.capacity_verification
        ]
    merged["idle_baseline_w"] = sorted(
        float(document["idle_baseline_w"]) for document in documents
    )[len(documents) // 2]
    for path, document, selected in zip(paths, documents, selections):
        source = {
            "path": str(path),
            "sha256": sha256(path),
            "run_id": document["run_id"],
        }
        if selected:
            if selected not in document["results"]:
                raise SystemExit(f"{path}: result {selected!r} not found")
            source["selected_engine"] = selected
        merged["source_artifacts"].append(source)
        for engine, batch in document["batches"].items():
            if selected and engine != selected:
                continue
            if engine in merged["batches"]:
                raise SystemExit(f"duplicate batch entry for {engine}")
            merged["batches"][engine] = batch
        for engine, result in document["results"].items():
            if selected and engine != selected:
                continue
            if engine in merged["results"]:
                raise SystemExit(f"duplicate result entry for {engine}")
            merged["results"][engine] = result

    if args.note:
        merged["protocol"] = dict(merged["protocol"])
        merged["protocol"]["merge_note"] = args.note

    base = merged["results"].get("opennn")
    if base and base.get("n_ok"):
        comparisons = {}
        for engine, result in merged["results"].items():
            if engine == "opennn" or not result.get("n_ok"):
                continue
            energy_ratio = (
                result["energy_total_j_median"]
                / base["energy_total_j_median"])
            time_ratio = (
                result["train_window_s_median"]
                / base["train_window_s_median"])
            result["energy_ratio_vs_opennn"] = round(energy_ratio, 6)
            result["time_ratio_vs_opennn"] = round(time_ratio, 6)
            comparisons[f"opennn_vs_{engine}"] = {
                "opennn_energy_ratio": round(1.0 / energy_ratio, 6),
                "opennn_time_ratio": round(1.0 / time_ratio, 6),
                "opennn_energy_percent_difference":
                    round((1.0 / energy_ratio - 1.0) * 100.0, 3),
                "opennn_time_percent_difference":
                    round((1.0 / time_ratio - 1.0) * 100.0, 3),
            }
        merged["comparisons"] = comparisons

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(output)

if __name__ == "__main__":
    main()
