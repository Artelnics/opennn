#!/usr/bin/env python3
"""Merge independently rerun ResNet-50 capacity engines with provenance."""

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--tensorflow", required=True)
    parser.add_argument("--opennn", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {name: Path(value).resolve() for name, value in (
        ("base", args.base), ("tensorflow", args.tensorflow))}
    if args.opennn:
        paths["opennn"] = Path(args.opennn).resolve()
    with paths["base"].open(encoding="utf-8") as stream:
        artifact = json.load(stream)
    with paths["tensorflow"].open(encoding="utf-8") as stream:
        tensorflow = json.load(stream)
    opennn = None
    if "opennn" in paths:
        with paths["opennn"].open(encoding="utf-8") as stream:
            opennn = json.load(stream)

    merged = deepcopy(artifact)
    for precision, rows in tensorflow["results"].items():
        merged["results"].setdefault(precision, {})["tensorflow"] = rows["tensorflow"]
        metrics = merged["metrics"]["max_train_batch"].setdefault(precision, {})
        metrics["tensorflow_xla"] = int(rows["tensorflow"]["max_batch"])
    if opennn:
        for precision, rows in opennn["results"].items():
            merged["results"].setdefault(precision, {})["opennn_pool1"] = rows["opennn_pool1"]
            metrics = merged["metrics"]["max_train_batch"].setdefault(precision, {})
            metrics["opennn_batch_pool_1"] = int(rows["opennn_pool1"]["max_batch"])

    merged["run_id"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    merged["merged_capacity_artifacts"] = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in paths.items()
    }
    output = Path(args.output).resolve()
    with output.open("w", encoding="utf-8") as stream:
        json.dump(merged, stream, indent=2)
    print(output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
