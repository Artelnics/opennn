#!/usr/bin/env python3
"""Prepare UCI HIGGS for dense benchmark runners.

The raw UCI file is label-first:

    label,feature_0,...,feature_27

OpenNN's tabular loader naturally treats the last column as the target, so this
script writes benchmark CSVs as:

    feature_0,...,feature_27,label

The canonical split follows the UCI recommendation used by the original HIGGS
benchmark: first 10,500,000 rows for training and last 500,000 rows for test.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


UCI_PAGE = "https://archive.ics.uci.edu/dataset/280/higgs"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/280/higgs.zip"
LEGACY_GZ_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
FEATURES = 28
DEFAULT_TRAIN_ROWS = 10_500_000
DEFAULT_TEST_ROWS = 500_000
DEFAULT_BENCH_DATA = Path(
    os.environ.get("OPENNN_BENCH_DATA", str(Path.home() / "opennn-benchmark-data"))
)
DEFAULT_OUT = DEFAULT_BENCH_DATA / "higgs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Path to HIGGS.csv.gz or HIGGS.csv. Required unless --download is used.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download higgs.zip from UCI into --out/raw before preparing.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=(
            "Output directory for higgs_train.csv, higgs_test.csv, and metadata. "
            "Defaults to $OPENNN_BENCH_DATA/higgs, or ~/opennn-benchmark-data/higgs."
        ),
    )
    parser.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS)
    parser.add_argument("--test-rows", type=int, default=DEFAULT_TEST_ROWS)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Write raw feature values instead of train-set z-score features.",
    )
    parser.add_argument(
        "--float-format",
        default="%.8g",
        help="printf-style format for prepared feature values.",
    )
    return parser.parse_args()


def open_higgs(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return open(path, "rt", newline="")


def download_raw(out_dir: Path) -> Path:
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "higgs.zip"
    gz_path = raw_dir / "HIGGS.csv.gz"

    if gz_path.exists():
        return gz_path

    print(f"downloading {UCI_ZIP_URL}", file=sys.stderr)
    urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if name.endswith("HIGGS.csv.gz")]
        if not members:
            raise RuntimeError("Downloaded zip does not contain HIGGS.csv.gz")
        zf.extract(members[0], raw_dir)
        extracted = raw_dir / members[0]
        if extracted != gz_path:
            extracted.replace(gz_path)

    return gz_path


def read_features_and_label(row: list[str], line_number: int) -> tuple[list[float], float]:
    if len(row) != FEATURES + 1:
        raise ValueError(
            f"line {line_number}: expected {FEATURES + 1} columns, got {len(row)}"
        )
    label = float(row[0])
    features = [float(value) for value in row[1:]]
    return features, label


def training_stats(raw_path: Path, train_rows: int) -> tuple[list[float], list[float]]:
    sums = [0.0] * FEATURES
    sums_sq = [0.0] * FEATURES
    count = 0

    with open_higgs(raw_path) as f:
        reader = csv.reader(f)
        for line_number, row in enumerate(reader, start=1):
            if count >= train_rows:
                break
            features, _ = read_features_and_label(row, line_number)
            for i, value in enumerate(features):
                sums[i] += value
                sums_sq[i] += value * value
            count += 1

    if count != train_rows:
        raise RuntimeError(f"requested {train_rows} training rows, found {count}")

    means = [total / count for total in sums]
    stds = []
    for mean, total_sq in zip(means, sums_sq):
        variance = max(total_sq / count - mean * mean, 0.0)
        std = math.sqrt(variance)
        stds.append(std if std > 1.0e-12 else 1.0)
    return means, stds


def transform(features: list[float], means: list[float], stds: list[float]) -> list[float]:
    return [(value - means[i]) / stds[i] for i, value in enumerate(features)]


def write_split(
    raw_path: Path,
    out_dir: Path,
    train_rows: int,
    test_rows: int,
    normalize: bool,
    float_format: str,
) -> dict:
    means = [0.0] * FEATURES
    stds = [1.0] * FEATURES
    if normalize:
        print("calculating train-set feature statistics", file=sys.stderr)
        means, stds = training_stats(raw_path, train_rows)

    train_path = out_dir / "higgs_train.csv"
    test_path = out_dir / "higgs_test.csv"
    counts = {"train_rows": 0, "test_rows": 0}

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing {train_path}", file=sys.stderr)
    print(f"writing {test_path}", file=sys.stderr)

    with open_higgs(raw_path) as f, open(train_path, "w", newline="") as train_f, open(
        test_path, "w", newline=""
    ) as test_f:
        reader = csv.reader(f)
        train_writer = csv.writer(train_f, lineterminator="\n")
        test_writer = csv.writer(test_f, lineterminator="\n")

        for line_number, row in enumerate(reader, start=1):
            row_index = line_number - 1
            if row_index >= train_rows + test_rows:
                break

            features, label = read_features_and_label(row, line_number)
            if normalize:
                features = transform(features, means, stds)

            prepared = [float_format % value for value in features]
            prepared.append("1" if label >= 0.5 else "0")

            if row_index < train_rows:
                train_writer.writerow(prepared)
                counts["train_rows"] += 1
            else:
                test_writer.writerow(prepared)
                counts["test_rows"] += 1

    if counts["train_rows"] != train_rows:
        raise RuntimeError(f"requested {train_rows} training rows, wrote {counts['train_rows']}")
    if counts["test_rows"] != test_rows:
        raise RuntimeError(f"requested {test_rows} test rows, wrote {counts['test_rows']}")

    return {
        "train_csv": str(train_path),
        "test_csv": str(test_path),
        "train_rows": counts["train_rows"],
        "test_rows": counts["test_rows"],
        "feature_means": means,
        "feature_stds": stds,
        "normalized": normalize,
    }


def main() -> int:
    args = parse_args()
    raw_path = args.raw
    if args.download:
        raw_path = download_raw(args.out)
    if raw_path is None:
        raise SystemExit("--raw is required unless --download is used")
    raw_path = raw_path.resolve()
    if not raw_path.exists():
        raise SystemExit(f"raw file not found: {raw_path}")

    split = write_split(
        raw_path=raw_path,
        out_dir=args.out,
        train_rows=args.train_rows,
        test_rows=args.test_rows,
        normalize=not args.no_normalize,
        float_format=args.float_format,
    )

    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "name": "HIGGS",
            "uci_page": UCI_PAGE,
            "uci_zip_url": UCI_ZIP_URL,
            "legacy_gz_url": LEGACY_GZ_URL,
            "raw_path": str(raw_path),
            "raw_layout": "label,feature_0,...,feature_27",
            "prepared_layout": "feature_0,...,feature_27,label",
        },
        "feature_count": FEATURES,
        "target": {"name": "signal", "negative": 0, "positive": 1},
        "split": split,
        "command": " ".join(sys.argv),
    }

    metadata_path = args.out / "higgs_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
