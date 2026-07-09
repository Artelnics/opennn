#!/usr/bin/env python3
"""Prepare UCI Beijing PM2.5 for the OpenNN forecasting benchmark.

The UCI file contains hourly PM2.5 and weather rows for 2010-2014. This script
keeps the full hourly series, interpolates missing numeric values, one-hot
encodes wind direction, and writes a numeric CSV whose last column is the target.
"""

import argparse
import csv
import math
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

DEFAULT_URL = "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip"
RAW_CSV_NAME = "PRSA_data_2010.1.1-2014.12.31.csv"
OUTPUT_NAME = "beijing_pm25_forecasting.csv"

WIND_DIRECTIONS = ["NE", "NW", "SE", "cv"]

SOURCE_TO_OUTPUT = [
    ("year", "year"),
    ("month", "month"),
    ("day", "day"),
    ("hour", "hour"),
    ("DEWP", "DEWP"),
    ("TEMP", "TEMP"),
    ("PRES", "PRES"),
    ("Iws", "Iws"),
    ("Is", "Is"),
    ("Ir", "Ir"),
    ("pm2.5", "pm2_5"),
]

OUTPUT_COLUMNS = (
    [name for _, name in SOURCE_TO_OUTPUT[:-1]]
    + [f"cbwd_{direction}" for direction in WIND_DIRECTIONS]
    + ["pm2_5"]
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=Path(os.environ.get("OPENNN_BENCH_DATA", str(Path.home() / "opennn-benchmark-data"))) / "beijing_pm25",
        type=Path,
        help="Directory for raw and prepared files (default: $OPENNN_BENCH_DATA/beijing_pm25)",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        help="Optional local UCI raw CSV or ZIP path",
    )
    parser.add_argument(
        "--source-url",
        default=DEFAULT_URL,
        help="UCI source URL used when --raw-path is not provided",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the UCI source again even if a raw file exists",
    )
    return parser.parse_args()


def download(url, destination):
    print(f"download_url={url}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "OpenNN-benchmark-prep/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with open(destination, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    return destination


def extract_csv_from_zip(zip_path, output_dir):
    with zipfile.ZipFile(zip_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV file found in {zip_path}")

        selected = next(
            (name for name in csv_names if Path(name).name == RAW_CSV_NAME),
            csv_names[0],
        )

        destination = output_dir / RAW_CSV_NAME
        with archive.open(selected) as source, open(destination, "wb") as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
        return destination


def ensure_raw_csv(args, output_dir):
    if args.raw_path:
        raw_path = args.raw_path.expanduser().resolve()
        if raw_path.suffix.lower() == ".zip":
            return extract_csv_from_zip(raw_path, output_dir)
        return raw_path

    raw_csv = output_dir / RAW_CSV_NAME
    if raw_csv.exists() and not args.force_download:
        return raw_csv

    source_url = args.source_url
    if source_url.lower().endswith(".zip"):
        zip_path = output_dir / "beijing_pm25_uci.zip"
        if args.force_download or not zip_path.exists():
            download(source_url, zip_path)
        return extract_csv_from_zip(zip_path, output_dir)

    return download(source_url, raw_csv)


def parse_float(value):
    text = value.strip()
    if not text or text.upper() == "NA":
        return None

    number = float(text)
    if not math.isfinite(number):
        return None
    return number


def interpolate(values, column_name):
    known = [index for index, value in enumerate(values) if value is not None]
    missing = len(values) - len(known)
    if not known:
        raise RuntimeError(f"Column {column_name} has no numeric values")
    if missing == 0:
        return values, 0

    filled = list(values)
    first = known[0]
    for index in range(0, first):
        filled[index] = filled[first]

    previous = first
    for current in known[1:]:
        gap = current - previous - 1
        if gap > 0:
            start = filled[previous]
            end = filled[current]
            for step in range(1, gap + 1):
                fraction = step / (gap + 1)
                filled[previous + step] = start + (end - start) * fraction
        previous = current

    last = known[-1]
    for index in range(last + 1, len(filled)):
        filled[index] = filled[last]

    return filled, missing


def format_number(value):
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.9g}"


def read_raw_rows(raw_csv):
    with open(raw_csv, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [source for source, _ in SOURCE_TO_OUTPUT if source not in reader.fieldnames]
        if "cbwd" not in reader.fieldnames:
            missing.append("cbwd")
        if missing:
            raise RuntimeError(f"Missing expected columns in {raw_csv}: {', '.join(missing)}")
        return list(reader)


def prepare(raw_csv, output_csv):
    rows = read_raw_rows(raw_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {raw_csv}")

    numeric = {}
    missing_by_column = {}
    for source, output in SOURCE_TO_OUTPUT:
        values = [parse_float(row[source]) for row in rows]
        numeric[output], missing_by_column[output] = interpolate(values, source)

    wind_values = set(WIND_DIRECTIONS)
    unknown_wind_rows = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_COLUMNS)

        for index, row in enumerate(rows):
            wind = row["cbwd"].strip()
            if wind not in wind_values:
                unknown_wind_rows += 1

            output_row = [format_number(numeric[name][index]) for _, name in SOURCE_TO_OUTPUT[:-1]]
            output_row.extend("1" if wind == direction else "0" for direction in WIND_DIRECTIONS)
            output_row.append(format_number(numeric["pm2_5"][index]))
            writer.writerow(output_row)

    return {
        "raw_rows": len(rows),
        "output_rows": len(rows),
        "missing_interpolated": missing_by_column,
        "unknown_wind_rows": unknown_wind_rows,
    }


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    os.makedirs(output_dir, exist_ok=True)

    raw_csv = ensure_raw_csv(args, output_dir)
    output_csv = output_dir / OUTPUT_NAME
    stats = prepare(raw_csv, output_csv)

    print(f"raw_csv={raw_csv}")
    print(f"output_csv={output_csv}")
    print(f"raw_rows={stats['raw_rows']}")
    print(f"output_rows={stats['output_rows']}")
    print(f"missing_interpolated={stats['missing_interpolated']}")
    print(f"unknown_wind_rows={stats['unknown_wind_rows']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
