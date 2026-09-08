#!/usr/bin/env python3
"""Download pinned sources, reconstruct example data outside Git, and verify it."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "airfoil": ("291.zip", "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip", "5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c"),
    "breast_cancer": ("15.zip", "https://archive.ics.uci.edu/static/public/15/breast+cancer+wisconsin+original.zip", "3f91e49bceb30c0de8ea988344357236f26ed8bd536415ac594226015b6fd84d"),
    "iris": ("53.zip", "https://archive.ics.uci.edu/static/public/53/iris.zip", "d11fe30213d36434a0879aab7cb00ce3c812eb7ba2495874438abff7b7b762e9"),
    "concrete": ("165.zip", "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip", "dad85d14de8aee4e07479daa774e6b569a313715b71a3b92c95a07cf91c2c9a7"),
    "amazon": ("331.zip", "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip", "afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343"),
    "mnist": ("mnist.npz", "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"),
    "ecg": ("tensorflow-ecg.csv", "https://storage.googleapis.com/download.tensorflow.org/data/ecg.csv", "72ce7b040ca0c6ed36c3368e570c6ac4ddf20100476e47373c63b2395e012df1"),
}
BUNDLES = dict(zip(SOURCES, ("airfoil_self_noise", "breast_cancer", "iris_plant", "concrete", "amazon_reviews", "mnist", "ecg5000_anomaly_detection")))
MISSING = ((10,5), (90,4), (98,4), (101,5), (102,7), (104,9), (105,2), (137,9), (159,3), (202,2), (209,7))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def indexed_files(bundle):
    """Read Git blobs once, without hydrating thousands of OneDrive images."""
    entries = subprocess.check_output(["git", "ls-files", "-s", "-z", f"examples/{bundle}/data"], cwd=ROOT).decode().split("\0")
    items = []
    for entry in filter(None, entries):
        metadata, path = entry.split("\t", 1)
        mode, oid, stage = metadata.split()
        if stage != "0":
            raise ValueError(f"Unresolved index entry: {path}")
        items.append((path.split("/data/", 1)[1], oid))
    data = subprocess.check_output(["git", "cat-file", "--batch"], cwd=ROOT, input="".join(oid + "\n" for _, oid in items).encode())
    result, offset = {}, 0
    for name, _ in items:
        end = data.index(b"\n", offset)
        size = int(data[offset:end].split()[-1])
        result[name] = data[end + 1:end + 1 + size]
        offset = end + size + 2
    return result


def source(name, cache):
    filename, url, digest = SOURCES[name]
    target = cache / filename
    if target.exists():
        data = target.read_bytes()
    else:
        print(f"Downloading {name}: {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
    if sha(data) != digest:
        raise ValueError(f"Source checksum mismatch: {filename}; review upstream changes")
    if not target.exists():
        target.write_bytes(data)
    return data


def member(data, suffix):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        matches = [name for name in archive.namelist() if name.split("/")[-1] == suffix]
        if len(matches) != 1:
            raise ValueError(f"Expected one archive member named {suffix}, got {matches}")
        return archive.read(matches[0])


def table(raw, delimiter):
    quoting = csv.QUOTE_NONE if delimiter == "\t" else csv.QUOTE_MINIMAL
    return list(csv.reader(io.StringIO(raw.decode("utf-8-sig")), delimiter=delimiter, quoting=quoting))


def verify_table(expected, actual):
    import math
    if len(expected) != len(actual):
        raise ValueError(f"Row count differs: {len(expected)} versus {len(actual)}")
    for i, (left, right) in enumerate(zip(expected, actual), 1):
        if len(left) != len(right):
            raise ValueError(f"Column count differs in row {i}")
        for j, (a, b) in enumerate(zip(left, right), 1):
            if str(a) == str(b):
                continue
            try:
                equal = math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)
            except ValueError:
                equal = False
            if not equal:
                raise ValueError(f"Cell differs at row {i}, column {j}: {a!r} versus {b!r}")


def reconstruct(name, raw, indexed, output):
    output.mkdir(parents=True, exist_ok=True)
    written = []

    def write(filename, data):
        path = output / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        written.append(filename)

    def write_table(filename, rows, delimiter=";", header=True):
        original = table(indexed[filename], delimiter)
        # Keep the reviewed feature names/order, including historical spelling.
        full = ([original[0]] if header else []) + rows
        verify_table(original, full)
        buffer = io.StringIO(newline="")
        if delimiter == "\t":
            # LanguageDataset consumes literal TSV, including quotation marks.
            buffer.write("".join("\t".join(row) + "\n" for row in full))
        else:
            csv.writer(buffer, delimiter=delimiter, lineterminator="\n").writerows(full)
        write(filename, buffer.getvalue().encode())

    if name == "airfoil":
        rows = [line.split() for line in member(raw, "airfoil_self_noise.dat").decode().splitlines() if line.strip()]
        write_table("airfoil_self_noise.csv", rows)
    elif name == "breast_cancer":
        rows = [row[1:-1] + ["0" if row[-1] == "2" else "1"] for row in table(member(raw, "breast-cancer-wisconsin.data"), ",") if row and "?" not in row]
        write_table("breast_cancer.csv", rows)
        altered = [row.copy() for row in rows]
        for row, column in MISSING:
            altered[row-1][column-1] = "NA"
        write_table("breast_cancer_with_missing_values.csv", altered)
    elif name == "iris":
        rows = [row[:-1] + [row[-1].lower().replace("-", "_")] for row in table(member(raw, "bezdekIris.data"), ",") if row]
        write_table("iris_plant_original.csv", rows)
    elif name == "concrete":
        import xlrd
        sheet = xlrd.open_workbook(file_contents=member(raw, "Concrete_Data.xls")).sheet_by_index(0)
        write_table("concrete_uci.csv", [sheet.row_values(i) for i in range(1, sheet.nrows)], ",")
    elif name == "amazon":
        rows = [row[:-1] + ["Bad" if row[-1] == "0" else "Good"] for row in table(member(raw, "amazon_cells_labelled.txt"), "\t") if row]
        rows[0][0] = "'" + rows[0][0]  # Retained local adaptation, documented in DATASETS.md.
        write_table("amazon_cells_labelled.txt", rows, "\t", header=False)
        reduced = [row.copy() for row in rows[:10]]
        reduced[0][0] = reduced[0][0][1:]
        write_table("amazon_cells_reduced.txt", reduced, "\t", header=False)
        small = [row.copy() for row in rows[:5]]
        small[0][0] = small[0][0][:-1] + " Bad."
        write_table("amazon_cells_labelled_small.txt", small + [[]], "\t", header=False)
    elif name == "ecg":
        import numpy as np
        filename = "ecg.csv"
        if raw.replace(b"\r\n", b"\n") != indexed[filename].replace(b"\r\n", b"\n"):
            raise ValueError("ECG CSV differs from the pinned TensorFlow derivative")
        write(filename, raw)
        rows = [[str(i)] for i in np.random.RandomState(21).permutation(4998)[:1000]]
        write_table("test_indices.csv", rows, ",", header=False)
    elif name == "mnist":
        import numpy as np
        from PIL import Image
        words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
        counts = [0] * 10
        expected = {path for path in indexed if path.endswith(".bmp")}
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            for pixels, label in zip(data["x_test"], data["y_test"]):
                digit = int(label)
                filename = f"{words[digit]}/{digit}_{counts[digit]}.bmp"
                counts[digit] += 1
                with Image.open(io.BytesIO(indexed[filename])) as image:
                    if image.size != (28, 28) or not np.array_equal(pixels, np.asarray(image.convert("L"))):
                        raise ValueError(f"MNIST pixels differ: {filename}")
                buffer = io.BytesIO()
                Image.fromarray(pixels).save(buffer, format="BMP")
                write(filename, buffer.getvalue())
        if set(written) != expected or len(written) != 10000:
            raise ValueError("MNIST inventory is not exactly the complete 10,000-image test set")
    for filename, data in indexed.items():
        if filename == "SOURCE.md":
            write(filename, data)
    return {filename: sha((output / filename).read_bytes()) for filename in sorted(written)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()
    for path in (args.cache, args.output):
        if path.resolve().is_relative_to(ROOT):
            parser.error("Cache and generated datasets must be outside the checkout")
        path.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "datasets": {}}
    for name in args.datasets:
        bundle = BUNDLES[name]
        files = reconstruct(name, source(name, args.cache), indexed_files(bundle), args.output / bundle)
        report["datasets"][name] = {"source_url": SOURCES[name][1], "source_sha256": SOURCES[name][2], "files": files}
        print(f"Verified and reconstructed {name}: {len(files)} files", flush=True)
    (args.output / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Verified equivalence does not clear unresolved redistribution terms; see DATASETS.md.")


if __name__ == "__main__":
    main()
