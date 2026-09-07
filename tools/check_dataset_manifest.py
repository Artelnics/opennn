#!/usr/bin/env python3
"""Check reviewed example assets; --release also requires redistribution clearance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets.manifest.json"
TEXT = {".csv", ".txt", ".json", ".md", ".js"}


def inventory():
    records = subprocess.check_output(
        ["git", "ls-files", "-s", "-z", "examples"], cwd=ROOT
    ).decode("utf-8").split("\0")
    paths = []
    for record in filter(None, records):
        metadata, name = record.split("\t", 1)
        if "/data/" in name or "/nn/" in name:
            paths.append((name, metadata.split()[1]))
    paths.sort()
    blobs = subprocess.run(
        ["git", "cat-file", "--batch"], cwd=ROOT, check=True,
        input="".join(f"{oid}\n" for _, oid in paths).encode("ascii"),
        stdout=subprocess.PIPE,
    ).stdout
    offset = 0
    groups = {}
    for name, _ in paths:
        end = blobs.index(b"\n", offset)
        size = int(blobs[offset:end].split()[-1])
        data = blobs[end + 1:end + 1 + size]
        offset = end + 2 + size
        path = ROOT / name
        # Git checkouts may use CRLF on Windows; content identity is portable.
        if path.suffix.lower() in TEXT:
            data = data.replace(b"\r\n", b"\n")
        group = name.split("/data/")[0].split("/nn/")[0]
        groups.setdefault(group, []).append(
            (name, hashlib.sha256(data).hexdigest(), len(data))
        )
    return {
        group: {
            "files": len(items),
            "normalized_bytes": sum(item[2] for item in items),
            "sha256": hashlib.sha256("".join(
                f"{name}\0{digest}\n" for name, digest, _ in items
            ).encode("utf-8")).hexdigest(),
        }
        for group, items in groups.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual = inventory()
    expected = {entry["path"]: entry for entry in manifest["datasets"]}
    errors = []
    for group in sorted(actual.keys() | expected.keys()):
        if group not in actual or group not in expected:
            errors.append(f"Unreviewed or removed asset group: {group}")
        elif actual[group] != expected[group]["inventory"]:
            errors.append(f"Changed asset content: {group}; repeat provenance review")
        if args.release and group in expected and not expected[group]["redistribution_cleared"]:
            errors.append(f"Unresolved redistribution: {group}")
    for error in errors:
        print(error)
    if errors:
        return 1
    print(f"Verified {len(actual)} asset groups ({sum(x['files'] for x in actual.values())} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
