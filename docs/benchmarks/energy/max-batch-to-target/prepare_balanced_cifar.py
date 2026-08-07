#!/usr/bin/env python3
"""Prepare a small, class-balanced CIFAR-10 capacity/target dataset."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, default=100)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    images = np.load(source / "cifar_images.npy", mmap_mode="r")
    labels = np.load(source / "cifar_labels.npy", mmap_mode="r")
    selected = []
    for label in range(len(CLASSES)):
        indices = np.flatnonzero(labels == label)
        if indices.size < args.per_class:
            raise ValueError(f"class {label} has only {indices.size} samples")
        selected.extend(int(index) for index in indices[:args.per_class])
    selected = np.asarray(selected, dtype=np.int64)

    subset_images = np.asarray(images[selected], dtype=np.float32)
    subset_labels = np.asarray(labels[selected], dtype=np.int64)
    np.save(output / "cifar_images.npy", subset_images)
    np.save(output / "cifar_labels.npy", subset_labels)

    train = output / "train"
    counters = [0] * len(CLASSES)
    for image, label_value in zip(subset_images, subset_labels):
        label = int(label_value)
        class_dir = train / CLASSES[label]
        class_dir.mkdir(parents=True, exist_ok=True)
        index = counters[label]
        counters[label] += 1
        Image.fromarray(image.astype(np.uint8)).save(
            class_dir / f"{CLASSES[label]}_{index}.bmp")

    metadata = {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "selection": "first N source-order samples per class",
        "per_class": args.per_class,
        "samples": int(subset_labels.size),
        "images_sha256": sha256(output / "cifar_images.npy"),
        "labels_sha256": sha256(output / "cifar_labels.npy"),
        "class_counts": counters,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
