"""Prepare shared, hashed tensors for paired training-quality experiments.

Training and test examples are disjoint. Transformations and vocabularies are
fitted on training data only. Binary arrays are little-endian float32 (images:
uint8 HWC); both engines consume these exact files.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("dense", "lstm", "cnn", "transformer")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def array_file(folder, name, values):
    values = np.asarray(values, dtype="u1" if values.dtype == np.uint8 else "<f4")
    path = folder / f"{name}.bin"
    values.tofile(path)
    return {
        "file": path.name,
        "shape": list(values.shape),
        "dtype": "uint8" if values.dtype == np.uint8 else "float32",
        "sha256": digest(path),
    }


def read_array(folder, specification):
    return np.memmap(
        Path(folder) / specification["file"],
        mode="r",
        dtype="u1" if specification["dtype"] == "uint8" else "<f4",
        shape=tuple(specification["shape"]),
    )


def validate_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "quality-data-v1"
        or manifest.get("model") not in MODELS
    ):
        raise ValueError("Unsupported quality manifest")
    for split in ("train", "test"):
        fields = manifest[split]
        count = fields["x"]["shape"][0]
        if count < 1:
            raise ValueError("Empty quality split")
        for spec in fields.values():
            item = (path.parent / spec["file"]).resolve()
            if not item.is_relative_to(path.parent):
                raise ValueError("Tensor escapes the manifest directory")
            if spec["shape"][0] != count or spec["dtype"] not in ("uint8", "float32"):
                raise ValueError("Tensor sample count or dtype mismatch")
            expected = int(np.prod(spec["shape"])) * (
                1 if spec["dtype"] == "uint8" else 4
            )
            if item.stat().st_size != expected or digest(item) != spec["sha256"]:
                raise ValueError(f"Tensor identity mismatch: {item}")
    if manifest["train"]["x"]["sha256"] == manifest["test"]["x"]["sha256"]:
        raise ValueError("Training and test inputs are identical")
    return manifest


def profile(model, smoke):
    return {
        "dense": {"hidden": 8 if smoke else 1024, "layers": 2},
        "lstm": {"hidden": 8 if smoke else 128, "past": 4 if smoke else 24},
        "cnn": {
            "size": 32 if smoke else 224,
            "blocks": [1, 1, 1, 1] if smoke else [3, 4, 6, 3],
        },
        "transformer": {
            "d_model": 16 if smoke else 512,
            "heads": 1 if smoke else 8,
            "ff": 32 if smoke else 2048,
            "layers": 1 if smoke else 6,
            "sequence": 6 if smoke else 130,
            "vocab_limit": 32 if smoke else 20000,
        },
    }[model]


def prepare_one(model, root, folder, smoke=False):
    folder.mkdir(parents=True, exist_ok=False)
    cfg = profile(model, smoke)
    rng = np.random.default_rng(20260910)
    result = {
        "schema": "quality-data-v1",
        "model": model,
        "smoke": smoke,
        "profile": cfg,
        "split_seed": 20260910,
        "initialization": "engine defaults; independent seeded initializations, not identical tensors",
        "preprocessing": "shared prepared tensors",
        "source_files": [],
    }

    def store(split, x, y, decoder=None):
        result[split] = {
            "x": array_file(folder, split + "-x", x),
            "y": array_file(folder, split + "-y", y),
        }
        if decoder is not None:
            result[split]["decoder"] = array_file(folder, split + "-decoder", decoder)

    def source(path):
        result["source_files"].append(
            {"path": str(path.resolve()), "sha256": digest(path)}
        )

    if smoke:
        for split, n in (("train", 4), ("test", 3)):
            if model == "dense":
                x = rng.normal(size=(n, 28)).astype("f4")
                store(split, x, (x[:, :1] > 0).astype("f4"))
            elif model == "lstm":
                x = rng.normal(size=(n, cfg["past"], 15)).astype("f4")
                store(split, x, x[:, -1, -1:])
            elif model == "cnn":
                x = rng.integers(
                    0, 256, (n, cfg["size"], cfg["size"], 3), dtype=np.uint8
                )
                store(split, x, np.eye(3, dtype="f4")[np.arange(n) % 3])
            else:
                seq = cfg["sequence"]
                x = rng.integers(4, 12, (n, seq)).astype("f4")
                x[:, 0], x[:, -1] = 2, 3
                y = np.zeros((n, seq), dtype="f4")
                y[:, :3] = rng.integers(4, 12, (n, 3))
                y[:, 3] = 3
                decoder = np.concatenate(
                    (np.full((n, 1), 2), y[:, :-1]), axis=1
                ).astype("f4")
                store(split, x, y, decoder)
        if model == "transformer":
            result["vocabulary"] = ["[PAD]", "[UNK]", "[START]", "[END]"] + [
                f"word{i}" for i in range(28)
            ]
            result["references"] = [
                " ".join(result["vocabulary"][int(t)] for t in row[:3])
                for row in read_array(folder, result["test"]["y"])
            ]
    elif model == "dense":
        import pandas as pd

        for split, name in (
            ("train", "higgs_train_250k.csv"),
            ("test", "higgs_test.csv"),
        ):
            path = root / "higgs" / name
            source(path)
            values = pd.read_csv(path, header=None, dtype=np.float32).to_numpy()
            store(split, values[:, :-1], values[:, -1:])
        result["preprocessing"] = (
            "prepare.py HIGGS normalization, fitted on training inputs"
        )
    elif model == "lstm":
        path = root / "beijing_pm25/beijing_pm25_forecasting.csv"
        source(path)
        values = np.loadtxt(path, delimiter=",", skiprows=1, dtype="f4")
        boundary = int(len(values) * 0.8)
        mean = values[:boundary].mean(0)
        std = values[:boundary].std(0)
        std = np.where(std > 1e-12, std, 1)
        standardized = (values - mean) / std
        # Split raw chronology first; no window crosses the split boundary.
        for split, start, end in (
            ("train", 0, boundary),
            ("test", boundary, len(values)),
        ):
            part = standardized[start:end]
            x = np.lib.stride_tricks.sliding_window_view(part, cfg["past"], axis=0)[
                :-1
            ].transpose(0, 2, 1)
            y = part[cfg["past"] :, -1:]
            store(split, np.ascontiguousarray(x), y)
        result["target_mean"], result["target_scale"] = float(mean[-1]), float(std[-1])
        result["preprocessing"] = (
            "Chronological 80/20 split; training-only standardization; separate windows. Prepared source retains prepare.py missing-value interpolation."
        )
    elif model == "cnn":
        from PIL import Image

        image_root = root / "imagenet_subset/train"
        classes = sorted(
            p for p in image_root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
        rows = {"train": [], "test": []}
        identities = []
        for label, directory in enumerate(classes):
            images = sorted(
                p
                for p in directory.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            if len(images) < 2:
                raise ValueError(f"Need at least two images per class: {directory}")
            order = rng.permutation(len(images))
            cut = max(1, min(len(images) - 1, int(0.8 * len(images))))
            for index, item in enumerate(order):
                rows["train" if index < cut else "test"].append((images[item], label))
        seen = {}
        for split, items in rows.items():
            path = folder / f"{split}-x.bin"
            labels = np.zeros((len(items), len(classes)), dtype="f4")
            with path.open("wb") as output:
                for index, (image, label) in enumerate(items):
                    sha = digest(image)
                    if sha in seen and seen[sha] != split:
                        raise ValueError(
                            "Identical image content appears in both splits"
                        )
                    seen[sha] = split
                    identities.append(
                        {
                            "path": str(image.relative_to(image_root)),
                            "split": split,
                            "sha256": sha,
                        }
                    )
                    with Image.open(image) as decoded:
                        np.asarray(
                            decoded.convert("RGB").resize(
                                (cfg["size"], cfg["size"]), Image.Resampling.BILINEAR
                            ),
                            dtype="u1",
                        ).tofile(output)
                    labels[index, label] = 1
            result[split] = {
                "x": {
                    "file": path.name,
                    "shape": [len(items), cfg["size"], cfg["size"], 3],
                    "dtype": "uint8",
                    "sha256": digest(path),
                },
                "y": array_file(folder, split + "-y", labels),
            }
        (folder / "image-split.json").write_text(json.dumps(identities, indent=2))
        result["classes"] = [p.name for p in classes]
        result["preprocessing"] = (
            "Stratified 80/20 subset split; one shared PIL bilinear RGB resize; input /255 in both models; no augmentation"
        )
    else:
        path = root / "wmt14/wmt14_pairs.txt"
        source(path)
        pairs = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                pairs.append(
                    tuple(
                        tuple(
                            re.findall(r"[a-z0-9]+|[^\w\s]", p.lower(), flags=re.ASCII)[
                                : cfg["sequence"] - 2
                            ]
                        )
                        for p in parts
                    )
                )
        # Deduplicate before splitting to prevent repeated pairs crossing splits.
        pairs = list(dict.fromkeys(pairs))
        order = rng.permutation(len(pairs))
        cut = int(0.9 * len(pairs))
        train = [pairs[i] for i in order[:cut]]
        test = [pairs[i] for i in order[cut:]]
        counter = Counter(token for pair in train for side in pair for token in side)
        vocab = ["[PAD]", "[UNK]", "[START]", "[END]"] + [
            t for t, _ in counter.most_common(cfg["vocab_limit"] - 4)
        ]
        lookup = {token: i for i, token in enumerate(vocab)}
        result["vocabulary"] = vocab
        result["references"] = [" ".join(t) for _, t in test]
        for split, items in (("train", train), ("test", test)):
            x = np.zeros((len(items), cfg["sequence"]), dtype="f4")
            y = np.zeros_like(x)
            decoder = np.zeros_like(x)
            for i, (s, t) in enumerate(items):
                source_ids = [2] + [lookup.get(k, 1) for k in s] + [3]
                target_ids = [lookup.get(k, 1) for k in t] + [3]
                x[i, : len(source_ids)] = source_ids
                y[i, : len(target_ids)] = target_ids
                decoder[i, : len(target_ids)] = [2] + target_ids[:-1]
            store(split, x, y, decoder)
        result["preprocessing"] = (
            "Deduplicated 90/10 split; training-only shared word vocabulary; teacher forcing; causal decoder; PAD=0 excluded from loss; dropout=0"
        )
    (folder / "manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    validate_manifest(folder / "manifest.json")
    return folder / "manifest.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get("OPENNN_BENCH_DATA", Path.home() / "opennn-benchmark-data")
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New prepared-data directory outside the checkout",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny synthetic fixtures; never an accuracy result",
    )
    args = parser.parse_args(argv)
    if args.out.resolve().is_relative_to(ROOT):
        parser.error("Prepared datasets must stay outside the checkout")
    args.out.mkdir(parents=True, exist_ok=False)
    for model in MODELS if args.model == "all" else (args.model,):
        print(
            prepare_one(model, args.data_root, args.out / model, args.smoke), flush=True
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
