#!/usr/bin/env python3
"""Create a tiny ImageNet class-folder subset for benchmark smoke tests.

The output keeps the same train/class/image layout the full benchmark expects,
but copies only a few images per class. Use this before paying the full
ImageNet cache/build/run cost.

The output root defaults to $OPENNN_BENCH_DATA/imagenet_smoke (fallback
~/opennn-benchmark-data/imagenet_smoke) so datasets stay outside the repository;
pass an explicit output path to override.

  usage: prepare_imagenet_smoke.py /path/to/imagenet [output]
         [--classes 4] [--images-per-class 4] [--symlink]
"""

import argparse
import os
import shutil
from pathlib import Path

EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg"}

DEFAULT_BENCH_DATA = Path(
    os.environ.get("OPENNN_BENCH_DATA", str(Path.home() / "opennn-benchmark-data")))
DEFAULT_OUT = DEFAULT_BENCH_DATA / "imagenet_smoke"


def train_dir_for(path):
    path = Path(path).expanduser()
    return path / "train" if (path / "train").is_dir() else path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="ImageNet root or train directory")
    parser.add_argument("output", nargs="?", default=str(DEFAULT_OUT),
                        help="Output root; a train/ folder is created inside it "
                             "(default $OPENNN_BENCH_DATA/imagenet_smoke)")
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--images-per-class", type=int, default=4)
    parser.add_argument("--symlink", action="store_true")
    args = parser.parse_args()

    source_train = train_dir_for(args.source)
    output_train = Path(args.output).expanduser() / "train"
    output_train.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted(p for p in source_train.iterdir() if p.is_dir() and not p.name.startswith("."))
    if len(class_dirs) < args.classes:
        raise SystemExit(f"Only found {len(class_dirs)} classes under {source_train}")

    written = 0
    for class_dir in class_dirs[: args.classes]:
        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        if len(images) < args.images_per_class:
            raise SystemExit(f"Only found {len(images)} images under {class_dir}")

        target_dir = output_train / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for src in images[: args.images_per_class]:
            dst = target_dir / src.name
            if dst.exists():
                continue
            if args.symlink:
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)
            written += 1

    print(f"wrote {written} images under {output_train}")


if __name__ == "__main__":
    main()
