#!/usr/bin/env python
"""Download Places365-Standard and convert it to class-folder BMPs.

The output root is intended for OpenNN ImageDataset: it contains only one
subdirectory per class, each with RGB BMP files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

TRAIN_URL = "https://data.csail.mit.edu/places/places365/train_256_places365standard.tar"
FILELIST_URL = "https://data.csail.mit.edu/places/places365/filelist_places365-standard.tar"

TRAIN_ARCHIVE = "train_256_places365standard.tar"
FILELIST_ARCHIVE = "filelist_places365-standard.tar"
EXPECTED_TRAIN_IMAGES = 1_803_460


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def sanitize_class_name(parts: list[str]) -> str:
    # Places uses category paths such as data_256_standard/a/airfield/image.jpg.
    # Keep the leading letter to preserve uniqueness and stable alphabetical sort.
    raw = "_".join(part.strip("/") for part in parts if part.strip("/"))
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    if not cleaned:
        raise ValueError(f"Cannot derive class name from parts: {parts!r}")
    return cleaned


def class_name_from_train_member(member_name: str) -> str | None:
    path = PurePosixPath(member_name)
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        return None

    parts = list(path.parts)
    if "data_256_standard" in parts:
        index = parts.index("data_256_standard")
        class_parts = parts[index + 1 : -1]
    else:
        # Fallback for mirrors with only class folders inside the tar.
        class_parts = parts[-3:-1] if len(parts) >= 3 else parts[:-1]

    if not class_parts:
        return None
    return sanitize_class_name(class_parts)


def download_with_curl(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    curl = "curl.exe" if os.name == "nt" else "curl"
    command = [
        curl,
        "-L",
        "--fail",
        "--retry",
        "30",
        "--retry-delay",
        "20",
        "--retry-all-errors",
        "-C",
        "-",
        "-o",
        str(destination),
        url,
    ]

    log(f"Downloading {url}")
    log(f"Destination: {destination}")
    subprocess.run(command, check=True)
    log(f"Download ready: {destination}")


def extract_filelist_metadata(filelist_tar: Path, work_root: Path) -> None:
    metadata_dir = work_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    names_to_extract = {
        "categories_places365.txt",
        "places365_train_standard.txt",
        "places365_val.txt",
    }

    extracted = []
    with tarfile.open(filelist_tar, "r:*") as archive:
        for member in archive:
            basename = PurePosixPath(member.name).name
            if basename not in names_to_extract or not member.isfile():
                continue
            src = archive.extractfile(member)
            if src is None:
                continue
            dst = metadata_dir / basename
            with dst.open("wb") as out:
                out.write(src.read())
            extracted.append(dst.name)

    log(f"Metadata extracted: {', '.join(extracted) if extracted else 'none'}")


def convert_train_archive(
    train_tar: Path,
    target_root: Path,
    work_root: Path,
    size: int,
    limit: int | None,
) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    state_path = work_root / "places365_bmp_state.json"
    error_log_path = work_root / "places365_bmp_errors.log"

    converted = 0
    skipped = 0
    errors = 0
    seen = 0
    classes: set[str] = set()
    started = time.time()

    log(f"Converting {train_tar}")
    log(f"Output root: {target_root}")
    log(f"Output size: {size}x{size} RGB BMP")

    with tarfile.open(train_tar, "r:*") as archive, error_log_path.open("a", encoding="utf-8") as error_log:
        for member in archive:
            if not member.isfile():
                continue

            class_name = class_name_from_train_member(member.name)
            if class_name is None:
                continue

            seen += 1
            classes.add(class_name)
            output_dir = target_root / class_name
            output_path = output_dir / (PurePosixPath(member.name).stem + ".bmp")

            if output_path.exists() and output_path.stat().st_size > 1024:
                skipped += 1
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = output_path.with_suffix(".bmp.tmp")
                try:
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError("archive.extractfile returned None")

                    with Image.open(source) as image:
                        image = image.convert("RGB")
                        if image.size != (size, size):
                            image = image.resize((size, size), Image.Resampling.BILINEAR)
                        image.save(tmp_path, format="BMP")

                    tmp_path.replace(output_path)
                    converted += 1
                except Exception as exc:  # noqa: BLE001 - keep long jobs alive.
                    errors += 1
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except OSError:
                        pass
                    error_log.write(f"{member.name}\t{exc}\n")
                    error_log.flush()

            if seen % 1000 == 0:
                elapsed = max(time.time() - started, 1.0)
                rate = seen / elapsed
                remaining = max(EXPECTED_TRAIN_IMAGES - seen, 0)
                eta_minutes = remaining / rate / 60.0
                state = {
                    "archive": str(train_tar),
                    "target_root": str(target_root),
                    "size": size,
                    "seen": seen,
                    "expected_train_images": EXPECTED_TRAIN_IMAGES,
                    "converted": converted,
                    "skipped": skipped,
                    "errors": errors,
                    "classes_seen": len(classes),
                    "images_per_second": round(rate, 2),
                    "eta_minutes": round(eta_minutes, 1),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                log(
                    "progress "
                    f"seen={seen:,}/{EXPECTED_TRAIN_IMAGES:,} "
                    f"converted={converted:,} skipped={skipped:,} "
                    f"errors={errors:,} classes={len(classes)} "
                    f"rate={rate:.1f}/s eta={eta_minutes:.1f}m"
                )

            if limit is not None and seen >= limit:
                log(f"Limit reached: {limit}")
                break

    final_state = {
        "archive": str(train_tar),
        "target_root": str(target_root),
        "size": size,
        "seen": seen,
        "expected_train_images": EXPECTED_TRAIN_IMAGES,
        "converted": converted,
        "skipped": skipped,
        "errors": errors,
        "classes_seen": len(classes),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    state_path.write_text(json.dumps(final_state, indent=2), encoding="utf-8")
    log(f"Finished. seen={seen:,} converted={converted:,} skipped={skipped:,} errors={errors:,} classes={len(classes)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", default=r"\\Artelnics\data_sets\places365_bmp_224")
    parser.add_argument("--work-root", default=r"\\Artelnics\data_sets\places365_work")
    parser.add_argument("--size", type=int, default=224, help="Output BMP side length.")
    parser.add_argument("--limit", type=int, default=None, help="Convert only N images, for smoke tests.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_root = Path(args.target_root)
    work_root = Path(args.work_root)
    archive_dir = work_root / "archives"
    log_dir = work_root / "logs"
    archive_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_tar = archive_dir / TRAIN_ARCHIVE
    filelist_tar = archive_dir / FILELIST_ARCHIVE

    log("Places365-Standard BMP preparation")
    log(f"Target root: {target_root}")
    log(f"Work root: {work_root}")
    log(f"Train archive: {train_tar}")
    log(f"Filelist archive: {filelist_tar}")

    if args.plan_only:
        return 0

    if not args.skip_download:
        download_with_curl(FILELIST_URL, filelist_tar)
        extract_filelist_metadata(filelist_tar, work_root)
        download_with_curl(TRAIN_URL, train_tar)
    else:
        log("Skipping downloads by request.")

    if not train_tar.exists():
        raise FileNotFoundError(train_tar)

    convert_train_archive(
        train_tar=train_tar,
        target_root=target_root,
        work_root=work_root,
        size=args.size,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
