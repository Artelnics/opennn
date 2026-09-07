#!/usr/bin/env python3
"""Every dataset the suite measures on, one subcommand per family.

This replaces eight `prepare_*.py` scattered through the old
metric directories. Each family gets its data from one place, so "how was this
prepared" has one answer per family instead of one per benchmark that happened
to need it.

    prepare.py dense         HIGGS, normalised, label last
    prepare.py cnn           ImageNet subset, 1000 classes x N, 224x224
    prepare.py transformer   WMT14 English-German sentence pairs
    prepare.py lstm          Beijing PM2.5 hourly
    prepare.py qwen          Pinned Qwen3-4B weights and runtimes

Nothing lands in the repository. Everything goes under $OPENNN_BENCH_DATA
(default ~/opennn-benchmark-data); the only committed artefact is the ImageNet
manifest, which is what makes that subset verifiable rather than trusted.

Every step is skipped when its output already exists, so re-running is cheap
and interrupted downloads resume rather than restart.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DATA = Path(os.environ.get("OPENNN_BENCH_DATA",
                                 str(Path.home() / "opennn-benchmark-data")))

def fetch(url: str, target: Path) -> Path:
    """Download unless already present. Partial files are re-fetched, not kept."""
    if target.exists() and target.stat().st_size > 0:
        print(f"  have {target.name}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    partial = target.with_suffix(target.suffix + ".part")

    with urllib.request.urlopen(urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120) as response:
        with open(partial, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)

    partial.rename(target)
    return target


def fetch_pinned(url: str, target: Path, specification: dict) -> Path:
    """Fetch an immutable benchmark asset and verify it before accepting it."""
    expected_bytes = specification.get("bytes")
    expected_hash = specification["sha256"].lower()

    def valid() -> bool:
        if not target.is_file():
            return False
        if expected_bytes is not None and target.stat().st_size != expected_bytes:
            return False
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            while chunk := handle.read(8 << 20):
                digest.update(chunk)
        return digest.hexdigest() == expected_hash

    if valid():
        print(f"  have verified {target.name}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "OpenNN reproducible benchmark/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        print(f"  resuming {target.name} at {offset:,} bytes")
    else:
        print(f"  downloading {url}")

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=180) as response:
        resumed = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        with partial.open(mode) as output:
            copied = offset if resumed else 0
            while chunk := response.read(8 << 20):
                output.write(chunk)
                copied += len(chunk)
                if expected_bytes and copied % (512 << 20) < len(chunk):
                    print(f"    {copied / (1 << 30):.1f}/{expected_bytes / (1 << 30):.1f} GiB")

    partial.replace(target)
    if not valid():
        raise SystemExit(f"downloaded asset failed size/SHA-256 validation: {target}")
    return target


def fetch_huggingface_pinned(repository: str, revision: str, filename: str,
                             target_directory: Path, specification: dict) -> Path:
    """Use Hugging Face's chunked/Xet client for multi-gigabyte model files."""
    target = target_directory / filename
    if target.is_file() and target.stat().st_size == specification["bytes"]:
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            while chunk := handle.read(8 << 20):
                digest.update(chunk)
        if digest.hexdigest() == specification["sha256"]:
            print(f"  have verified {target.name}")
            return target

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        base = f"https://huggingface.co/{repository}/resolve/{revision}"
        return fetch_pinned(f"{base}/{filename}?download=true", target, specification)

    # This exact .part name belongs to fetch_pinned above.  Leaving it would
    # waste space; hf_hub_download has its own chunk cache and cannot reuse it.
    target.with_suffix(target.suffix + ".part").unlink(missing_ok=True)
    target_directory.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {repository}/{filename} via Hugging Face Xet")
    downloaded = Path(hf_hub_download(
        repo_id=repository, filename=filename, revision=revision,
        local_dir=target_directory,
    ))
    if downloaded.resolve() != target.resolve():
        shutil.copy2(downloaded, target)
    return fetch_pinned("", target, specification)

# --------------------------------------------------------------------------
# dense -- UCI HIGGS
# --------------------------------------------------------------------------

HIGGS_URL = "https://archive.ics.uci.edu/static/public/280/higgs.zip"
HIGGS_FEATURES = 28

def prepare_dense(root: Path, args) -> None:
    """HIGGS, normalised by training statistics, label moved last.

    The raw file is label-first; OpenNN's tabular loader treats the last column
    as the target, so the benchmark CSVs are written feature-first. Both
    engines then read the identical file and neither pays a transformation the
    other does not.

    Normalisation uses training statistics only -- computing them over the
    whole file would leak the test split into the scaling, and every engine
    then reports a slightly optimistic loss for the same reason.
    """
    out = root / "higgs"
    out.mkdir(parents=True, exist_ok=True)
    train_csv, test_csv = out / "higgs_train.csv", out / "higgs_test.csv"

    if train_csv.exists() and test_csv.exists() and not args.force:
        print(f"  have {train_csv.name} and {test_csv.name}")
        return

    archive = fetch(HIGGS_URL, out / "raw" / "higgs.zip")
    raw = out / "raw" / "HIGGS.csv.gz"
    if not raw.exists():
        with zipfile.ZipFile(archive) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith((".csv.gz", ".csv")))
            print(f"  extracting {name}")
            raw.write_bytes(zf.read(name))

    def rows():
        opener = gzip.open if raw.suffix == ".gz" else open
        with opener(raw, "rt") as handle:
            for line in handle:
                parts = line.rstrip("\n").split(",")
                if len(parts) >= HIGGS_FEATURES + 1:
                    yield parts

    print(f"  computing training statistics over {args.train_rows:,} rows")
    sums = [0.0] * HIGGS_FEATURES
    squares = [0.0] * HIGGS_FEATURES
    count = 0
    for row in rows():
        if count >= args.train_rows:
            break
        for i in range(HIGGS_FEATURES):
            value = float(row[i + 1])
            sums[i] += value
            squares[i] += value * value
        count += 1

    means = [total / count for total in sums]
    stds = [max(math.sqrt(max(sq / count - m * m, 0.0)), 1.0e-12)
            for m, sq in zip(means, squares)]
    stds = [s if s > 1.0e-12 else 1.0 for s in stds]

    print(f"  writing {train_csv.name} and {test_csv.name}")
    with open(train_csv, "w", newline="") as ftrain, open(test_csv, "w", newline="") as ftest:
        train_writer, test_writer = csv.writer(ftrain), csv.writer(ftest)
        for index, row in enumerate(rows()):
            scaled = [f"{(float(row[i + 1]) - means[i]) / stds[i]:.6g}"
                      for i in range(HIGGS_FEATURES)]
            scaled.append(row[0])
            (train_writer if index < args.train_rows else test_writer).writerow(scaled)

# --------------------------------------------------------------------------
# cnn -- ImageNet subset
# --------------------------------------------------------------------------

VAL_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
DEVKIT_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
MANIFEST = HERE / "manifests" / "imagenet_subset.manifest"

def prepare_cnn(root: Path, args) -> None:
    """ResNet-50 keeps its true 2048x1000 head, so the subset keeps all 1000
    classes and varies only the images per class -- a 10-class subset would be
    a different network.

    Source is the ILSVRC2012 *validation* archive: 50,000 images over all 1000
    synsets, 50 each, and unlike the training archive it is served without a
    login. Labels come from the devkit, which numbers validation images in its
    own ILSVRC2012_ID order; meta.mat maps those to WNIDs.

    A subset cannot carry an accuracy claim -- at 50 images per class
    ResNet-50 will not reach a meaningful top-1 -- so the CNN cells carry no
    accuracy gate: run.py's quality gate only sees a `test_accuracy` field,
    which today only the dense training drivers print, and the CNN family is
    held to the shape gate (parameter count, samples per epoch) alone. The
    drivers' `quality` mode exists for longer runs that can compare loss.
    """
    from PIL import Image
    from scipy.io import loadmat

    out = root / "imagenet_subset"
    train = out / "train"
    if train.exists() and len(list(train.glob("*/*.JPEG"))) >= 1000 and not args.force:
        print(f"  have {len(list(train.glob('*/*.JPEG'))):,} images in {train}")
        return

    val_tar = fetch(VAL_URL, out / "raw" / "ILSVRC2012_img_val.tar")
    devkit = fetch(DEVKIT_URL, out / "raw" / "ILSVRC2012_devkit_t12.tar.gz")

    with tarfile.open(devkit) as tar:
        meta = tar.extractfile("ILSVRC2012_devkit_t12/data/meta.mat")
        truth = tar.extractfile(
            "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")
        if meta is None or truth is None:
            raise SystemExit(f"{devkit} is not a usable ILSVRC2012 devkit")
        synsets = loadmat(io.BytesIO(meta.read()), squeeze_me=True, struct_as_record=False)
        ids = [int(line) for line in truth.read().split()]

    wnid_of = {int(s.ILSVRC2012_ID): str(s.WNID)
               for s in synsets["synsets"] if int(s.ILSVRC2012_ID) <= 1000}
    labels = [wnid_of[i] for i in ids]
    print(f"  {len(labels):,} validation labels over {len(set(labels))} synsets")

    # Deterministic: sorted synsets, sorted members, seeded pick. Same
    # arguments give the same subset on any machine, which is what lets the
    # manifest mean anything.
    members: dict[str, list[str]] = {}
    for index, wnid in enumerate(labels, start=1):
        members.setdefault(wnid, []).append(f"ILSVRC2012_val_{index:08d}.JPEG")

    chosen: dict[str, str] = {}
    for wnid in sorted(members):
        names = sorted(members[wnid])
        rng = random.Random(f"{args.seed}:{wnid}")
        keep = names if args.per_class >= len(names) else rng.sample(names, args.per_class)
        for name in sorted(keep):
            chosen[name] = wnid
        (train / wnid).mkdir(parents=True, exist_ok=True)

    def square(data: bytes) -> bytes:
        """Resize shortest side then centre crop -- the conventional eval crop."""
        image = Image.open(io.BytesIO(data)).convert("RGB")
        scale = (args.size * 256 / 224) / min(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)),
                             Image.BILINEAR)
        left, top = (image.width - args.size) // 2, (image.height - args.size) // 2
        buffer = io.BytesIO()
        image.crop((left, top, left + args.size, top + args.size)).save(
            buffer, format="JPEG", quality=95)
        return buffer.getvalue()

    # One pass: the archive is 6.3 GB and streams in name order, so seeking per
    # image would cost far more than reading straight through.
    rows, written = [], 0
    with tarfile.open(val_tar) as tar:
        for member in tar:
            wnid = chosen.get(Path(member.name).name)
            if wnid is None:
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            name = Path(member.name).name
            data = square(source.read())
            (train / wnid / name).write_bytes(data)
            rows.append((wnid, name, hashlib.sha256(data).hexdigest()))
            written += 1
            if written % 5000 == 0:
                print(f"    {written:,}/{len(chosen):,}")

    rows.sort()
    MANIFEST.write_text(
        f"# ImageNet subset, ILSVRC2012 validation split\n"
        f"# per_class={args.per_class} size={args.size} seed={args.seed} "
        f"classes={len(members)} images={len(rows)}\n"
        f"# wnid\tfilename\tsha256\n"
        + "".join(f"{w}\t{n}\t{h}\n" for w, n, h in rows))

    print(f"  wrote {written:,} images and {MANIFEST.name}")

# --------------------------------------------------------------------------
# transformer -- WMT14 English-German
# --------------------------------------------------------------------------

# News Commentary v9 is one of the three official WMT14 En-De training
# corpora. The Stanford NMT preprocessed mirror the old script used now
# answers 403, and statmt.org is the primary source anyway.
NC_URL = "https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz"

def opennn_tokens(text: str) -> list[str]:
    """Exactly what OpenNN's WordLevelTokenizer sees, reproduced here.

    `opennn::tokenize_views` ASCII-lowercases, then emits runs of `isalnum` as
    one token and every `ispunct` character as its own, skipping everything
    else. In the C locale "everything else" includes every non-ASCII byte, so
    German text loses its umlauts and the surrounding word splits -- `aeussert`
    written with the umlaut yields two tokens, not one.

    Reproducing the rule rather than approximating it is the point. Writing the
    corpus already split this way makes OpenNN's own tokeniser idempotent over
    it and lets a whitespace split give PyTorch the identical tokens, so the
    two engines process the same sequence length and the same vocabulary. When
    they did not, OpenNN was doing 158 positions per sequence against PyTorch's
    128 -- 23% more arithmetic, invisible in a throughput number.
    """
    raw = text.encode("utf-8")
    tokens: list[str] = []
    index = 0

    while index < len(raw):
        byte = raw[index]
        character = chr(byte)

        if character.isascii() and character.isalnum():
            start = index
            while index < len(raw):
                nxt = chr(raw[index])
                if not (nxt.isascii() and nxt.isalnum()):
                    break
                index += 1
            tokens.append(raw[start:index].decode("ascii").lower())
        elif character.isascii() and 33 <= byte <= 126 and not character.isalnum():
            tokens.append(character)
            index += 1
        else:
            index += 1

    return tokens

def prepare_transformer(root: Path, args) -> None:
    """`source <TAB> target` pairs, bounded on both axes.

    Capacity depends on the corpus only through the derived vocabulary and the
    sequence lengths, so both are capped: `--max-tokens` bounds the padded
    length every engine builds, and `--max-pairs` keeps the per-trial corpus
    load small, since every fresh-process trial re-reads it.
    """
    out = root / "wmt14"
    pairs = out / "wmt14_pairs.txt"
    if pairs.exists() and not args.force:
        print(f"  have {pairs.name}")
        return

    archive = fetch(NC_URL, out / "raw" / "training-parallel-nc-v9.tgz")

    english = german = None
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.name.endswith("news-commentary-v9.de-en.en"):
                english = tar.extractfile(member).read().decode("utf-8", "replace")
            elif member.name.endswith("news-commentary-v9.de-en.de"):
                german = tar.extractfile(member).read().decode("utf-8", "replace")

    if english is None or german is None:
        raise SystemExit(f"{archive} did not contain the de-en pair files")

    out.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(pairs, "w", encoding="utf-8") as handle:
        # splitlines() would also break on VT, FF, NEL and U+2028, which occur
        # inside natural prose and would silently misalign the two sides
        # against each other from that point on. Only newlines separate records.
        for en_line, de_line in zip(english.split("\n"), german.split("\n")):
            en, de = en_line.strip(), de_line.strip()
            if not en or not de or "\t" in en or "\t" in de:
                continue

            # LanguageDataset reads this file with CSV quote semantics, so an
            # unterminated double quote swallows the tab and the record then
            # "does not contain exactly two fields". Natural prose has
            # unbalanced quotes routinely -- a sentence opening with one and
            # closing with a typographic one, for instance -- so those pairs
            # are dropped rather than rewritten. It costs 0.5% of the corpus
            # and keeps every surviving line exactly as the source wrote it.
            if en.count('"') % 2 or de.count('"') % 2:
                continue
            # Pre-tokenised with OpenNN's own rule, so both engines agree on
            # the token stream, and truncated after tokenising so the cap
            # bounds what the model actually sees.
            en_tokens = opennn_tokens(en)[:args.max_tokens]
            de_tokens = opennn_tokens(de)[:args.max_tokens]
            if not en_tokens or not de_tokens:
                continue

            handle.write(f"{' '.join(en_tokens)}\t{' '.join(de_tokens)}\n")
            kept += 1
            if args.max_pairs and kept >= args.max_pairs:
                break

    print(f"  wrote {kept:,} pairs to {pairs}")

# --------------------------------------------------------------------------
# lstm -- Beijing PM2.5
# --------------------------------------------------------------------------

PM25_URL = "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip"

PM25_COLUMNS = ("year", "month", "day", "hour", "DEWP", "TEMP", "PRES",
                "Iws", "Is", "Ir", "cbwd_NE", "cbwd_NW", "cbwd_SE", "cbwd_cv", "pm2_5")

def prepare_lstm(root: Path, args) -> None:
    """Hourly Beijing PM2.5, one-hot wind direction, target last.

    The raw series has gaps in pm2.5, which are linearly interpolated rather
    than dropped: dropping them would silently change the sampling interval
    the recurrent model sees, which is the one thing a forecasting benchmark
    must keep constant.
    """
    out = root / "beijing_pm25"
    prepared = out / "beijing_pm25_forecasting.csv"
    if prepared.exists() and not args.force:
        print(f"  have {prepared.name}")
        return

    archive = fetch(PM25_URL, out / "raw" / "beijing_pm25_uci.zip")
    with zipfile.ZipFile(archive) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw_rows = list(csv.DictReader(io.StringIO(zf.read(name).decode("utf-8", "replace"))))

    directions = ("NE", "NW", "SE", "cv")
    values = [None if row["pm2.5"] in ("NA", "") else float(row["pm2.5"]) for row in raw_rows]

    # Linear interpolation across gaps, holding the endpoints flat.
    known = [i for i, v in enumerate(values) if v is not None]
    if not known:
        raise SystemExit("no pm2.5 readings in the raw file")
    for i, value in enumerate(values):
        if value is not None:
            continue
        before = max((k for k in known if k < i), default=None)
        after = min((k for k in known if k > i), default=None)
        if before is None:
            values[i] = values[after]
        elif after is None:
            values[i] = values[before]
        else:
            span = after - before
            values[i] = values[before] + (values[after] - values[before]) * (i - before) / span

    out.mkdir(parents=True, exist_ok=True)
    with open(prepared, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PM25_COLUMNS)
        for row, pm25 in zip(raw_rows, values):
            one_hot = [1 if row["cbwd"] == d else 0 for d in directions]
            writer.writerow([row["year"], row["month"], row["day"], row["hour"],
                             row["DEWP"], row["TEMP"], row["PRES"],
                             row["Iws"], row["Is"], row["Ir"], *one_hot,
                             f"{pm25:g}"])

    print(f"  wrote {len(raw_rows):,} hourly rows to {prepared}")


# --------------------------------------------------------------------------
# qwen -- one canonical BF16 model, two serializations, pinned runtimes
# --------------------------------------------------------------------------

QWEN_TOOLS = HERE / "tools"
QWEN_MANIFEST = HERE / "manifests" / "qwen_manifest.json"


def _clone_llama(destination: Path, specification: dict) -> None:
    commit = specification["commit"]
    if (destination / ".git").is_dir():
        current = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip()
        if current == commit:
            print(f"  have llama.cpp {commit[:12]}")
            return
        subprocess.check_call(["git", "-C", str(destination), "fetch", "origin", commit])
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", "--filter=blob:none", "--no-checkout",
                               specification["repository"], str(destination)])
    subprocess.check_call(["git", "-C", str(destination), "checkout", "--detach", commit])
    current = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if current != commit:
        raise SystemExit(f"llama.cpp checkout is {current}, expected {commit}")


def _apply_llama_benchmark_patch(llama: Path, specification: dict) -> None:
    patch_spec = specification["benchmark_patch"]
    patch = QWEN_TOOLS / patch_spec["path"]
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    if digest != patch_spec["sha256"]:
        raise SystemExit(f"llama-bench patch hash is {digest}, expected {patch_spec['sha256']}")
    reverse = subprocess.run(["git", "-C", str(llama), "apply", "--reverse",
                              "--check", str(patch)], capture_output=True, text=True)
    if reverse.returncode == 0:
        print("  have llama-bench fixed-context instrumentation")
        return
    check = subprocess.run(["git", "-C", str(llama), "apply", "--check", str(patch)],
                           capture_output=True, text=True)
    if check.returncode:
        raise SystemExit(f"llama-bench patch does not apply cleanly:\n{check.stderr}")
    subprocess.check_call(["git", "-C", str(llama), "apply", str(patch)])
    print("  applied llama-bench fixed-context instrumentation")


def _install_converter_requirements(llama: Path) -> None:
    candidates = [
        llama / "requirements" / "requirements-convert_hf_to_gguf.txt",
        llama / "requirements.txt",
    ]
    requirements = next((path for path in candidates if path.is_file()), None)
    if requirements is None:
        raise SystemExit("llama.cpp has no converter requirements file")

    uv = os.environ.get("OPENNN_BENCH_UV")
    if uv:
        command = [uv, "pip", "install", "--python", sys.executable,
                   "-r", str(requirements), "safetensors"]
    else:
        command = [sys.executable, "-m", "pip", "install", "-r", str(requirements),
                   "safetensors"]
    subprocess.check_call(command)


def prepare_qwen(root: Path, args) -> None:
    manifest = json.loads(QWEN_MANIFEST.read_text(encoding="utf-8"))
    qwen_root = root / "qwen3"
    opennn_dir = qwen_root / "models" / "opennn"
    canonical_dir = qwen_root / "models" / "canonical"
    tool_root = qwen_root / "tools"

    for key, destination in (("opennn_model", opennn_dir),
                             ("canonical_model", canonical_dir)):
        model = manifest[key]
        for name, specification in model["files"].items():
            fetch_huggingface_pinned(model["repository"], model["revision"],
                                     name, destination, specification)

    llama = tool_root / "llama.cpp"
    _clone_llama(llama, manifest["llama_cpp"])
    _apply_llama_benchmark_patch(llama, manifest["llama_cpp"])

    ollama_archive = tool_root / "downloads" / "ollama-windows-amd64.zip"
    fetch_pinned(manifest["ollama"]["url"], ollama_archive,
                 {"sha256": manifest["ollama"]["sha256"]})
    ollama_dir = tool_root / "ollama" / manifest["ollama"]["version"]
    if not (ollama_dir / "ollama.exe").is_file():
        print(f"  extracting Ollama to {ollama_dir}")
        ollama_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ollama_archive) as archive:
            archive.extractall(ollama_dir)
    if not (ollama_dir / "ollama.exe").is_file():
        raise SystemExit(f"Ollama archive did not contain ollama.exe: {ollama_archive}")

    gguf = qwen_root / "models" / "qwen3-4b-bf16.gguf"
    if not gguf.is_file() or args.force:
        _install_converter_requirements(llama)
        converter = llama / "convert_hf_to_gguf.py"
        print(f"  converting canonical BF16 weights to {gguf.name}")
        subprocess.check_call([
            sys.executable, str(converter), str(canonical_dir),
            "--outfile", str(gguf), "--outtype", "bf16",
        ])

    validator = QWEN_TOOLS / "validate_qwen.py"
    subprocess.check_call([
        sys.executable, str(validator), "--data-root", str(root),
        "--write-report", str(qwen_root / "validation.json"),
    ])

FAMILIES = {
    "dense": prepare_dense,
    "cnn": prepare_cnn,
    "transformer": prepare_transformer,
    "lstm": prepare_lstm,
    "qwen": prepare_qwen,
}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("family", nargs="+", choices=sorted(FAMILIES) + ["all"])
    parser.add_argument("--data-root", type=Path, default=BENCH_DATA)
    parser.add_argument("--force", action="store_true", help="rebuild even if present")
    parser.add_argument("--train-rows", type=int, default=10_500_000, help="dense")
    parser.add_argument("--per-class", type=int, default=50, help="cnn")
    parser.add_argument("--size", type=int, default=224, help="cnn")
    parser.add_argument("--seed", type=int, default=42, help="cnn")
    parser.add_argument("--max-pairs", type=int, default=200_000, help="transformer")
    parser.add_argument("--max-tokens", type=int, default=128, help="transformer")
    args = parser.parse_args()

    wanted = sorted(FAMILIES) if "all" in args.family else args.family
    for family in wanted:
        print(f"{family}:")
        FAMILIES[family](args.data_root, args)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
