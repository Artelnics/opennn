# ISIC 2016 melanoma images

`images.zip` contains 102 images derived directly from the **ISIC 2016,
Part 3 training release**: 50 `benign` and 52 `malignant` (melanoma).
The release publisher identifies the images and ground-truth labels as
**CC0 1.0 Universal** on the [official data page](https://challenge.isic-archive.com/data/#2016)
(checked 2026-09-30).

CC0 permits copying, modification and redistribution, including commercial use,
without asking permission or requiring attribution. See the
[CC0 dedication](https://creativecommons.org/publicdomain/zero/1.0/) and its
[legal text](https://creativecommons.org/publicdomain/zero/1.0/legalcode).
Keep this provenance notice and `manifest.csv` with the example data. OpenNN's
LGPL software licence does not replace this dedication.

## Source and credit

Publisher: International Skin Imaging Collaboration (ISIC).

Dataset publication: David Gutman, Noel C. F. Codella, Emre Celebi, Brian Helba,
Michael Marchetti, Nabin Mishra and Allan Halpern. *Skin Lesion Analysis toward
Melanoma Detection: A Challenge at the International Symposium on Biomedical
Imaging (ISBI) 2016, hosted by the International Skin Imaging Collaboration
(ISIC)*, 2016. [arXiv:1605.01397](https://arxiv.org/abs/1605.01397).

Original release files:

- [ISBI2016_ISIC_Part3_Training_Data.zip](https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part3_Training_Data.zip)
  — SHA-256 `d78bd1de08511b514e58784c91fd2d8dab24e407cafd023a7debbceb9155ffb2`.
- [ISBI2016_ISIC_Part3_Training_GroundTruth.csv](https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv)
  — SHA-256 `0ffadd63b7cec650cddfbe5df54d1d3167453fe6004c2bbf05824184b0c4d405`.

## OpenNN preparation

Sort the original identifiers within each class and take the first 50 benign
images and the first 52 malignant images. Keep the original labels and ISIC
identifiers. Convert each JPEG to RGB and resize the whole image to 300 x 300
with Lanczos resampling, without cropping; this changes its aspect ratio when
the original is not square. Save it as BMP, without copying source metadata.
Archive paths are `isic2016/<class>/<ISIC identifier>.bmp`.

`manifest.csv` records every original archive member, label, output path and
the SHA-256 of both the source JPEG and resulting BMP. All names are unique.
The `isic2016` subdirectory keeps the replacement separate from any previously
staged images when reusing a build directory.

Bundled file SHA-256:

- `images.zip`: `309abd5cf598a878f153c2c661b3637e8c040877158923e84804e721aa1a7310`.
- `manifest.csv`: `cf7944987c06415d5bd30f7901dee15b5df032a4e094088c7a6a4f6819c85648`.

The example creates its own random training/validation/testing split. This
small, deterministically selected subset is for demonstrating the application;
it is not the official ISIC challenge evaluation split.

## Reproduce

Run the following with Python 3.12 and Pillow 12.3.0 from the repository root.
Downloads and generated files go outside the checkout. It verifies the original
release hashes before generating the archive and manifest. Fixed ZIP metadata
keeps the archive reproducible in the same Python/Pillow/zlib environment.
No Python script needs to be saved.

```python
import csv
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen
import zipfile
from PIL import Image

CACHE = Path("../opennn-isic2016-source")
OUT = Path("../opennn-isic2016-regenerated")
BASE = "https://isic-archive.s3.amazonaws.com/challenges/2016/"
SOURCES = {
    "ISBI2016_ISIC_Part3_Training_Data.zip":
        "d78bd1de08511b514e58784c91fd2d8dab24e407cafd023a7debbceb9155ffb2",
    "ISBI2016_ISIC_Part3_Training_GroundTruth.csv":
        "0ffadd63b7cec650cddfbe5df54d1d3167453fe6004c2bbf05824184b0c4d405",
}

def sha256(data):
    return hashlib.sha256(data).hexdigest()

CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
for name, expected in SOURCES.items():
    path = CACHE / name
    if not path.exists():
        with urlopen(BASE + name, timeout=60) as response, path.open("wb") as f:
            while block := response.read(1024 * 1024):
                f.write(block)
    if sha256(path.read_bytes()) != expected:
        raise ValueError(f"Source hash mismatch: {name}")

with (CACHE / "ISBI2016_ISIC_Part3_Training_GroundTruth.csv").open(
        encoding="utf-8", newline="") as f:
    labels = list(csv.reader(f))
assert len(labels) == 900
assert len({image_id for image_id, _ in labels}) == 900
rows = []
with zipfile.ZipFile(CACHE / "ISBI2016_ISIC_Part3_Training_Data.zip") as source:
    with zipfile.ZipFile(OUT / "images.zip", "w") as destination:
        for category, count in (("benign", 50), ("malignant", 52)):
            selected = sorted(i for i, label in labels if label == category)[:count]
            assert len(selected) == count
            for image_id in selected:
                member = f"ISBI2016_ISIC_Part3_Training_Data/{image_id}.jpg"
                original = source.read(member)
                with Image.open(io.BytesIO(original)) as image:
                    resized = image.convert("RGB").resize(
                        (300, 300), Image.Resampling.LANCZOS)
                    output = io.BytesIO()
                    resized.save(output, format="BMP")
                bmp = output.getvalue()
                filename = f"isic2016/{category}/{image_id}.bmp"
                info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                destination.writestr(info, bmp, compresslevel=9)
                rows.append((member, image_id, category, filename,
                             sha256(original), sha256(bmp)))

with (OUT / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(("source_archive_member", "image_id", "class", "output_file",
                     "source_sha256", "bmp_sha256"))
    writer.writerows(rows)
assert len(rows) == 102
for name in ("images.zip", "manifest.csv"):
    print(name, sha256((OUT / name).read_bytes()))
```
