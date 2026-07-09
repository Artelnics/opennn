# Download CIFAR-10 (binary version) and write the two representations the
# benchmark consumes: BMP class folders for OpenNN's ImageDataset and
# images/labels .npy files for PyTorch. Training split only (50,000 images).
#
# Writes to $OPENNN_BENCH_DATA/cifar10 (fallback ~/opennn-benchmark-data/cifar10)
# so datasets stay outside the repository; pass [data_dir] to override.
#
# usage: python prepare_cifar10.py [data_dir]

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

DEFAULT_BENCH_DATA = Path(
    os.environ.get("OPENNN_BENCH_DATA", str(Path.home() / "opennn-benchmark-data")))
DEFAULT_OUT = DEFAULT_BENCH_DATA / "cifar10"

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
data_dir.mkdir(parents=True, exist_ok=True)

archive = data_dir / "cifar-10-binary.tar.gz"
if not archive.exists():
    print(f"downloading {URL} ...")
    urllib.request.urlretrieve(URL, archive)

extracted = data_dir / "cifar-10-batches-bin"
if not extracted.exists():
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir)

records = []
for i in range(1, 6):
    raw = (extracted / f"data_batch_{i}.bin").read_bytes()
    records.append(np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3073))
batch = np.concatenate(records)

labels = batch[:, 0].astype(np.int64)
images = batch[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC

np.save(data_dir / "cifar_images.npy", images.astype(np.float32))
np.save(data_dir / "cifar_labels.npy", labels)
print(f"wrote {data_dir}/cifar_images.npy {images.shape} and cifar_labels.npy")

folders = data_dir / "train"
if not (folders / CLASSES[0]).exists():
    counters = [0] * 10
    for name in CLASSES:
        (folders / name).mkdir(parents=True, exist_ok=True)
    for image, label in zip(images, labels):
        index = counters[label]
        counters[label] += 1
        Image.fromarray(image).save(folders / CLASSES[label] / f"{CLASSES[label]}_{index}.bmp")
    print(f"wrote {sum(counters)} BMPs under {folders}")
else:
    print(f"BMP folders already present under {folders}")
