# Download CIFAR-100 (binary version) and write the two representations the
# benchmark consumes: BMP class folders for OpenNN's ImageDataset and
# images/labels .npy files for PyTorch. Training split only (50,000 images),
# fine labels (100 classes). Mirrors prepare_cifar10.py so the same harness
# (opennn_resnet50_speed, pytorch_resnet50_speed, run_resnet50.py) runs on it.
#
# CIFAR-100 binary records are 3074 bytes: [coarse_label, fine_label, 3072 px];
# CIFAR-10 records are 3073 bytes: [label, 3072 px]. We keep the fine label.
#
# Writes to $OPENNN_BENCH_DATA/cifar100 (fallback ~/opennn-benchmark-data/cifar100)
# so datasets stay outside the repository; pass [data_dir] to override.
#
# usage: python prepare_cifar100.py [data_dir]

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

URL = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"

DEFAULT_BENCH_DATA = Path(
    os.environ.get("OPENNN_BENCH_DATA", str(Path.home() / "opennn-benchmark-data")))
DEFAULT_OUT = DEFAULT_BENCH_DATA / "cifar100"

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
data_dir.mkdir(parents=True, exist_ok=True)

archive = data_dir / "cifar-100-binary.tar.gz"
if not archive.exists():
    print(f"downloading {URL} ...")
    urllib.request.urlretrieve(URL, archive)

extracted = data_dir / "cifar-100-binary"
if not extracted.exists():
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir)

raw = (extracted / "train.bin").read_bytes()
batch = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3074)

labels = batch[:, 1].astype(np.int64)  # fine label (0..99); byte 0 is coarse
images = batch[:, 2:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC

np.save(data_dir / "cifar_images.npy", images.astype(np.float32))
np.save(data_dir / "cifar_labels.npy", labels)
print(f"wrote {data_dir}/cifar_images.npy {images.shape} and cifar_labels.npy"
      f" ({labels.max() + 1} classes)")

folders = data_dir / "train"
classes = int(labels.max()) + 1
if not (folders / f"class_{0:03d}").exists():
    counters = [0] * classes
    for label in range(classes):
        (folders / f"class_{label:03d}").mkdir(parents=True, exist_ok=True)
    for image, label in zip(images, labels):
        name = f"class_{label:03d}"
        index = counters[label]
        counters[label] += 1
        Image.fromarray(image).save(folders / name / f"{name}_{index}.bmp")
    print(f"wrote {sum(counters)} BMPs under {folders}")
else:
    print(f"BMP folders already present under {folders}")
