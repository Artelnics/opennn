# Build an ImageNet-like proxy training-speed dataset from CIFAR content:
# 50,000 images upsampled to 224x224x3, laid out across 1000 class folders so
# ResNet-50 gets a true 1000-way classifier head and a 1000-way softmax. The
# image *content* is irrelevant to a speed benchmark -- only the 224x224x3 /
# 1000-class shape and the per-batch disk-decode cost matter, and this avoids
# ImageNet's 150 GB download and license wall.
#
# Both engines read the same BMP class folders from disk and lazy-load per
# batch (OpenNN's ImageDataset BinaryFile cache; PyTorch's DataLoader): at this
# size 50k*224*224*3 fp32 = 30 GB cannot be GPU-resident, so this measures
# conv-FLOP throughput PLUS input-pipeline efficiency, not launch overhead.
#
# usage: python prepare_imagenet_like.py [data_dir] [classes] [size]

import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "imagenet_like")
classes = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
size = int(sys.argv[3]) if len(sys.argv) > 3 else 224
data_dir.mkdir(parents=True, exist_ok=True)

# Source pixels: reuse CIFAR-10 (any content works; we relabel across `classes`).
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
images = batch[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC, 50000
n = images.shape[0]

# Spread the 50,000 images evenly across `classes` folders so every class is
# non-empty (ImageDataset requires every folder populated): label = i % classes.
labels = np.arange(n, dtype=np.int64) % classes
print(f"{n} images -> {classes} classes ({n // classes} per class), {size}x{size}")

folders = data_dir / "train"
sentinel = folders / f"class_{classes - 1:04d}"
if not sentinel.exists():
    for c in range(classes):
        (folders / f"class_{c:04d}").mkdir(parents=True, exist_ok=True)
    counters = [0] * classes
    for image, label in zip(images, labels):
        name = f"class_{label:04d}"
        index = counters[label]
        counters[label] += 1
        big = Image.fromarray(image).resize((size, size), Image.BILINEAR)
        big.save(folders / name / f"{name}_{index}.bmp")
        if (sum(counters)) % 5000 == 0:
            print(f"  wrote {sum(counters)}/{n}")
    print(f"wrote {sum(counters)} BMPs under {folders}")
else:
    print(f"BMP folders already present under {folders}")
