# Decode the repository's MNIST BMP folders (examples/mnist/data: 10,000
# 28x28 grayscale images in 10 class folders) into mnist_images.npy
# (N, 28, 28, 1) float32 raw 0-255 and mnist_labels.npy (N,) int64, so the
# PyTorch and TensorFlow benchmarks consume exactly the same samples that
# OpenNN's ImageDataset reads from the folders.
#
# usage: python prepare_mnist.py <path_to_examples/mnist/data>

import sys
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1] if len(sys.argv) > 1 else "../../../../examples/mnist/data")

classes = sorted(d.name for d in root.iterdir()
                 if d.is_dir() and not d.name.startswith(".") and any(d.glob("*.bmp")))
print(f"classes: {classes}")

images, labels = [], []
for label, name in enumerate(classes):
    for bmp in sorted((root / name).glob("*.bmp")):
        with Image.open(bmp) as im:
            images.append(np.asarray(im.convert("L"), dtype=np.float32))
        labels.append(label)

x = np.stack(images)[..., None]
y = np.array(labels, dtype=np.int64)

np.save("mnist_images.npy", x)
np.save("mnist_labels.npy", y)
print(f"wrote mnist_images.npy {x.shape} and mnist_labels.npy {y.shape}")
