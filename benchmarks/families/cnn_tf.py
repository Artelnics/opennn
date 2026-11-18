#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

"""The CNN family in TensorFlow, the counterpart of cnn.cpp and cnn.py.

  cnn_tf.py train    <train_dir> <test_dir> [epochs] [batch,...] [size] [dev] [prec]
  cnn_tf.py infer    <test_dir>             [reps]   [batch,...] [size] [dev] [prec]
  cnn_tf.py capacity <train_dir>            [batch]              [size] [dev] [prec]

ResNet-50 v1.5, built here rather than taken from `keras.applications`:
that model is v1 -- its stride sits in the first 1x1 convolution of each
downsampling block, not the 3x3 -- and gives its convolutions biases, for
25,636,712 parameters against the 25,557,032 of torchvision's v1.5 and
OpenNN's ResNet. Bottleneck blocks [3,4,6,3] over widths [64,128,256,512],
convolutions without bias, batch normalization with torchvision's epsilon and
momentum, explicit symmetric padding where a stride-2 "same" would pad
asymmetrically.

Images come from `keras.utils.image_dataset_from_directory`, Keras's own
folder loader: decoded every epoch by tf.data's parallel map, resized with
centre crop and scaled to [0, 1], shuffled, whole batches only -- the
counterpart of cnn.py's ImageFolder DataLoader. Channels-last is TensorFlow's
default layout.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import tensorflow_engine as engine  # noqa: E402

CLASSES = 1000

def build(size: int, blocks: tuple[int, ...] = (3, 4, 6, 3), classes: int = CLASSES):
    """The CNN family. Nothing else here constructs the network. quality_tf.py
    reuses it with the quality profile's blocks and class count."""
    keras = engine.keras
    layers = keras.layers
    engine.seed()

    def conv_bn(x, filters: int, kernel: int, stride: int, relu: bool = True):
        if stride > 1 and kernel > 1:
            x = layers.ZeroPadding2D(kernel // 2)(x)
            padding = "valid"
        else:
            padding = "same" if kernel > 1 else "valid"
        x = layers.Conv2D(filters, kernel, strides=stride, padding=padding, use_bias=False)(x)
        x = layers.BatchNormalization(epsilon=1.0e-5, momentum=0.9)(x)
        return layers.ReLU()(x) if relu else x

    def bottleneck(x, width: int, stride: int, downsample: bool):
        shortcut = conv_bn(x, 4 * width, 1, stride, relu=False) if downsample else x
        y = conv_bn(x, width, 1, 1)
        y = conv_bn(y, width, 3, stride)
        y = conv_bn(y, 4 * width, 1, 1, relu=False)
        return layers.ReLU()(layers.Add()([y, shortcut]))

    inputs = keras.Input((size, size, 3))
    x = conv_bn(inputs, 64, 7, 2)
    x = layers.ZeroPadding2D(1)(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    for stage, (count, width) in enumerate(zip(blocks, (64, 128, 256, 512))):
        for block in range(count):
            stride = 2 if stage > 0 and block == 0 else 1
            x = bottleneck(x, width, stride, downsample=block == 0)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(classes)(x)
    return keras.Model(inputs, outputs)

def compile_for_training(model) -> None:
    keras = engine.keras
    # torch.optim.Adam's defaults, as cnn.py uses them.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1.0e-3, epsilon=1.0e-8),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

def class_names(path: str) -> list[str]:
    """Sorted class folders, skipping dot-directories such as OpenNN's `.cache/`."""
    return sorted(entry.name for entry in os.scandir(path)
                  if entry.is_dir() and not entry.name.startswith("."))

def dataset_for(path: str, batch: int, size: int, shuffle: bool):
    """Keras's folder loader, decoded per epoch, whole batches. Returns (dataset, samples)."""
    keras, tf = engine.keras, engine.tf
    names = class_names(path)

    images = keras.utils.image_dataset_from_directory(
        path, labels="inferred", label_mode="int", class_names=names,
        batch_size=None, image_size=(size, size), shuffle=shuffle, seed=engine.SEED,
        interpolation="bilinear", crop_to_aspect_ratio=True, verbose=False)
    count = int(images.cardinality().numpy())

    images = images.map(lambda image, label: (image / 255.0, label),
                        num_parallel_calls=tf.data.AUTOTUNE)
    images = images.batch(batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return images, (count // batch) * batch

def parse_opts(argv: list[str], first: int) -> dict:
    def at(index: int, default: str) -> str:
        return argv[index] if len(argv) > index else default

    size = int(at(first, "224"))
    device = at(first + 1, "cuda")
    precision = at(first + 2, "fp32")
    engine.start(device, precision)
    return {"size": size, "device": device, "precision": precision}

def header(mode: str, opts: dict) -> None:
    engine.report_baselines()
    print(f"engine=tensorflow\nmode={mode}\ndevice={opts['device']}")
    engine.report_blas()

def train_like(argv: list[str], mode: str) -> int:
    epochs = int(argv[4]) if len(argv) > 4 else 1
    batches = engine.batches_of(argv[5] if len(argv) > 5 else "128")
    opts = parse_opts(argv, 6)
    policy = engine.set_policy(opts["precision"], "train")

    header(mode, opts)
    print(f"dtype_policy={policy}")
    engine.report_opened(argv[2])
    warmup = 1 if mode == "train" else 0

    for batch in batches:
        images, samples = dataset_for(argv[2], batch, opts["size"], shuffle=True)
        print(f"samples={samples}")

        model = build(opts["size"])
        if batch == batches[0]:
            print(f"parameters={engine.trainable_parameters(model)}", flush=True)
        compile_for_training(model)

        timer = engine.epoch_timer(warmup)
        with engine.tf.device(engine.device_name(opts["device"])):
            model.fit(images, epochs=warmup + epochs, verbose=0, callbacks=[timer])

        times = sorted(timer.times)
        median = times[len(times) // 2]

        print(f"batch_{batch}_samples_per_sec={int(samples / median)}"
              f" median_epoch_s={median:.6g}")
        print(f"batch_{batch}_epoch_times={','.join(f'{t:.6g}' for t in times)}")
        print(f"batch_{batch}_mode={engine.execution(model)}", flush=True)

    print("RESULT=OK")
    return 0

def infer(argv: list[str]) -> int:
    reps = int(argv[3]) if len(argv) > 3 else 1
    batches = engine.batches_of(argv[4] if len(argv) > 4 else "128")
    opts = parse_opts(argv, 5)
    policy = engine.set_policy(opts["precision"], "infer")
    tf = engine.tf

    header("infer", opts)
    print(f"dtype_policy={policy}")
    engine.report_opened(argv[2])

    for batch in batches:
        images, samples = dataset_for(argv[2], batch, opts["size"], shuffle=False)
        print(f"samples={samples}")

        model = build(opts["size"])
        model.compile()
        if batch == batches[0]:
            print(f"parameters={engine.trainable_parameters(model)}", flush=True)
        forward = engine.predict_step(model)

        # One batch, filled once and replayed, matching cnn.cpp and cnn.py:
        # the resident forward pass, not the image decode.
        first = next(iter(images))[0]
        with tf.device(engine.device_name(opts["device"])):
            batch_images = tf.identity(tf.cast(first, engine.compute_dtype()))

        def run_pass():
            with tf.device(engine.device_name(opts["device"])):
                for _ in range(samples // batch):
                    forward(batch_images)
            engine.sync()

        run_pass()

        print(f"TIMED_START_UNIX={time.time():.3f}", flush=True)
        times = []
        for _ in range(reps):
            mark = time.perf_counter()
            run_pass()
            times.append(time.perf_counter() - mark)
        print(f"TIMED_END_UNIX={time.time():.3f}", flush=True)

        print(f"batch_{batch}_pass_times={','.join(f'{t:.6g}' for t in times)}")
        times.sort()
        median = times[len(times) // 2]
        print(f"batch_{batch}_samples_per_sec={int(samples / median)}"
              f" median_pass_s={median:.6g}", flush=True)
        print(f"batch_{batch}_mode={engine.execution(model)}", flush=True)

    print("RESULT=OK")
    return 0

def capacity(argv: list[str]) -> int:
    batch = int(argv[3]) if len(argv) > 3 else 128
    opts = parse_opts(argv, 4)
    engine.set_policy(opts["precision"], "train")

    header("capacity", opts)
    print(f"batch={batch}")

    try:
        images, _ = dataset_for(argv[2], batch, opts["size"], shuffle=False)
        first, labels = next(iter(images))
        model = build(opts["size"])
        print(f"parameters={engine.trainable_parameters(model)}", flush=True)
        compile_for_training(model)
        with engine.tf.device(engine.device_name(opts["device"])):
            model.train_on_batch(first, labels)
        engine.sync()
    except Exception as error:      # noqa: BLE001 -- only allocation failures are OOM
        if not engine.is_oom(error):
            raise
        return engine.oom_exit(error)

    print("fits=1\nRESULT=OK", flush=True)
    return 0

def main() -> int:
    argv = sys.argv
    mode = argv[1] if len(argv) > 1 else ""

    if mode in ("train", "quality") and len(argv) > 3:
        return train_like(argv, mode)
    if mode == "infer" and len(argv) > 2:
        return infer(argv)
    if mode == "capacity" and len(argv) > 2:
        return capacity(argv)

    return engine.usage(__doc__)

if __name__ == "__main__":
    raise SystemExit(main())
