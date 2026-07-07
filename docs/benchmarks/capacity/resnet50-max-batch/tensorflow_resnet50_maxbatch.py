#!/usr/bin/env python3
"""TensorFlow ResNet-50/CIFAR-10 max training batch trial."""

import argparse
import gc
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


def configure_gpu(memory_limit_mb):
    gpus = tf.config.list_physical_devices("GPU")
    assert gpus, "CUDA GPU required"
    if memory_limit_mb:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)],
        )
    else:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    return gpus[0].name


K = tf.keras.layers


def bottleneck(x, mid, stride):
    out = mid * 4
    shortcut = x
    if stride != 1 or x.shape[-1] != out:
        shortcut = K.Conv2D(out, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = K.BatchNormalization()(shortcut)
    y = K.Conv2D(mid, 1, use_bias=False)(x)
    y = K.ReLU()(K.BatchNormalization()(y))
    y = K.Conv2D(mid, 3, strides=stride, padding="same", use_bias=False)(y)
    y = K.ReLU()(K.BatchNormalization()(y))
    y = K.Conv2D(out, 1, use_bias=False)(y)
    y = K.BatchNormalization()(y)
    return K.ReLU()(y + shortcut)


def build_resnet50(classes):
    inp = K.Input(shape=(32, 32, 3))
    x = K.Conv2D(64, 7, strides=2, padding="same", use_bias=False)(inp)
    x = K.ReLU()(K.BatchNormalization()(x))
    x = K.MaxPool2D(3, strides=2, padding="same")(x)
    for stage, (mid, blocks) in enumerate(zip([64, 128, 256, 512], [3, 4, 6, 3])):
        for block in range(blocks):
            stride = 2 if (block == 0 and stage > 0) else 1
            x = bottleneck(x, mid, stride)
    x = K.GlobalAveragePooling2D()(x)
    out = K.Dense(classes, dtype="float32")(x)
    return tf.keras.Model(inp, out)


def make_batch(data_dir, batch):
    images = np.load(os.path.join(data_dir, "cifar_images.npy"), mmap_mode="r")
    labels = np.load(os.path.join(data_dir, "cifar_labels.npy"), mmap_mode="r")
    idx = np.arange(batch, dtype=np.int64) % images.shape[0]
    xb = np.asarray(images[idx], dtype=np.float32) / 255.0
    yb = np.asarray(labels[idx], dtype=np.int64)
    return xb, yb, int(labels.max()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../throughput/resnet50-training-speed/cifar10")
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--memory-limit-mb", type=int, default=0)
    args = ap.parse_args()

    device_name = configure_gpu(args.memory_limit_mb or None)
    tf.random.set_seed(42)

    xb_np, yb_np, classes = make_batch(args.data, args.batch)
    with tf.device("/GPU:0"):
        xb = tf.constant(xb_np)
        yb = tf.constant(yb_np)
    del xb_np, yb_np
    gc.collect()

    with tf.device("/GPU:0"):
        model = build_resnet50(classes)
        optimizer = tf.keras.optimizers.Adam(1e-3)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        @tf.function(jit_compile=True)
        def train_step(x, y):
            with tf.GradientTape() as tape:
                loss = loss_fn(y, model(x, training=True))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        loss0 = train_step(xb, yb)
        loss1 = train_step(xb, yb)
        loss0_value = float(loss0.numpy())
        loss1_value = float(loss1.numpy())

    if not np.isfinite(loss1_value):
        raise RuntimeError("loss is not finite")

    print("engine=tensorflow_xla")
    print("path=xla")
    print(f"device={device_name}")
    print(f"samples={args.batch} batch={args.batch} precision=fp32 classes={classes}")
    print(f"parameters={model.count_params()}")
    print(f"loss_warmup={loss0_value:.6g}")
    print(f"loss_final={loss1_value:.6g}")
    print("RESULT=OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL : {exc}", file=sys.stderr)
        print("RESULT=ERROR")
        raise SystemExit(1)
