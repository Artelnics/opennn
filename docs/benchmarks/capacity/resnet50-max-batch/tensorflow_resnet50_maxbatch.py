#!/usr/bin/env python3
"""TensorFlow ResNet-50/CIFAR-10 max training batch trial."""

import argparse
import gc
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


def configure_gpu(memory_limit_mb):
    gpus = tf.config.list_physical_devices("GPU")
    assert gpus, "CUDA GPU required"




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
    classes = int(labels.max()) + 1
    class_indices = [np.flatnonzero(labels == label) for label in range(classes)]
    positions = np.arange(batch, dtype=np.int64)
    idx = np.fromiter(
        (class_indices[int(position % classes)]
         [int(position // classes)
          % class_indices[int(position % classes)].size]
         for position in positions),
        dtype=np.int64,
        count=batch,
    )
    xb = np.asarray(images[idx], dtype=np.float32) / 255.0
    yb = np.asarray(labels[idx], dtype=np.int64)
    return xb, yb, classes


def default_data():
    root = os.environ.get("OPENNN_BENCH_DATA",
                          os.path.expanduser("~/opennn-benchmark-data"))
    return os.path.join(root, "cifar10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=default_data())
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--memory-limit-mb", type=int, default=0)
    ap.add_argument("--target", type=float, default=None,
                    help="optional training-loss target")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device_name = configure_gpu(args.memory_limit_mb or None)
    if args.precision == "bf16":
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    tf.keras.utils.set_random_seed(args.seed)

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
        use_xla = os.environ.get("TF_XLA", "1") != "0"

        @tf.function(jit_compile=use_xla)
        def train_step(x, y):
            with tf.GradientTape() as tape:
                loss = loss_fn(y, model(x, training=True))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        if args.target is None:
            loss_history = [float(train_step(xb, yb).numpy()),
                            float(train_step(xb, yb).numpy())]
        else:
            print(f"TRAIN_START_UNIX={time.time():.3f}", flush=True)
            started = time.perf_counter()
            loss_history = []
            for _ in range(args.max_steps):
                value = float(train_step(xb, yb).numpy())
                loss_history.append(value)
                if value <= args.target:
                    break
            wall_s = time.perf_counter() - started
            print(f"TRAIN_END_UNIX={time.time():.3f}", flush=True)

    if not np.isfinite(loss_history[-1]):
        raise RuntimeError("loss is not finite")

    print(f"engine=tensorflow_{'xla' if use_xla else 'graph'}")
    print(f"path={'xla' if use_xla else 'graph'}")
    print(f"device={device_name}")
    print(f"samples={args.batch} batch={args.batch} precision={args.precision} classes={classes}")
    print(f"parameters={model.count_params()}")
    if args.target is None:
        print(f"loss_warmup={loss_history[0]:.6g}")
        print(f"loss_final={loss_history[-1]:.6g}")
    else:
        reached = loss_history[-1] <= args.target
        print(f"target={args.target}")
        print(f"steps_run={len(loss_history)}")
        print(f"epochs_run={len(loss_history)}")
        print(f"final_error={loss_history[-1]:.9g}")
        print(f"reached_goal={1 if reached else 0}")
        print("loss_history=" + ",".join(f"{v:.9g}" for v in loss_history))
        print(f"wall_s={wall_s:.9g}")
        print(f"samples_per_sec={args.batch * len(loss_history) / wall_s:.9g}")
    print("RESULT=OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL : {exc}", file=sys.stderr)
        print("RESULT=ERROR")
        raise SystemExit(1)
