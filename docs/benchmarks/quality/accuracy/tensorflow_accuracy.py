#!/usr/bin/env python3
"""TensorFlow accuracy-parity benchmark on the HIGGS classification task,
written the way a TensorFlow user writes it: keras Sequential, a tf.function
train step, a plain forward over the test split. The protocol lives in
run_accuracy.py.

Trains the canonical HIGGS dense classifier (28 -> 1024 -> 1024 -> 1, ReLU
hidden, sigmoid output, binary cross entropy, Adam, fixed epochs) on the shared
prepared split and prints the test-set quality so parity with OpenNN and
PyTorch can be checked at a fixed training budget. The three engines are scored
by the same metrics.py, so no framework's own reduction can bias the
comparison.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from metrics import binary_metrics

HIGGS_DIR = Path(os.environ.get("OPENNN_BENCH_DATA",
                                str(Path.home() / "opennn-benchmark-data"))) / "higgs"


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # np.loadtxt on the 10.5M-row split is minutes of wall clock, none of it
    # measured, so the parsed array is cached next to the CSV.
    cache = path.with_suffix(path.suffix + ".npy")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        data = np.load(cache, mmap_mode="r")
    else:
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
        try:
            np.save(cache, data)
        except OSError:              # read-only data directory: parse next time
            pass
    x = np.ascontiguousarray(data[:, :-1])
    y = np.ascontiguousarray(data[:, -1:].astype(np.float32))
    return x, y


def batches(n: int, batch: int):
    stop = (n // batch) * batch
    for start in range(0, stop, batch):
        yield start, start + batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=HIGGS_DIR / "higgs_train.csv")
    parser.add_argument("--test", type=Path, default=HIGGS_DIR / "higgs_test.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=2)
    # Accepted because the runner passes it to every engine; this driver has
    # always left TensorFlow's thread pools at their defaults (quality is not
    # timed here).
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    import tensorflow as tf

    tf.random.set_seed(42)
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    x_np, y_np = load_csv(args.train)
    xt_np, yt_np = load_csv(args.test)
    x = tf.constant(x_np)
    y = tf.constant(y_np)
    xt = tf.constant(xt_np)

    layers = [tf.keras.layers.Input(shape=(x_np.shape[1],))]
    for _ in range(args.hidden_layers):
        layers.append(tf.keras.layers.Dense(args.hidden, activation="relu"))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    model = tf.keras.Sequential(layers)

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    @tf.function(jit_compile=False)
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            loss = loss_fn(yb, model(xb, training=True))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for _ in range(args.epochs):
        for start, end in batches(x_np.shape[0], args.batch):
            train_step(x[start:end], y[start:end])

    predictions = np.empty((xt_np.shape[0] // args.batch * args.batch, 1), dtype=np.float32)
    for start, end in batches(xt_np.shape[0], args.batch):
        predictions[start:end] = model(xt[start:end], training=False).numpy()
    metrics = binary_metrics(yt_np[: len(predictions)], predictions)

    print("engine=tensorflow")
    print("device=cpu")
    print(f"samples={x_np.shape[0]}")
    print(f"batch={args.batch}")
    print(f"epochs={args.epochs}")
    print(f"hidden={args.hidden}")
    print(f"hidden_layers={args.hidden_layers}")
    print("activation=relu")
    print(f"test_samples={len(predictions)}")
    for key, value in metrics.items():
        print(f"{key}={value:.9g}")
    print("RESULT=OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        print("RESULT=ERROR")
        raise
