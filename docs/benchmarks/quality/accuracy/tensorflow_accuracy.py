#!/usr/bin/env python3
"""TensorFlow accuracy-parity benchmark on the HIGGS classification task.

Trains the canonical HIGGS dense classifier (28 -> 1024 -> 1024 -> 1, ReLU
hidden, sigmoid output, binary cross entropy, Adam, fixed epochs) on the shared
prepared split and prints the test-set quality so parity with OpenNN and
PyTorch can be checked at a fixed training budget.
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


def bench_data_dir() -> Path:
    root = os.environ.get("OPENNN_BENCH_DATA", str(Path.home() / "opennn-benchmark-data"))
    return Path(root) / "higgs"


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    x = np.ascontiguousarray(data[:, :-1])
    y = np.ascontiguousarray(data[:, -1:].astype(np.float32))
    return x, y


def batches(n: int, batch: int):
    stop = (n // batch) * batch
    for start in range(0, stop, batch):
        yield start, start + batch


def run(args: argparse.Namespace) -> None:
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

    layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=(x_np.shape[1],))]
    for _ in range(args.hidden_layers):
        layers.append(tf.keras.layers.Dense(args.hidden, activation="relu"))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    model = tf.keras.Sequential(layers)

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    @tf.function(jit_compile=False)
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            pred = model(xb, training=True)
            loss = loss_fn(yb, pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for _ in range(args.epochs):
        for start, end in batches(x_np.shape[0], args.batch):
            train_step(x[start:end], y[start:end])

    preds = []
    for start, end in batches(xt_np.shape[0], args.batch):
        preds.append(model(xt[start:end], training=False).numpy())
    pred_np = np.vstack(preds) if preds else np.empty((0, 1), dtype=np.float32)
    metrics = binary_metrics(yt_np[: pred_np.shape[0]], pred_np)

    print("engine=tensorflow")
    print("device=cpu")
    print(f"samples={x_np.shape[0]}")
    print(f"batch={args.batch}")
    print(f"epochs={args.epochs}")
    print(f"hidden={args.hidden}")
    print(f"hidden_layers={args.hidden_layers}")
    print("activation=relu")
    print(f"test_samples={pred_np.shape[0]}")
    for key, value in metrics.items():
        print(f"{key}={value:.9g}")
    print("RESULT=OK")


def parse_args() -> argparse.Namespace:
    data_dir = bench_data_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=data_dir / "higgs_train.csv")
    parser.add_argument("--test", type=Path, default=data_dir / "higgs_test.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        print("RESULT=ERROR")
        raise
