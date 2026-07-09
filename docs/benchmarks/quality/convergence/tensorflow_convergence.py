#!/usr/bin/env python3
"""TensorFlow convergence-gate benchmark on the HIGGS classification dataset.

MLPerf-style metric: WALL-CLOCK TIME TO REACH A FIXED QUALITY TARGET, not
throughput at a fixed epoch count. Trains the canonical HIGGS dense classifier
(28 -> 1024 -> 1024 -> 1, ReLU, sigmoid, BCE, Adam) and, after each epoch,
evaluates the HELD-OUT (test) log-loss. When it reaches the target the clock
stops and we report the wall-clock time, epochs taken, and final held-out
metric. Same data / arch / optimizer / target as opennn_convergence and
pytorch_convergence. Per-epoch evaluation is excluded from the clock.

  usage: python tensorflow_convergence.py --train TRAIN.csv --test TEST.csv
                                          [--target 0.60] [--max-epochs 50]
                                          [--batch 1024] [--hidden 1024]
                                          [--hidden-layers 2] [--threads 0]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np


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
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)
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

    def run_epoch() -> None:
        for start, end in batches(x_np.shape[0], args.batch):
            train_step(x[start:end], y[start:end])

    # The convergence gate is the HELD-OUT test log-loss, not the training loss.
    # Evaluate the test set after each epoch and stop when it reaches the target.
    # Evaluation time is excluded from the clock.
    def eval_log_loss() -> float:
        preds = []
        for start, end in batches(xt_np.shape[0], args.batch):
            preds.append(model(xt[start:end], training=False).numpy())
        if not preds:
            return float("nan")
        p = np.clip(np.vstack(preds).reshape(-1), 1.0e-7, 1.0 - 1.0e-7)
        yb = yt_np[: p.shape[0]].reshape(-1)
        return float(-(yb * np.log(p) + (1.0 - yb) * np.log(1.0 - p)).mean())

    reached = False
    epochs_taken = args.max_epochs
    test_log_loss = float("nan")
    train_s = 0.0
    for epoch in range(1, args.max_epochs + 1):
        t0 = time.perf_counter()
        run_epoch()
        train_s += time.perf_counter() - t0
        test_log_loss = eval_log_loss()
        if test_log_loss <= args.target:
            reached = True
            epochs_taken = epoch
            break

    print("engine=tensorflow")
    print("device=cpu")
    print("dataset=HIGGS")
    print(f"train_samples={x_np.shape[0]}")
    print(f"batch={args.batch}")
    print(f"hidden={args.hidden}")
    print(f"hidden_layers={args.hidden_layers}")
    print(f"target_log_loss={args.target}")
    print(f"reached_goal={1 if reached else 0}")
    print(f"epochs_to_target={epochs_taken}")
    print(f"test_log_loss={test_log_loss:.10f}")
    print(f"time_to_target_s={train_s:.6f}")
    print(f"RESULT={'OK' if reached else 'DID_NOT_CONVERGE'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--target", type=float, default=0.60)
    parser.add_argument("--max-epochs", type=int, default=50)
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
