#!/usr/bin/env python3
"""TensorFlow CPU HIGGS dense benchmark, written the way a TensorFlow user
writes it: keras Sequential, a tf.function train step, XLA on because that is
TensorFlow's fast path (TF_PLAIN=1 turns it off, which the sweep runner's
--plain arm uses). The measurement protocol lives in run_higgs_cpu_sweep.py,
not here.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # np.loadtxt on the 500k-row split is minutes of wall clock, none of it
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


def batch_list(args: argparse.Namespace) -> list[int]:
    return [int(item) for item in args.batches.split(",") if item] or [args.batch]


def jit() -> bool:
    return not os.environ.get("TF_PLAIN")


def print_common(args: argparse.Namespace) -> None:
    print("engine=tensorflow")
    print(f"mode={args.mode}")
    print("device=cpu")
    print(f"hidden={args.hidden}")
    print(f"hidden_layers={args.hidden_layers}")
    print(f"activation={args.activation}")


def make_model(tf, features: int, args: argparse.Namespace):
    activation = "relu" if args.activation == "relu" else "tanh"
    layers = [tf.keras.layers.Input(shape=(features,))]
    for _ in range(args.hidden_layers):
        layers.append(tf.keras.layers.Dense(args.hidden, activation=activation))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    return tf.keras.Sequential(layers)


def run_train(tf, args: argparse.Namespace) -> None:
    x_np, y_np = load_csv(args.train)
    xt_np, yt_np = load_csv(args.test)
    x = tf.constant(np.array(x_np))
    y = tf.constant(np.array(y_np))
    xt = tf.constant(np.array(xt_np))

    print_common(args)
    print(f"samples={x_np.shape[0]}")
    print(f"epochs={args.epochs}")

    for batch in batch_list(args):
        tf.random.set_seed(42)
        model = make_model(tf, x_np.shape[1], args)
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        @tf.function(jit_compile=jit())
        def train_step(xb, yb):
            with tf.GradientTape() as tape:
                loss = loss_fn(yb, model(xb, training=True))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        def run_epoch() -> None:
            for start, end in batches(x_np.shape[0], batch):
                train_step(x[start:end], y[start:end])

        for _ in range(args.warmup_epochs):
            run_epoch()

        times: list[float] = []
        for _ in range(args.epochs):
            t0 = time.perf_counter()
            run_epoch()
            times.append(time.perf_counter() - t0)

        print(f"batch_{batch}_epoch_times=" + ",".join(f"{t:.6f}" for t in times))

        median_epoch_s = sorted(times)[len(times) // 2]

        accuracy = tf.keras.metrics.BinaryAccuracy()
        auc = tf.keras.metrics.AUC()
        for start, end in batches(xt_np.shape[0], 8192):
            predictions = model(xt[start:end], training=False)
            accuracy.update_state(yt_np[start:end], predictions)
            auc.update_state(yt_np[start:end], predictions)

        print(f"batch_{batch}_samples_per_sec={x_np.shape[0] / median_epoch_s:.0f}"
              f" median_epoch_s={median_epoch_s:.9g}")
        print(f"batch_{batch}_test_accuracy={float(accuracy.result()):.9g}"
              f" test_roc_auc={float(auc.result()):.9g}", flush=True)


    print("RESULT=OK")


def run_infer(tf, args: argparse.Namespace) -> None:
    x_np, _ = load_csv(args.test)
    x = tf.constant(np.array(x_np))
    tf.random.set_seed(42)
    model = make_model(tf, x_np.shape[1], args)

    @tf.function(jit_compile=jit())
    def infer_step(xb):
        return model(xb, training=False)

    print_common(args)
    print(f"reps={args.reps}")

    for batch in batch_list(args):
        processed = (x_np.shape[0] // batch) * batch

        def run_pass() -> None:
            for start, end in batches(x_np.shape[0], batch):
                infer_step(x[start:end])

        run_pass()                      # each batch size is its own XLA compile
        run_pass()
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            run_pass()
            times.append(time.perf_counter() - t0)

        print(f"batch_{batch}_pass_times=" + ",".join(f"{t:.6f}" for t in times))

        median_pass_s = sorted(times)[len(times) // 2]
        print(f"batch_{batch}_samples_per_sec={processed / median_pass_s:.0f}"
              f" median_pass_s={median_pass_s:.9g}", flush=True)


    print("RESULT=OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "infer"])
    parser.add_argument("--train", type=Path)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--batches", default="")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    if args.threads:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    import tensorflow as tf

    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    if args.mode == "train":
        run_train(tf, args)
    else:
        run_infer(tf, args)


if __name__ == "__main__":
    main()
