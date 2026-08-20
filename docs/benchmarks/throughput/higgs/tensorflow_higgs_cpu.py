#!/usr/bin/env python3
"""TensorFlow CPU HIGGS dense benchmark counterpart."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from metrics import binary_metrics

def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # np.loadtxt on the full split is minutes of wall clock per engine per run,
    # none of it measured, so the parsed array is cached next to the CSV and
    # re-read with np.load. OpenNN's own driver reads the CSV directly and is
    # unaffected; nothing here is inside a timed region either way.
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

# XLA is TensorFlow's fast path on CPU as well as GPU, and the GPU family
# has always measured it with XLA on; this one was pinned to jit_compile=False,
# which measured TensorFlow below its own best. TF_PLAIN=1 restores that for
# an A/B.
def tensorflow_jit() -> bool:
    return not os.environ.get("TF_PLAIN")

def run_tensorflow(args: argparse.Namespace) -> None:

    if args.threads:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    import tensorflow as tf

    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)

    tf.random.set_seed(42)
    activation = "relu" if args.activation == "relu" else "tanh"
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    def make_model(features: int):
        layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=(features,))]
        for _ in range(args.hidden_layers):
            layers.append(tf.keras.layers.Dense(args.hidden, activation=activation))
        layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
        return tf.keras.Sequential(layers)

    if args.mode == "train":
        x_np, y_np = load_csv(args.train)
        xt_np, yt_np = load_csv(args.test)
        x = tf.constant(x_np)
        y = tf.constant(y_np)
        xt = tf.constant(xt_np)
        model = make_model(x_np.shape[1])
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        @tf.function(jit_compile=tensorflow_jit())
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

        for _ in range(args.warmup_epochs):
            run_epoch()

        times: list[float] = []
        for _ in range(args.epochs):
            t0 = time.perf_counter()
            run_epoch()
            times.append(time.perf_counter() - t0)
        times.sort()
        median_epoch_s = times[len(times) // 2]

        preds = []
        for start, end in batches(xt_np.shape[0], args.batch):
            preds.append(model(xt[start:end], training=False).numpy())
        pred_np = np.vstack(preds) if preds else np.empty((0, 1), dtype=np.float32)
        metrics = binary_metrics(yt_np[: pred_np.shape[0]], pred_np)

        print_common("tensorflow", args, x_np.shape[0])
        print(f"median_epoch_s={median_epoch_s:.9g}")
        print(f"samples_per_sec={x_np.shape[0] / median_epoch_s:.0f}")
        print(f"test_samples={pred_np.shape[0]}")
        for key, value in metrics.items():
            print(f"{key}={value:.9g}")
        print("RESULT=OK")
        return

    x_np, _ = load_csv(args.test)
    x = tf.constant(x_np)
    model = make_model(x_np.shape[1])

    @tf.function(jit_compile=tensorflow_jit())
    def infer_step(xb):
        return model(xb, training=False)

    def measure(batch: int) -> tuple[int, float]:
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
        times.sort()
        return (x_np.shape[0] // batch) * batch, times[len(times) // 2]

    batch_list = [int(item) for item in args.batches.split(",") if item] or [args.batch]

    if len(batch_list) == 1:
        processed, median_pass_s = measure(batch_list[0])
        print_common("tensorflow", args, processed)
        print(f"median_pass_s={median_pass_s:.9g}")
        print(f"samples_per_sec={processed / median_pass_s:.0f}")
        print("RESULT=OK")
        return

    print_common("tensorflow", args, x_np.shape[0])
    for batch in batch_list:
        processed, median_pass_s = measure(batch)
        print(f"batch_{batch}_samples_per_sec={processed / median_pass_s:.0f}"
              f" median_pass_s={median_pass_s:.9g}", flush=True)
    print("RESULT=OK")

def print_common(engine: str, args: argparse.Namespace, samples: int) -> None:
    print(f"engine={engine}")
    print(f"mode={args.mode}")
    print("device=cpu")
    print(f"samples={samples}")
    print(f"batch={args.batch}")
    print(f"hidden={args.hidden}")
    print(f"hidden_layers={args.hidden_layers}")
    print(f"activation={args.activation}")
    if args.mode == "train":
        print(f"epochs={args.epochs}")
    else:
        print(f"reps={args.reps}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "infer"])
    parser.add_argument("--train", type=Path)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1024)
    # Inference only: a comma-separated list is measured in one process, so the
    # whole batch-size row of a comparison shares one model, one load and one
    # thermal window on a laptop that drifts ten per cent over a sweep.
    parser.add_argument("--batches", default="")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "train" and args.train is None:
        parser.error("--train is required in train mode")
    return args

def main() -> None:
    args = parse_args()
    run_tensorflow(args)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        print("RESULT=ERROR")
        raise
