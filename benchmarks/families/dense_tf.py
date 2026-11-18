#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

"""TensorFlow dense benchmark matching dense.cpp and dense.py. Modes: train, infer,
capacity and quality, with the same positional arguments and `key=value` output.

  dense_tf.py train    <train_csv> <test_csv> [epochs] [batch,...] [hidden] [layers] [act] [dev] [prec]
  dense_tf.py infer    <test_csv>             [reps]   [batch,...] [hidden] [layers] [act] [dev] [prec]
  dense_tf.py capacity <train_csv>            [batch]              [hidden] [layers] [act] [dev] [prec]

Keras defaults throughout (see tools/tensorflow_engine.py): `compile()` and
`fit()` for training, Keras's compiled predict step for inference. The work
matches the other engines: the same network, Adam with the PyTorch driver's
hyperparameters, data in file order, whole batches only, the same warm-up.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import tensorflow_engine as engine  # noqa: E402
from common import binary_metrics  # noqa: E402

def load_csv(path: str, role: str = "input") -> tuple[np.ndarray, np.ndarray]:
    """Parse float32 inputs and targets from the same CSV supplied to OpenNN."""
    import pandas as pd

    engine.report_opened(path, role)
    frame = pd.read_csv(path, header=None, dtype=np.float32)

    data = np.ascontiguousarray(frame.to_numpy(dtype=np.float32))
    return data[:, :-1], data[:, -1:]

def build(features: int, opts: dict):
    """The dense family. Nothing else here constructs the network."""
    keras = engine.keras
    engine.seed()

    activation = "relu" if opts["activation"] == "relu" else "tanh"
    layers = [keras.Input((features,))]
    layers += [keras.layers.Dense(opts["hidden"], activation=activation)
               for _ in range(opts["layers"])]
    layers.append(keras.layers.Dense(1))
    return keras.Sequential(layers)

def compile_for_training(model) -> None:
    keras = engine.keras
    # torch.optim.Adam's defaults, which the PyTorch driver uses: the Keras
    # default epsilon is 1e-7. A hyperparameter of the training algorithm, not
    # an engine setting, so it is matched; it does not change the arithmetic cost.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1.0e-3, epsilon=1.0e-8),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True))

def parse_opts(argv: list[str], first: int) -> dict:
    """Trailing options, positional and shared by every mode -- the same five
    in the same order as dense.cpp."""
    def at(index: int, default: str) -> str:
        return argv[index] if len(argv) > index else default

    device = at(first + 3, "cuda")
    precision = at(first + 4, "fp32")
    engine.start(device, precision)

    return {
        "hidden": int(at(first, "1024")),
        "layers": int(at(first + 1, "2")),
        "activation": at(first + 2, "relu").lower(),
        "device": device,
        "precision": precision,
    }

def header(mode: str, opts: dict) -> None:
    engine.report_baselines()
    print(f"engine=tensorflow\nmode={mode}\ndevice={opts['device']}")
    engine.report_blas()

def train_like(argv: list[str], mode: str) -> int:
    """train and quality: the same loop, differing only in warmup."""
    epochs = int(argv[4]) if len(argv) > 4 else 1
    batches = engine.batches_of(argv[5] if len(argv) > 5 else "1024")
    opts = parse_opts(argv, 6)
    policy = engine.set_policy(opts["precision"], "train")

    header(mode, opts)
    x, y = load_csv(argv[2], "train")
    xt, yt = load_csv(argv[3], "test")
    print(f"dtype_policy={policy}")
    warmup = 2 if mode == "train" else 0

    for batch in batches:
        model = build(x.shape[1], opts)
        if batch == batches[0]:
            print(f"parameters={engine.trainable_parameters(model)}", flush=True)
        compile_for_training(model)

        # Whole batches only, the rule dense.cpp and dense.py apply.
        whole = (x.shape[0] // batch) * batch
        timer = engine.epoch_timer(warmup)
        with engine.tf.device(engine.device_name(opts["device"])):
            model.fit(x[:whole], y[:whole], batch_size=batch, epochs=warmup + epochs,
                      shuffle=False, verbose=0, callbacks=[timer])

        times = timer.times
        print(f"batch_{batch}_epoch_order={','.join(f'{t:.6g}' for t in times)}")
        times = sorted(times)
        median = times[len(times) // 2]

        metrics = evaluate(model, xt, yt, batch, opts)

        print(f"batch_{batch}_samples_per_sec={int(whole / median)}"
              f" median_epoch_s={median:.6g}")
        print(f"batch_{batch}_epoch_times={','.join(f'{t:.6g}' for t in times)}")
        print(f"batch_{batch}_mode={engine.execution(model)}")
        print(f"batch_{batch}_test_accuracy={metrics['test_accuracy']:.6g}"
              f" test_roc_auc={metrics['test_roc_auc']:.6g}", flush=True)

    print("RESULT=OK")
    return 0

def evaluate(model, xt: np.ndarray, yt: np.ndarray, batch: int, opts: dict) -> dict:
    processed = (xt.shape[0] // batch) * batch
    with engine.tf.device(engine.device_name(opts["device"])):
        logits = model.predict(xt[:processed], batch_size=batch, verbose=0)
    logits = np.asarray(logits, dtype=np.float32)
    return binary_metrics(yt[:processed], 1.0 / (1.0 + np.exp(-logits)))

def infer(argv: list[str]) -> int:
    reps = int(argv[3]) if len(argv) > 3 else 1
    batches = engine.batches_of(argv[4] if len(argv) > 4 else "1024")
    opts = parse_opts(argv, 5)
    policy = engine.set_policy(opts["precision"], "infer")
    tf = engine.tf

    header("infer", opts)
    x_np, _ = load_csv(argv[2], "test")
    print(f"dtype_policy={policy}")

    with tf.device(engine.device_name(opts["device"])):
        x = tf.constant(x_np, dtype=engine.compute_dtype())
    samples = x_np.shape[0]

    for batch in batches:
        model = build(x_np.shape[1], opts)
        model.compile()
        if batch == batches[0]:
            print(f"parameters={engine.trainable_parameters(model)}", flush=True)
        forward = engine.predict_step(model)
        processed = (samples // batch) * batch

        def run_pass():
            with tf.device(engine.device_name(opts["device"])):
                for start in range(0, processed, batch):
                    forward(x[start:start + batch])
            engine.sync()

        run_pass()
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
        print(f"batch_{batch}_samples_per_sec={int(processed / median)}"
              f" median_pass_s={median:.6g}", flush=True)
        print(f"batch_{batch}_mode={engine.execution(model)}", flush=True)

    print("RESULT=OK")
    return 0

def capacity(argv: list[str]) -> int:
    """One attempt, then exit -- the contract the other drivers honour."""
    batch = int(argv[3]) if len(argv) > 3 else 1024
    opts = parse_opts(argv, 4)
    engine.set_policy(opts["precision"], "train")

    header("capacity", opts)
    x, y = load_csv(argv[2])
    print(f"batch={batch}")

    try:
        model = build(x.shape[1], opts)
        compile_for_training(model)
        with engine.tf.device(engine.device_name(opts["device"])):
            model.train_on_batch(x[:batch], y[:batch])
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
