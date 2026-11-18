#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

"""The LSTM family in TensorFlow, the counterpart of lstm.cpp and lstm.py.

LSTM forecasting on UCI Beijing PM2.5, predicting the next hourly reading
from a window of past ones.

  lstm_tf.py train    <csv> <csv> [epochs] [batch,...] [hidden] [past] [dev] [prec]
  lstm_tf.py infer    <csv>       [reps]   [batch,...] [hidden] [past] [dev] [prec]
  lstm_tf.py capacity <csv>       [batch]              [hidden] [past] [dev] [prec]

`keras.layers.LSTM` with its defaults reaches cuDNN's fused RNN on a GPU, as
nn.LSTM and OpenNN's LSTMOperator do, and Keras then compiles the model without
XLA because cuDNN ops are not XLA-compatible -- its own default, recorded as
`mode=graph`. Keras's LSTM has a single bias, like OpenNN's, so the
parameter count matches without the frozen second bias the PyTorch driver
needs.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import tensorflow_engine as engine  # noqa: E402

def build(features: int, opts: dict):
    """The LSTM family. Nothing else here constructs the network."""
    keras = engine.keras
    engine.seed()
    return keras.Sequential([
        keras.Input((opts["past"], features)),
        keras.layers.LSTM(opts["hidden"]),
        keras.layers.Dense(1),
    ])

def compile_for_training(model) -> None:
    keras = engine.keras
    # torch.optim.Adam's defaults, as the PyTorch driver uses them.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1.0e-3, epsilon=1.0e-8),
                  loss=keras.losses.MeanSquaredError())

def load_series(path: str, past: int, *,
                standardize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Windows of `past` hourly rows, predicting the next target -- lstm.py's loader."""
    engine.report_opened(path)

    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))

    values = np.asarray(rows[1:], dtype=np.float32)
    features, target = values, values[:, -1:]

    if standardize:
        mean, std = features.mean(axis=0), features.std(axis=0)
        features = (features - mean) / np.where(std > 1.0e-12, std, 1.0)

    count = len(values) - past
    windows = np.lib.stride_tricks.sliding_window_view(features, past, axis=0)
    windows = np.ascontiguousarray(windows[:count].transpose(0, 2, 1))

    return windows, np.ascontiguousarray(target[past:past + count])

def parse_opts(argv: list[str], first: int) -> dict:
    def at(index: int, default: str) -> str:
        return argv[index] if len(argv) > index else default

    device = at(first + 2, "cuda")
    precision = at(first + 3, "fp32")
    engine.start(device, precision)

    return {"hidden": int(at(first, "128")), "past": int(at(first + 1, "24")),
            "device": device, "precision": precision}

def header(mode: str, opts: dict) -> None:
    engine.report_baselines()
    print(f"engine=tensorflow\nmode={mode}\ndevice={opts['device']}")
    engine.report_blas()

def describe(samples: int, features: int, model, opts: dict) -> None:
    print(f"samples={samples} inputs={features} past={opts['past']} "
          f"hidden={opts['hidden']} "
          f"parameters={engine.trainable_parameters(model)}", flush=True)

def train_like(argv: list[str], mode: str) -> int:
    epochs = int(argv[4]) if len(argv) > 4 else 1
    batches = engine.batches_of(argv[5] if len(argv) > 5 else "256")
    opts = parse_opts(argv, 6)
    policy = engine.set_policy(opts["precision"], "train")

    header(mode, opts)
    print(f"dtype_policy={policy}")
    x, y = load_series(argv[2], opts["past"])
    warmup = 2 if mode == "train" else 0

    for batch in batches:
        model = build(x.shape[2], opts)
        if batch == batches[0]:
            describe(x.shape[0], x.shape[2], model, opts)
        compile_for_training(model)

        whole = (x.shape[0] // batch) * batch
        timer = engine.epoch_timer(warmup)
        with engine.tf.device(engine.device_name(opts["device"])):
            model.fit(x[:whole], y[:whole], batch_size=batch, epochs=warmup + epochs,
                      shuffle=False, verbose=0, callbacks=[timer])

        times = sorted(timer.times)
        median = times[len(times) // 2]

        print(f"batch_{batch}_samples_per_sec={int(whole / median)}"
              f" median_epoch_s={median:.6g}")
        print(f"batch_{batch}_epoch_times={','.join(f'{t:.6g}' for t in times)}")
        print(f"batch_{batch}_mode={engine.execution(model)}", flush=True)

    print("RESULT=OK")
    return 0

def infer(argv: list[str]) -> int:
    reps = int(argv[3]) if len(argv) > 3 else 1
    batches = engine.batches_of(argv[4] if len(argv) > 4 else "256")
    opts = parse_opts(argv, 5)
    policy = engine.set_policy(opts["precision"], "infer")
    tf = engine.tf

    header("infer", opts)
    print(f"dtype_policy={policy}")
    windows, _ = load_series(argv[2], opts["past"], standardize=False)

    for batch in batches:
        model = build(windows.shape[2], opts)
        model.compile()
        if batch == batches[0]:
            describe(windows.shape[0], windows.shape[2], model, opts)

        forward = engine.predict_step(model)
        processed = (windows.shape[0] // batch) * batch
        with tf.device(engine.device_name(opts["device"])):
            window = tf.constant(windows[:batch], dtype=engine.compute_dtype())

        def run_pass():
            with tf.device(engine.device_name(opts["device"])):
                for _ in range(processed // batch):
                    forward(window)
            engine.sync()

        # One warm call, then a complete pass, before collecting samples --
        # what both other drivers do.
        forward(window)
        engine.sync()
        run_pass()

        print(f"TIMED_START_UNIX={time.time():.3f}", flush=True)
        times = []
        for _ in range(reps):
            mark = time.perf_counter()
            run_pass()
            times.append(time.perf_counter() - mark)
        print(f"TIMED_END_UNIX={time.time():.3f}", flush=True)

        times.sort()
        median = times[len(times) // 2]
        print(f"batch_{batch}_samples_per_sec={int(processed / median)}"
              f" median_pass_s={median:.6g}", flush=True)
        print(f"batch_{batch}_mode={engine.execution(model)}", flush=True)

    print("RESULT=OK")
    return 0

def capacity(argv: list[str]) -> int:
    batch = int(argv[3]) if len(argv) > 3 else 256
    opts = parse_opts(argv, 4)
    engine.set_policy(opts["precision"], "train")

    header("capacity", opts)
    print(f"batch={batch}")

    try:
        windows, targets = load_series(argv[2], opts["past"])
        model = build(windows.shape[2], opts)
        describe(windows.shape[0], windows.shape[2], model, opts)
        compile_for_training(model)
        with engine.tf.device(engine.device_name(opts["device"])):
            model.train_on_batch(windows[:batch], targets[:batch])
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
