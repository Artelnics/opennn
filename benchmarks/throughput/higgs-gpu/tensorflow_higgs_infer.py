#!/usr/bin/env python3
"""TensorFlow GPU HIGGS dense inference benchmark, written the way a TensorFlow
user writes it: keras Sequential, the forward in an XLA-compiled tf.function,
the test split GPU-resident. The measurement protocol - rotation, soaking,
medians over rounds - lives in run_higgs_infer.py and run_higgs_infer_sweep.py,
not here.

    python tensorflow_higgs_infer.py <test_csv> [batch[,batch...]] [runs] [fp32|bf16] [hidden] [hidden_layers] [activation]

Precision: fp32 (float32 policy, TF32 tensor cores as in every engine of this
benchmark) or bf16 (mixed_bfloat16 policy), matching opennn_higgs_infer's
precision; "strict" turns TF32 off.

Two dispatch paths are timed at every rung and the faster is reported, named
in `tf_path`. The OpenNN and PyTorch drivers replay a captured graph, so they
pay one cheap launch per batch; calling a tf.function per batch from Python
costs ~0.23 ms of eager dispatch, which at batch 1024 is the entire
measurement - the GPU sat idle while the work was enqueued. Compiling the batch
loop into one tf.function is TensorFlow's own answer to that. It is decisive
when the GPU work per batch is shorter than the dispatch, which is what the
small rungs of a sweep are, and a small loss when it is not, so neither path
can be pinned without under-reporting TensorFlow somewhere. The per-iteration
shape is unchanged either way, so the GEMMs are the ones the other engines run.
`acc` keeps every iteration of the compiled loop live: the model is pure and
only the last output is returned, so without it XLA is free to elide the loop.
TF_NOLOOP=1 times only the per-batch path, mirroring PT_NOGRAPH on the PyTorch
side.

TF_BF16_INPUT=1 holds the resident rows as bfloat16 in the bf16 cell, which is
what the other two engines do (PyTorch's PT_BF16_WEIGHTS, OpenNN's staged type)
and what saves them a cast kernel and half the staging bytes per batch. It is
OFF by default because it does not help TensorFlow and hurts it at small batch:
alternated pairs at batch 256/8,192/65,536 measured 10.92/32.68/24.57 M
samples/s with it against 11.96/32.66/24.44 M without, so -9% at 256 and a tie
above. XLA already folds the cast into the first layer's operand read, and
slicing a bf16 constant costs more than it saves. Recorded because the question
had to be asked before comparing: an engine measured below its own best is the
failure mode this benchmark family has hit before.

The batch sizes run inside one process, matching the other drivers, so they
share one load and one thermal window; each rung compiles its own pair of
tf.functions, so every rung is measured on a statically compiled shape.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


def load_csv(path):
    # np.loadtxt on the 500k-row split is minutes of wall clock, none of it
    # measured, so the parsed array is cached next to the CSV.
    path = Path(path)
    cache = path.with_suffix(path.suffix + ".npy")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        data = np.load(cache, mmap_mode="r")
    else:
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
        try:
            np.save(cache, data)
        except OSError:              # read-only data directory: parse next time
            pass
    return np.ascontiguousarray(data[:, :-1])


def main():
    test_csv = sys.argv[1] if len(sys.argv) > 1 else "higgs_test.csv"
    batch_text = sys.argv[2] if len(sys.argv) > 2 else "8192"
    batch_list = [int(item) for item in batch_text.split(",") if item.strip()] or [8192]
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    precision = sys.argv[4] if len(sys.argv) > 4 else "fp32"
    hidden = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
    hidden_layers = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    activation = (sys.argv[7] if len(sys.argv) > 7 else "relu").lower()

    gpus = tf.config.list_physical_devices("GPU")
    assert gpus, "CUDA GPU required"
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.random.set_seed(42)

    if precision == "bf16":
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")

    # fp32 runs with TF32 tensor cores in every engine of this benchmark
    # (OpenNN's fp32 GEMMs are CUBLAS_COMPUTE_32F_FAST_TF32); "strict" disables it.
    tf.config.experimental.enable_tensor_float_32_execution(precision != "strict")

    act = "relu" if activation == "relu" else "tanh"
    x_np = load_csv(test_csv)
    features = x_np.shape[1]
    samples = x_np.shape[0]

    print("engine=tensorflow")
    print("mode=infer")
    print(f"gpus={[g.name for g in gpus]}")
    print(f"runs={runs}")
    print(f"hidden={hidden}")
    print(f"hidden_layers={hidden_layers}")
    print(f"activation={activation}")
    print(f"precision={precision}")

    bf16_input = precision == "bf16" and os.environ.get("TF_BF16_INPUT", "0") != "0"
    print(f"bf16_input={int(bf16_input)}")

    with tf.device("/GPU:0"):

        x = tf.constant(np.ascontiguousarray(x_np))
        if bf16_input:
            x = tf.cast(x, tf.bfloat16)

        model_layers = [tf.keras.layers.Input(shape=(features,))]
        for _ in range(hidden_layers):
            model_layers.append(tf.keras.layers.Dense(hidden, activation=act))
        model_layers.append(tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32"))
        model = tf.keras.Sequential(model_layers)
        print(f"parameters={model.count_params()}")

        single = len(batch_list) == 1
        for batch in batch_list:
            processed = (samples // batch) * batch
            if processed <= 0:
                print(f"batch_{batch}_error=batch larger than the test split")
                continue

            n_batches = processed // batch

            @tf.function(jit_compile=True)
            def infer_step(xb):
                return model(xb, training=False)

            @tf.function(jit_compile=True)
            def whole_pass(data):
                acc = tf.zeros([batch, 1], tf.float32)
                for i in tf.range(n_batches):
                    xb = tf.slice(data, [i * batch, 0], [batch, features])
                    acc += tf.cast(model(xb, training=False), tf.float32)
                return acc

            data = x[:processed]

            def compiled_pass():
                return whole_pass(data)

            def per_batch_pass():
                out = None
                for s in range(0, processed, batch):
                    out = infer_step(data[s:s + batch])
                return out

            paths = [("compiled_loop", compiled_pass), ("per_batch", per_batch_pass)]
            if os.environ.get("TF_NOLOOP") is not None:
                paths = [("per_batch", per_batch_pass)]

            medians = {}
            for name, run_pass in paths:
                run_pass().numpy()                    # XLA compiles here
                run_pass().numpy()

                times = []
                for _ in range(runs):
                    t0 = time.perf_counter()
                    run_pass().numpy()
                    times.append(time.perf_counter() - t0)

                # Temporal order, before the sort: a median hides a drifting machine.
                print(f"batch_{batch}_{name}_pass_times=" + ",".join(f"{t:.9g}" for t in times))

                times.sort()
                medians[name] = times[len(times) // 2]
                print(f"batch_{batch}_samples_per_sec_{name}={processed / medians[name]:.0f}")
                print(f"batch_{batch}_ms_per_batch_{name}={medians[name] * 1000.0 / n_batches:.6f}")

            tf_path = min(medians, key=medians.get)
            median_pass_s = medians[tf_path]
            samples_per_sec = processed / median_pass_s
            ms_per_batch = median_pass_s * 1000.0 / n_batches

            print(f"batch_{batch}_tf_path={tf_path}")

            if single:
                print(f"samples={processed}")
                print(f"batch={batch}")
                print(f"tf_path={tf_path}")
                print(f"median_pass_s={median_pass_s:.9g}")
                print(f"samples_per_sec={samples_per_sec:.0f}")
                print(f"ms_per_batch={ms_per_batch:.6f}")
            else:
                print(f"batch_{batch}_samples={processed}")
                print(f"batch_{batch}_samples_per_sec={samples_per_sec:.0f}"
                      f" median_pass_s={median_pass_s:.9g} ms_per_batch={ms_per_batch:.6f}")
            sys.stdout.flush()

    print("RESULT=OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        print("RESULT=ERROR")
        raise
