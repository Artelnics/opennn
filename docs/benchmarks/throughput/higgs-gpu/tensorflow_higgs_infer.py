# TensorFlow GPU HIGGS dense inference-speed benchmark, the counterpart to
# opennn_higgs_infer.
#
# Mirrors the canonical HIGGS dense classifier (28 -> hidden -> hidden -> 1,
# ReLU hidden, sigmoid output -- see docs/benchmarks/throughput/higgs/README.md).
# Inference-only: the model is called with training=False inside an XLA-compiled
# tf.function (jit_compile=True). The whole (batch-aligned) test slice is made
# GPU-resident once and only the forward is timed, after a warmup. Reports
# samples/sec and ms/batch.
#
# The batch loop lives INSIDE the compiled function. The OpenNN and PyTorch
# drivers replay a captured graph, so they pay one cheap launch per batch;
# calling a tf.function per batch from Python instead costs ~0.23 ms of eager
# dispatch, which at batch 1024 is the entire measurement -- enqueueing the
# work took as long as enqueueing and executing it, i.e. the GPU sat idle.
# Compiling the loop is TensorFlow's own answer to that and makes the dispatch
# contract match. The per-iteration shape is unchanged, so the GEMMs are the
# same ones the other engines run.
#
# Both paths are timed and the faster is reported, because which one wins
# depends on the precision and on the batch size -- see the comment at the
# timing loop. `tf_path` names the winner; per-path metrics are emitted too.
# Set TF_NOLOOP=1 to time only the per-batch path, mirroring PT_NOGRAPH on the
# PyTorch side.
#
# `acc` exists to keep every iteration live: the model is pure and only the
# last output is returned, so without it XLA is free to elide the loop.
#
# Precision: fp32 (float32 policy, tensor cores off) or bf16 (mixed_bfloat16
# policy, TF32 tensor cores on) -- matching opennn_higgs_infer's precision.
#
# TF_BF16_INPUT=1 holds the resident rows as bfloat16 in the bf16 cell, which is
# what the other two engines do (PyTorch's PT_BF16_WEIGHTS, OpenNN's staged
# type) and what saves them a cast kernel and half the staging bytes per batch.
# It is OFF by default because it does not help TensorFlow and hurts it at small
# batch: alternated pairs at batch 256/8,192/65,536 measured 10.92/32.68/24.57 M
# samples/s with it against 11.96/32.66/24.44 M without, so -9% at 256 and a
# tie above. XLA already folds the cast into the first layer's operand read, and
# slicing a bf16 constant costs more than it saves. Recorded because the
# question had to be asked before comparing: an engine measured below its own
# best is the failure mode this benchmark family has hit before.
#
#   usage:  python tensorflow_higgs_infer.py <test_csv> [batch[,batch...]] [runs]
#                                            [fp32|bf16] [hidden] [hidden_layers] [activation]
#
# A comma-separated batch list is swept inside one process, matching the OpenNN
# and PyTorch drivers: the batch sizes then share one load and one thermal
# window. Each rung compiles its own pair of tf.functions, so every rung is
# measured on a statically compiled shape.

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

def load_csv(path):
    # np.loadtxt on the full split is minutes of wall clock per engine per run,
    # none of it measured, so the parsed array is cached next to the CSV and
    # re-read with np.load.
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

def parse_batches(text):
    values = [int(item) for item in str(text).split(",") if item.strip()]
    return values or [8192]

def measure(batch, x, samples, features, model, runs, single):
    processed = (samples // batch) * batch
    if processed <= 0:
        print(f"batch_{batch}_error=batch larger than the test split")
        return

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

    # Which path wins depends on the precision and on the batch size, so
    # neither can be hardcoded without under-reporting TensorFlow somewhere:
    # compiling the loop removes the per-batch eager dispatch, decisive when the
    # GPU work per batch is shorter than that dispatch -- which is exactly what
    # the small rungs of a batch sweep are -- and a small loss when it is not.
    # Time both and report the better one, which is the configuration the engine
    # would actually be deployed in.
    paths = [("compiled_loop", compiled_pass), ("per_batch", per_batch_pass)]
    if os.environ.get("TF_NOLOOP") is not None:
        paths = [("per_batch", per_batch_pass)]

    medians = {}
    for name, run_pass in paths:
        _ = run_pass().numpy()                        # XLA compiles here
        _ = run_pass().numpy()

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            out = run_pass()
            _ = out.numpy()
            times.append(time.perf_counter() - t0)

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

def main():
    test_csv = sys.argv[1] if len(sys.argv) > 1 else "higgs_test.csv"
    batch_list = parse_batches(sys.argv[2] if len(sys.argv) > 2 else "8192")
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

    print(f"engine=tensorflow")
    print(f"mode=infer")
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
            measure(batch, x, samples, features, model, runs, single)

    print("RESULT=OK")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        print("RESULT=ERROR")
        raise
