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
# depends on the precision -- see the comment at the timing loop. `tf_path`
# names the winner; per-path metrics are emitted too. Set TF_NOLOOP=1 to time
# only the per-batch path, mirroring PT_NOGRAPH on the PyTorch side.
#
# `acc` exists to keep every iteration live: the model is pure and only the
# last output is returned, so without it XLA is free to elide the loop.
#
# Precision: fp32 (float32 policy, tensor cores off) or bf16 (mixed_bfloat16
# policy, TF32 tensor cores on) -- matching opennn_higgs_infer's precision.
#
#   usage:  python tensorflow_higgs_infer.py <test_csv> [batch] [runs] [fp32|bf16]
#                                            [hidden] [hidden_layers] [activation]

import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

def load_csv(path):
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)

    return np.ascontiguousarray(data[:, :-1])

def main():
    test_csv = sys.argv[1] if len(sys.argv) > 1 else "higgs_test.csv"
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
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
    processed = (samples // batch) * batch

    print(f"engine=tensorflow")
    print(f"mode=infer")
    print(f"gpus={[g.name for g in gpus]}")
    print(f"samples={processed}")
    print(f"batch={batch}")
    print(f"runs={runs}")
    print(f"hidden={hidden}")
    print(f"hidden_layers={hidden_layers}")
    print(f"activation={activation}")
    print(f"precision={precision}")

    if processed <= 0:
        print("RESULT=ERROR")
        raise SystemExit("batch larger than the test split")

    with tf.device("/GPU:0"):

        x = tf.constant(x_np[:processed])

        model_layers = [tf.keras.layers.Input(shape=(features,))]
        for _ in range(hidden_layers):
            model_layers.append(tf.keras.layers.Dense(hidden, activation=act))

        model_layers.append(tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32"))
        model = tf.keras.Sequential(model_layers)
        print(f"parameters={model.count_params()}")

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

        def compiled_pass():
            return whole_pass(x)

        def per_batch_pass():
            out = None
            for s in range(0, processed, batch):
                out = infer_step(x[s:s + batch])
            return out

        # Which path wins depends on the precision, so neither can be hardcoded
        # without under-reporting TensorFlow somewhere: compiling the loop
        # removes the per-batch eager dispatch, decisive when the GPU work per
        # batch is shorter than that dispatch (bf16 here, +11%) and a small loss
        # when it is not (fp32, -5%). Time both and report the better one, which
        # is the configuration the engine would actually be deployed in.
        paths = [("compiled_loop", compiled_pass), ("per_batch", per_batch_pass)]
        if os.environ.get("TF_NOLOOP") is not None:
            paths = [("per_batch", per_batch_pass)]

        medians = {}
        for name, run_pass in paths:
            print(f"warmup {name} (XLA compiling)...")
            _ = run_pass().numpy()

            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                out = run_pass()
                _ = out.numpy()
                times.append(time.perf_counter() - t0)

            times.sort()
            medians[name] = times[len(times) // 2]
            print(f"samples_per_sec_{name}={processed / medians[name]:.0f}")
            print(f"ms_per_batch_{name}={medians[name] * 1000.0 / n_batches:.6f}")

    tf_path = min(medians, key=medians.get)
    median_pass_s = medians[tf_path]
    samples_per_sec = processed / median_pass_s
    ms_per_batch = median_pass_s * 1000.0 / n_batches

    print(f"tf_path={tf_path}")

    print(f"median_pass_s={median_pass_s:.9g}")
    print(f"samples_per_sec={samples_per_sec:.0f}")
    print(f"ms_per_batch={ms_per_batch:.6f}")
    print("RESULT=OK")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        print("RESULT=ERROR")
        raise
