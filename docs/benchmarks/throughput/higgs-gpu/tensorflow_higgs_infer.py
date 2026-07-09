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
    # x = features (all but the last column); the last column (label) is ignored.
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
    # TF32 tensor cores: on for bf16, off for strict fp32.
    tf.config.experimental.enable_tensor_float_32_execution(precision in ("bf16", "tf32"))

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
        # Whole batch-aligned test slice resident on the GPU once.
        x = tf.constant(x_np[:processed])

        model_layers = [tf.keras.layers.Input(shape=(features,))]
        for _ in range(hidden_layers):
            model_layers.append(tf.keras.layers.Dense(hidden, activation=act))
        # Keep the sigmoid output in float32 even under mixed_bfloat16.
        model_layers.append(tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32"))
        model = tf.keras.Sequential(model_layers)
        print(f"parameters={model.count_params()}")

        n_batches = processed // batch

        @tf.function(jit_compile=True)
        def infer_step(xb):
            return model(xb, training=False)

        def run_pass():
            for s in range(0, processed, batch):
                infer_step(x[s:s + batch])

        print("warmup (XLA compiling)...")
        run_pass()
        _ = infer_step(x[:batch]).numpy()

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            run_pass()
            out = infer_step(x[:batch])
            _ = out.numpy()
            times.append(time.perf_counter() - t0)

    times.sort()
    median_pass_s = times[len(times) // 2]
    samples_per_sec = processed / median_pass_s
    ms_per_batch = median_pass_s * 1000.0 / n_batches

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
