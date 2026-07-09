# TensorFlow GPU HIGGS dense training-speed benchmark, the counterpart to
# opennn_speed (GPU HIGGS training).
#
# Mirrors the canonical HIGGS dense classifier (28 -> hidden -> hidden -> 1,
# ReLU hidden, sigmoid output, binary cross entropy -- see
# docs/benchmarks/throughput/higgs/README.md). The train and test CSVs are loaded
# (features then last-column label), the training tensors are made GPU-resident
# once, and Adam runs for N epochs at the given batch inside an XLA-compiled train
# step. After training the test set is scored and accuracy / log-loss / ROC-AUC
# are reported for the quality gate.
#
# "Highest performance" path (adapted from higgs/higgs_framework_cpu.py's
# run_tensorflow to the GPU):
#   * whole dataset resident on the GPU,
#   * XLA compilation of the train step (jit_compile=True),
#   * mixed_bfloat16 precision policy (bf16 mode) with TF32 tensor cores,
#   * per-epoch GPU-resident reshuffle (matches OpenNN).
#
#   usage:  python tensorflow_speed.py <train_csv> <epochs> <batch> <precision>
#                                      <shuffle> <hidden> <activation>
#                                      <hidden_layers> <test_csv>
#                                      <min_accuracy> <max_log_loss> <min_auc>
#           precision  = "bf16" (mixed_bfloat16 + TF32) or "fp32" (strict)
#           shuffle    = "shuffle" to reshuffle every epoch (matches OpenNN)
#           activation = "relu" (default) or "tanh"
#           thresholds = "none" when unset

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "higgs"))
from metrics import binary_metrics, parse_optional_float, passes_quality_gate


def load_csv(path):
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    x = np.ascontiguousarray(data[:, :-1])
    y = np.ascontiguousarray(data[:, -1:].astype(np.float32))
    return x, y


def main():
    train_csv = sys.argv[1] if len(sys.argv) > 1 else "higgs_train.csv"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 7000
    precision = sys.argv[4] if len(sys.argv) > 4 else "bf16"
    shuffle = (sys.argv[5] if len(sys.argv) > 5 else "shuffle") in ("shuffle", "1", "true")
    hidden = int(sys.argv[6]) if len(sys.argv) > 6 else 1024
    activation = (sys.argv[7] if len(sys.argv) > 7 else "relu").lower()
    hidden_layers = int(sys.argv[8]) if len(sys.argv) > 8 else 2
    test_csv = sys.argv[9] if len(sys.argv) > 9 else "higgs_test.csv"

    min_accuracy = parse_optional_float(sys.argv[10] if len(sys.argv) > 10 else None)
    max_log_loss = parse_optional_float(sys.argv[11] if len(sys.argv) > 11 else None)
    min_auc = parse_optional_float(sys.argv[12] if len(sys.argv) > 12 else None)

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
    x_np, y_np = load_csv(train_csv)
    xt_np, yt_np = load_csv(test_csv)
    features = x_np.shape[1]
    samples = x_np.shape[0]

    print("engine=tensorflow")
    print("mode=train")
    print(f"gpus={[g.name for g in gpus]}")
    print(f"samples={samples}")
    print(f"batch={batch}")
    print(f"epochs={epochs}")
    print(f"hidden={hidden}")
    print(f"hidden_layers={hidden_layers}")
    print(f"activation={activation}")
    print(f"precision={precision} shuffle={shuffle}")

    with tf.device("/GPU:0"):
        # Whole training set resident on the GPU once.
        x = tf.constant(x_np)
        y = tf.constant(y_np)

        model_layers = [tf.keras.layers.Input(shape=(features,))]
        for _ in range(hidden_layers):
            model_layers.append(tf.keras.layers.Dense(hidden, activation=act))
        # Keep the sigmoid output in float32 even under mixed_bfloat16.
        model_layers.append(tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32"))
        model = tf.keras.Sequential(model_layers)
        print(f"parameters={model.count_params()}")

        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        @tf.function(jit_compile=True)
        def train_step(xb, yb):
            with tf.GradientTape() as tape:
                pred = model(xb, training=True)
                loss = loss_fn(yb, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        n = x_np.shape[0]
        starts = list(range(0, n - batch + 1, batch))

        def run_epoch():
            if shuffle:
                # Fresh permutation each epoch, resident on the GPU.
                perm = tf.random.shuffle(tf.range(n))
                for s in starts:
                    idx = perm[s:s + batch]
                    train_step(tf.gather(x, idx), tf.gather(y, idx))
            else:
                for s in starts:
                    train_step(x[s:s + batch], y[s:s + batch])

        print("warmup (XLA compiling)...")
        run_epoch()
        run_epoch()

        times = []
        for _ in range(epochs):
            t0 = time.perf_counter()
            run_epoch()
            times.append(time.perf_counter() - t0)

        # Score the test set: whole batch-aligned slice on the GPU, forward-only.
        processed = (xt_np.shape[0] // batch) * batch
        xt = tf.constant(xt_np[:processed])
        preds = []
        for s in range(0, processed, batch):
            preds.append(model(xt[s:s + batch], training=False).numpy())

    times.sort()
    median_epoch_s = times[len(times) // 2]
    samples_per_sec = samples / median_epoch_s

    pred_np = np.vstack(preds) if preds else np.empty((0, 1), dtype=np.float32)
    metrics = binary_metrics(yt_np[: pred_np.shape[0]], pred_np)

    print(f"median_epoch_s={median_epoch_s:.9g}")
    print(f"samples_per_sec={samples_per_sec:.0f}")
    print(f"test_samples={pred_np.shape[0]}")
    for key, value in metrics.items():
        print(f"{key}={value:.9g}")

    if min_accuracy is not None or max_log_loss is not None or min_auc is not None:
        gate = passes_quality_gate(metrics, min_accuracy, max_log_loss, min_auc)
        print(f"quality_gate={'PASS' if gate else 'FAIL'}")

    print("RESULT=OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        print("RESULT=ERROR")
        raise
