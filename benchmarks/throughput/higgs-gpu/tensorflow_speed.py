# TensorFlow GPU HIGGS dense training-speed benchmark, the counterpart to
# opennn_speed (GPU HIGGS training).
#
# Mirrors the canonical HIGGS dense classifier (28 -> hidden -> hidden -> 1,
# ReLU hidden, sigmoid output, binary cross entropy -- see
# benchmarks/throughput/higgs/README.md). The train and test CSVs are loaded
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

    # fp32 runs with TF32 tensor cores in every engine of this benchmark
    # (OpenNN's fp32 GEMMs are CUBLAS_COMPUTE_32F_FAST_TF32); "strict" disables it.
    tf.config.experimental.enable_tensor_float_32_execution(precision != "strict")

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

        x = tf.constant(x_np)
        y = tf.constant(y_np)

        model_layers = [tf.keras.layers.Input(shape=(features,))]
        for _ in range(hidden_layers):
            model_layers.append(tf.keras.layers.Dense(hidden, activation=act))

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

        # Shuffle once per epoch into a permuted copy, then slice, rather than
        # gathering per batch. Both shuffle identically; the difference is that
        # a per-batch gather is two eager ops dispatched from Python on top of
        # the step call, and TensorFlow enqueues asynchronously, so that host
        # cost is only hidden while it stays under the GPU's. It did not:
        # measured at batch 7000, enqueueing an epoch cost 0.9167 ms/batch
        # against 0.9295 ms to enqueue and run it -- the GPU was idle waiting on
        # Python, and the benchmark was measuring dispatch. Gathering once moves
        # 1.1026 -> 0.6878 ms/batch here with the same loss curve. The permuted
        # copy costs one dataset's worth of device memory for the epoch.
        @tf.function
        def shuffled_epoch_data():
            perm = tf.random.shuffle(tf.range(n))
            return tf.gather(x, perm), tf.gather(y, perm)

        def run_epoch():
            last = None
            xe, ye = shuffled_epoch_data() if shuffle else (x, y)
            for s in starts:
                last = train_step(xe[s:s + batch], ye[s:s + batch])
            return last

        print("warmup (XLA compiling)...")
        run_epoch()
        run_epoch()

        print(f"TRAIN_START_UNIX={time.time():.3f}", flush=True)
        times = []
        last_loss = None
        for _ in range(epochs):
            t0 = time.perf_counter()
            last_loss = run_epoch()
            times.append(time.perf_counter() - t0)
        if last_loss is not None:
            float(last_loss)
        print(f"TRAIN_END_UNIX={time.time():.3f}", flush=True)

        processed = (xt_np.shape[0] // batch) * batch
        xt = tf.constant(xt_np[:processed])
        preds = []
        for s in range(0, processed, batch):
            preds.append(model(xt[s:s + batch], training=False).numpy())

    times.sort()
    median_epoch_s = times[len(times) // 2]
    # An epoch runs whole batches only; dividing the full split by the epoch time
    # overstates throughput by up to one batch, which is 6.5% at batch 896,000.
    samples_per_epoch = len(starts) * batch
    samples_per_sec = samples_per_epoch / median_epoch_s

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
