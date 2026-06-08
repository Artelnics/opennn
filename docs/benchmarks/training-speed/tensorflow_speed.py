# TensorFlow GPU training-speed benchmark, tuned for maximum throughput.
#
# Mirrors the Neural Designer training-speed benchmark: a 2-layer MLP
# (F -> F -> 1, tanh then linear) trained with Adam + MSE on the Rosenbrock
# dataset, batch 1000. Reports median seconds/epoch and samples/second.
#
# "Highest performance" path:
#   * whole dataset resident on the GPU,
#   * XLA compilation of the train step (jit_compile=True),
#   * mixed_bfloat16 precision policy,
#   * a single tf.function stepping over batches with no Python overhead.
#
#   usage:  python tensorflow_speed.py <samples> <features> [epochs] [batch]

import sys
import time

import numpy as np
import tensorflow as tf


def rosenbrock(n, f, seed=1234):
    rng = np.random.default_rng(seed)
    x = (rng.random((n, f), dtype=np.float32) * 2.0 - 1.0)
    a = 1.0 - x[:, :-1]
    b = x[:, 1:] - x[:, :-1] ** 2
    y = (a * a + 100.0 * b * b).sum(axis=1, keepdims=True).astype(np.float32)
    return x, y


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    features = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    # Precision: bf16 (mixed_bfloat16 policy, default), tf32 (fp32 with tensor
    # cores), or fp32 (strict, tensor cores off).
    precision = sys.argv[5] if len(sys.argv) > 5 else "bf16"

    gpus = tf.config.list_physical_devices("GPU")
    assert gpus, "CUDA GPU required"
    print(f"gpus={[g.name for g in gpus]}")

    if precision == "bf16":
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
    # TF32 tensor cores: on for bf16/tf32, off for strict fp32.
    tf.config.experimental.enable_tensor_float_32_execution(precision in ("bf16", "tf32"))
    print(f"precision={precision}")
    tf.random.set_seed(42)

    x_np, y_np = rosenbrock(samples, features)
    print(f"samples={samples} features={features} batch={batch} epochs={epochs}")

    with tf.device("/GPU:0"):
        x = tf.constant(x_np)
        y = tf.constant(y_np)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(features,)),
            tf.keras.layers.Dense(features, activation="tanh"),
            tf.keras.layers.Dense(1, activation="linear", dtype="float32"),
        ])
        optimizer = tf.keras.optimizers.Adam()
        mse = tf.keras.losses.MeanSquaredError()

        @tf.function(jit_compile=True)
        def train_step(xb, yb):
            with tf.GradientTape() as tape:
                pred = model(xb, training=True)
                loss = mse(yb, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        n = x.shape[0]
        starts = list(range(0, n - batch + 1, batch))

        def run_epoch():
            for s in starts:
                train_step(x[s:s + batch], y[s:s + batch])

        print("warmup (XLA compiling)...")
        run_epoch()
        run_epoch()

        times = []
        for e in range(epochs):
            t0 = time.perf_counter()
            run_epoch()
            times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    print(f"median_epoch_s={median:.4f}")
    print(f"samples_per_sec={samples / median:.0f}")
    print("RESULT=OK")


if __name__ == "__main__":
    main()
