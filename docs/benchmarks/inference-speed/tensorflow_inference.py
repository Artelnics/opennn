# TensorFlow CPU inference-speed benchmark.
#
# Mirrors the training-speed model so the two read together: a 2-layer MLP
# (F -> F -> 1, tanh then linear) on the Rosenbrock dataset. Here the network
# only does inference: model(x, training=False) runs a pure forward pass (no
# gradient tape, no dropout) -- the apples-to-apples equivalent of OpenNN's
# calculate_outputs().
#
# Reports median seconds per full pass over the dataset, samples/second
# (throughput), and milliseconds per batch (latency).
#
#   usage:  python tensorflow_inference.py <samples> <features> [batch] [reps]

import sys
import time

import numpy as np
import tensorflow as tf


def rosenbrock(n, f, seed=1234):
    rng = np.random.default_rng(seed)
    return (rng.random((n, f), dtype=np.float32) * 2.0 - 1.0)


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    features = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 30

    tf.random.set_seed(42)
    print(f"samples={samples} features={features} batch={batch} reps={reps}")

    with tf.device("/CPU:0"):
        x = tf.constant(rosenbrock(samples, features))

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(features,)),
            tf.keras.layers.Dense(features, activation="tanh"),
            tf.keras.layers.Dense(1, activation="linear"),
        ])

        # Compile the forward pass into a single graph (the inference equivalent
        # of the XLA train step used on the training-speed side).
        infer = tf.function(lambda xb: model(xb, training=False), jit_compile=True)

        n = int(x.shape[0])
        starts = list(range(0, n - batch + 1, batch))

        def run_pass():
            for s in starts:
                out = infer(x[s:s + batch])
                float(out[0, 0])

        print("warmup (graph tracing)...")
        run_pass()
        run_pass()

        batched_samples = len(starts) * batch
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            run_pass()
            times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    print(f"median_pass_s={median:.6f}")
    print(f"samples_per_sec={batched_samples / median:.0f}")
    print(f"ms_per_batch={median / len(starts) * 1000.0:.4f}")
    print("RESULT=OK")


if __name__ == "__main__":
    main()
