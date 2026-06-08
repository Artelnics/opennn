# TensorFlow peak-memory benchmark: load sum.csv (1000 x 101, regression), build
# the same MLP, train, and report resident-set-size (RSS) at two points:
# baseline (framework loaded + model built, before training) and peak.

import resource
import csv

import tensorflow as tf


def peak_rss_mb():
    # ru_maxrss is kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# sum.csv: 100 inputs + 1 target, ';'-separated, no header.
rows = []
with open("sum.csv", newline="") as f:
    for r in csv.reader(f, delimiter=";"):
        rows.append([float(x) for x in r])

data = tf.constant(rows, dtype=tf.float32)
inputs = data[:, :-1]
targets = data[:, -1:]

tf.random.set_seed(42)
net = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(100,)),
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(1),
])

print(f"baseline_rss_mb {peak_rss_mb():.1f}")

net.compile(optimizer="adam", loss="mse")
net.fit(inputs, targets, epochs=50, batch_size=32, verbose=0)

print(f"peak_rss_mb {peak_rss_mb():.1f}")
