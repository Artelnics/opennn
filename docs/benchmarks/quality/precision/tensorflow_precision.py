# TensorFlow precision benchmark on the Rosenbrock dataset (10 inputs), after
# the Neural Designer blog protocol: 10 -> 10 (tanh) -> 1 (linear), U(-1, 1)
# init, Adam (lr 0.001), MSE, batch 1000, 10,000 epochs, trained on all
# samples. Prints the training wall time and writes full-dataset predictions.
#
# usage: python tensorflow_precision.py <seed>

import sys
import csv
import time

import numpy as np
import tensorflow as tf

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
tf.keras.utils.set_random_seed(seed)


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            rows.append([float(v) for v in r])
    data = np.array(rows, dtype=np.float32)
    return data[:, :-1], data[:, -1:]


x, y = load("rosenbrock.csv")

init = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="tanh", kernel_initializer=init, bias_initializer=init),
    tf.keras.layers.Dense(1, activation="linear", kernel_initializer=init, bias_initializer=init),
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

start = time.perf_counter()
model.fit(x, y, batch_size=1000, epochs=10000, shuffle=True, verbose=0)
print(f"train_time={time.perf_counter() - start}")

preds = model.predict(x, batch_size=10000, verbose=0)

with open("pred_tensorflow.txt", "w") as f:
    for v in preds[:, 0].tolist():
        f.write(f"{v:.10f}\n")
