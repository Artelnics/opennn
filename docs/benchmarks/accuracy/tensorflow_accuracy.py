# TensorFlow accuracy-parity benchmark on the Rosenbrock dataset (10 inputs).
# Same architecture / loss / optimizer / epochs as the OpenNN and PyTorch
# programs; consumes the shared normalized split and writes test predictions.

import sys
import csv

import tensorflow as tf

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
tf.random.set_seed(seed)


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            rows.append([float(x) for x in r])
    data = tf.constant(rows, dtype=tf.float32)
    return data[:, :-1], data[:, -1:]


train_x, train_y = load("rosenbrock_train.csv")
test_x, _ = load("rosenbrock_test.csv")

# Glorot/Xavier uniform is Keras's Dense default — matches OpenNN and the
# PyTorch program's explicit init.
net = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(50, activation="tanh", kernel_initializer="glorot_uniform"),
    tf.keras.layers.Dense(50, activation="tanh", kernel_initializer="glorot_uniform"),
    tf.keras.layers.Dense(1, kernel_initializer="glorot_uniform"),
])

net.compile(optimizer="adam", loss="mse")
net.fit(train_x, train_y, epochs=200, batch_size=64, verbose=0)

preds = net.predict(test_x, verbose=0)

with open("pred_tensorflow.txt", "w") as f:
    for v in preds[:, 0]:
        f.write(f"{float(v):.10f}\n")
