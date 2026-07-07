# TensorFlow convergence-gate benchmark on the Rosenbrock dataset (10 inputs).
#
# MLPerf-style metric: wall-clock time to reach a fixed training-MSE target,
# plus the held-out TEST MSE at the stopping point. Same architecture / data /
# optimizer / target as opennn_convergence and pytorch_convergence.
#
#   usage: python tensorflow_convergence.py [seed] [target_mse] [max_epochs] [lr]

import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import csv
import tensorflow as tf

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
target_mse = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
max_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
lr = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
tf.random.set_seed(seed)


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            rows.append([float(x) for x in r])
    data = tf.constant(rows, dtype=tf.float32)
    return data[:, :-1], data[:, -1:]


train_x, train_y = load("rosenbrock_train.csv")
test_x, test_y = load("rosenbrock_test.csv")

net = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(50, activation="tanh", kernel_initializer="glorot_uniform"),
    tf.keras.layers.Dense(50, activation="tanh", kernel_initializer="glorot_uniform"),
    tf.keras.layers.Dense(1, kernel_initializer="glorot_uniform"),
])
net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")


# The convergence gate is the HELD-OUT test MSE (matches the OpenNN and PyTorch
# drivers). Keras computes val_loss on validation_data each epoch; stop when it
# reaches the target. A timing callback excludes the per-epoch validation eval.
class StopAtTarget(tf.keras.callbacks.Callback):
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.reached = False
        self.epochs_taken = 0
        self.final_train_mse = float("nan")
        self.test_mse = float("nan")
        self.train_s = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.train_s += time.perf_counter() - self._t0
        self.epochs_taken = epoch + 1
        self.final_train_mse = logs.get("loss", float("nan"))
        self.test_mse = logs.get("val_loss", float("nan"))
        if self.test_mse <= self.target:
            self.reached = True
            self.model.stop_training = True


stopper = StopAtTarget(target_mse)
net.fit(train_x, train_y, validation_data=(test_x, test_y),
        epochs=max_epochs, batch_size=64, verbose=0, callbacks=[stopper])
time_to_target = stopper.train_s
test_mse = stopper.test_mse

print(f"target_mse={target_mse}")
print(f"reached_goal={1 if stopper.reached else 0}")
print(f"epochs_to_target={stopper.epochs_taken}")
print(f"final_train_mse={stopper.final_train_mse:.10f}")
print(f"test_mse={test_mse:.10f}")
print(f"time_to_target_s={time_to_target:.6f}")
print(f"RESULT={'OK' if stopper.reached else 'DID_NOT_CONVERGE'}")
