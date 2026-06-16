# TensorFlow GPU CNN training-speed benchmark on MNIST.
#
# Same model as the OpenNN and PyTorch programs: 28x28x1 -> Conv 16@3x3
# (Same, ReLU) -> MaxPool 2x2 -> Flatten -> Dense 10 (softmax),
# cross-entropy, Adam, batch 128. Reports the median epoch time after a
# 2-epoch warmup, measured with an epoch callback.
#
# TensorFlow's fair fast path, so the comparison against OpenNN is honest:
#   default     -> plain Keras fit() fp32 (framework default, TF32 matmul).
#   TF_FAST=1   -> jit_compile=True (XLA graph fusion), TF's optimized path.
#   TF_BF16=1   -> mixed_bfloat16 global policy (matches OpenNN's bf16 column).
# TF's native conv layout is already NHWC, so no layout flag is needed.
#
#   usage:  python tensorflow_cnn_speed.py [epochs] [batch]
#   env:    TF_FAST=1 -> XLA ; TF_BF16=1 -> mixed_bfloat16

import os
import sys
import time

import numpy as np
import tensorflow as tf

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 128
fast = os.environ.get("TF_FAST") is not None
bf16 = os.environ.get("TF_BF16") is not None

tf.keras.utils.set_random_seed(42)
if bf16:
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

x = np.load("mnist_images.npy") / np.float32(255.0)
y = np.load("mnist_labels.npy")
n = x.shape[0]
print(f"devices={tf.config.list_physical_devices('GPU')}")
print(f"path={'fast(XLA)' if fast else 'eager(fit)'}{' +bf16' if bf16 else ''}")
print(f"samples={n} batch={batch} epochs={epochs}")

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    # Keep the softmax output in fp32 under the mixed_bfloat16 policy for a
    # numerically stable loss (Keras's recommended mixed-precision practice).
    tf.keras.layers.Dense(10, activation="softmax", dtype="float32"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy",
              jit_compile=fast)


class EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self.t0)


model.fit(x, y, batch_size=batch, epochs=2, shuffle=True, verbose=0)

timer = EpochTimer()
model.fit(x, y, batch_size=batch, epochs=epochs, shuffle=True, verbose=0, callbacks=[timer])

times = sorted(timer.times)
median = times[len(times) // 2]
print(f"epoch_s={median:.4f}")
print(f"samples_per_sec={n / median:.0f}")
print("RESULT=OK")
