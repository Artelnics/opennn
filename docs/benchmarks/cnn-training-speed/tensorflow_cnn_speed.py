# TensorFlow GPU CNN training-speed benchmark on MNIST.
#
# Same model as the OpenNN and PyTorch programs: 28x28x1 -> Conv 16@3x3
# (Same, ReLU) -> MaxPool 2x2 -> Flatten -> Dense 10 (softmax),
# cross-entropy, Adam, batch 128. Plain Keras fit() in fp32 with
# framework-default TF32 settings. Reports the median epoch time after a
# 2-epoch warmup, measured with an epoch callback.
#
#   usage:  python tensorflow_cnn_speed.py [epochs] [batch]

import sys
import time

import numpy as np
import tensorflow as tf

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 128

tf.keras.utils.set_random_seed(42)

x = np.load("mnist_images.npy") / np.float32(255.0)
y = np.load("mnist_labels.npy")
n = x.shape[0]
print(f"devices={tf.config.list_physical_devices('GPU')}")
print(f"samples={n} batch={batch} epochs={epochs}")

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy")


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
