# TensorFlow GPU ResNet-50 training-speed benchmark on CIFAR-10.
#
# Counterpart to opennn_resnet50_speed / pytorch_resnet50_speed. Same v1.5
# bottleneck ResNet-50 (stride on the 3x3), written explicitly so the
# architecture matches the others. CIFAR-10 held GPU-resident; cross-entropy +
# Adam; median epoch time after a 2-epoch warmup.
#
# TensorFlow's fair fast path: NHWC (its native layout, the same OpenNN uses) +
# @tf.function with XLA (jit_compile=True). TF on GPU already uses NHWC + tensor
# cores, so this is the like-for-like opponent to OpenNN's CUDA-graph path.
#
#   usage:  python tensorflow_resnet50_speed.py [epochs] [batch] [data_dir]
#   env:    TF_BF16=1 -> mixed_bfloat16 policy (matches OpenNN's bf16 column)

import sys
import time
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 128
data_dir = sys.argv[3] if len(sys.argv) > 3 else "cifar10"
bf16 = os.environ.get("TF_BF16") is not None

gpus = tf.config.list_physical_devices("GPU")
assert gpus, "CUDA GPU required"
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
tf.random.set_seed(42)
if bf16:
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

K = tf.keras.layers


def bottleneck(x, mid, stride, name):
    out = mid * 4
    shortcut = x
    if stride != 1 or x.shape[-1] != out:
        shortcut = K.Conv2D(out, 1, strides=stride, use_bias=False)(x)
        shortcut = K.BatchNormalization()(shortcut)
    y = K.Conv2D(mid, 1, use_bias=False)(x)
    y = K.ReLU()(K.BatchNormalization()(y))
    y = K.Conv2D(mid, 3, strides=stride, padding="same", use_bias=False)(y)
    y = K.ReLU()(K.BatchNormalization()(y))
    y = K.Conv2D(out, 1, use_bias=False)(y)
    y = K.BatchNormalization()(y)
    return K.ReLU()(y + shortcut)


def build_resnet50(classes, hw):
    inp = K.Input(shape=(hw, hw, 3))
    x = K.Conv2D(64, 7, strides=2, padding="same", use_bias=False)(inp)
    x = K.ReLU()(K.BatchNormalization()(x))
    x = K.MaxPool2D(3, strides=2, padding="same")(x)
    in_ch = 64
    for stage, (mid, blocks) in enumerate(zip([64, 128, 256, 512], [3, 4, 6, 3])):
        for block in range(blocks):
            stride = 2 if (block == 0 and stage > 0) else 1
            x = bottleneck(x, mid, stride, f"s{stage}b{block}")
            in_ch = mid * 4
    x = K.GlobalAveragePooling2D()(x)
    # fp32 logits head under mixed_bfloat16 for a numerically stable loss.
    out = K.Dense(classes, dtype="float32")(x)
    return tf.keras.Model(inp, out)


x = np.load(f"{data_dir}/cifar_images.npy").astype("float32") / 255.0  # NHWC already
y = np.load(f"{data_dir}/cifar_labels.npy").astype("int64")
n = x.shape[0]
hw = x.shape[1]
classes = int(y.max()) + 1
print(f"device={gpus[0].name}")
print(f"path=fast(NHWC+XLA){' +bf16' if bf16 else ''}")
print(f"samples={n} batch={batch} epochs={epochs} classes={classes}")

with tf.device("/GPU:0"):
    xg = tf.constant(x)
    yg = tf.constant(y)
    model = build_resnet50(classes, hw)
    print(f"parameters={model.count_params()}")
    opt = tf.keras.optimizers.Adam(1e-3)
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function(jit_compile=True)
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            loss = scce(yb, model(xb, training=True))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    starts = list(range(0, n - batch + 1, batch))

    def run_epoch():
        perm = tf.random.shuffle(tf.range(n))
        for s in starts:
            idx = perm[s:s + batch]
            train_step(tf.gather(xg, idx), tf.gather(yg, idx))

    run_epoch()  # warmup (traces + compiles)
    run_epoch()

    times = []
    for _ in range(epochs):
        t0 = time.perf_counter()
        run_epoch()
        times.append(time.perf_counter() - t0)

times.sort()
median = times[len(times) // 2]
print(f"epoch_s={median:.4f}")
print(f"samples_per_sec={n / median:.0f}")
print("RESULT=OK")
