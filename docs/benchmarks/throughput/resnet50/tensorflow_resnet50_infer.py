# TensorFlow GPU ResNet-50 inference-speed benchmark on CIFAR-10 (FORWARD ONLY).
#
# Inference twin of tensorflow_resnet50_speed.py. Same v1.5 bottleneck ResNet-50
# (stride on the 3x3), written explicitly so the architecture matches the others,
# same GPU-resident CIFAR-10 batch. The difference is that the loop is
# forward-only: model(xb, training=False) under @tf.function XLA, one batch held
# on the GPU, warmup then N timed forward passes -- no GradientTape, no optimizer.
# This is the fair counterpart to OpenNN's device-resident inference path.
#
# TensorFlow's fair fast path: NHWC (its native layout, the same OpenNN uses) +
# @tf.function with XLA (jit_compile=True). TF on GPU already uses NHWC + tensor
# cores, so this is the like-for-like opponent to OpenNN's resident path.
#
#   usage:  python tensorflow_resnet50_infer.py [batch] [runs] [data_dir]
#   env:    TF_BF16=1 -> mixed_bfloat16 policy (matches OpenNN's bf16 column)

import sys
import time
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 128
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
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
    # fp32 logits head under mixed_bfloat16 for a numerically stable head.
    out = K.Dense(classes, dtype="float32")(x)
    return tf.keras.Model(inp, out)


x = np.load(f"{data_dir}/cifar_images.npy").astype("float32") / 255.0  # NHWC already
y = np.load(f"{data_dir}/cifar_labels.npy").astype("int64")
n = x.shape[0]
hw = x.shape[1]
batch = min(batch, n)
classes = int(y.max()) + 1
print(f"device={gpus[0].name}")
print(f"path=fast(NHWC+XLA){' +bf16' if bf16 else ''}")
print(f"samples={n} batch={batch} runs={runs} classes={classes}")

with tf.device("/GPU:0"):
    xg = tf.constant(x)
    model = build_resnet50(classes, hw)
    print(f"parameters={model.count_params()}")

    @tf.function(jit_compile=True)
    def infer_step(xb):
        return model(xb, training=False)

    # One GPU-resident batch, held constant so the timed loop is pure forward.
    xb = tf.gather(xg, tf.range(batch))

    infer_step(xb)  # warmup (traces + compiles)
    infer_step(xb)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = infer_step(xb)
        _ = out.numpy()  # force the device forward to complete before stopping the clock
        times.append(time.perf_counter() - t0)

times.sort()
median = times[len(times) // 2]
print(f"ms_per_batch={median * 1000:.4f}")
print(f"samples_per_sec={batch / median:.0f}")
print("RESULT=OK")
