# Equal-batch steady-state speed for the shallow Rosenbrock MLP in TensorFlow,
# the counterpart to opennn_rosenbrock_throughput / pytorch_rosenbrock_throughput.
# Same net (inputs->hidden->1, tanh, MSE, Adam, fp32), GPU-resident synthetic
# data, warmup excluded. Uses TensorFlow's fair fast path: @tf.function with XLA
# (jit_compile=True) so the step is fused/compiled, not eager.
#
#   usage: python tensorflow_rosenbrock_throughput.py <train|inference> [batch] [iters] [inputs] [hidden]

import sys
import time
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

mode   = sys.argv[1] if len(sys.argv) > 1 else "train"
batch  = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
iters  = int(sys.argv[3]) if len(sys.argv) > 3 else 100
inputs = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
hidden = int(sys.argv[5]) if len(sys.argv) > 5 else 1000

gpus = tf.config.list_physical_devices("GPU")
assert gpus, "CUDA GPU required"
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
tf.random.set_seed(0)

with tf.device("/GPU:0"):
    net = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden, activation="tanh", input_shape=(inputs,)),
        tf.keras.layers.Dense(1),
    ])
    x = tf.random.normal((batch, inputs))
    y = tf.random.normal((batch, 1))

    if mode == "inference":
        @tf.function(jit_compile=True)
        def fwd():
            return net(x, training=False)

        fwd()  # warmup (traces + compiles)
        _ = fwd().numpy()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fwd()
        _ = out.numpy()  # force completion
        per = (time.perf_counter() - t0) / iters
    else:
        opt = tf.keras.optimizers.Adam(1e-3)
        mse = tf.keras.losses.MeanSquaredError()

        @tf.function(jit_compile=True)
        def step():
            with tf.GradientTape() as tape:
                loss = mse(y, net(x, training=True))
            grads = tape.gradient(loss, net.trainable_variables)
            opt.apply_gradients(zip(grads, net.trainable_variables))
            return loss

        step()  # warmup (traces + compiles)
        _ = step().numpy()
        t0 = time.perf_counter()
        for _ in range(iters):
            loss = step()
        _ = loss.numpy()  # force completion
        per = (time.perf_counter() - t0) / iters

print(f"mode={mode} batch={batch}")
print(f"step_s={per:.6f}")
print(f"samples_per_sec={int(batch / per)}")
