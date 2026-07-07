# TensorFlow/Keras HIGGS dense max-batch counterpart to
# opennn_higgs_maxbatch_trial. Same canonical model (28 -> hidden -> hidden ->
# 1, ReLU hidden, binary cross-entropy, Adam -- see
# docs/benchmarks/higgs/README.md). The output layer produces logits kept in
# float32 and the loss is BinaryCrossentropy(from_logits=True), the standard
# TF formulation of the same objective.
#
# TF config: mixed_bfloat16 policy (bf16 path), TF32 for fp32 matmuls (on by
# default), standard graph mode (@tf.function, no XLA) -- the same
# no-whole-graph-compiler execution as the other engines in this suite.
#
# mode "train": one training step per --steps (forward + backward + Adam).
# mode "infer": forward-only, training=False, no GradientTape, no optimizer.
#
# Data is synthetic with the HIGGS contract shapes: capacity depends on the
# shapes and the step, not the feature values.
#
#   usage: python tensorflow_higgs_maxbatch.py --mode train|infer --batch B
#              [--hidden 1024] [--layers 2] [--steps N] [--warmup W]
#              [--device cuda|cpu]
#   env:   TF_BF16=1 -> mixed_bfloat16 (CUDA only)

import argparse, os, time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf

INPUTS = 28   # HIGGS contract: 28 features, 1 target

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["train", "infer"], default="train")
ap.add_argument("--batch", type=int, required=True)
ap.add_argument("--hidden", type=int, default=1024)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--steps", type=int, default=1)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
args = ap.parse_args()

use_cuda = args.device == "cuda"
if use_cuda:
    assert tf.config.list_physical_devices("GPU"), "CUDA GPU required"
else:
    tf.config.set_visible_devices([], "GPU")
tf.random.set_seed(0)

use_bf16 = use_cuda and os.environ.get("TF_BF16") is not None
if use_bf16:
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
print(f"precision={'bf16' if use_bf16 else 'fp32'} mode={args.mode} "
      f"device={args.device} "
      f"inputs={INPUTS} hidden={args.hidden} hidden_layers={args.layers} "
      f"batch={args.batch} steps={args.steps} xla=False")

L = tf.keras.layers
inp = L.Input(shape=(INPUTS,), dtype="float32")
h = inp
for _ in range(args.layers):
    h = L.Dense(args.hidden, activation="relu")(h)
# logits kept in float32 under mixed precision (standard practice)
logits = L.Dense(1, dtype="float32")(h)
model = tf.keras.Model(inp, logits)
print(f"parameters={model.count_params()}")

rng = np.random.default_rng(0)

# Real HIGGS rows (float32 bin, rows x 29: features then label) when
# HIGGS_BIN is set; rows repeat modulo beyond the file (np.resize), the same
# convention as the ResNet-50 capacity runner. Synthetic otherwise.
higgs_bin = os.environ.get("HIGGS_BIN")
if higgs_bin:
    raw = np.fromfile(higgs_bin, dtype=np.float32).reshape(-1, INPUTS + 1)
    print(f"data=higgs_bin rows={raw.shape[0]}")
    x = tf.constant(np.resize(np.ascontiguousarray(raw[:, :INPUTS]), (args.batch, INPUTS)))
    y_host = np.resize(np.ascontiguousarray(raw[:, INPUTS:]), (args.batch, 1))
else:
    print("data=synthetic")
    x = tf.constant(rng.standard_normal((args.batch, INPUTS), dtype=np.float32))
    y_host = None

if args.mode == "train":
    y = tf.constant(y_host) if y_host is not None \
        else tf.constant(rng.integers(0, 2, (args.batch, 1)).astype(np.float32))

    opt = tf.keras.optimizers.Adam(1e-3)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(x_b, y_b):
        with tf.GradientTape() as tape:
            loss = loss_fn(y_b, model(x_b, training=True))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    last = None
    for _ in range(args.warmup):
        last = train_step(x, y)
    _ = last.numpy()   # force sync after warmup

    t0 = time.perf_counter()
    for _ in range(args.steps):
        last = train_step(x, y)
    last_host = last.numpy()   # force sync
    wall_s = time.perf_counter() - t0

    assert np.isfinite(float(last_host)), "non-finite loss"
    print(f"final_loss={float(last_host):.5f}")
else:
    @tf.function
    def infer_step(x_b):
        out = model(x_b, training=False)
        return out[:8, 0]

    probe = None
    for _ in range(args.warmup):
        probe = infer_step(x)
    _ = probe.numpy()   # force sync after warmup

    t0 = time.perf_counter()
    for _ in range(args.steps):
        probe = infer_step(x)
    probe_host = probe.numpy()   # force sync
    wall_s = time.perf_counter() - t0

    assert np.isfinite(np.asarray(probe_host, dtype=np.float32)).all(), "non-finite outputs"

samples_per_s = args.steps * args.batch / wall_s
try:   # peak memory for the CPU-capped runs (POSIX only)
    import resource
    print(f"peak_rss_mib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}")
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmPeak:"):
                print(f"vm_peak_mib={int(line.split()[1]) // 1024}")
                break
except Exception:
    pass
print(f"wall_s={wall_s:.5f}")
print(f"samples_per_sec={samples_per_s:.2f}")
print("RESULT=OK")
