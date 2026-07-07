# TensorFlow/Keras encoder-decoder Transformer max-batch / training-speed
# counterpart to opennn_transformer_maxbatch_trial. Same architecture (token
# embeddings + sinusoidal positional encoding, N encoder + N decoder layers with
# multi-head attention + feed-forward + layer norm, linear vocab projection),
# same optimizer (Adam), same token cross-entropy, same model shape.
#
# TF config: mixed_bfloat16 policy (bf16 path), TF32 for fp32 matmuls (on by
# default), standard graph mode (@tf.function, no XLA). XLA is deliberately OFF
# so all three engines run the same kind of execution -- op-by-op kernels with
# no whole-graph compiler fusion/rematerialization (OpenNN and eager PyTorch
# have no equivalent).
#
# --mode infer measures forward-only capacity/throughput: model([...],
# training=False) inside @tf.function, no GradientTape, no optimizer state.
#
#   usage: python tensorflow_transformer_maxbatch.py --in-vocab V --out-vocab V \
#              --in-seq S --dec-seq S --d 512 --h 8 --ff 2048 --layers 6 \
#              --batch B --steps N [--warmup W] [--mode train|infer]
#   env:   TF_BF16=1 -> mixed_bfloat16

import argparse, math, os, time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--in-vocab", type=int, required=True)
ap.add_argument("--out-vocab", type=int, required=True)
ap.add_argument("--in-seq", type=int, required=True)
ap.add_argument("--dec-seq", type=int, required=True)
ap.add_argument("--d", type=int, default=512)
ap.add_argument("--h", type=int, default=8)
ap.add_argument("--ff", type=int, default=2048)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--mode", choices=["train", "infer"], default="train")
args = ap.parse_args()

assert tf.config.list_physical_devices("GPU"), "CUDA GPU required"
tf.random.set_seed(0)

use_bf16 = os.environ.get("TF_BF16") is not None
if use_bf16:
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
print(f"precision={'bf16' if use_bf16 else 'fp32'} mode={args.mode} "
      f"in_vocab={args.in_vocab} "
      f"out_vocab={args.out_vocab} in_seq={args.in_seq} dec_seq={args.dec_seq} "
      f"d_model={args.d} heads={args.h} ff={args.ff} layers={args.layers} "
      f"batch={args.batch} steps={args.steps} xla=False")


def positional_encoding(length, depth):
    pos = np.arange(length)[:, None]
    i = np.arange(depth)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / np.float32(depth))
    pe = np.zeros((length, depth), dtype=np.float32)
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return tf.constant(pe[None], dtype=tf.float32)


L = tf.keras.layers


def ffn(x, d, ff):
    h = L.Dense(ff, activation="relu")(x)
    return L.Dense(d)(h)


def build_model():
    d, h, ff, n = args.d, args.h, args.ff, args.layers
    src = L.Input(shape=(args.in_seq,), dtype="int32", name="src")
    dec = L.Input(shape=(args.dec_seq,), dtype="int32", name="dec")

    pe_enc = positional_encoding(args.in_seq, d)
    pe_dec = positional_encoding(args.dec_seq, d)

    x = L.Embedding(args.in_vocab, d)(src) * math.sqrt(d) + pe_enc
    for _ in range(n):
        a = L.MultiHeadAttention(h, d // h)(x, x)
        x = L.LayerNormalization()(x + a)
        x = L.LayerNormalization()(x + ffn(x, d, ff))
    enc = x

    y = L.Embedding(args.out_vocab, d)(dec) * math.sqrt(d) + pe_dec
    for _ in range(n):
        sa = L.MultiHeadAttention(h, d // h)(y, y, use_causal_mask=True)
        y = L.LayerNormalization()(y + sa)
        ca = L.MultiHeadAttention(h, d // h)(y, enc)
        y = L.LayerNormalization()(y + ca)
        y = L.LayerNormalization()(y + ffn(y, d, ff))

    # keep the vocab logits in float32 under mixed precision (standard practice)
    logits = L.Dense(args.out_vocab, dtype="float32")(y)
    return tf.keras.Model([src, dec], logits)


model = build_model()
print(f"parameters={model.count_params()}")

pool = max(8, args.warmup + args.steps)
rng = np.random.default_rng(0)
src = tf.constant(rng.integers(0, args.in_vocab, (pool, args.batch, args.in_seq), dtype=np.int32))
dec = tf.constant(rng.integers(0, args.out_vocab, (pool, args.batch, args.dec_seq), dtype=np.int32))

if args.mode == "train":
    tgt = tf.constant(rng.integers(0, args.out_vocab, (pool, args.batch, args.dec_seq), dtype=np.int32))

    opt = tf.keras.optimizers.Adam(1e-4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_step(src_b, dec_b, tgt_b):
        with tf.GradientTape() as tape:
            logits = model([src_b, dec_b], training=True)
            loss = loss_fn(tgt_b, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    def one(i):
        j = i % pool
        return train_step(src[j], dec[j], tgt[j])

    for i in range(args.warmup):
        one(i)
    last = one(0)
    _ = last.numpy()   # force sync after warmup

    t0 = time.perf_counter()
    for i in range(args.steps):
        last = one(args.warmup + i)
    _ = last.numpy()   # force sync
    wall_s = time.perf_counter() - t0
    print(f"final_loss={float(last):.5f}")
else:
    # Forward-only: no GradientTape, no optimizer state. The tiny logits slice
    # returned per step forces execution and gives the finiteness probe.
    @tf.function
    def infer_step(src_b, dec_b):
        logits = model([src_b, dec_b], training=False)
        return logits[0, 0, :8]

    def one(i):
        j = i % pool
        return infer_step(src[j], dec[j])

    probe = None
    for i in range(args.warmup):
        probe = one(i)
    _ = probe.numpy()   # force sync after warmup

    t0 = time.perf_counter()
    for i in range(args.steps):
        probe = one(args.warmup + i)
    probe_host = probe.numpy()   # force sync
    wall_s = time.perf_counter() - t0

    assert np.isfinite(np.asarray(probe_host, dtype=np.float32)).all(), "non-finite logits"

samples_per_s = args.steps * args.batch / wall_s
print(f"wall_s={wall_s:.5f}")
print(f"samples_per_sec={samples_per_s:.2f}")
print(f"tokens_per_sec={samples_per_s * (args.in_seq + args.dec_seq):.2f}")
print("RESULT=OK")
