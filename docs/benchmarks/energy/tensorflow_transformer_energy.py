# TensorFlow/Keras energy-to-target counterpart to opennn_transformer_energy.
# Same model (encoder-decoder Transformer, scaled token embeddings + sinusoidal
# positional encoding, MHA + FFN + post-LayerNorm, linear vocab projection),
# same token ids (OpenNN's tokens.bin cache), same gate (epoch-mean token
# cross-entropy over non-PAD targets <= target), in TensorFlow's fastest
# configuration: mixed_bfloat16 + @tf.function with XLA (TF_XLA=0 to disable),
# and the attention masks OpenNN also applies (PAD keys masked everywhere,
# causal decoder self-attention).
#
#   usage: python tensorflow_transformer_energy.py --tokens-bin F --in-seq S \
#              --dec-seq S --in-vocab V --out-vocab V --target T [--batch B]
#              [--max-epochs N] [--lr LR] [--d 512] [--h 8] [--ff 2048] [--layers 6]
#   env:   TF_BF16=1 -> mixed_bfloat16;  TF_XLA=0 -> disable XLA (default on)

import argparse, math, os, time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--tokens-bin", required=True)
ap.add_argument("--in-seq", type=int, required=True)
ap.add_argument("--dec-seq", type=int, required=True)
ap.add_argument("--in-vocab", type=int, required=True)
ap.add_argument("--out-vocab", type=int, required=True)
ap.add_argument("--target", type=float, required=True)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--max-epochs", type=int, default=40)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--d", type=int, default=512)
ap.add_argument("--h", type=int, default=8)
ap.add_argument("--ff", type=int, default=2048)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

assert tf.config.list_physical_devices("GPU"), "CUDA GPU required"
tf.random.set_seed(args.seed)

use_bf16 = os.environ.get("TF_BF16") is not None
use_xla = os.environ.get("TF_XLA", "1") != "0"
if use_bf16:
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

START_INDEX = 2

records = np.fromfile(args.tokens_bin, dtype=np.int32)
record_len = args.in_seq + args.dec_seq
assert records.size % record_len == 0, \
    f"tokens.bin size {records.size} not divisible by record {record_len}"
records = records.reshape(-1, record_len)
n_samples = records.shape[0]

src_np = records[:, :args.in_seq]
tgt_np = records[:, args.in_seq:]
dec_np = np.concatenate([np.full((n_samples, 1), START_INDEX, dtype=np.int32),
                         tgt_np[:, :-1]], axis=1)

src_all = tf.constant(src_np)
dec_all = tf.constant(dec_np)
tgt_all = tf.constant(tgt_np)

print(f"precision={'bf16' if use_bf16 else 'fp32'} xla={use_xla} "
      f"samples={n_samples} in_seq={args.in_seq} dec_seq={args.dec_seq} "
      f"in_vocab={args.in_vocab} out_vocab={args.out_vocab} "
      f"target={args.target} batch={args.batch} max_epochs={args.max_epochs} "
      f"lr={args.lr} d_model={args.d} heads={args.h} ff={args.ff} layers={args.layers}",
      flush=True)


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
    d, h, ff_dim, n = args.d, args.h, args.ff, args.layers
    src = L.Input(shape=(args.in_seq,), dtype="int32", name="src")
    dec = L.Input(shape=(args.dec_seq,), dtype="int32", name="dec")

    # PAD keys masked out, as in OpenNN's attention (mask=True means attend).
    src_keys = L.Lambda(lambda t: tf.not_equal(t, 0)[:, None, :])(src)
    dec_keys = L.Lambda(lambda t: tf.not_equal(t, 0)[:, None, :])(dec)

    pe_enc = positional_encoding(args.in_seq, d)
    pe_dec = positional_encoding(args.dec_seq, d)

    x = L.Embedding(args.in_vocab, d,
                    embeddings_initializer="glorot_uniform")(src) * math.sqrt(d) + pe_enc
    for _ in range(n):
        a = L.MultiHeadAttention(h, d // h)(x, x, attention_mask=src_keys)
        x = L.LayerNormalization()(x + a)
        x = L.LayerNormalization()(x + ffn(x, d, ff_dim))
    enc = x

    y = L.Embedding(args.out_vocab, d,
                    embeddings_initializer="glorot_uniform")(dec) * math.sqrt(d) + pe_dec
    for _ in range(n):
        sa = L.MultiHeadAttention(h, d // h)(y, y, attention_mask=dec_keys,
                                             use_causal_mask=True)
        y = L.LayerNormalization()(y + sa)
        ca = L.MultiHeadAttention(h, d // h)(y, enc, attention_mask=src_keys)
        y = L.LayerNormalization()(y + ca)
        y = L.LayerNormalization()(y + ffn(y, d, ff_dim))

    # keep the vocab logits in float32 under mixed precision (standard practice)
    logits = L.Dense(args.out_vocab, dtype="float32")(y)
    return tf.keras.Model([src, dec], logits)


model = build_model()

# Match OpenNN's init: Dense/MHA kernels are already glorot_uniform with zero
# biases (Keras defaults); embeddings set above; zero the PAD embedding row.
for layer in model.layers:
    if isinstance(layer, L.Embedding):
        w = layer.embeddings.numpy()
        w[0] = 0.0
        layer.embeddings.assign(w)

print(f"parameters={model.count_params()}", flush=True)

opt = tf.keras.optimizers.Adam(args.lr)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                        reduction="none")


@tf.function(jit_compile=use_xla)
def train_step(src_b, dec_b, tgt_b):
    with tf.GradientTape() as tape:
        logits = model([src_b, dec_b], training=True)
        per_token = loss_fn(tgt_b, logits)
        mask = tf.cast(tgt_b > 0, per_token.dtype)
        # mean over non-PAD targets, as OpenNN's CrossEntropyError3d
        loss = tf.reduce_sum(per_token * mask) / tf.reduce_sum(mask)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss


rng = np.random.default_rng(args.seed)

# The energy window starts here: it includes the XLA/graph compilation, exactly
# as OpenNN's window includes its in-train() warmup and graph capture.
print(f"TRAIN_START_UNIX={time.time():.3f}", flush=True)
t0 = time.perf_counter()

loss_history = []
reached = False
epochs_run = 0

for epoch in range(args.max_epochs + 1):
    perm = rng.permutation(n_samples)
    batch_losses = []
    for start in range(0, n_samples, args.batch):
        idx = perm[start:start + args.batch]
        batch_losses.append(train_step(tf.gather(src_all, idx),
                                       tf.gather(dec_all, idx),
                                       tf.gather(tgt_all, idx)))

    # single host sync per epoch (a per-step float() would stall the pipeline)
    mean_loss = float(tf.add_n(batch_losses)) / len(batch_losses)
    loss_history.append(mean_loss)
    epochs_run = epoch
    print(f"epoch={epoch} loss={mean_loss:.6f} elapsed={time.perf_counter() - t0:.1f}s",
          flush=True)
    if mean_loss < args.target:
        reached = True
        break

wall_s = time.perf_counter() - t0
print(f"TRAIN_END_UNIX={time.time():.3f}", flush=True)

print("loss_history=" + ",".join(f"{v:.6f}" for v in loss_history))
print(f"epochs={epochs_run}")
print(f"final_error={loss_history[-1]:.6f}")
print(f"reached_goal={1 if reached else 0}")
print(f"wall_s={wall_s:.3f}")
print(f"samples_per_sec={n_samples * (epochs_run + 1) / wall_s:.2f}")
print("RESULT=OK")
