# TensorFlow/Keras Transformer TRAINING throughput counterpart to
# opennn_transformer_train / pytorch_transformer_train.
#
# Mirrors the OpenNN encoder-decoder Transformer training loop: same architecture
# (token embeddings + sinusoidal positional encoding, N encoder + N decoder layers,
# multi-head attention + position-wise feed-forward, post-LayerNorm, linear
# projection to vocab), same optimizer (Adam), same token cross-entropy loss over
# the vocabulary, same synthetic corpus shape (vocab / sequence length / sample
# count are read from the SAME corpus file the OpenNN side trains on, so the FLOPs
# match token-for-token). Times a fixed number of epochs after a warmup epoch and
# reports samples/sec and tokens/sec.
#
# TensorFlow's fair fast path: the timed train step is a @tf.function with XLA
# (jit_compile) and, for the bf16 comparison, a mixed-precision policy (TF's fused
# attention runs in bf16 on tensor cores). Uses a loss-scale-free bf16 setup to
# match PyTorch's plain autocast(bf16) (bf16 has fp32's exponent range, so no loss
# scaling is needed) and to keep the update math comparable across engines.
#
#   usage: python tensorflow_transformer_train.py CORPUS.txt [d_model] [heads] [ff] [layers] [batch] [epochs]
#   env:   TF_BF16=1 -> train under a mixed_bfloat16 policy (matches OpenNN OPENNN_BF16 / PyTorch PT_BF16)

import sys
import os
import math
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

corpus  = sys.argv[1] if len(sys.argv) > 1 else "synthetic_corpus.txt"
d_model = int(sys.argv[2]) if len(sys.argv) > 2 else 256
heads   = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ff      = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
layers  = int(sys.argv[5]) if len(sys.argv) > 5 else 2
batch   = int(sys.argv[6]) if len(sys.argv) > 6 else 32
epochs  = int(sys.argv[7]) if len(sys.argv) > 7 else 30

gpus = tf.config.list_physical_devices("GPU")
assert gpus, "CUDA GPU required"
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
tf.random.set_seed(0)

use_bf16 = os.environ.get("TF_BF16") is not None
if use_bf16:
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")


# --- read the SAME corpus OpenNN trains on, to match shapes exactly -------------
# OpenNN's TextDataset: input_seq = max(#input tokens) + 2 (START/END),
# decoder_seq = max(#target tokens) + 1, vocab = distinct tokens + 4 reserved.
def read_corpus(path):
    in_lens, tgt_lens, vocab = [], [], set()
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        a, _, b = line.partition("\t")
        ina, tgta = a.split(), b.split()
        in_lens.append(len(ina))
        tgt_lens.append(len(tgta))
        vocab.update(ina); vocab.update(tgta)
    input_seq = max(in_lens) + 2
    decoder_seq = max(tgt_lens) + 1
    vocab_size = len(vocab) + 4   # [PAD] [UNK] [START] [END]
    return len(in_lens), input_seq, decoder_seq, vocab_size


samples, input_seq, decoder_seq, vocab = read_corpus(corpus)
print(f"precision={'bf16' if use_bf16 else 'fp32'} samples={samples} "
      f"input_seq={input_seq} decoder_seq={decoder_seq} input_vocab={vocab} output_vocab={vocab} "
      f"d_model={d_model} heads={heads} ff={ff} layers={layers} batch={batch} epochs={epochs}")

K = tf.keras.layers


def sinusoidal_pe(length, depth):
    pos = np.arange(length)[:, None]
    i = np.arange(depth)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / depth)
    pe = np.zeros((length, depth), dtype="float32")
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return tf.constant(pe[None])


def encoder_layer(x):
    a = K.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads)(x, x)
    x = K.LayerNormalization()(x + a)
    h = K.Dense(ff, activation="relu")(x)
    h = K.Dense(d_model)(h)
    return K.LayerNormalization()(x + h)


def decoder_layer(x, mem, causal_mask):
    a = K.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads)(
        x, x, attention_mask=causal_mask)
    x = K.LayerNormalization()(x + a)
    c = K.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads)(x, mem)
    x = K.LayerNormalization()(x + c)
    h = K.Dense(ff, activation="relu")(x)
    h = K.Dense(d_model)(h)
    return K.LayerNormalization()(x + h)


def build():
    src = K.Input(shape=(input_seq,), dtype="int32")
    tgt = K.Input(shape=(decoder_seq,), dtype="int32")
    src_emb = K.Embedding(vocab, d_model)
    tgt_emb = K.Embedding(vocab, d_model)
    scale = math.sqrt(d_model)
    s = src_emb(src) * scale + sinusoidal_pe(input_seq, d_model)
    t = tgt_emb(tgt) * scale + sinusoidal_pe(decoder_seq, d_model)
    causal = tf.linalg.band_part(
        tf.ones((decoder_seq, decoder_seq), dtype=tf.bool), -1, 0)
    for _ in range(layers):
        s = encoder_layer(s)
    for _ in range(layers):
        t = decoder_layer(t, s, causal)
    # logits kept in fp32 so cross-entropy/softmax is numerically identical to
    # the fp32 path even under the bf16 policy (Keras mixed-precision convention).
    out = K.Dense(vocab, dtype="float32")(t)
    return tf.keras.Model([src, tgt], out)


with tf.device("/GPU:0"):
    model = build()
    params = model.count_params()
    print(f"parameters={params}")

    # Synthetic data matching OpenNN's corpus shape (same #samples, seq lengths,
    # vocab). Throughput is shape/FLOP bound, so matched shapes make the
    # comparison fair; the token values differ but fwd+bwd+optimizer cost is equal.
    src = tf.random.uniform((samples, input_seq), 0, vocab, dtype=tf.int32)
    dec = tf.random.uniform((samples, decoder_seq), 0, vocab, dtype=tf.int32)
    tgt = tf.random.uniform((samples, decoder_seq), 0, vocab, dtype=tf.int32)

    lr = float(os.environ.get("OPENNN_LR", "0.0001"))
    print(f"learning_rate={lr}")
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function(jit_compile=True)
    def train_step(s, d, y):
        with tf.GradientTape() as tape:
            logits = model([s, d], training=True)     # (batch, decoder_seq, vocab)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    n_batches = samples // batch

    def run_epoch():
        last = tf.constant(0.0)
        for b in range(n_batches):
            i = b * batch
            last = train_step(src[i:i + batch], dec[i:i + batch], tgt[i:i + batch])
        return float(last)

    # warmup epoch (traces + XLA-compiles the step, matches OpenNN train()'s
    # internal CUDA warmup and PyTorch's warmup epoch), excluded from timing.
    run_epoch()

    t0 = time.perf_counter()
    last = 0.0
    timed_passes = max(1, epochs)                      # OpenNN's train() runs max(1, maximum_epochs) passes
    for _ in range(timed_passes):
        last = run_epoch()
    wall_s = time.perf_counter() - t0

total_samples = n_batches * batch * timed_passes
samples_per_s = total_samples / wall_s
tokens_per_s = samples_per_s * (input_seq + decoder_seq)

print(f"final_loss={last:.5f}")
print(f"wall_s={wall_s:.5f}")
print(f"samples_per_sec={samples_per_s:.2f}")
print(f"tokens_per_sec={tokens_per_s:.2f}")
print("RESULT=OK")
