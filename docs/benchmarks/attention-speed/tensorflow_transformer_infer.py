# TensorFlow Transformer inference throughput, the counterpart to
# opennn_transformer_resident / pytorch_transformer_infer. Encoder-decoder
# Transformer ("Attention Is All You Need"): scaled token embeddings + sinusoidal
# positional encoding, N encoder + N decoder layers (multi-head attention +
# position-wise feed-forward, post-LayerNorm), linear projection to vocab.
#
# TensorFlow's fair fast path: @tf.function with XLA (jit_compile) and, for the
# bf16 comparison, a mixed-precision policy (TF's fused attention runs in bf16 on
# tensor cores). Times the steady-state forward pass after warmup; reports tok/s.
#
#   usage: python tensorflow_transformer_infer.py [seq] [d_model] [heads] [ff] [layers] [vocab] [batch] [iters]
#   env:   TF_BF16=1 -> mixed_bfloat16 policy

import sys
import os
import math
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

seq     = int(sys.argv[1]) if len(sys.argv) > 1 else 64
d_model = int(sys.argv[2]) if len(sys.argv) > 2 else 512
heads   = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ff      = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
layers  = int(sys.argv[5]) if len(sys.argv) > 5 else 6
vocab   = int(sys.argv[6]) if len(sys.argv) > 6 else 10000
batch   = int(sys.argv[7]) if len(sys.argv) > 7 else 8
iters   = int(sys.argv[8]) if len(sys.argv) > 8 else 50

gpus = tf.config.list_physical_devices("GPU")
assert gpus, "CUDA GPU required"
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
tf.random.set_seed(0)

use_bf16 = os.environ.get("TF_BF16") is not None
if use_bf16:
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
print(f"precision={'bf16' if use_bf16 else 'fp32'}")

K = tf.keras.layers


def sinusoidal_pe(length, depth):
    pos = np.arange(length)[:, None]
    i = np.arange(depth)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / depth)
    pe = np.zeros((length, depth), dtype="float32")
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return tf.constant(pe[None])


def encoder_layer(x, pe_unused):
    a = K.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads)(x, x)
    x = K.LayerNormalization()(x + a)
    h = K.Dense(ff, activation="relu")(x)
    h = K.Dense(d_model)(h)
    return K.LayerNormalization()(x + h)


def decoder_layer(x, mem):
    a = K.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads)(x, x)
    x = K.LayerNormalization()(x + a)
    c = K.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads)(x, mem)
    x = K.LayerNormalization()(x + c)
    h = K.Dense(ff, activation="relu")(x)
    h = K.Dense(d_model)(h)
    return K.LayerNormalization()(x + h)


def build():
    src = K.Input(shape=(seq,), dtype="int32")
    tgt = K.Input(shape=(seq,), dtype="int32")
    src_emb = K.Embedding(vocab, d_model)
    tgt_emb = K.Embedding(vocab, d_model)
    scale = math.sqrt(d_model)
    pe = sinusoidal_pe(seq, d_model)
    s = src_emb(src) * scale + pe
    t = tgt_emb(tgt) * scale + pe
    for _ in range(layers):
        s = encoder_layer(s, pe)
    for _ in range(layers):
        t = decoder_layer(t, s)
    out = K.Dense(vocab)(t)
    return tf.keras.Model([src, tgt], out)


with tf.device("/GPU:0"):
    model = build()
    print(f"config seq={seq} d_model={d_model} heads={heads} ff={ff} layers={layers} vocab={vocab} batch={batch}")
    print(f"parameters={model.count_params()}")
    src = tf.random.uniform((batch, seq), 0, vocab, dtype=tf.int32)
    tgt = tf.random.uniform((batch, seq), 0, vocab, dtype=tf.int32)

    @tf.function(jit_compile=True)
    def fwd():
        return model([src, tgt], training=False)

    fwd()  # warmup (trace + compile)
    _ = fwd().numpy()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fwd()
    _ = out.numpy()
    per = (time.perf_counter() - t0) / iters

tokens = batch * seq
print(f"step_s={per:.6f}")
print(f"tokens_per_sec={int(tokens / per)}")
print("RESULT=OK")
