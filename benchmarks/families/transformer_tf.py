#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

"""The transformer family in TensorFlow, the counterpart of transformer.cpp and
transformer.py.

Encoder-decoder, d_model 512, 8 heads, feed-forward 2048, 6 layers -- the
"Attention Is All You Need" base model -- with scaled token embeddings,
sinusoidal positions and a final projection to the vocabulary.

  transformer_tf.py train    <corpus> <corpus> [epochs] [batch,...] [d_model] [layers] [dev] [prec]
  transformer_tf.py infer    <corpus>          [reps]   [batch,...] [d_model] [layers] [dev] [prec]
  transformer_tf.py capacity <corpus>          [batch]              [d_model] [layers] [dev] [prec]

Keras has no encoder or decoder layer of its own, so both are assembled from
`keras.layers.MultiHeadAttention`, `LayerNormalization` and `Dense`, in the
post-norm arrangement nn.TransformerEncoderLayer/DecoderLayer use, with no
final stack norm (transformer.py passes norm=None) and no attention mask
(transformer.py passes none). The parameter count is therefore identical.

Dropout is 0, which is OpenNN's transformer and the default of the official
Keras transformer blocks (KerasHub). transformer.py keeps nn.Transformer*'s
default of 0.1. TF_DROPOUT=0.1 is the controlled variant that matches it.

Tokenisation is transformer.py's: whitespace tokens ranked by frequency,
capped at 20,000, with OpenNN's PAD/UNK/START/END ids, so the padded shape
matches both other engines.
"""

from __future__ import annotations

import collections
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import tensorflow_engine as engine  # noqa: E402

VOCAB_CAP = 20_000
DROPOUT = float(os.environ.get("TF_DROPOUT", "0"))
LAYER_NORM_EPSILON = 1.0e-5     # nn.LayerNorm's; the Keras default is 1e-3

def positional_table(length: int, d_model: int) -> np.ndarray:
    """transformer.py's PositionalEncoding, as a constant."""
    position = np.arange(length, dtype=np.float32)[:, None]
    divisor = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model))
    table = np.zeros((length, d_model), dtype=np.float32)
    table[:, 0::2] = np.sin(position * divisor)
    table[:, 1::2] = np.cos(position * divisor)
    return table[None]

def layer_classes():
    """Keras layer classes, defined once Keras is imported."""
    keras = engine.keras
    layers, ops = keras.layers, keras.ops

    class Embedding(layers.Layer):
        """Token embedding scaled by sqrt(d_model) plus sinusoidal positions."""

        def __init__(self, vocab: int, d_model: int, sequence: int, **kwargs):
            super().__init__(**kwargs)
            self.embedding = layers.Embedding(vocab, d_model)
            self.scale = math.sqrt(d_model)
            self.table = positional_table(sequence + 1, d_model)

        def build(self, input_shape):
            self.embedding.build(input_shape)

        def call(self, tokens):
            x = self.embedding(tokens) * self.scale
            length = ops.shape(tokens)[1]
            return x + ops.cast(self.table[:, :length], x.dtype)

    class FeedForward(layers.Layer):
        def __init__(self, d_model: int, width: int, **kwargs):
            super().__init__(**kwargs)
            self.inner = layers.Dense(width, activation="relu")
            self.dropout = layers.Dropout(DROPOUT)
            self.outer = layers.Dense(d_model)

        def build(self, input_shape):
            self.inner.build(input_shape)
            self.outer.build((*input_shape[:-1], self.inner.units))

        def call(self, x, training=None):
            return self.outer(self.dropout(self.inner(x), training=training))

    class EncoderLayer(layers.Layer):
        """nn.TransformerEncoderLayer, norm_first=False, ReLU."""

        def __init__(self, d_model: int, heads: int, width: int, **kwargs):
            super().__init__(**kwargs)
            self.attention = layers.MultiHeadAttention(heads, d_model // heads, dropout=DROPOUT)
            self.feed_forward = FeedForward(d_model, width)
            self.norm1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)
            self.norm2 = layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)
            self.dropout1 = layers.Dropout(DROPOUT)
            self.dropout2 = layers.Dropout(DROPOUT)

        def build(self, input_shape):
            self.attention.build(input_shape, input_shape)
            self.feed_forward.build(input_shape)
            self.norm1.build(input_shape)
            self.norm2.build(input_shape)

        def call(self, x, training=None):
            attended = self.attention(x, x, training=training)
            x = self.norm1(x + self.dropout1(attended, training=training))
            transformed = self.feed_forward(x, training=training)
            return self.norm2(x + self.dropout2(transformed, training=training))

    class DecoderLayer(layers.Layer):
        """nn.TransformerDecoderLayer, norm_first=False, ReLU, no mask."""

        def __init__(self, d_model: int, heads: int, width: int, **kwargs):
            super().__init__(**kwargs)
            self.self_attention = layers.MultiHeadAttention(heads, d_model // heads,
                                                            dropout=DROPOUT)
            self.cross_attention = layers.MultiHeadAttention(heads, d_model // heads,
                                                             dropout=DROPOUT)
            self.feed_forward = FeedForward(d_model, width)
            self.norm1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)
            self.norm2 = layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)
            self.norm3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)
            self.dropout1 = layers.Dropout(DROPOUT)
            self.dropout2 = layers.Dropout(DROPOUT)
            self.dropout3 = layers.Dropout(DROPOUT)

        def build(self, x_shape, memory_shape):
            self.self_attention.build(x_shape, x_shape)
            self.cross_attention.build(x_shape, memory_shape)
            self.feed_forward.build(x_shape)
            for norm in (self.norm1, self.norm2, self.norm3):
                norm.build(x_shape)

        def call(self, x, memory, training=None):
            attended = self.self_attention(x, x, training=training)
            x = self.norm1(x + self.dropout1(attended, training=training))
            attended = self.cross_attention(x, memory, training=training)
            x = self.norm2(x + self.dropout2(attended, training=training))
            transformed = self.feed_forward(x, training=training)
            return self.norm3(x + self.dropout3(transformed, training=training))

    return Embedding, EncoderLayer, DecoderLayer

def build(vocab: int, sequence: int, opts: dict):
    """The transformer family. Nothing else here constructs the network."""
    keras = engine.keras
    engine.seed()
    Embedding, EncoderLayer, DecoderLayer = layer_classes()

    d_model, layers = opts["d_model"], opts["layers"]
    heads = max(1, d_model // 64)

    source = keras.Input((sequence,), dtype="int32", name="source")
    target = keras.Input((sequence,), dtype="int32", name="target")

    memory = Embedding(vocab, d_model, sequence)(source)
    for _ in range(layers):
        memory = EncoderLayer(d_model, heads, 4 * d_model)(memory)

    x = Embedding(vocab, d_model, sequence)(target)
    for _ in range(layers):
        x = DecoderLayer(d_model, heads, 4 * d_model)(x, memory)

    logits = keras.layers.Dense(vocab)(x)
    return keras.Model([source, target], logits)

def compile_for_training(model) -> None:
    keras = engine.keras
    # transformer.py's optimiser: Adam, lr 1e-4, torch's default epsilon.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1.0e-4, epsilon=1.0e-8),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

def load_corpus(path: str, sequence: int | None = None) -> tuple[np.ndarray, np.ndarray, int, int]:
    """`source <TAB> target` to padded id arrays -- transformer.py's tokenisation."""
    sources, targets = [], []
    counter: collections.Counter = collections.Counter()

    engine.report_opened(path)

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            source, target = parts[0].split(), parts[1].split()
            counter.update(source)
            counter.update(target)
            sources.append(source)
            targets.append(target)

    PAD, UNK, START, END = 0, 1, 2, 3
    vocabulary = {token: index + 4
                  for index, (token, _) in enumerate(counter.most_common(VOCAB_CAP - 4))}

    longest = max(max(len(s) for s in sources), max(len(t) for t in targets))
    length = sequence or longest + 2

    def encode(rows):
        array = np.full((len(rows), length), PAD, dtype=np.int32)
        for row_index, row in enumerate(rows):
            ids = [START] + [vocabulary.get(t, UNK) for t in row[: length - 2]] + [END]
            array[row_index, : len(ids)] = ids
        return array

    return encode(sources), encode(targets), VOCAB_CAP, length

def parse_opts(argv: list[str], first: int) -> dict:
    def at(index: int, default: str) -> str:
        return argv[index] if len(argv) > index else default

    device = at(first + 2, "cuda")
    precision = at(first + 3, "fp32")
    engine.start(device, precision)

    return {"d_model": int(at(first, "512")), "layers": int(at(first + 1, "6")),
            "device": device, "precision": precision}

def header(mode: str, opts: dict) -> None:
    engine.report_baselines()
    print(f"engine=tensorflow\nmode={mode}\ndevice={opts['device']}")
    engine.report_blas()
    print(f"dropout={DROPOUT}")

def report(batch: int, seconds: list[float], sequences: int, tokens_per_sequence: int) -> None:
    seconds = sorted(seconds)
    median = seconds[len(seconds) // 2]
    print(f"batch_{batch}_samples_per_sec={int(sequences / median)}"
          f" median_epoch_s={median:.6g}")
    print(f"batch_{batch}_tokens_per_sec={int(sequences * tokens_per_sequence / median)}")
    print(f"batch_{batch}_epoch_times={','.join(f'{t:.6g}' for t in seconds)}", flush=True)

def train_like(argv: list[str], mode: str) -> int:
    epochs = int(argv[4]) if len(argv) > 4 else 1
    batches = engine.batches_of(argv[5] if len(argv) > 5 else "32")
    opts = parse_opts(argv, 6)
    policy = engine.set_policy(opts["precision"], "train")

    header(mode, opts)
    print(f"dtype_policy={policy}")
    source, target, vocab, sequence = load_corpus(argv[2])
    print(f"samples={source.shape[0]} sequence={sequence} "
          f"input_vocab={vocab} target_vocab={vocab} "
          f"d_model={opts['d_model']} layers={opts['layers']}")
    warmup = 1 if mode == "train" else 0

    for batch in batches:
        model = build(vocab, sequence, opts)
        if batch == batches[0]:
            print(f"parameters={engine.trainable_parameters(model)}", flush=True)
        compile_for_training(model)

        whole = (source.shape[0] // batch) * batch
        timer = engine.epoch_timer(warmup)
        with engine.tf.device(engine.device_name(opts["device"])):
            model.fit([source[:whole], target[:whole]], target[:whole], batch_size=batch,
                      epochs=warmup + epochs, shuffle=False, verbose=0, callbacks=[timer])

        print(f"batch_{batch}_mode={engine.execution(model)}")
        report(batch, timer.times, whole, sequence)

    print("RESULT=OK")
    return 0

def infer(argv: list[str]) -> int:
    reps = int(argv[3]) if len(argv) > 3 else 1
    batches = engine.batches_of(argv[4] if len(argv) > 4 else "32")
    opts = parse_opts(argv, 5)
    policy = engine.set_policy(opts["precision"], "infer")
    tf = engine.tf

    header("infer", opts)
    print(f"dtype_policy={policy}")
    source, target, vocab, sequence = load_corpus(argv[2])
    print(f"samples={source.shape[0]} sequence={sequence} "
          f"input_vocab={vocab} target_vocab={vocab}")

    for batch in batches:
        model = build(vocab, sequence, opts)
        model.compile()
        if batch == batches[0]:
            print(f"parameters={engine.trainable_parameters(model)}", flush=True)
        forward = engine.predict_step(model)
        processed = (source.shape[0] // batch) * batch

        with tf.device(engine.device_name(opts["device"])):
            s, t = tf.constant(source[:batch]), tf.constant(target[:batch])

        def run_pass():
            with tf.device(engine.device_name(opts["device"])):
                for _ in range(processed // batch):
                    forward(s, t)
            engine.sync()

        run_pass()

        print(f"TIMED_START_UNIX={time.time():.3f}", flush=True)
        times = []
        for _ in range(reps):
            mark = time.perf_counter()
            run_pass()
            times.append(time.perf_counter() - mark)
        print(f"TIMED_END_UNIX={time.time():.3f}", flush=True)

        print(f"batch_{batch}_mode={engine.execution(model)}")
        report(batch, times, processed, sequence)

    print("RESULT=OK")
    return 0

def capacity(argv: list[str]) -> int:
    batch = int(argv[3]) if len(argv) > 3 else 32
    opts = parse_opts(argv, 4)
    engine.set_policy(opts["precision"], "train")

    header("capacity", opts)
    print(f"batch={batch}")

    try:
        source, target, vocab, sequence = load_corpus(argv[2])
        model = build(vocab, sequence, opts)
        print(f"parameters={engine.trainable_parameters(model)}", flush=True)
        compile_for_training(model)
        with engine.tf.device(engine.device_name(opts["device"])):
            model.train_on_batch([source[:batch], target[:batch]], target[:batch])
        engine.sync()
    except Exception as error:      # noqa: BLE001 -- only allocation failures are OOM
        if not engine.is_oom(error):
            raise
        return engine.oom_exit(error)

    print("fits=1\nRESULT=OK", flush=True)
    return 0

def main() -> int:
    argv = sys.argv
    mode = argv[1] if len(argv) > 1 else ""

    if mode in ("train", "quality") and len(argv) > 3:
        return train_like(argv, mode)
    if mode == "infer" and len(argv) > 2:
        return infer(argv)
    if mode == "capacity" and len(argv) > 2:
        return capacity(argv)

    return engine.usage(__doc__)

if __name__ == "__main__":
    raise SystemExit(main())
