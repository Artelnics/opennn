#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

"""TensorFlow/Keras counterpart of quality.cpp and quality.py; scoring lives in
quality_runner.py.

  quality_tf.py manifest out epochs batch seed cpu|cuda fp32|bf16 learning_rate

Training is Keras `compile()` + `fit()` over the shared prepared tensors, in
file order, every sample including the tail batch, Adam with beta1 .9, beta2
.999 and float32 epsilon, no regularization, clipping, early stopping or
warm-up -- the rules of PROTOCOL.md section 13. TF32 is disabled, as in the
other drivers. Initialization is Keras's own: Glorot-uniform with zero biases
for dense layers (the protocol's dense rule), and Keras's LSTM, convolution
and attention defaults, which the protocol leaves to each engine.

The transformer is quality.py's: post-norm encoder-decoder, ReLU, dropout 0,
LayerNorm epsilon 1e-6, source padding masked in encoder and cross attention,
causal plus padding mask in decoder self-attention, cross-entropy averaged
over non-PAD target tokens, and greedy generation through EOS or the fixed
length for the test predictions.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import tensorflow_engine as engine  # noqa: E402
from quality_data import read_array  # noqa: E402

LAYER_NORM_EPSILON = 1.0e-6


def translator_class():
    keras = engine.keras
    layers, ops = keras.layers, keras.ops

    class FeedForward(layers.Layer):
        def __init__(self, width, ff, **kwargs):
            super().__init__(**kwargs)
            self.inner = layers.Dense(ff, activation="relu")
            self.outer = layers.Dense(width)

        def call(self, x):
            return self.outer(self.inner(x))

    class EncoderLayer(layers.Layer):
        def __init__(self, width, heads, ff, **kwargs):
            super().__init__(**kwargs)
            self.attention = layers.MultiHeadAttention(heads, width // heads)
            self.feed_forward = FeedForward(width, ff)
            self.norm1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)
            self.norm2 = layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)

        def call(self, x, attention_mask=None):
            x = self.norm1(x + self.attention(x, x, attention_mask=attention_mask))
            return self.norm2(x + self.feed_forward(x))

    class DecoderLayer(layers.Layer):
        def __init__(self, width, heads, ff, **kwargs):
            super().__init__(**kwargs)
            self.self_attention = layers.MultiHeadAttention(heads, width // heads)
            self.cross_attention = layers.MultiHeadAttention(heads, width // heads)
            self.feed_forward = FeedForward(width, ff)
            self.norm1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)
            self.norm2 = layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)
            self.norm3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)

        def call(self, x, memory, self_mask=None, memory_mask=None):
            x = self.norm1(x + self.self_attention(x, x, attention_mask=self_mask))
            x = self.norm2(x + self.cross_attention(x, memory, attention_mask=memory_mask))
            return self.norm3(x + self.feed_forward(x))

    def pair_mask(query_tokens, key_keep):
        """(B, T) queries x (B, S) attendable keys -> (B, T, S)."""
        queries = ops.ones_like(query_tokens, dtype="bool")
        return ops.logical_and(ops.expand_dims(queries, -1), ops.expand_dims(key_keep, 1))

    class Translator(keras.Model):
        def __init__(self, vocabulary, cfg, **kwargs):
            super().__init__(**kwargs)
            width, heads, ff = cfg["d_model"], cfg["heads"], cfg["ff"]
            self.sequence = cfg["sequence"]
            self.source_embedding = layers.Embedding(vocabulary, width)
            self.target_embedding = layers.Embedding(vocabulary, width)
            self.encoders = [EncoderLayer(width, heads, ff) for _ in range(cfg["layers"])]
            self.decoders = [DecoderLayer(width, heads, ff) for _ in range(cfg["layers"])]
            self.projection = layers.Dense(vocabulary)
            position = np.arange(self.sequence, dtype=np.float32)[:, None]
            divisor = np.exp(np.arange(0, width, 2, dtype=np.float32)
                             * (-math.log(10000.0) / width))
            table = np.zeros((self.sequence, width), dtype=np.float32)
            table[:, 0::2], table[:, 1::2] = np.sin(position * divisor), np.cos(position * divisor)
            self.table = table[None]
            self.scale = math.sqrt(width)
            self.causal = np.tril(np.ones((self.sequence, self.sequence), dtype=bool))[None]

        def call(self, inputs):
            source, decoder = inputs
            source = ops.cast(source, "int32")
            decoder = ops.cast(decoder, "int32")
            source_keep = ops.not_equal(source, 0)
            decoder_keep = ops.not_equal(decoder, 0)

            s = self.source_embedding(source) * self.scale
            s = s + ops.cast(self.table, s.dtype)
            d = self.target_embedding(decoder) * self.scale
            d = d + ops.cast(self.table, d.dtype)

            encoder_mask = pair_mask(source, source_keep)
            for layer in self.encoders:
                s = layer(s, attention_mask=encoder_mask)

            self_mask = ops.logical_and(pair_mask(decoder, decoder_keep), self.causal)
            memory_mask = pair_mask(decoder, source_keep)
            for layer in self.decoders:
                d = layer(d, s, self_mask=self_mask, memory_mask=memory_mask)
            return self.projection(d)

    return Translator


def build(manifest):
    keras = engine.keras
    kind, cfg = manifest["model"], manifest["profile"]
    train = manifest["train"]

    if kind == "dense":
        width = train["x"]["shape"][-1]
        model = keras.Sequential(
            [keras.Input((width,))]
            + [keras.layers.Dense(cfg["hidden"], activation="relu") for _ in range(cfg["layers"])]
            + [keras.layers.Dense(1)])
        return model
    if kind == "lstm":
        return keras.Sequential([
            keras.Input(tuple(train["x"]["shape"][1:])),
            keras.layers.LSTM(cfg["hidden"]),
            keras.layers.Dense(1),
        ])
    if kind == "cnn":
        import cnn_tf

        return cnn_tf.build(train["x"]["shape"][1], tuple(cfg["blocks"]),
                            train["y"]["shape"][-1])

    model = translator_class()(len(manifest["vocabulary"]), cfg)
    sequence = cfg["sequence"]
    dummy = np.ones((1, sequence), dtype=np.float32)
    model([dummy, dummy])
    return model


def loss_for(kind: str):
    losses = engine.keras.losses
    if kind == "dense":
        return losses.BinaryCrossentropy(from_logits=True)
    if kind == "lstm":
        return losses.MeanSquaredError()
    if kind == "cnn":
        return losses.CategoricalCrossentropy(from_logits=True)
    # Averaged over non-PAD target tokens: the PAD mask arrives as sample weights.
    return losses.SparseCategoricalCrossentropy(from_logits=True,
                                               reduction="mean_with_sample_weight")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 8:
        raise ValueError(
            "usage: quality_tf.py manifest out epochs batch seed cpu|cuda fp32|bf16 learning_rate"
        )
    manifest_path, out = Path(argv[0]), Path(argv[1])
    epochs, batch, seed = map(int, argv[2:5])
    device, precision = argv[5:7]
    rate = float(argv[7])
    if (
        epochs < 1
        or batch < 1
        or device not in ("cpu", "cuda")
        or precision not in ("fp32", "bf16")
    ):
        raise ValueError("Invalid quality options")
    if device == "cpu" and precision != "fp32":
        raise ValueError("CPU quality uses FP32")

    os.environ.setdefault("TF_NUM_INTRAOP_THREADS",
                          os.environ.get("OMP_NUM_THREADS", "2"))
    tf, keras = engine.start(device, "strict")       # TF32 off, as in quality.py
    engine.set_policy(precision, "train")
    keras.utils.set_random_seed(seed)

    manifest = json.loads(manifest_path.read_text())
    kind = manifest["model"]
    data = {
        split: {
            key: read_array(manifest_path.parent, spec)
            for key, spec in manifest[split].items()
        }
        for split in ("train", "test")
    }
    out.mkdir(parents=True, exist_ok=True)
    place = tf.device(engine.device_name(device))

    def inputs(split, start, end):
        x = np.array(data[split]["x"][start:end], dtype=np.float32)
        return x / 255.0 if kind == "cnn" else x

    class Batches(keras.utils.PyDataset):
        """The prepared training split in file order, every sample, tail included."""

        def __init__(self):
            super().__init__()
            self.count = len(data["train"]["x"])

        def __len__(self):
            return math.ceil(self.count / batch)

        def __getitem__(self, index):
            start = index * batch
            end = min(start + batch, self.count)
            x = inputs("train", start, end)
            y = np.array(data["train"]["y"][start:end], dtype=np.float32)
            if kind != "transformer":
                return x, y
            decoder = np.array(data["train"]["decoder"][start:end], dtype=np.float32)
            return (x, decoder), y.astype(np.int32), (y != 0).astype(np.float32)

    class History(keras.callbacks.Callback):
        def __init__(self, stream):
            super().__init__()
            self.stream = stream

        def on_epoch_end(self, epoch, logs=None):
            loss = float((logs or {}).get("loss", float("nan")))
            if not math.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            self.stream.write(f"{epoch + 1},{loss:.9g}\n")
            self.stream.flush()
            print(f"epoch={epoch + 1} training_loss={loss:.9g}", flush=True)

    with place:
        model = build(manifest)
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=rate, beta_1=0.9, beta_2=0.999,
                epsilon=float(np.finfo(np.float32).eps)),
            loss=loss_for(kind))
        print(f"execution={engine.execution(model)}", flush=True)

        with (out / "history.csv").open("w") as history:
            history.write("epoch,training_loss\n")
            model.fit(Batches(), epochs=epochs, verbose=0, callbacks=[History(history)])

        forward = engine.predict_step(model)

        @tf.function(jit_compile=model.jit_compile, reduce_retracing=True)
        def generate_step(x, decoder):
            logits = model([x, decoder], training=False)
            return (tf.argmax(logits, axis=-1, output_type=tf.int32),
                    tf.reduce_all(tf.math.is_finite(tf.cast(logits, tf.float32))))

        with (out / "predictions.bin").open("wb") as predictions:
            for start in range(0, len(data["test"]["x"]), batch):
                end = min(start + batch, len(data["test"]["x"]))
                x = inputs("test", start, end)
                if kind == "transformer":
                    sequence = manifest["profile"]["sequence"]
                    decoder = np.zeros((end - start, sequence), dtype=np.float32)
                    decoder[:, 0] = 2
                    output = np.zeros_like(decoder)
                    done = np.zeros(end - start, dtype=bool)
                    for position in range(sequence):
                        tokens, finite = generate_step(x, decoder)
                        if not bool(finite):
                            raise RuntimeError("Nonfinite generation logits")
                        token = tokens.numpy()[:, position]
                        output[:, position] = np.where(done, 0, token)
                        if position + 1 < sequence:
                            decoder[:, position + 1] = np.where(done, 0, token)
                        done |= token == 3
                        if done.all():
                            break
                else:
                    output = tf.cast(forward(x), tf.float32)
                    if kind == "dense":
                        output = tf.sigmoid(output)
                    elif kind == "cnn":
                        output = tf.nn.softmax(output, axis=-1)
                    output = output.numpy()
                if not np.isfinite(output).all():
                    raise RuntimeError("Nonfinite prediction")
                np.asarray(output, dtype=np.float32).astype("<f4").tofile(predictions)

    record = {
        "engine": "tensorflow",
        "interface": "python",
        "device": device,
        "precision": precision,
        "model": kind,
        "seed": seed,
        "epochs": epochs,
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "execution": engine.execution(model),
        "parameters": engine.trainable_parameters(model),
        "train_samples": len(data["train"]["x"]),
        "test_samples": len(data["test"]["x"]),
        "status": "ok",
    }
    (out / "driver.json").write_text(json.dumps(record, indent=2))
    print("RESULT=OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
