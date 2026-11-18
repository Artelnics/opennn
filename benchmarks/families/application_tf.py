# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

"""Small native-tensor applications through the standard TensorFlow/Keras Python API.

The counterpart of application.py: the same four small models, the same
constructed inputs, eager inference (no tf.function, no XLA warm-up), two
intra-op threads and one inter-op thread, and the same STARTUP_READY record.
"""

import time

entered_ns = time.monotonic_ns()
import sys
import os
import json
import math

family, device_name, precision = sys.argv[1:4]
if device_name == "cpu":
    # A CPU application: TensorFlow must not place work on a GPU it can see.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    # TF 2.21 finds every NVIDIA pip library except cuSOLVER, and without it
    # registers no GPU. Loading it by path lets TensorFlow's dlopen succeed.
    import ctypes
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    for root in (spec.submodule_search_locations if spec else []):
        candidate = os.path.join(root, "cusolver", "lib", "libcusolver.so.11")
        if os.path.exists(candidate):
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            break

import numpy as np
import tensorflow as tf
import keras

threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
tf.config.threading.set_intra_op_parallelism_threads(threads)
tf.config.threading.set_inter_op_parallelism_threads(1)
keras.utils.set_random_seed(42)
dtype = "bfloat16" if precision == "bf16" else "float32"
keras.mixed_precision.set_dtype_policy(dtype)
layers, ops = keras.layers, keras.ops
models = []


def prepare(model):
    models.append(model)
    return model


def encoder_layer(x):
    attended = layers.MultiHeadAttention(4, 8)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attended)
    transformed = layers.Dense(32)(layers.Dense(64, activation="relu")(x))
    return layers.LayerNormalization(epsilon=1e-6)(x + transformed)


def decoder_layer(x, memory):
    attended = layers.MultiHeadAttention(4, 8)(x, x, use_causal_mask=True)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attended)
    attended = layers.MultiHeadAttention(4, 8)(x, memory)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attended)
    transformed = layers.Dense(32)(layers.Dense(64, activation="relu")(x))
    return layers.LayerNormalization(epsilon=1e-6)(x + transformed)


with tf.device("/GPU:0" if device_name == "cuda" else "/CPU:0"):
    if family == "dense":
        model = prepare(keras.Sequential([
            keras.Input((28,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]))
        output = model(tf.fill((2, 28), tf.constant(0.1, dtype)), training=False)
    elif family == "lstm":
        model = prepare(keras.Sequential([
            keras.Input((8, 15)),
            layers.LSTM(128),
            layers.Dense(1),
        ]))
        output = model(tf.fill((2, 8, 15), tf.constant(0.1, dtype)), training=False)
    elif family == "cnn":
        model = prepare(keras.Sequential([
            keras.Input((32, 32, 3)),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]))
        inputs = tf.fill((2, 32, 32, 3), tf.constant(0.1, dtype))
        output = model(inputs / 255.0, training=False)
    elif family == "transformer":
        positions = np.arange(8, dtype=np.float32)[:, None]
        frequencies = np.exp(np.arange(0, 32, 2, dtype=np.float32) * (-math.log(10000.0) / 32))
        angles = positions * frequencies
        table = np.stack((np.sin(angles), np.cos(angles)), axis=-1).reshape(1, 8, 32)

        source = keras.Input((8,), dtype="int32")
        target = keras.Input((8,), dtype="int32")
        position = ops.convert_to_tensor(table)
        s = layers.Embedding(128, 32)(source) * math.sqrt(32.0) + ops.cast(position, dtype)
        t = layers.Embedding(128, 32)(target) * math.sqrt(32.0) + ops.cast(position, dtype)
        logits = layers.Dense(128)(decoder_layer(t, encoder_layer(s)))
        model = prepare(keras.Model([source, target], logits))
        tokens = tf.ones((2, 8), dtype=tf.int32)
        output = model([tokens, tokens], training=False)
    else:
        raise ValueError("Unknown architecture: " + family)
    actual_device = "cuda" if "GPU" in output.device else "cpu"
    output = np.asarray(tf.cast(output, tf.float32))
    ready_ns = time.monotonic_ns()

assert actual_device == device_name, "Unexpected output device"
assert np.isfinite(output).all(), "Nonfinite output"


def count(weights):
    return sum(int(np.prod(w.shape)) for w in weights)


parameters = sum(count(model.trainable_weights) for model in models)
stored = sum(count(model.weights) for model in models)
build = tf.sysconfig.get_build_info()
record = {
    "main_ns": entered_ns,
    "ready_ns": ready_ns,
    "engine": "tensorflow",
    "interface": "python",
    "device": actual_device,
    "precision": precision,
    "parameters": parameters,
    "stored_parameters": stored,
    "output_values": int(output.size),
    "first": float(output.flatten()[0]),
    "tensorflow_version": tf.__version__,
    "keras_version": keras.__version__,
    "python_version": sys.version.split()[0],
    "tensorflow_file": tf.__file__,
    "python_prefix": sys.prefix,
    "cuda_version": build.get("cuda_version"),
    "tf32": tf.config.experimental.tensor_float_32_execution_enabled(),
}
print("STARTUP_READY " + json.dumps(record), flush=True)

if len(sys.argv) > 4:
    from pathlib import Path

    audit = Path(sys.argv[4])
    audit.mkdir(parents=True, exist_ok=True)
    modules = []
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and Path(file).is_file():
            modules.append(
                {
                    "name": name,
                    "file": str(Path(file).resolve()),
                    "cached": getattr(module, "__cached__", None),
                }
            )
    (audit / "modules.json").write_text(json.dumps(modules, indent=2))
    (audit / "maps.txt").write_text(Path("/proc/self/maps").read_text())
    import sysconfig

    (audit / "python-runtime.json").write_text(
        json.dumps(
            {
                "executable": sys.executable,
                "base_prefix": sys.base_prefix,
                "prefix": sys.prefix,
                "paths": sysconfig.get_paths(),
                "sys_path": sys.path,
            },
            indent=2,
        )
    )
    import importlib.metadata

    packages = sorted(
        [
            {"name": d.metadata["Name"], "version": d.version}
            for d in importlib.metadata.distributions()
        ],
        key=lambda d: d["name"].lower(),
    )
    (audit / "packages.json").write_text(json.dumps(packages, indent=2))
