"""Optimized TensorFlow/Keras RNN/LSTM forecasting training benchmark.

The batch-size, seed-count and fixed-work controls are shared with the OpenNN
driver through OPENNN_FORECASTING_{BATCH_SIZES,SEEDS,EPOCHS}. The Keras path
uses its native compiled fit loop, cuDNN-compatible layers and one execution
call per epoch to minimize Python dispatch overhead.
"""

import math
import os
import statistics
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from xf_common import SCENARIOS, make_windows


def env_ints(name, default):
    text = os.environ.get(name, "").strip()
    if not text:
        return list(default)
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive comma-separated integers")
    return values


ALLOW_CPU = "--allow-cpu" in sys.argv[1:] or os.environ.get("CUDA_VISIBLE_DEVICES") == ""
GPUS = tf.config.list_physical_devices("GPU")
if not GPUS and not ALLOW_CPU:
    print("ERROR device_mismatch engine=tensorflow expected=cuda actual=cpu "
          "(install tensorflow[and-cuda], or pass --allow-cpu / "
          "CUDA_VISIBLE_DEVICES=\"\" for a deliberate CPU run)", file=sys.stderr)
    sys.exit(2)
for gpu in GPUS:
    tf.config.experimental.set_memory_growth(gpu, True)

PHASE = "GPU" if GPUS else "CPU"
DEV = "cuda" if GPUS else "cpu"
PRECISION = os.environ.get("OPENNN_FORECASTING_PRECISION", "fp32").strip().lower()
if PRECISION not in ("fp32", "bf16"):
    raise ValueError("OPENNN_FORECASTING_PRECISION must be fp32 or bf16")
if PRECISION == "bf16" and not GPUS:
    raise ValueError("BF16 forecasting benchmark currently requires CUDA")
tf.keras.mixed_precision.set_global_policy(
    "mixed_bfloat16" if PRECISION == "bf16" else "float32")
SEEDS = list(range(min(env_ints("OPENNN_FORECASTING_SEEDS", [5])[0], 5)))
FIXED_EPOCHS = int(os.environ.get("OPENNN_FORECASTING_EPOCHS", "0"))
JIT_TEXT = os.environ.get("OPENNN_FORECASTING_TF_JIT", "auto").lower()
JIT_COMPILE = "auto" if JIT_TEXT == "auto" else JIT_TEXT not in ("0", "false", "off")
CPU_THREADS = int(os.environ.get("OPENNN_FORECASTING_CPU_THREADS", "0"))
if CPU_THREADS > 0:
    tf.config.threading.set_intra_op_parallelism_threads(CPU_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(1)

cli_scenarios = [arg for arg in sys.argv[1:] if arg != "--allow-cpu"]
env_scenarios = [item for item in os.environ.get(
    "OPENNN_FORECASTING_SCENARIOS", "").split(",") if item]
want = cli_scenarios or env_scenarios or [s[0] for s in SCENARIOS]

for sid, past, future, hidden, lr, default_batch, max_ep, patience, multi in SCENARIOS:
    if sid not in want:
        continue

    batch_sizes = env_ints("OPENNN_FORECASTING_BATCH_SIZES", [default_batch])
    epochs_limit = FIXED_EPOCHS or max_ep
    Xtr, Ytr, Xva, Yva, Xte, Yte, y_mean, y_std = make_windows(past, future, multi)
    n = Xtr.shape[0]

    for batch in batch_sizes:
        for kind in ("Recurrent", "LSTM"):
            rmses, times, epochs_l, throughputs = [], [], [], []
            for seed in SEEDS:
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed)
                inp = tf.keras.layers.Input(shape=(past, Xtr.shape[2]))
                if kind == "Recurrent":
                    state = tf.keras.layers.SimpleRNN(hidden, activation="tanh")(inp)
                else:
                    state = tf.keras.layers.LSTM(hidden, use_cudnn="auto")(inp)
                output = tf.keras.layers.Dense(Ytr.shape[1])(state)
                model = tf.keras.Model(inp, output)
                steps_per_execution = max(1, math.ceil(n / batch))
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(lr),
                    loss="mse",
                    jit_compile=JIT_COMPILE,
                    steps_per_execution=steps_per_execution)
                callbacks = []
                if not FIXED_EPOCHS:
                    callbacks.append(tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=patience,
                        restore_best_weights=True, min_delta=1e-7))

                # Compile/autotune the exact fit and validation functions before
                # timing, then restore both model and Adam to their initial state.
                initial_weights = model.get_weights()
                model.optimizer.build(model.trainable_variables)
                initial_optimizer_values = [variable.numpy().copy()
                                            for variable in model.optimizer.variables]
                model.fit(
                    Xtr, Ytr, validation_data=(Xva, Yva), epochs=1,
                    validation_freq=1, batch_size=batch, shuffle=True,
                    verbose=0)
                if GPUS and hasattr(tf.experimental, "async_wait"):
                    tf.experimental.async_wait()
                model.set_weights(initial_weights)
                for variable, value in zip(model.optimizer.variables,
                                           initial_optimizer_values):
                    variable.assign(value)
                tf.keras.utils.set_random_seed(seed)

                t0 = time.perf_counter()
                history = model.fit(
                    Xtr, Ytr,
                    validation_data=None if FIXED_EPOCHS else (Xva, Yva),
                    epochs=epochs_limit,
                    validation_freq=epochs_limit if FIXED_EPOCHS else 1,
                    batch_size=batch, shuffle=True, verbose=0, callbacks=callbacks)
                if GPUS and hasattr(tf.experimental, "async_wait"):
                    tf.experimental.async_wait()
                train_s = time.perf_counter() - t0
                ran = len(history.history["loss"])

                pred = model.predict(Xte, batch_size=4096, verbose=0).astype(np.float32)
                pred_orig = pred * y_std + y_mean
                true_orig = Yte * y_std + y_mean
                rmse = float(np.sqrt(np.mean((pred_orig - true_orig) ** 2)))
                params = model.count_params()
                throughput = (n * ran) / train_s if train_s > 0 else 0.0
                rmses.append(rmse)
                times.append(train_s)
                epochs_l.append(ran)
                throughputs.append(throughput)
                print(f"METRIC engine=tensorflow phase={PHASE} scenario={sid} net={kind} "
                      f"batch_size={batch} seed={seed} params={params} epochs={ran} "
                      f"test_rmse={rmse:.6f} time_s={train_s:.6f} "
                      f"samples_per_sec={throughput:.1f} train_windows={n} device={DEV} "
                      f"jit_compile={JIT_COMPILE} steps_per_execution={steps_per_execution} "
                      f"precision={PRECISION} warmup=one_epoch_reset")

            std = statistics.stdev(rmses) if len(rmses) > 1 else 0.0
            print(f"METRIC engine=tensorflow phase={PHASE} scenario={sid} net={kind} "
                  f"batch_size={batch} seed=aggregate params={params} "
                  f"epochs_mean={round(statistics.fmean(epochs_l))} successful_runs={len(rmses)} "
                  f"test_rmse_mean={statistics.fmean(rmses):.6f} test_rmse_std={std:.6f} "
                  f"test_rmse_best={min(rmses):.6f} time_s_mean={statistics.fmean(times):.6f} "
                  f"samples_per_sec_mean={statistics.fmean(throughputs):.1f} "
                  f"train_windows={n} device={DEV} jit_compile={JIT_COMPILE} "
                  f"steps_per_execution={steps_per_execution} "
                  f"precision={PRECISION} warmup=one_epoch_reset", flush=True)
