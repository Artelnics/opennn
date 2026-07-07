"""TensorFlow/Keras counterpart of the OpenNN recurrent-vs-LSTM forecasting
benchmark. Same scenarios, architecture, training protocol and SEEDS as
pt_forecasting.py, so the three engines (OpenNN / PyTorch / TensorFlow) are
directly comparable on aggregated test RMSE and training throughput.

RMSE convention: standard sqrt(mean((pred-true)^2)), matching pt_forecasting.py
and the OpenNN C++ driver (errs(2)*sqrt(2)).

  usage: tf_forecasting.py [--allow-cpu] [B1 B2 ...]   (default: all)

Aborts (exit 2) if no GPU is visible (e.g. missing CUDA libs -> silent CPU
fallback), so CPU timings can never be recorded as a GPU run. Pass --allow-cpu
(or set CUDA_VISIBLE_DEVICES="") to run a deliberate CPU pass.
"""
import os
import statistics
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from xf_common import SCENARIOS, make_windows

ALLOW_CPU = "--allow-cpu" in sys.argv[1:] or os.environ.get("CUDA_VISIBLE_DEVICES") == ""
GPU = bool(tf.config.list_physical_devices("GPU"))
if not GPU and not ALLOW_CPU:
    print("ERROR device_mismatch engine=tensorflow expected=cuda actual=cpu "
          "(install tensorflow[and-cuda], or pass --allow-cpu / CUDA_VISIBLE_DEVICES=\"\" "
          "for a deliberate CPU run)",
          file=sys.stderr)
    sys.exit(2)
PHASE = "GPU" if GPU else "CPU"
DEV = "cuda" if GPU else "cpu"
SEEDS = [0, 1, 2, 3, 4]


def build(kind, n_feat, past, hidden, out):
    inp = tf.keras.layers.Input(shape=(past, n_feat))
    if kind == "Recurrent":
        h = tf.keras.layers.SimpleRNN(hidden, activation="tanh")(inp)
    else:
        h = tf.keras.layers.LSTM(hidden)(inp)
    y = tf.keras.layers.Dense(out)(h)
    return tf.keras.Model(inp, y)


def train_eval_once(kind, sc, data, seed):
    sid, past, future, hidden, lr, batch, max_ep, patience, multi = sc
    Xtr, Ytr, Xva, Yva, Xte, Yte, y_mean, y_std = data
    tf.keras.utils.set_random_seed(seed)

    model = build(kind, Xtr.shape[2], past, hidden, Ytr.shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                          restore_best_weights=True, min_delta=1e-7)
    t0 = time.perf_counter()
    hist = model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=max_ep,
                     batch_size=batch, shuffle=True, verbose=0, callbacks=[es])
    train_s = time.perf_counter() - t0
    ran = len(hist.history["loss"])

    pred = model.predict(Xte, batch_size=4096, verbose=0)
    pred_orig = pred * y_std + y_mean
    true_orig = Yte * y_std + y_mean
    rmse = float(np.sqrt(np.mean((pred_orig - true_orig) ** 2)))

    n = Xtr.shape[0]
    return {
        "rmse": rmse,
        "time_s": train_s,
        "epochs": ran,
        "params": model.count_params(),
        "sps": (n * ran) / train_s if train_s > 0 else 0.0,
        "n": n,
    }


def run(kind, sc, data):
    sid = sc[0]
    rmses, times, epochs_l, spss = [], [], [], []
    params = 0; n = 0
    for seed in SEEDS:
        res = train_eval_once(kind, sc, data, seed)
        rmses.append(res["rmse"]); times.append(res["time_s"])
        epochs_l.append(res["epochs"]); spss.append(res["sps"])
        params = res["params"]; n = res["n"]
        print(f"METRIC engine=tensorflow phase={PHASE} scenario={sid} net={kind} seed={seed} "
              f"params={params} epochs={res['epochs']} test_rmse={res['rmse']:.6f} "
              f"time_s={res['time_s']:.3f} samples_per_sec={res['sps']:.1f} "
              f"train_windows={n} device={DEV}")

    std = statistics.stdev(rmses) if len(rmses) > 1 else 0.0
    print(f"METRIC engine=tensorflow phase={PHASE} scenario={sid} net={kind} seed=aggregate "
          f"params={params} epochs_mean={round(statistics.fmean(epochs_l))} "
          f"successful_runs={len(rmses)} test_rmse_mean={statistics.fmean(rmses):.6f} "
          f"test_rmse_std={std:.6f} test_rmse_best={min(rmses):.6f} "
          f"time_s_mean={statistics.fmean(times):.3f} "
          f"samples_per_sec_mean={statistics.fmean(spss):.1f} "
          f"train_windows={n} device={DEV}")
    sys.stdout.flush()


def main():
    want = [a for a in sys.argv[1:] if a != "--allow-cpu"] or [s[0] for s in SCENARIOS]
    for sc in SCENARIOS:
        if sc[0] not in want:
            continue
        data = make_windows(sc[1], sc[2], sc[8])
        for kind in ("Recurrent", "LSTM"):
            run(kind, sc, data)


if __name__ == "__main__":
    main()
