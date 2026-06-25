"""TensorFlow/Keras counterpart of the OpenNN recurrent-vs-LSTM forecasting
benchmark. Same scenarios, architecture and training protocol as
pt_forecasting.py, so the three engines (OpenNN / PyTorch / TensorFlow) are
directly comparable on test RMSE and training throughput.

  usage: tf_forecasting.py [B1 B2 ...]   (default: all)
"""
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from xf_common import SCENARIOS, make_windows

GPU = bool(tf.config.list_physical_devices("GPU"))


def build(kind, n_feat, past, hidden, out):
    inp = tf.keras.layers.Input(shape=(past, n_feat))
    if kind == "Recurrent":
        h = tf.keras.layers.SimpleRNN(hidden, activation="tanh")(inp)
    else:
        h = tf.keras.layers.LSTM(hidden)(inp)
    y = tf.keras.layers.Dense(out)(h)
    return tf.keras.Model(inp, y)


def run(kind, sc, data):
    sid, past, future, hidden, lr, batch, max_ep, patience, multi = sc
    Xtr, Ytr, Xva, Yva, Xte, Yte, y_mean, y_std = data
    tf.keras.utils.set_random_seed(42)

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
    sps = (n * ran) / train_s if train_s > 0 else 0.0
    print(f"METRIC engine=tensorflow scenario={sid} net={kind} params={model.count_params()} "
          f"epochs={ran} test_rmse={rmse:.4f} time_s={train_s:.3f} samples_per_sec={sps:.1f} "
          f"train_windows={n} device={'cuda' if GPU else 'cpu'}")
    sys.stdout.flush()


def main():
    want = sys.argv[1:] or [s[0] for s in SCENARIOS]
    for sc in SCENARIOS:
        if sc[0] not in want:
            continue
        data = make_windows(sc[1], sc[2], sc[8])
        for kind in ("Recurrent", "LSTM"):
            run(kind, sc, data)


if __name__ == "__main__":
    main()
