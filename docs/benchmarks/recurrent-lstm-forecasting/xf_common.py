"""Shared data preparation for the cross-framework recurrent-vs-LSTM forecasting
comparison (OpenNN vs PyTorch vs TensorFlow), replicating the OpenNN
recurrent_lstm_forecasting_benchmark setup on UCI Beijing PM2.5:

  * input window = past P hours of ALL 15 columns (pm2_5 is InputTarget in
    OpenNN, so its past values are an input too),
  * target      = pm2_5 over the next F hours (1 value if future==1, else F),
  * sequential 60/20/20 train/val/test split (TimeSeriesDataset default),
  * z-score standardisation fit on the training rows only,
  * RMSE reported in the original pm2_5 units (predictions un-standardised).

Same four scenarios B1..B4 as examples/recurrent_lstm_forecasting_benchmark/main.cpp.
"""
import csv
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data", "beijing_pm25_forecasting.csv")

# id, past, future, hidden, lr, batch, max_epochs, patience, multi_target
SCENARIOS = [
    ("B1", 24, 1, 32, 0.003, 128, 120, 20, False),
    ("B2", 48, 1, 48, 0.003, 128, 100, 20, False),
    ("B3", 72, 24, 64, 0.002, 128, 80, 20, True),
    ("B4", 168, 24, 64, 0.001, 128, 60, 15, True),
]


def _load_columns():
    rows = []
    with open(CSV, newline="") as f:
        r = csv.reader(f)
        next(r)  # header
        for line in r:
            rows.append([float(x) for x in line])
    data = np.asarray(rows, dtype=np.float32)   # (T, 15); last col = pm2_5
    return data


def make_windows(past, future, multi_target):
    """Return (Xtr,Ytr, Xva,Yva, Xte,Yte, y_mean,y_std) as float32 numpy.

    X: (N, past, 15)   Y: (N, F_out)  with F_out = future if multi else 1.
    Standardisation stats come from the training split only; Y is returned
    standardised, and (y_mean,y_std) let callers un-standardise predictions to
    compute RMSE in original pm2_5 units.
    """
    data = _load_columns()
    T, n_feat = data.shape
    target_col = n_feat - 1

    # Sequential 60/20/20 on the raw rows (windows inherit the split by start row).
    n_train = int(0.6 * T)
    n_val = int(0.2 * T)

    mean = data[:n_train].mean(axis=0)
    std = data[:n_train].std(axis=0)
    std[std == 0.0] = 1.0
    scaled = (data - mean) / std
    y_mean, y_std = float(mean[target_col]), float(std[target_col])

    f_out = future if multi_target else 1

    def build(lo, hi):
        # window start rows s with [s, s+past) input and [s+past, s+past+future) target,
        # restricted so the whole window+horizon stays inside [lo, hi).
        Xs, Ys = [], []
        for s in range(lo, hi - past - future + 1):
            Xs.append(scaled[s:s + past, :])
            tgt = scaled[s + past: s + past + future, target_col]
            Ys.append(tgt if multi_target else tgt[-1:])
        if not Xs:
            return (np.zeros((0, past, n_feat), np.float32),
                    np.zeros((0, f_out), np.float32))
        return np.asarray(Xs, np.float32), np.asarray(Ys, np.float32)

    Xtr, Ytr = build(0, n_train)
    Xva, Yva = build(n_train, n_train + n_val)
    Xte, Yte = build(n_train + n_val, T)
    return Xtr, Ytr, Xva, Yva, Xte, Yte, y_mean, y_std
