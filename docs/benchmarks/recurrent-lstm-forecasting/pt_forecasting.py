"""PyTorch counterpart of the OpenNN recurrent-vs-LSTM forecasting benchmark.

For each Beijing PM2.5 scenario B1..B4 it trains a 1-layer SimpleRNN(tanh) and a
1-layer LSTM (same hidden size, lr, batch, epochs, early-stop patience as
OpenNN), over SEEDS seeds, then reports aggregated test RMSE (mean/std/best in
original pm2_5 units), training wall time and throughput. Runs on CUDA if
available.

RMSE convention: standard sqrt(mean((pred-true)^2)), matching the OpenNN C++
driver which reports errs(2)*sqrt(2) so all three engines are apples-to-apples.

  usage: pt_forecasting.py [B1 B2 ...]   (default: all)
"""
import statistics
import sys
import time
import numpy as np
import torch
import torch.nn as nn

from xf_common import SCENARIOS, make_windows

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PHASE = "GPU" if DEV == "cuda" else "CPU"
SEEDS = [0, 1, 2, 3, 4]


class Net(nn.Module):
    def __init__(self, kind, n_feat, hidden, out):
        super().__init__()
        if kind == "Recurrent":
            self.rnn = nn.RNN(n_feat, hidden, batch_first=True, nonlinearity="tanh")
        else:
            self.rnn = nn.LSTM(n_feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


def train_eval_once(kind, sc, data, seed):
    sid, past, future, hidden, lr, batch, max_ep, patience, multi = sc
    Xtr, Ytr, Xva, Yva, Xte, Yte, y_mean, y_std = data
    torch.manual_seed(seed)

    xtr = torch.from_numpy(Xtr).to(DEV); ytr = torch.from_numpy(Ytr).to(DEV)
    xva = torch.from_numpy(Xva).to(DEV); yva = torch.from_numpy(Yva).to(DEV)
    xte = torch.from_numpy(Xte).to(DEV)

    net = Net(kind, Xtr.shape[2], hidden, Ytr.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    n = xtr.shape[0]

    best_val = float("inf"); best_state = None; fails = 0; ran = 0
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(max_ep):
        net.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(net(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        ran += 1
        net.eval()
        with torch.no_grad():
            vloss = lossf(net(xva), yva).item()
        if vloss < best_val - 1e-7:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            fails = 0
        else:
            fails += 1
            if fails >= patience:
                break
    if DEV == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(xte).cpu().numpy()
    pred_orig = pred * y_std + y_mean
    true_orig = Yte * y_std + y_mean
    rmse = float(np.sqrt(np.mean((pred_orig - true_orig) ** 2)))

    return {
        "rmse": rmse,
        "time_s": train_s,
        "epochs": ran,
        "params": sum(p.numel() for p in net.parameters()),
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
        print(f"METRIC engine=pytorch phase={PHASE} scenario={sid} net={kind} seed={seed} "
              f"params={params} epochs={res['epochs']} test_rmse={res['rmse']:.6f} "
              f"time_s={res['time_s']:.3f} samples_per_sec={res['sps']:.1f} "
              f"train_windows={n} device={DEV}")

    std = statistics.stdev(rmses) if len(rmses) > 1 else 0.0
    print(f"METRIC engine=pytorch phase={PHASE} scenario={sid} net={kind} seed=aggregate "
          f"params={params} epochs_mean={round(statistics.fmean(epochs_l))} "
          f"successful_runs={len(rmses)} test_rmse_mean={statistics.fmean(rmses):.6f} "
          f"test_rmse_std={std:.6f} test_rmse_best={min(rmses):.6f} "
          f"time_s_mean={statistics.fmean(times):.3f} "
          f"samples_per_sec_mean={statistics.fmean(spss):.1f} "
          f"train_windows={n} device={DEV}")
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
