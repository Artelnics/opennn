#!/usr/bin/env python3
"""PyTorch convergence-gate benchmark on the HIGGS classification dataset,
written the way a PyTorch user writes it: nn.Sequential, an eager training
loop, a no_grad evaluation. The protocol lives in run_convergence.py.

MLPerf-style metric: WALL-CLOCK TIME TO REACH A FIXED QUALITY TARGET, not
throughput at a fixed epoch count. Trains the canonical HIGGS dense classifier
(28 -> 1024 -> 1024 -> 1, ReLU, sigmoid, BCE, Adam) and, after each epoch,
evaluates the HELD-OUT (test) log-loss. When it reaches the target the clock
stops and we report the wall-clock time, epochs taken, and final held-out
metric. Same data / arch / optimizer / target as opennn_convergence and
tensorflow_convergence, so the only thing compared is how fast each engine
reaches the same held-out quality. Per-epoch evaluation is excluded from the
clock.

  usage: python pytorch_convergence.py --train TRAIN.csv --test TEST.csv
                                       [--target 0.60] [--max-epochs 50]
                                       [--batch 1024] [--hidden 1024]
                                       [--hidden-layers 2] [--threads 0]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # np.loadtxt on the 10.5M-row split is minutes of wall clock, none of it
    # measured, so the parsed array is cached next to the CSV.
    cache = path.with_suffix(path.suffix + ".npy")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        data = np.load(cache, mmap_mode="r")
    else:
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
        try:
            np.save(cache, data)
        except OSError:              # read-only data directory: parse next time
            pass
    x = np.ascontiguousarray(data[:, :-1])
    y = np.ascontiguousarray(data[:, -1:].astype(np.float32))
    return x, y


def batches(n: int, batch: int):
    stop = (n // batch) * batch
    for start in range(0, stop, batch):
        yield start, start + batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--target", type=float, default=0.60)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    import torch

    torch.manual_seed(42)
    if args.threads:
        torch.set_num_threads(args.threads)

    x_np, y_np = load_csv(args.train)
    xt_np, yt_np = load_csv(args.test)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    xt = torch.from_numpy(xt_np)
    yt = torch.from_numpy(yt_np)

    layers: list[torch.nn.Module] = []
    current = x.shape[1]
    for _ in range(args.hidden_layers):
        layers += [torch.nn.Linear(current, args.hidden), torch.nn.ReLU()]
        current = args.hidden
    layers += [torch.nn.Linear(current, 1), torch.nn.Sigmoid()]
    model = torch.nn.Sequential(*layers)

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    reached = False
    epochs_taken = args.max_epochs
    test_log_loss = float("nan")
    train_s = 0.0

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        for start, end in batches(x.shape[0], args.batch):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[start:end]), y[start:end])
            loss.backward()
            optimizer.step()
        train_s += time.perf_counter() - t0

        # The gate: mean clamped binary cross entropy over the held-out split
        # (whole batches only), outside the clock.
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start, end in batches(xt.shape[0], args.batch):
                p = model(xt[start:end]).clamp(1.0e-7, 1.0 - 1.0e-7)
                yb = yt[start:end]
                bce = -(yb * torch.log(p) + (1.0 - yb) * torch.log(1.0 - p))
                total += float(bce.sum())
                count += yb.shape[0]
        test_log_loss = total / count if count else float("nan")

        if test_log_loss <= args.target:
            reached = True
            epochs_taken = epoch
            break

    print("engine=pytorch")
    print("device=cpu")
    print("dataset=HIGGS")
    print(f"train_samples={x.shape[0]}")
    print(f"batch={args.batch}")
    print(f"hidden={args.hidden}")
    print(f"hidden_layers={args.hidden_layers}")
    print(f"target_log_loss={args.target}")
    print(f"reached_goal={1 if reached else 0}")
    print(f"epochs_to_target={epochs_taken}")
    print(f"test_log_loss={test_log_loss:.10f}")
    print(f"time_to_target_s={train_s:.6f}")
    print(f"RESULT={'OK' if reached else 'DID_NOT_CONVERGE'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        print("RESULT=ERROR")
        raise
