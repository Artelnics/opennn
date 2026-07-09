#!/usr/bin/env python3
"""PyTorch accuracy-parity benchmark on the HIGGS classification task.

Trains the canonical HIGGS dense classifier (28 -> 1024 -> 1024 -> 1, ReLU
hidden, sigmoid output, binary cross entropy, Adam, fixed epochs) on the shared
prepared split and prints the test-set quality so parity with OpenNN and
TensorFlow can be checked at a fixed training budget.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from metrics import binary_metrics


def bench_data_dir() -> Path:
    root = os.environ.get("OPENNN_BENCH_DATA", str(Path.home() / "opennn-benchmark-data"))
    return Path(root) / "higgs"


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    x = np.ascontiguousarray(data[:, :-1])
    y = np.ascontiguousarray(data[:, -1:].astype(np.float32))
    return x, y


def batches(n: int, batch: int):
    stop = (n // batch) * batch
    for start in range(0, stop, batch):
        yield start, start + batch


def run(args: argparse.Namespace) -> None:
    import torch

    torch.manual_seed(42)
    torch.set_num_threads(args.threads or torch.get_num_threads())

    x_np, y_np = load_csv(args.train)
    xt_np, yt_np = load_csv(args.test)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    xt = torch.from_numpy(xt_np)

    layers: list[torch.nn.Module] = []
    current = x.shape[1]
    for _ in range(args.hidden_layers):
        layers.append(torch.nn.Linear(current, args.hidden))
        layers.append(torch.nn.ReLU())
        current = args.hidden
    layers.append(torch.nn.Linear(current, 1))
    layers.append(torch.nn.Sigmoid())
    model = torch.nn.Sequential(*layers)

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for _ in range(args.epochs):
        model.train()
        for start, end in batches(x.shape[0], args.batch):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[start:end]), y[start:end])
            loss.backward()
            optimizer.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for start, end in batches(xt.shape[0], args.batch):
            preds.append(model(xt[start:end]).numpy())
    pred_np = np.vstack(preds) if preds else np.empty((0, 1), dtype=np.float32)
    metrics = binary_metrics(yt_np[: pred_np.shape[0]], pred_np)

    print("engine=pytorch")
    print("device=cpu")
    print(f"samples={x.shape[0]}")
    print(f"batch={args.batch}")
    print(f"epochs={args.epochs}")
    print(f"hidden={args.hidden}")
    print(f"hidden_layers={args.hidden_layers}")
    print("activation=relu")
    print(f"test_samples={pred_np.shape[0]}")
    for key, value in metrics.items():
        print(f"{key}={value:.9g}")
    print("RESULT=OK")


def parse_args() -> argparse.Namespace:
    data_dir = bench_data_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=data_dir / "higgs_train.csv")
    parser.add_argument("--test", type=Path, default=data_dir / "higgs_test.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        print("RESULT=ERROR")
        raise
