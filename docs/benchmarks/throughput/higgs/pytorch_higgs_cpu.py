#!/usr/bin/env python3
"""PyTorch CPU HIGGS dense benchmark, written the way a PyTorch user writes it:
nn.Sequential, an eager training loop, inference_mode for the forward. The
measurement protocol - arm rotation, soaking, medians over rounds - lives in
run_higgs_cpu_sweep.py, not here.

Eager is this model's fast path, not a simplification: torch.compile measured
29,449 samples/s against eager's 41,523 on this machine, inductor's CPU codegen
losing to eager on a three-GEMM MLP.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

from metrics import binary_metrics


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # np.loadtxt on the 500k-row split is minutes of wall clock, none of it
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


def batch_list(args: argparse.Namespace) -> list[int]:
    return [int(item) for item in args.batches.split(",") if item] or [args.batch]


def make_model(torch, features: int, args: argparse.Namespace):
    activation = torch.nn.ReLU if args.activation == "relu" else torch.nn.Tanh
    layers: list = []
    current = features
    for _ in range(args.hidden_layers):
        layers += [torch.nn.Linear(current, args.hidden), activation()]
        current = args.hidden
    layers += [torch.nn.Linear(current, 1), torch.nn.Sigmoid()]
    return torch.nn.Sequential(*layers)


def print_common(args: argparse.Namespace) -> None:
    print("engine=pytorch")
    print(f"mode={args.mode}")
    print("device=cpu")
    print(f"hidden={args.hidden}")
    print(f"hidden_layers={args.hidden_layers}")
    print(f"activation={args.activation}")


def run_train(torch, args: argparse.Namespace) -> None:
    x_np, y_np = load_csv(args.train)
    xt_np, yt_np = load_csv(args.test)
    x = torch.from_numpy(np.array(x_np))
    y = torch.from_numpy(np.array(y_np))
    xt = torch.from_numpy(np.array(xt_np))

    print_common(args)
    print(f"samples={x.shape[0]}")
    print(f"epochs={args.epochs}")

    for batch in batch_list(args):
        torch.manual_seed(42)
        model = make_model(torch, x.shape[1], args)
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters())

        def run_epoch() -> None:
            model.train()
            for start, end in batches(x.shape[0], batch):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x[start:end]), y[start:end])
                loss.backward()
                optimizer.step()

        for _ in range(args.warmup_epochs):
            run_epoch()

        times: list[float] = []
        for _ in range(args.epochs):
            t0 = time.perf_counter()
            run_epoch()
            times.append(time.perf_counter() - t0)

        print(f"batch_{batch}_epoch_times=" + ",".join(f"{t:.6f}" for t in times))

        median_epoch_s = sorted(times)[len(times) // 2]

        model.eval()
        preds = []
        with torch.inference_mode():
            for start, end in batches(xt.shape[0], batch):
                preds.append(model(xt[start:end]).numpy())
        pred_np = np.vstack(preds) if preds else np.empty((0, 1), dtype=np.float32)
        m = binary_metrics(yt_np[: pred_np.shape[0]], pred_np)

        print(f"batch_{batch}_samples_per_sec={x.shape[0] / median_epoch_s:.0f}"
              f" median_epoch_s={median_epoch_s:.9g}")
        print(f"batch_{batch}_test_accuracy={m['test_accuracy']:.9g}"
              f" test_roc_auc={m['test_roc_auc']:.9g}", flush=True)


    print("RESULT=OK")


def run_infer(torch, args: argparse.Namespace) -> None:
    x_np, _ = load_csv(args.test)
    x = torch.from_numpy(np.array(x_np))
    torch.manual_seed(42)
    model = make_model(torch, x.shape[1], args)
    model.eval()

    print_common(args)
    print(f"reps={args.reps}")

    for batch in batch_list(args):
        processed = (x.shape[0] // batch) * batch

        def run_pass() -> None:
            with torch.inference_mode():
                for start, end in batches(x.shape[0], batch):
                    model(x[start:end])

        run_pass()
        run_pass()
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            run_pass()
            times.append(time.perf_counter() - t0)

        print(f"batch_{batch}_pass_times=" + ",".join(f"{t:.6f}" for t in times))

        median_pass_s = sorted(times)[len(times) // 2]
        print(f"batch_{batch}_samples_per_sec={processed / median_pass_s:.0f}"
              f" median_pass_s={median_pass_s:.9g}", flush=True)


    print("RESULT=OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "infer"])
    parser.add_argument("--train", type=Path)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--batches", default="")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)

    if args.mode == "train":
        run_train(torch, args)
    else:
        run_infer(torch, args)


if __name__ == "__main__":
    main()
