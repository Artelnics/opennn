#!/usr/bin/env python3
"""PyTorch CPU HIGGS dense benchmark counterpart."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from metrics import binary_metrics

def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # np.loadtxt on the full split is minutes of wall clock per engine per run,
    # none of it measured, so the parsed array is cached next to the CSV and
    # re-read with np.load. OpenNN's own driver reads the CSV directly and is
    # unaffected; nothing here is inside a timed region either way.
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

# PyTorch's fast path on this workload is eager, which is worth stating because
# it is not the obvious answer: on an i7-12700H, torch.compile measured 29,449
# samples/s against eager's 41,523 on the same 1M-row epoch, inductor's CPU
# codegen losing to eager for a three-GEMM MLP. So compilation is opt-in
# (PT_COMPILE=1, mode via PT_COMPILE_MODE) rather than the default, and what
# the driver always takes is inference_mode over no_grad.
def compiled(fn, torch):
    if not os.environ.get("PT_COMPILE"):
        return fn
    mode = os.environ.get("PT_COMPILE_MODE", "default")
    try:
        return torch.compile(fn, mode=mode)
    except Exception as error:          # inductor needs a C++ toolchain
        print(f"note=torch.compile unavailable ({error})")
        return fn

def run_pytorch(args: argparse.Namespace) -> None:
    import torch

    torch.manual_seed(42)
    torch.set_num_threads(args.threads or torch.get_num_threads())
    activation_layer = torch.nn.ReLU if args.activation == "relu" else torch.nn.Tanh

    def make_model(features: int):
        layers: list[torch.nn.Module] = []
        current = features
        for _ in range(args.hidden_layers):
            layers.append(torch.nn.Linear(current, args.hidden))
            layers.append(activation_layer())
            current = args.hidden
        layers.append(torch.nn.Linear(current, 1))
        layers.append(torch.nn.Sigmoid())
        return torch.nn.Sequential(*layers)

    if args.mode == "train":
        x_np, y_np = load_csv(args.train)
        xt_np, yt_np = load_csv(args.test)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        xt = torch.from_numpy(xt_np)
        model = make_model(x.shape[1])
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters())

        def train_step(xb, yb):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            return loss

        step = compiled(train_step, torch)

        def run_epoch() -> None:
            model.train()
            for start, end in batches(x.shape[0], args.batch):
                step(x[start:end], y[start:end])

        for _ in range(args.warmup_epochs):
            run_epoch()

        times: list[float] = []
        for _ in range(args.epochs):
            t0 = time.perf_counter()
            run_epoch()
            times.append(time.perf_counter() - t0)
        times.sort()
        median_epoch_s = times[len(times) // 2]

        model.eval()
        preds = []
        with torch.no_grad():
            for start, end in batches(xt.shape[0], args.batch):
                preds.append(model(xt[start:end]).numpy())
        pred_np = np.vstack(preds) if preds else np.empty((0, 1), dtype=np.float32)
        metrics = binary_metrics(yt_np[: pred_np.shape[0]], pred_np)

        print_common("pytorch", args, x.shape[0])
        print(f"median_epoch_s={median_epoch_s:.9g}")
        print(f"samples_per_sec={x.shape[0] / median_epoch_s:.0f}")
        print(f"test_samples={pred_np.shape[0]}")
        for key, value in metrics.items():
            print(f"{key}={value:.9g}")
        print("RESULT=OK")
        return

    x_np, _ = load_csv(args.test)
    x = torch.from_numpy(x_np)
    model = make_model(x.shape[1])
    model.eval()

    forward = compiled(model, torch)

    def measure(batch: int) -> tuple[int, float]:
        def run_pass() -> None:
            with torch.inference_mode():
                for start, end in batches(x.shape[0], batch):
                    forward(x[start:end])

        run_pass()
        run_pass()
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            run_pass()
            times.append(time.perf_counter() - t0)
        # In temporal order, before the sort: a median hides a drifting machine,
        # and this one drifts - whatever is measured first after an idle gap runs
        # in the processor's boost window and what follows does not.
        print(f"batch_{batch}_pass_times=" + ",".join(f"{t:.6f}" for t in times), flush=True)
        times.sort()
        return (x.shape[0] // batch) * batch, times[len(times) // 2]

    batch_list = [int(item) for item in args.batches.split(",") if item] or [args.batch]

    if len(batch_list) == 1:
        processed, median_pass_s = measure(batch_list[0])
        print_common("pytorch", args, processed)
        print(f"median_pass_s={median_pass_s:.9g}")
        print(f"samples_per_sec={processed / median_pass_s:.0f}")
        print("RESULT=OK")
        return

    print_common("pytorch", args, x.shape[0])
    for batch in batch_list:
        processed, median_pass_s = measure(batch)
        print(f"batch_{batch}_samples_per_sec={processed / median_pass_s:.0f}"
              f" median_pass_s={median_pass_s:.9g}", flush=True)
    print("RESULT=OK")

def print_common(engine: str, args: argparse.Namespace, samples: int) -> None:
    print(f"engine={engine}")
    print(f"mode={args.mode}")
    print("device=cpu")
    print(f"samples={samples}")
    print(f"batch={args.batch}")
    print(f"hidden={args.hidden}")
    print(f"hidden_layers={args.hidden_layers}")
    print(f"activation={args.activation}")
    if args.mode == "train":
        print(f"epochs={args.epochs}")
    else:
        print(f"reps={args.reps}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "infer"])
    parser.add_argument("--train", type=Path)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1024)
    # Inference only: a comma-separated list is measured in one process, so the
    # whole batch-size row of a comparison shares one model, one load and one
    # thermal window on a laptop that drifts ten per cent over a sweep.
    parser.add_argument("--batches", default="")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "train" and args.train is None:
        parser.error("--train is required in train mode")
    return args

def main() -> None:
    args = parse_args()
    run_pytorch(args)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        print("RESULT=ERROR")
        raise
