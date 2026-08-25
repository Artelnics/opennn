#!/usr/bin/env python3
"""The dense family in PyTorch, defined once, driven four ways.

REORGANIZATION_PLAN.md sections 4 and 8; the counterpart of model_opennn.cpp
and deliberately its mirror image. Same modes, same positional arguments, same
`key=value` output, so a protocol drives either engine by swapping the command
prefix and never learns which one it is talking to.

  dense.py train    <train_csv> <test_csv> [epochs] [batch,...] [opts]
  dense.py infer    <test_csv>             [reps]   [batch,...] [opts]
  dense.py capacity <train_csv>            [batch]              [opts]
  dense.py quality  <train_csv> <test_csv> [epochs] [batch]     [opts]

  opts: [hidden] [layers] [relu|tanh] [cpu|cuda] [fp32|bf16|strict]

The definition is `Linear -> activation -> ... -> Linear(1)`, which is what
model_opennn.cpp builds: bare layers, no scaling stage, since prepare_higgs.py
normalises the CSV beforehand.

Contract item 3 -- each engine at its best -- means `torch.compile` here, the
way it means captured CUDA graphs and a device-resident split for OpenNN.
`reduce-overhead` is the default because it adds CUDA graphs, which is the
closest analogue to what OpenNN is doing; PT_COMPILE_MODE=eager opts out.

`fp32` allows TF32 tensor cores, as it does in every engine of this suite --
OpenNN's fp32 GEMMs are CUBLAS_COMPUTE_32F_FAST_TF32. `strict` is the escape
hatch that turns TF32 off, and is not the published fp32 cell.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import binary_metrics  # noqa: E402

SEED = 42

def load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Inputs and target, float32. `.npy` beside the CSV wins -- same content,
    and parsing 3 GB of text per trial would dominate a capacity sweep."""
    cached = Path(str(path) + ".npy")
    data = np.load(cached) if cached.exists() else np.loadtxt(path, delimiter=",", dtype=np.float32)
    data = np.ascontiguousarray(data, dtype=np.float32)
    return data[:, :-1], data[:, -1:]

def build(features: int, opts: dict) -> torch.nn.Module:
    """The dense family. Nothing else here constructs the network."""
    torch.manual_seed(SEED)

    act = torch.nn.ReLU if opts["activation"] == "relu" else torch.nn.Tanh
    layers: list[torch.nn.Module] = []
    current = features

    for _ in range(opts["layers"]):
        layers.append(torch.nn.Linear(current, opts["hidden"]))
        layers.append(act())
        current = opts["hidden"]

    layers.append(torch.nn.Linear(current, 1))
    return torch.nn.Sequential(*layers).to(opts["device"])

def parse_opts(argv: list[str], first: int) -> dict:
    """Trailing options, positional and shared by every mode -- the same five
    in the same order as model_opennn.cpp."""
    def at(index: int, default: str) -> str:
        return argv[index] if len(argv) > index else default

    precision = at(first + 4, "fp32")
    device = at(first + 3, "cuda")

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")

    allow_tf32 = precision != "strict"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = True

    return {
        "hidden": int(at(first, "1024")),
        "layers": int(at(first + 1, "2")),
        "activation": at(first + 2, "relu").lower(),
        "device": device,
        "precision": precision,
        "autocast": precision == "bf16",
        "tf32": allow_tf32,
    }

def autocast_ctx(opts: dict):
    if not opts["autocast"]:
        return contextlib.nullcontext()
    return torch.autocast(device_type=opts["device"], dtype=torch.bfloat16)

def compiled(fn, opts: dict):
    """torch.compile unless PT_COMPILE_MODE=eager. reduce-overhead brings CUDA
    graphs, which is the analogue of what OpenNN is measured with."""
    mode = os.environ.get("PT_COMPILE_MODE", "reduce-overhead")
    if mode == "eager" or opts["device"] != "cuda":
        return fn, "eager"
    return torch.compile(fn, mode=None if mode == "default" else mode), f"compile:{mode}"

def batches_of(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]

def sync(opts: dict) -> None:
    if opts["device"] == "cuda":
        torch.cuda.synchronize()

def train_like(argv: list[str], mode: str) -> int:
    """train and quality: the same loop, differing only in warmup.

    quality takes no warmup epochs -- they would train a different network than
    the one whose accuracy is reported.
    """
    epochs = int(argv[4]) if len(argv) > 4 else 1
    batches = batches_of(argv[5] if len(argv) > 5 else "1024")
    opts = parse_opts(argv, 6)

    x_np, y_np = load_csv(argv[2])
    xt_np, yt_np = load_csv(argv[3])

    print(f"engine=pytorch\nmode={mode}\ndevice={opts['device']}")

    x = torch.from_numpy(x_np).to(opts["device"]).contiguous()
    y = torch.from_numpy(y_np).to(opts["device"]).contiguous()
    warmup = 2 if mode == "train" else 0

    for batch in batches:
        model = build(x.shape[1], opts)
        if batch == batches[0]:
            print(f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters())
        ctx = autocast_ctx(opts)

        def step(xb, yb):
            optimizer.zero_grad(set_to_none=True)
            with ctx:
                loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        step_fn, how = compiled(step, opts)

        # Whole batches only, the same rule model_opennn.cpp applies by
        # dropping the tail: the remainder is not trained.
        starts = range(0, x.shape[0] - batch + 1, batch)

        def run_epoch():
            model.train()
            for start in starts:
                step_fn(x[start:start + batch], y[start:start + batch])
            sync(opts)

        for _ in range(warmup):
            run_epoch()

        print(f"TIMED_START_UNIX={time.time():.3f}", flush=True)
        times = []
        for _ in range(epochs):
            mark = time.perf_counter()
            run_epoch()
            times.append(time.perf_counter() - mark)
        print(f"TIMED_END_UNIX={time.time():.3f}", flush=True)

        times.sort()
        median = times[len(times) // 2]
        samples_per_epoch = len(list(starts)) * batch

        metrics = evaluate(model, xt_np, yt_np, batch, opts)

        print(f"batch_{batch}_samples_per_sec={int(samples_per_epoch / median)}"
              f" median_epoch_s={median:.6g}")
        print(f"batch_{batch}_epoch_times={','.join(f'{t:.6g}' for t in times)}")
        print(f"batch_{batch}_mode={how}")
        print(f"batch_{batch}_test_accuracy={metrics['test_accuracy']:.6g}"
              f" test_roc_auc={metrics['test_roc_auc']:.6g}", flush=True)

    print("RESULT=OK")
    return 0

def evaluate(model, xt_np, yt_np, batch: int, opts: dict) -> dict:
    processed = (xt_np.shape[0] // batch) * batch
    xt = torch.from_numpy(xt_np[:processed]).to(opts["device"]).contiguous()

    model.eval()
    chunks = []
    with torch.no_grad(), autocast_ctx(opts):
        for start in range(0, processed, batch):
            chunks.append(torch.sigmoid(model(xt[start:start + batch])).float().cpu())

    return binary_metrics(yt_np[:processed], torch.cat(chunks).numpy())

def infer(argv: list[str]) -> int:
    reps = int(argv[3]) if len(argv) > 3 else 1
    batches = batches_of(argv[4] if len(argv) > 4 else "1024")
    opts = parse_opts(argv, 5)

    x_np, _ = load_csv(argv[2])
    print(f"engine=pytorch\nmode=infer\ndevice={opts['device']}")

    x = torch.from_numpy(x_np).to(opts["device"]).contiguous()
    samples = x.shape[0]

    for batch in batches:
        model = build(x.shape[1], opts).eval()
        if batch == batches[0]:
            print(f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)
        forward, _ = compiled(model, opts)
        processed = (samples // batch) * batch

        def run_pass():
            with torch.no_grad(), autocast_ctx(opts):
                for start in range(0, processed, batch):
                    forward(x[start:start + batch])
            sync(opts)

        run_pass()
        run_pass()

        # The same marks train prints: energy is integrated between them, so
        # interpreter startup and torch.compile stay outside the window.
        print(f"TIMED_START_UNIX={time.time():.3f}", flush=True)
        times = []
        for _ in range(reps):
            mark = time.perf_counter()
            run_pass()
            times.append(time.perf_counter() - mark)
        print(f"TIMED_END_UNIX={time.time():.3f}", flush=True)

        print(f"batch_{batch}_pass_times={','.join(f'{t:.6g}' for t in times)}")
        times.sort()
        median = times[len(times) // 2]
        print(f"batch_{batch}_samples_per_sec={int(processed / median)}"
              f" median_pass_s={median:.6g}", flush=True)

    print("RESULT=OK")
    return 0

def capacity(argv: list[str]) -> int:
    """One attempt, then exit -- the same contract model_opennn.cpp honours,
    because an out-of-memory fault leaves the CUDA context unusable and the
    next attempt in this process would measure the wreck of the last."""
    batch = int(argv[3]) if len(argv) > 3 else 1024
    opts = parse_opts(argv, 4)

    x_np, y_np = load_csv(argv[2])
    print(f"engine=pytorch\nmode=capacity\ndevice={opts['device']}\nbatch={batch}")

    try:
        x = torch.from_numpy(x_np).to(opts["device"]).contiguous()
        y = torch.from_numpy(y_np).to(opts["device"]).contiguous()

        model = build(x.shape[1], opts)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters())

        with autocast_ctx(opts):
            loss = loss_fn(model(x[:batch]), y[:batch])
        loss.backward()
        optimizer.step()
        sync(opts)
    except torch.cuda.OutOfMemoryError as error:
        print(f"fits=0\nreason={error}\nRESULT=OOM", flush=True)
        return 1
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        print(f"fits=0\nreason={error}\nRESULT=OOM", flush=True)
        return 1

    print("fits=1\nRESULT=OK", flush=True)
    return 0

def main() -> int:
    argv = sys.argv
    mode = argv[1] if len(argv) > 1 else ""

    if mode in ("train", "quality") and len(argv) > 3:
        return train_like(argv, mode)
    if mode == "infer" and len(argv) > 2:
        return infer(argv)
    if mode == "capacity" and len(argv) > 2:
        return capacity(argv)

    print(__doc__, file=sys.stderr)
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
