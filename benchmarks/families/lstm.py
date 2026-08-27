#!/usr/bin/env python3
"""The LSTM family in PyTorch, defined once, driven four ways.

PLAN.md; the counterpart of lstm.cpp. LSTM forecasting on UCI Beijing PM2.5,
predicting the next hourly reading from a window of past ones.

  lstm.py train    <csv> <csv> [epochs] [batch,...] [hidden] [past] [dev] [prec]
  lstm.py infer    <csv>       [reps]   [batch,...] [hidden] [past] [dev] [prec]
  lstm.py capacity <csv>       [batch]              [hidden] [past] [dev] [prec]

`nn.LSTM` reaches cuDNN's fused RNN, and so does OpenNN's
`LongShortTermMemoryOperator`. Both engines therefore run the *same NVIDIA
kernel*, which makes this the cleanest cell in the matrix: the arithmetic is
identical, so what is measured is the surrounding machinery -- data movement,
launch overhead, the optimiser -- rather than two hand-written kernels.

`nn.LSTM(features, hidden) + nn.Linear(hidden, 1)` is what lstm.cpp's
ForecastingLstmNetwork builds, and the parameter counts are printed by both so
the claim is checkable rather than asserted: 4*(hidden*features + hidden^2 +
2*hidden) + hidden + 1, which is 73,857 at the defaults.
"""

from __future__ import annotations

import contextlib
import csv
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED = 42

class Forecaster(nn.Module):
    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])

def report_blas() -> None:
    """Which BLAS this engine dispatches to, printed like OpenNN prints it.

    PyTorch's is fixed at build time -- the wheels are MKL-backed -- while
    OpenNN's is chosen at runtime. Recording both is what stops a comparison
    quietly turning into Eigen against MKL.
    """
    settings = torch.__config__.show()
    match = re.search(r"BLAS_INFO=(\w+)", settings)

    print(f"blas={match.group(1) if match else 'unknown'}")


def build(features: int, opts: dict) -> nn.Module:
    """The LSTM family. Nothing else here constructs the network."""
    torch.manual_seed(SEED)
    return Forecaster(features, opts["hidden"]).to(opts["device"])

def load_series(path: str, past: int) -> tuple[np.ndarray, np.ndarray]:
    """Windows of `past` hourly rows, predicting the next target.

    The last column is the target, matching what the tabular loader on the
    OpenNN side treats as such, and it is standardised on training statistics
    so neither engine pays a scaling stage the other does not.
    """
    report_opened(path)

    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))

    values = np.asarray(rows[1:], dtype=np.float32)
    features, target = values[:, :-1], values[:, -1:]

    mean, std = features.mean(axis=0), features.std(axis=0)
    features = (features - mean) / np.where(std > 1.0e-12, std, 1.0)

    count = len(values) - past
    windows = np.lib.stride_tricks.sliding_window_view(features, past, axis=0)
    windows = np.ascontiguousarray(windows[:count].transpose(0, 2, 1))

    return windows, np.ascontiguousarray(target[past:past + count])

def resident_mib() -> float:
    """Resident set, MiB. The framework baseline is subtracted from the peak,
    because torch's import alone is ~816 MiB here against OpenNN's 209 -- so a
    raw RSS comparison measures which framework is bigger, not which run costs
    more, and the answer flips with dataset size."""
    try:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
    except Exception:
        return 0.0

def report_opened(path: str) -> None:
    """Announce the file actually opened, not the one passed in."""
    print(f"dataset_opened={Path(path).resolve()}", flush=True)

def parse_opts(argv: list[str], first: int) -> dict:
    def at(index: int, default: str) -> str:
        return argv[index] if len(argv) > index else default

    device = at(first + 2, "cuda")
    precision = at(first + 3, "fp32")

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")

    # The runner pins CPU work to performance cores and sets a matched thread
    # count for both engines; honour it, or this side quietly takes every
    # logical CPU and the comparison becomes a thread-count comparison.
    threads = os.environ.get("TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    if threads:
        torch.set_num_threads(int(threads))

    allow_tf32 = precision != "strict"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = True

    return {"hidden": int(at(first, "128")), "past": int(at(first + 1, "24")),
            "device": device, "precision": precision,
            "autocast": precision == "bf16"}

def autocast_ctx(opts: dict):
    if not opts["autocast"]:
        return contextlib.nullcontext()
    return torch.autocast(device_type=opts["device"], dtype=torch.bfloat16)

def compiled(fn, opts: dict):
    """torch.compile on CUDA, eager on CPU -- both measured, not assumed.

    On CPU, eager is PyTorch's fast path for these models, not a shortcut:
    inductor's CPU codegen loses on a small stack of GEMMs. Measured on this
    machine, dense CPU training, batch 4,096: eager 93,156 samples/s against
    compiled 83,722. The previous suite measured the same thing on different
    hardware (41,523 against 29,449), so it is the codegen and not the box.

    Compiling here anyway would hand OpenNN a win against a PyTorch nobody
    would ship, which is the mirror image of the eager-on-GPU mistake that
    made dense training read 1.29x when it was 1.06x.

    PT_COMPILE_MODE overrides either way, so the choice stays measurable.
    """
    mode = os.environ.get("PT_COMPILE_MODE", "default")
    if mode == "eager" or opts["device"] != "cuda":
        return fn, "eager"
    return torch.compile(fn, mode=None if mode == "default" else mode), f"compile:{mode}"

def batches_of(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]

def sync(opts: dict) -> None:
    if opts["device"] == "cuda":
        torch.cuda.synchronize()

def describe(samples: int, features: int, model: nn.Module, opts: dict) -> None:
    """`features` stays in the signature as documentation of the call site;
    it is not reported, because the OpenNN side has no unambiguous
    counterpart and `parameters` pins the model exactly."""
    print(f"samples={samples} past={opts['past']} "
          f"hidden={opts['hidden']} "
          f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)

def train_like(argv: list[str], mode: str) -> int:
    epochs = int(argv[4]) if len(argv) > 4 else 1
    batches = batches_of(argv[5] if len(argv) > 5 else "256")
    opts = parse_opts(argv, 6)

    print(f"baseline_rss_mib={resident_mib():.1f}")
    print(f"engine=pytorch\nmode={mode}\ndevice={opts['device']}")
    report_blas()

    windows, targets = load_series(argv[2], opts["past"])
    x = torch.from_numpy(windows).to(opts["device"])
    y = torch.from_numpy(targets).to(opts["device"])
    warmup = 2 if mode == "train" else 0

    for batch in batches:
        model = build(x.shape[2], opts)
        if batch == batches[0]:
            describe(x.shape[0], x.shape[2], model, opts)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        ctx = autocast_ctx(opts)

        def step(xb, yb):
            optimizer.zero_grad(set_to_none=True)
            with ctx:
                loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        step_fn, how = compiled(step, opts)
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

        print(f"batch_{batch}_samples_per_sec={int(len(list(starts)) * batch / median)}"
              f" median_epoch_s={median:.6g}")
        print(f"batch_{batch}_mode={how}", flush=True)

    print("RESULT=OK")
    return 0

def infer(argv: list[str]) -> int:
    reps = int(argv[3]) if len(argv) > 3 else 1
    batches = batches_of(argv[4] if len(argv) > 4 else "256")
    opts = parse_opts(argv, 5)

    print(f"baseline_rss_mib={resident_mib():.1f}")
    print(f"engine=pytorch\nmode=infer\ndevice={opts['device']}")
    report_blas()

    windows, _ = load_series(argv[2], opts["past"])
    x = torch.from_numpy(windows).to(opts["device"])

    for batch in batches:
        model = build(x.shape[2], opts).eval()
        if batch == batches[0]:
            describe(x.shape[0], x.shape[2], model, opts)

        forward, _ = compiled(model, opts)
        processed = (x.shape[0] // batch) * batch
        window = x[:batch]

        def run_pass():
            with torch.no_grad(), autocast_ctx(opts):
                for _ in range(processed // batch):
                    forward(window)
            sync(opts)

        run_pass()

        print(f"TIMED_START_UNIX={time.time():.3f}", flush=True)
        times = []
        for _ in range(reps):
            mark = time.perf_counter()
            run_pass()
            times.append(time.perf_counter() - mark)
        print(f"TIMED_END_UNIX={time.time():.3f}", flush=True)

        times.sort()
        median = times[len(times) // 2]
        print(f"batch_{batch}_samples_per_sec={int(processed / median)}"
              f" median_pass_s={median:.6g}", flush=True)

    print("RESULT=OK")
    return 0

def capacity(argv: list[str]) -> int:
    batch = int(argv[3]) if len(argv) > 3 else 256
    opts = parse_opts(argv, 4)

    print(f"baseline_rss_mib={resident_mib():.1f}")
    print(f"engine=pytorch\nmode=capacity\ndevice={opts['device']}\nbatch={batch}")
    report_blas()

    try:
        windows, targets = load_series(argv[2], opts["past"])
        x = torch.from_numpy(windows[:batch]).to(opts["device"])
        y = torch.from_numpy(targets[:batch]).to(opts["device"])

        model = build(x.shape[2], opts)
        describe(windows.shape[0], x.shape[2], model, opts)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        with autocast_ctx(opts):
            loss = loss_fn(model(x), y)
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
