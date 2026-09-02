#!/usr/bin/env python3
"""The CNN family in PyTorch, defined once, driven four ways.

PLAN.md; the counterpart of cnn.cpp and deliberately its mirror image. Same
modes, same positional arguments, same `key=value` output, so run.py drives
either engine by swapping the command prefix.

  cnn.py train    <train_dir> <test_dir> [epochs] [batch,...] [size] [dev] [prec]
  cnn.py infer    <test_dir>             [reps]   [batch,...] [size] [dev] [prec]
  cnn.py capacity <train_dir>            [batch]              [size] [dev] [prec]
  cnn.py quality  <train_dir> <test_dir> [epochs] [batch]     [size] [dev] [prec]

`torchvision.models.resnet50` *is* ResNet-50 v1.5 -- the stride sits in the
3x3 convolution -- which is what cnn.cpp builds from bottleneck blocks
[3,4,6,3] over widths [64,128,256,512]. Using the library definition rather
than hand-rolling one keeps the comparison against the citable network instead
of against our transcription of it.

Images are lazy-loaded per batch from class folders by a DataLoader with
worker processes that decode the JPEGs every epoch, against OpenNN's
per-batch reads of its pre-decoded image cache: at 50,000 x 224x224x3 the
split cannot be resident, so this measures convolution throughput *plus*
input-pipeline efficiency in both engines. The two pipelines are not the same
work -- PT_INPUT=cache feeds PyTorch from OpenNN's cache file to measure how
much of the training margin is the decode.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED = 42
WORKERS = int(os.environ.get("PT_WORKERS", "8"))
INPUT = os.environ.get("PT_INPUT", "jpeg")
# Inference in bf16 can keep autocast on (weights re-cast on every call, which
# torch.compile does not fold without freezing) or store the weights in bf16
# once, the way OpenNN keeps a bf16 mirror of its parameters. PT_INFER_CAST
# selects it; the published cell uses whichever measured faster.
INFER_CAST = os.environ.get("PT_INFER_CAST", "autocast")

def report_blas() -> None:
    """Which BLAS this engine dispatches to, printed like OpenNN prints it.

    PyTorch's is fixed at build time -- the wheels are MKL-backed -- while
    OpenNN's is chosen at runtime. Recording both is what stops a comparison
    quietly turning into Eigen against MKL.
    """
    settings = torch.__config__.show()
    match = re.search(r"BLAS_INFO=(\w+)", settings)

    print(f"blas={match.group(1) if match else 'unknown'}")


def build(opts: dict) -> torch.nn.Module:
    """The CNN family. Nothing else here constructs the network."""
    torch.manual_seed(SEED)

    model = resnet50(weights=None, num_classes=1000).to(opts["device"])
    if opts["device"] == "cuda":
        # channels_last is how a convolution reaches the tensor cores; without
        # it PyTorch is measured on a layout it would never ship.
        model = model.to(memory_format=torch.channels_last)
    return model

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

def parse_opts(argv: list[str], first: int) -> dict:
    def at(index: int, default: str) -> str:
        return argv[index] if len(argv) > index else default

    size = int(at(first, "224"))
    device = at(first + 1, "cuda")
    precision = at(first + 2, "fp32")

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

    return {"size": size, "device": device, "precision": precision,
            "autocast": precision == "bf16", "tf32": allow_tf32}

class Folders(ImageFolder):
    """ImageFolder that ignores dot-directories.

    OpenNN's ImageDataset writes its per-batch binary cache into `.cache/`
    beside the class folders, and ImageFolder would otherwise enrol it as a
    1001st class and then fail on it for holding no images. Skipping hidden
    directories keeps both engines reading the same 1000 classes from the same
    tree, which is the point of the shared layout.
    """

    def find_classes(self, directory):
        names = sorted(e.name for e in os.scandir(directory)
                       if e.is_dir() and not e.name.startswith("."))
        return names, {name: index for index, name in enumerate(names)}

class CachedImages(torch.utils.data.Dataset):
    """OpenNN's pre-decoded image cache, read the way OpenNN reads it.

    A controlled variant (`PT_INPUT=cache`), not the published cell. The
    published training cell has PyTorch decode JPEGs per epoch while OpenNN
    reads `.cache/images.bin` -- uint8 HxWxC per image, sorted-folder /
    sorted-file order, a signature trailer -- which OpenNN's ImageDataset
    writes beside the class folders on its first pass at a given size. Feeding
    PyTorch from the same file isolates the input-pipeline asymmetry from the
    convolution throughput; it requires OpenNN to have opened the folder at
    this size once. Labels still come from ImageFolder, whose class and file
    order the cache shares.
    """

    def __init__(self, path: str, size: int):
        folder = Folders(path)
        self.targets = folder.targets
        self.shape = (size, size, 3)
        self.bytes = size * size * 3
        self.cache = os.path.join(path, ".cache", "images.bin")
        expected = len(self.targets) * self.bytes
        actual = os.path.getsize(self.cache) if os.path.exists(self.cache) else 0
        if actual < expected:
            raise SystemExit(f"{self.cache}: {actual} bytes, at least {expected} expected "
                             f"for {len(self.targets)} images at {size}x{size}")
        self.fd = None

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        if self.fd is None:                        # one descriptor per worker process
            self.fd = os.open(self.cache, os.O_RDONLY)
        raw = os.pread(self.fd, self.bytes, index * self.bytes)
        pixels = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(self.shape)
        return pixels.permute(2, 0, 1).float().div_(255.0), self.targets[index]

def loader_for(path: str, batch: int, opts: dict, shuffle: bool) -> DataLoader:
    """Class folders, decoded per batch by worker processes -- or, with
    PT_INPUT=cache, OpenNN's pre-decoded cache read by the same workers."""
    if INPUT == "cache":
        dataset = CachedImages(path, opts["size"])
    else:
        dataset = Folders(path, transforms.Compose([
            transforms.Resize(opts["size"]),
            transforms.CenterCrop(opts["size"]),
            transforms.ToTensor(),
        ]))
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle,
                      num_workers=WORKERS, pin_memory=True, drop_last=True,
                      persistent_workers=WORKERS > 0)

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

def to_device(images, opts: dict):
    if opts["device"] != "cuda":
        return images
    return images.to("cuda", non_blocking=True, memory_format=torch.channels_last)

def train_like(argv: list[str], mode: str) -> int:
    epochs = int(argv[4]) if len(argv) > 4 else 1
    batches = batches_of(argv[5] if len(argv) > 5 else "128")
    opts = parse_opts(argv, 6)

    print(f"baseline_rss_mib={resident_mib():.1f}")
    print(f"engine=pytorch\nmode={mode}\ndevice={opts['device']}")
    report_blas()
    warmup = 1 if mode == "train" else 0

    for batch in batches:
        loader = loader_for(argv[2], batch, opts, shuffle=True)
        print(f"samples={len(loader) * batch}")

        model = build(opts)
        if batch == batches[0]:
            print(f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        ctx = autocast_ctx(opts)

        def step(images, labels):
            optimizer.zero_grad(set_to_none=True)
            with ctx:
                loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

        step_fn, how = compiled(step, opts)

        def run_epoch():
            model.train()
            for images, labels in loader:
                step_fn(to_device(images, opts), labels.to(opts["device"], non_blocking=True))
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

        print(f"batch_{batch}_samples_per_sec={int(len(loader) * batch / median)}"
              f" median_epoch_s={median:.6g}")
        print(f"batch_{batch}_epoch_times={','.join(f'{t:.6g}' for t in times)}")
        print(f"batch_{batch}_mode={how}", flush=True)

    print("RESULT=OK")
    return 0

def infer(argv: list[str]) -> int:
    reps = int(argv[3]) if len(argv) > 3 else 1
    batches = batches_of(argv[4] if len(argv) > 4 else "128")
    opts = parse_opts(argv, 5)

    print(f"baseline_rss_mib={resident_mib():.1f}")
    print(f"engine=pytorch\nmode=infer\ndevice={opts['device']}")
    report_blas()

    for batch in batches:
        loader = loader_for(argv[2], batch, opts, shuffle=False)
        samples = len(loader) * batch
        print(f"samples={samples}")

        model = build(opts).eval()
        weights_bf16 = opts["autocast"] and INFER_CAST == "weights"
        if weights_bf16:
            model = model.to(torch.bfloat16)
            opts = dict(opts, autocast=False)
        if batch == batches[0]:
            print(f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)
        forward, _ = compiled(model, opts)

        # One batch, filled once and replayed, matching cnn.cpp: this times the
        # resident forward pass rather than the image decode, which the
        # training cell already accounts for.
        images = to_device(next(iter(loader))[0], opts)
        if weights_bf16:
            images = images.to(torch.bfloat16)

        def run_pass():
            with torch.no_grad(), autocast_ctx(opts):
                for _ in range(len(loader)):
                    forward(images)
            sync(opts)

        run_pass()

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
        print(f"batch_{batch}_samples_per_sec={int(samples / median)}"
              f" median_pass_s={median:.6g}", flush=True)

    print("RESULT=OK")
    return 0

def capacity(argv: list[str]) -> int:
    batch = int(argv[3]) if len(argv) > 3 else 128
    opts = parse_opts(argv, 4)

    print(f"baseline_rss_mib={resident_mib():.1f}")
    print(f"engine=pytorch\nmode=capacity\ndevice={opts['device']}\nbatch={batch}")
    report_blas()

    try:
        loader = loader_for(argv[2], batch, opts, shuffle=False)
        images, labels = next(iter(loader))

        model = build(opts)
        print(f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        with autocast_ctx(opts):
            loss = loss_fn(model(to_device(images, opts)),
                           labels.to(opts["device"], non_blocking=True))
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
