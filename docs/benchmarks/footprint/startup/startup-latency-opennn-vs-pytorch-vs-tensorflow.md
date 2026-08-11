# Startup latency: OpenNN vs PyTorch vs TensorFlow

Size on disk is one cost of a heavy framework; **time** is another. Many applications run a model in
short, frequent bursts rather than one long session: a command-line tool invoked per file, a
serverless function that cold-starts on each request, a desktop app that must feel instant, an edge
device that wakes, predicts, and sleeps. For all of these, the clock that matters is
**time-to-first-prediction** — from process launch to the first output — and it is paid *every* time
the process starts.

## Contents

- [The numbers](#the-numbers)
- [Where the second-plus goes](#where-the-second-plus-goes)
- [Why OpenNN is faster to start](#why-opennn-is-faster-to-start)
- [Why it matters](#why-it-matters)
- [Caveats](#caveats)
- [References](#references)

## The numbers

|  | OpenNN | PyTorch | TensorFlow |
| --- | --- | --- | --- |
| **Time-to-first-prediction (median)** | **100 ms** (MKL build) | **1,536 ms** | **2,938 ms** |
| vs OpenNN | 1× | ≈15× | ≈29× |

Each program does the same thing: prepare a small MLP (10 → 64 → 1), run one forward pass, print the
result, and exit. We time the whole process, launch to exit, over several warm rounds, and report
the median.

**2026-08-11 measurement** (native Linux, i9-12900K, commit 52e21e15d, 7 rounds
per engine, process wall clock, GPU hidden so all three predict on CPU, the
same Python environment as the rest of the suite): OpenNN 100 ms median with
the MKL-linked binary used by the CPU benchmarks (the CUDA-less fallback start
of the CUDA build measures 93 ms, so the MKL runtime load costs only a few
milliseconds) vs PyTorch 1,536 ms and TensorFlow 2,938 ms. With the GPU
visible, the CUDA-build OpenNN binary pays context creation and starts in
~405 ms — still ~3.8× before PyTorch's CPU-only first prediction. This
supersedes the 2026-08-10 note revision, whose PyTorch/TensorFlow medians
(950 / 2,083 ms) came from the 2026-07 WSL run with a CPU-only `torch
2.12.0+cpu` wheel and were internally inconsistent with the measured cost of
the imports alone.

## Where the second-plus goes

The gap is almost entirely framework startup, not model work — the model here is trivial. Timed on
the same machine:

| Step | Time (2026-08-11) |
| --- | --- |
| OpenNN: whole process (launch → first prediction) | ~100 ms MKL build / ~405 ms CUDA build with GPU visible |
| Bare Python interpreter (`python -c pass`, no framework) | ~12 ms |
| Python + `import torch` alone | ~1,470 ms |
| Python + `import torch` + model + predict | ~1,536 ms |
| Python + `import tensorflow` alone | ~2,930 ms |
| Python + `import tensorflow` + model + predict | ~2,938 ms |

The standout: **importing the framework costs 1.5–3 seconds** — loading and initializing its large
native library dominates everything else (the model build and prediction add under 70 ms on top).
Python's own interpreter starts in ~12 milliseconds; it is the framework, not the language, that is
slow to load.

## Why OpenNN is faster to start

OpenNN is a native executable with the library statically linked in: the OS maps a ~3 MB binary and
jumps to `main`. There is no interpreter to boot and no large shared library to load and initialize.
PyTorch pays for the Python runtime plus the load-time initialization of `libtorch` (the same large
library measured in the `CPU size benchmark`) on every process start.

## Why it matters

- **Cold-start / serverless:** when you pay startup per invocation, ~1.5–3 s vs ~100 ms is the
  difference between a responsive function and a sluggish one.
- **CLI tools:** a command run once per file feels instant at ~100 ms and laggy at seconds.
- **Edge / duty-cycled devices:** a sensor that wakes, predicts, and sleeps spends far less energy
  and wall-clock time with a native binary.
- **Interactivity:** short-lived UI helper processes start without a visible delay.

## Caveats

- This is a **startup** benchmark: it measures time-to-first-prediction, not steady-state training
  or inference throughput on large models, where the picture is different and depends on the
  workload.
- The model is deliberately tiny so the numbers reflect framework startup, which is the point. A
  larger model adds compute time to *both* sides on top of these baselines.
- Measured on Linux x86_64 (g++ 13.3 MKL-linked OpenNN; PyTorch 2.13.0+cu130 and TensorFlow
  2.21.0 on CPython 3.12.3 — the same environment every other benchmark in this suite uses, with
  the GPU hidden so all three predict on CPU). Absolute numbers vary with machine, disk, and OS,
  but the order-of-magnitude gap is structural — interpreter + large shared library vs. a native
  binary.
- The CUDA-enabled `torch` wheel loads its GPU libraries at import even when no GPU is used; a
  CPU-only wheel imports faster (the 2026-07 WSL run measured 950 ms time-to-first-prediction
  with `2.12.0+cpu`). Deploying the slim wheel is the right move where available — the gap to a
  native binary remains ~10×.

## References

- [OpenNN](https://www.opennn.net/).
- [PyTorch](https://pytorch.org/).
- [TensorFlow](https://www.tensorflow.org/).
