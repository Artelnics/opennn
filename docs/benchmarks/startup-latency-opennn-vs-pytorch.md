# Startup latency: OpenNN vs PyTorch (Linux)

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-07. Linux x86_64.*

Size on disk is one cost of a heavy framework; **time** is another. Many applications run a
model in short, frequent bursts rather than one long session: a command-line tool invoked per
file, a serverless function that cold-starts on each request, a desktop app that must feel
instant, an edge device that wakes, predicts, and sleeps. For all of these, the clock that
matters is **time-to-first-prediction** — from process launch to the first output — and it is
paid *every* time the process starts.

A framework with a large runtime pays that cost up front. Before PyTorch can run a single
layer, it must start the Python interpreter and `import torch`, which loads its multi-hundred-
megabyte native library. OpenNN, a native binary with the library linked in, simply runs.

## The two numbers

| | OpenNN | PyTorch | PyTorch / OpenNN |
|---|---|---|---|
| **Time-to-first-prediction (median)** | **36 ms** | **1,005 ms** | **≈ 28×** |

Each program does the same thing: build a small MLP (10 → 64 → 1), run one forward pass, print
the result, and exit. We time the whole process, launch to exit, over 15 runs after warm-up,
and report the median. (Measured on Linux x86_64; OpenNN built with g++ 13.3, CPU-only;
PyTorch 2.12.0+cpu on CPython 3.12.)

## Where PyTorch's second goes

The gap is almost entirely framework startup, not model work — the model here is trivial. Timed
on the same machine:

| Step | Time |
|---|---|
| OpenNN: whole process (launch → first prediction) | ~36 ms |
| Bare Python interpreter (`python -c pass`, no torch) | ~9 ms |
| Python + `import torch` + model + predict | ~1,005 ms |
| → `import torch` alone adds | **~995 ms** |

The standout: **`import torch` by itself costs ~1 second** — loading and initializing the
framework's large native library dominates everything else. Python's own interpreter starts in
single-digit milliseconds; it is the framework, not the language, that is slow to load.

## Why OpenNN is faster to start

OpenNN is a native executable with the library statically linked in: the OS maps a ~3 MB binary
and jumps to `main`. There is no interpreter to boot and no large shared library to load and
initialize. PyTorch pays for the Python runtime plus the load-time initialization of
`libtorch` (the same large library measured in the
[CPU size benchmark](size-cpu-opennn-vs-pytorch.md)) on every process start.

## Why it matters

* **Cold-start / serverless:** when you pay startup per invocation, ~1 s vs ~36 ms is the
  difference between a responsive function and a sluggish one.
* **CLI tools:** a command run once per file feels instant at ~36 ms and laggy at ~1 s.
* **Edge / duty-cycled devices:** a sensor that wakes, predicts, and sleeps spends far less
  energy and wall-clock time with a native binary.
* **Interactivity:** short-lived UI helper processes start without a visible delay.

## Caveats

* This is a **startup** benchmark: it measures time-to-first-prediction, not steady-state
  training or inference throughput on large models, where the picture is different and depends
  on the workload.
* The model is deliberately tiny so the numbers reflect framework startup, which is the point.
  A larger model adds compute time to *both* sides on top of these baselines.
* Measured on Linux x86_64 (g++ 13.3 CPU-only OpenNN; PyTorch 2.12.0+cpu, CPython 3.12).
  Absolute numbers vary with machine, disk, and OS, but the order-of-magnitude gap is
  structural — interpreter + large shared library vs. a native binary.
* PyTorch numbers are CPU-only; a CUDA build's `import torch` is typically slower still, as it
  also initializes the GPU libraries.

## Reproducing

The two equivalent programs are in [`docs/benchmarks/startup/`](startup/):
`opennn_startup.cpp` (build it against the OpenNN library) and `pytorch_startup.py`. Time each
end-to-end and take the median after a few warm-up runs, e.g.:

```bash
hyperfine --warmup 3 ./opennn_startup 'python pytorch_startup.py'
# or a simple loop with: t0=$(date +%s.%N); ./opennn_startup; t1=$(date +%s.%N)
```
