# Peak memory: OpenNN vs PyTorch (Linux)

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-07. Linux x86_64.*

Disk size and startup time are two costs of a heavy runtime; **RAM** is a third. On a
constrained target — a small edge device, a memory-capped container, a function with a tight
memory limit, or simply a machine running many model processes at once — what matters is how
much resident memory a model process actually holds. A framework that loads a large runtime
pays for it in RAM the moment it starts, before any data is touched.

This note measures the resident-set size (RSS) of an OpenNN process versus a PyTorch process
doing the **same job**: load a tabular regression dataset, build an identical MLP, and train
it. We report RSS at two points — a **baseline** (framework loaded and model built, before
training) and the **peak** during training — so the source of the difference is visible.

## The numbers

| | OpenNN | PyTorch | TensorFlow |
|---|---|---|---|
| **Baseline RSS** (model built, no training) | **9 MB** | **221 MB** | **485 MB** |
| **Peak RSS** (during training) | **9 MB** | **295 MB** | **521 MB** |
| Peak vs OpenNN | 1× | ≈32× | ≈56× |

All three programs do the same thing on the same data: load `sum.csv` (1,000 rows × 100 numeric
inputs + 1 target), build a 100 → 64 → 1 MLP, and train for 50 epochs (Adam, batch size 32,
single-threaded). Each reports its own peak RSS via the OS (`getrusage` / `resource`).

## What the numbers show

* **OpenNN holds ~9 MB and barely moves.** Baseline and peak are the same to within
  measurement noise — the dataset and training buffers are tiny next to the already-small
  working set. The whole process, code and data, fits in single-digit megabytes.
* **PyTorch starts at ~221 MB** before training — the Python interpreter plus the `libtorch`
  runtime (and NumPy) resident in memory — and **rises to ~295 MB** during training as autograd
  and optimizer buffers are allocated.
* **TensorFlow starts at ~485 MB** and **rises to ~521 MB** — the Keras/TF runtime carries an
  even larger resident footprint than PyTorch before any training.

So the peak gap is ~32× for PyTorch and ~56× for TensorFlow. As with startup time, most of it is
fixed framework overhead that is paid regardless of how small the model is.

## Why OpenNN uses so little

OpenNN is a native binary with the library linked in and Eigen (header-only) for math. The
resident memory is essentially the model parameters, the data, and a small amount of code —
there is no interpreter and no large general-purpose tensor runtime mapped into the process.
PyTorch keeps the Python runtime and the full `libtorch` engine resident for the life of the
process, which sets a high floor independent of model size.

## Why it matters

* **Memory-capped containers / functions:** a 256 MB function can host the OpenNN process many
  times over; PyTorch's baseline alone nearly fills it before any work.
* **Many concurrent model processes:** at ~9 MB each, you can run far more OpenNN workers per
  machine than ~250–300 MB PyTorch ones.
* **Small edge devices:** RAM is often scarcer than disk; a single-digit-MB footprint leaves
  room for the rest of the application.

## Caveats

* This is a **memory** benchmark on a small model, chosen so the numbers reflect framework
  overhead — the structural difference — rather than a specific large workload. A bigger model
  or dataset adds parameter/activation memory to *both* sides on top of these baselines.
* Measured on Linux x86_64, single-threaded (`OMP_NUM_THREADS=1`) for both, to avoid
  thread-pool arenas inflating RSS differently. OpenNN built with g++ 13.3 (CPU-only); PyTorch
  2.12.0+cpu on CPython 3.12 with NumPy installed.
* RSS is the OS's peak resident size (`ru_maxrss`); absolute values vary with allocator, glibc,
  and thread settings, but the order-of-magnitude gap is structural.
* CPU-only on both sides. A CUDA build adds GPU-side memory, which this note does not cover.

## Reproducing

The two equivalent programs are in [`docs/benchmarks/memory/`](memory/):
`opennn_memory.cpp` (build it against the OpenNN library) and `pytorch_memory.py`. Put
`sum.csv` (from `datasets/`) in the working directory and run each with `OMP_NUM_THREADS=1`;
both print `baseline_rss_mb` and `peak_rss_mb`.
