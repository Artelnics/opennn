# OpenNN benchmarks: OpenNN vs PyTorch vs TensorFlow

*For [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-08.*

These notes compare the **deployment characteristics** of OpenNN against PyTorch and
TensorFlow — how much a trained model costs to ship, start, and run, not how fast it trains.
The theme is the same throughout: OpenNN is a native C++ library that links into your
executable, so its footprint is a small binary; PyTorch and TensorFlow are large general-purpose
frameworks with a runtime that must be loaded, installed, or shipped alongside the model.

Every number below is **measured on Linux x86_64** (OpenNN built with g++ 13.3; PyTorch
2.12.0, TensorFlow 2.21.0, CPython 3.12), or read from a primary source where noted. Each row
links to the full note with methodology and reproduction steps.

## Summary

| Benchmark | OpenNN | PyTorch | TensorFlow |
|---|---:|---:|---:|
| [CPU runtime size](size-cpu-opennn-vs-pytorch-vs-tensorflow.md) | **3.2 MB** | 442 MB | 752 MB |
| [GPU (CNN) deployment](size-gpu-opennn-vs-pytorch-vs-tensorflow.md) | **~1.3 GB** | ~5.0 GB | ~6.2 GB |
| [Startup latency](startup-latency-opennn-vs-pytorch-vs-tensorflow.md) | **36 ms** | 1,005 ms | 1,685 ms |
| [Peak memory (training)](peak-memory-opennn-vs-pytorch-vs-tensorflow.md) | **9 MB** | 295 MB | 521 MB |
| [Install size / packages](dependencies-opennn-vs-pytorch-vs-tensorflow.md) | **1 file, 0 pkgs** | 946 MB, 12 pkgs | 1.6 GB, 33 pkgs |
| [Native source LOC](loc-opennn-vs-pytorch-vs-tensorflow.md) | **34,926** | 834,319 | 1,792,182 |
| [Standalone code export](code-export-opennn-vs-pytorch-vs-tensorflow.md) | **C/Py/JS/PHP source** | needs a runtime | needs a runtime |

OpenNN is the smallest on every footprint axis, starts an order of magnitude faster, and is the
only one of the three that can export a trained model as dependency-free source.

## The benchmarks

* **[Deployment size on CPU](size-cpu-opennn-vs-pytorch-vs-tensorflow.md)** — the runtime library a CPU app
  ships: a 3.2 MB OpenNN executable vs the 442 MB `libtorch_cpu` / 752 MB `libtensorflow_cc`.
* **[Deployment size on GPU](size-gpu-opennn-vs-pytorch-vs-tensorflow.md)** — a CNN's CUDA footprint. Here the
  gap narrows because NVIDIA's cuBLAS/cuDNN dominate all three; OpenNN still wins by shipping
  only the libraries its model loads.
* **[Startup latency](startup-latency-opennn-vs-pytorch-vs-tensorflow.md)** — time-to-first-prediction. A
  native binary starts in tens of milliseconds; `import torch` / `import tensorflow` cost ~1–1.7 s.
* **[Peak memory](peak-memory-opennn-vs-pytorch-vs-tensorflow.md)** — resident memory for the same small
  training job: ~9 MB vs hundreds of MB of fixed framework overhead.
* **[Dependencies & install friction](dependencies-opennn-vs-pytorch-vs-tensorflow.md)** — one self-contained
  file that runs on a clean machine, vs Python plus a 12–33 package tree.
* **[Source lines of code](loc-opennn-vs-pytorch-vs-tensorflow.md)** — the size of the native library layer
  behind each project.
* **[Model export to standalone code](code-export-opennn-vs-pytorch-vs-tensorflow.md)** — OpenNN emits a
  trained model as compilable C/Python/JavaScript/PHP; the frameworks export model files that
  still need their runtime.

## Scope and fairness

* These are **size, startup, memory, and packaging** benchmarks — not training or inference
  throughput. PyTorch and TensorFlow are general-purpose frameworks with vast operator sets,
  autograd, JIT, and large ecosystems; OpenNN is a focused native library. The footprint
  differences follow directly from that difference in scope.
* The comparisons are deliberately like-for-like on the same hardware, same data, same model,
  and (for memory) single-threaded on all sides. Each note states its method and how to
  reproduce it.
* Where a figure favors OpenNN by a smaller margin (GPU size) or carries a methodology caveat
  (LOC test files), the note says so explicitly.
