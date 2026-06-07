# Deployment size on CPU: OpenNN vs PyTorch (Linux)

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-07. Linux x86_64.*

When you deploy a trained model to a CPU-only target — an edge device, an embedded board, a
slim container — the size of the inference runtime matters. This note compares the on-disk
footprint of an **OpenNN** application against **PyTorch**, CPU-only, on Linux x86_64.

## The two numbers

| | OpenNN | PyTorch |
|---|---|---|
| **CPU deployment size** | **3.2 MB** | **442 MB** |

OpenNN is **~140× smaller**. The OpenNN figure is the complete, ready-to-run executable; the
PyTorch figure is its core runtime library (`libtorch_cpu.so`) alone, before adding your
application on top.

## What each number is

**OpenNN — 3.2 MB (measured).** OpenNN is a C++ library that links statically into your
executable; its math backend, Eigen, is header-only and gets inlined, so there is no
separate runtime, interpreter, or BLAS library to ship. We built a complete tabular
example (`examples/iris_plant`: load a CSV, build and train a network, evaluate) on Linux
x86_64 with g++ 13.3, Release, `-DOpenNN_DISABLE_CUDA=ON`. The resulting stripped
executable is **3.23 MB** (3.70 MB unstripped). Its only shared-library dependencies, per
`ldd`, are the standard Linux runtime that ships with the OS/toolchain (`libstdc++`,
`libc`, `libm`, `libgcc_s`, `libgomp` for OpenMP, `libtbb` for parallel STL) — no CUDA, no
bundled BLAS.

**PyTorch — 442 MB (primary source).** `libtorch_cpu.so` from the official LibTorch CPU
distribution (`libtorch-shared-with-deps-2.12.0+cpu.zip`, Linux x86_64) is **441.8 MB**
(421.3 MiB) — 99% of the runtime libraries in that package. This single library holds the
ATen tensor/autograd engine with Intel MKL and oneDNN statically linked in, which is why it
is so large. Read directly from `download.pytorch.org/libtorch/cpu/` on 2026-06-07.

## Why the gap is so large

OpenNN links its (focused, fixed) set of layers and training algorithms into your binary,
keeping only the code you call, with header-only math. PyTorch ships a complete
general-purpose tensor runtime — plus statically-linked MKL/oneDNN — as one large library,
regardless of how little of it a given model uses.

## Caveats

* This is a **size** benchmark, not a capability or speed one. PyTorch is a general-purpose
  framework with a vast operator set, autograd, a JIT, and a large ecosystem; OpenNN is a
  focused native library. The size difference follows directly from that difference in scope.
* The comparison is deliberately like-for-like at the "what you ship" level: OpenNN's number
  already includes the application, whereas PyTorch's 442 MB is the runtime **before** your
  app or Python. Counting PyTorch's full installed footprint (the CPU wheel is ~703 MB
  installed, plus CPython and NumPy) widens the gap further.
* On **GPU** the picture differs: both stacks are dominated by NVIDIA's cuBLAS/cuDNN, so the
  size gap narrows. This note is specifically about CPU deployment.
* PyTorch can be trimmed with custom builds or the Edge runtime (ExecuTorch), at the cost of
  extra build effort; the 442 MB figure is the standard CPU distribution.

## Reproducing the OpenNN number

```bash
cmake -S . -B build-cpu -DOpenNN_DISABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DOpenNN_BUILD_EXAMPLES=ON
cmake --build build-cpu --config Release --target iris_plant -j
strip -o iris_plant.stripped build-cpu/examples/iris_plant/iris_plant   # or wherever the exe lands
ls -l iris_plant.stripped     # ~3.2 MB
ldd  build-cpu/.../iris_plant # standard Linux libs only — no CUDA, no BLAS
```
