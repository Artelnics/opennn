# Deployment size on CPU: OpenNN vs PyTorch

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-07.*

When you deploy a trained model to a CPU-only target — an edge device, an embedded
board, a slim container, a desktop app that must not ship gigabytes — the size of the
inference runtime matters. This note compares the on-disk footprint of an **OpenNN**
application against **PyTorch**, CPU-only, with every number either measured directly or
cited from a primary source.

## Summary

| | OpenNN (CPU) | PyTorch (CPU) |
|---|---|---|
| What you ship | your `.exe` with OpenNN linked in | your app **+** the PyTorch runtime library |
| Runtime library size | **0 MB** (statically linked into the exe) | **~442 MB** (`libtorch_cpu`) |
| Minimal real app | **~1.4 MB** (tabular model exe) | runtime alone is **~442 MB** |
| Extra math runtime to ship | none (header-only Eigen, inlined) | none separate — MKL/oneDNN are **inside** the 442 MB lib |

For a CPU-only deployment, an OpenNN application is roughly **two orders of magnitude
smaller** than a PyTorch one. The reason is structural, not incidental: OpenNN is a C++
library that links into your binary, whereas PyTorch ships a large general-purpose tensor
runtime.

## OpenNN — measured

OpenNN is a header-and-source C++ library. It compiles **into** your executable (static
link by default), and its linear-algebra backend, Eigen, is header-only and gets inlined.
The linker keeps only the code your program actually references. There is no framework
process, no interpreter, and no separate math DLL to redistribute.

Measured on Windows (MSVC 19.50 / Visual Studio 2026, x64, Release,
`-DOpenNN_DISABLE_CUDA=ON`), a complete tabular classification example
(`examples/iris_plant`: load a CSV, build and train a network, evaluate):

| Component | Size |
|---|---|
| `iris_plant.exe` (OpenNN statically linked in) | **1.44 MB** |
| Visual C++ runtime DLLs (`MSVCP140`, `VCRUNTIME140`, `VCRUNTIME140_1`, `MSVCP140_ATOMIC_WAIT`, `VCOMP140`) | ~0.94 MB |
| **Total redistributable** | **~2.4 MB** |

The only non-application dependencies are the Microsoft Visual C++ Redistributable DLLs
(commonly already present on Windows) and Windows' own system libraries — confirmed by
`dumpbin /dependents` on the binary. No CUDA, no BLAS/LAPACK DLL.

A larger app that uses the vision stack (convolution, attention) links more OpenNN code:
the YOLO object-detection example builds to **~3.3 MB**. Either way, the deliverable is a
single-digit-megabyte executable plus the ~1 MB MSVC runtime.

## PyTorch — primary sources

The figures below are for **`torch` / `libtorch` 2.12.0 (CPU)**, the latest stable
release at the time of writing, read directly from PyTorch's official distribution servers
(`download.pytorch.org`, `pypi.org`) on 2026-06-07. They are *not* estimates. We
distinguish **compressed** (download) from **installed** (on-disk) sizes, and we note the
platform, because both are common sources of error in size comparisons.

**LibTorch — the C++ runtime (the fair comparison to "OpenNN code in your binary"):**

| Item | Size | Source |
|---|---|---|
| `libtorch-shared-with-deps-2.12.0+cpu.zip` (Linux x86_64), download | **128.0 MB** (127,970,405 B) | download.pytorch.org/libtorch/cpu/ |
| …unpacked (libs + headers) | **484.5 MB** (462.1 MiB) | `unzip -l` of the above |
| `libtorch_cpu.so` (the runtime library you actually link/ship) | **441.8 MB** (421.3 MiB) — 99% of the runtime libs | inside the zip |
| `libtorch-win-shared-with-deps-2.12.0+cpu.zip` (Windows), download | **201.8 MB** (201,827,610 B) | download.pytorch.org/libtorch/cpu/ |

**The `torch` Python wheel (CPU):**

| Item | Size | Source |
|---|---|---|
| `torch-2.12.0+cpu` wheel, Windows x86_64 (cp312), download | **122.9 MB** (122,900,744 B) | download.pytorch.org/whl/cpu |
| `torch-2.12.0+cpu` wheel, Linux x86_64 (cp312), download | **192.3 MB** (192,268,791 B) | download.pytorch.org/whl/cpu |
| …installed on disk (Linux) | **~703 MB** (670.7 MiB, 12,704 files) | unzip of the wheel |
| dominant file: `libtorch_cpu.so` | **442.2 MB** (421.7 MiB) — ~63% of the install | unzip of the wheel |

Two accuracy caveats worth stating plainly:

* **`pip install torch` does not give you the CPU build on Linux.** PyPI's default Linux
  `torch` wheel **bundles CUDA** and is ~532 MB to download; the CPU build requires
  `--index-url https://download.pytorch.org/whl/cpu`. PyTorch maintainers note this
  explicitly ([pytorch/pytorch#17621](https://github.com/pytorch/pytorch/issues/17621)).
* **MKL and oneDNN are not separate files** — they are statically linked *into*
  `libtorch_cpu`, which is why that single library is ~442 MB.

A Python deployment also needs the CPython interpreter (~150 MB+) and dependencies
(NumPy, SymPy, etc.), which we exclude here to keep the comparison to the ML runtime
itself.

## Why the gap is so large

1. **Static C++ linking vs. a shipped runtime.** OpenNN's code is pulled into your exe and
   trimmed to what you call. PyTorch ships a complete tensor/autograd engine as a
   redistributable library regardless of how little of it a given model uses.
2. **Header-only math vs. a bundled BLAS.** OpenNN uses Eigen (header-only, inlined). PyTorch
   statically links Intel MKL + oneDNN, which dominate `libtorch_cpu`.
3. **No interpreter.** OpenNN deployments are native executables. The PyTorch Python path
   additionally requires CPython and its package tree.

## Caveats and scope

* **Platform.** OpenNN figures are measured on Windows/MSVC; the OpenNN code size is
  comparable on Linux/GCC (same static-link, header-only-Eigen model). PyTorch's
  most-cited figures (`libtorch_cpu.so` 442 MB, Linux wheel 192 MB) are Linux; the Windows
  wheel is smaller (123 MB). We give both platforms above so nothing is misattributed.
* **This is a size benchmark, not a capability or speed benchmark.** PyTorch is a
  general-purpose framework with an enormous operator set, autograd, a JIT, and a vast
  ecosystem. OpenNN implements a focused, fixed set of layers, networks, and training
  algorithms. The size advantage follows directly from that narrower, native scope.
* **Trimming PyTorch.** Custom/stripped builds and the PyTorch Edge runtime (ExecuTorch)
  can be much smaller — ExecuTorch advertises a "~50 KB" core runtime *without* operators
  or backends ([executorch README](https://github.com/pytorch/executorch)), though real
  deployable binaries grow once the kernels a model needs are linked in. These require
  extra build effort; the figures above are for the standard distributions.
* **GPU is a different story.** On GPU both stacks are dominated by NVIDIA's libraries
  (cuBLAS, cuDNN), so the size gap narrows considerably. This note is specifically about
  **CPU** deployment, where OpenNN's native, dependency-light design is most visible.

## Reproducing the OpenNN numbers

```bash
cmake -S . -B build-cpu -DOpenNN_DISABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DOpenNN_BUILD_EXAMPLES=ON
cmake --build build-cpu --config Release --target iris_plant
# then measure build-cpu/bin/Release/iris_plant.exe and check its imports:
#   dumpbin /dependents iris_plant.exe   (Windows)
#   ldd iris_plant                       (Linux)
```
