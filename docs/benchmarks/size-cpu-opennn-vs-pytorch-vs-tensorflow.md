# Deployment size on CPU: OpenNN vs PyTorch vs TensorFlow (Linux)

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-07. Linux x86_64.*

Not every model gets to run in a data center. A vibration monitor on a factory motor, a
camera on a drone, a sensor hub in a medical device, a point-of-sale terminal, a microcontroller
gateway on a remote pipeline — for a growing number of applications, the model has to run
*on the device*, close to the data, with no GPU and often no network. On those targets the
constraint is rarely raw compute; it is **space**. Storage is measured in tens or hundreds of
megabytes, firmware images have hard size budgets, and over-the-air updates have to fit
through a thin pipe.

That is exactly where the usual deep-learning stack becomes the problem. A mainstream
framework's CPU runtime is enormous: PyTorch's `libtorch_cpu` library alone is **~442 MB** —
before you add your own application, the Python interpreter, or anything else. On a device
with a 256 MB partition, the inference engine doesn't even fit. The model isn't the problem;
the library is.

This is the gap **OpenNN** is built for. Because it is a C++ library that compiles directly
into your executable — with header-only math and no bundled tensor runtime — an equivalent
CPU application ships in **~3.2 MB**. Same job, ≈138× less to deploy.

This note compares the two, CPU-only, on Linux x86_64.

## The numbers

| | OpenNN | ONNX Runtime | PyTorch | TensorFlow |
|---|---|---|---|---|
| **CPU runtime library** | **3.2 MB** | **22 MB** | **442 MB** | **752 MB** |
| vs OpenNN | 1× | ≈7× | ≈138× | ≈235× |

Note this is generous to the others: the OpenNN figure is a complete, ready-to-run
application, while the rest are only the core runtime library (`libonnxruntime.so` /
`libtorch_cpu.so` / `libtensorflow_cc.so`), before your own code or the Python interpreter are
added on top. ONNX Runtime is an inference-only engine (it cannot train), included here because
its runtime library is directly comparable.

## What each number is

**OpenNN — 3.2 MB (measured).** OpenNN is a C++ library that links statically into your
executable; its math backend, Eigen, is header-only and gets inlined, so there is no
separate runtime, interpreter, or BLAS library to ship. A complete OpenNN application,
built on Linux x86_64 (g++ 13.3, Release, `-DOpenNN_DISABLE_CUDA=ON`), is **~3.2 MB** as a
stripped executable. The size is set by the OpenNN code linked into the binary rather than
by the network you build, so it is representative rather than a hand-picked best case. Its
only shared-library dependencies, per `ldd`, are the standard Linux runtime that ships with
the OS/toolchain (`libstdc++`, `libc`, `libm`, `libgcc_s`, `libgomp` for OpenMP, `libtbb`
for parallel STL) — no CUDA, no bundled BLAS.

And that ~3.2 MB is the *whole* library: **the same binary both trains and runs inference.**
The measured application builds a network, trains it, and evaluates it — all in one
executable. There is no separate, slimmed-down inference runtime to export to, as is usual
when a model is trained with a heavy framework and then shipped to a lightweight engine. A
device running OpenNN can not only run a model but also retrain or fine-tune it on-device,
at the same footprint.

**PyTorch — 442 MB (primary source).** `libtorch_cpu.so` from the official LibTorch CPU
distribution (`libtorch-shared-with-deps-2.12.0+cpu.zip`, Linux x86_64) is **441.8 MB**
(421.3 MiB) — 99% of the runtime libraries in that package. This single library holds the
ATen tensor/autograd engine with Intel MKL and oneDNN statically linked in, which is why it
is so large. Read directly from `download.pytorch.org/libtorch/cpu/` on 2026-06-07.

**TensorFlow — 752 MB (measured).** `libtensorflow_cc.so.2`, the core C++ runtime in the
`tensorflow-cpu` 2.21.0 wheel (Linux x86_64, CPython 3.12), is **751.6 MB** — the single
largest file in the install. With its companion `libtensorflow_framework.so.2` (62 MB) the
runtime is ~814 MB, and the whole `tensorflow/` package directory is ~1.3 GB. Measured by
sizing the `.so` files in a clean virtual environment.

**ONNX Runtime — 22 MB (measured).** `libonnxruntime.so` from the `onnxruntime` 1.26.0 wheel
(Linux x86_64) is **22.0 MB** — far smaller than the full frameworks because ORT is a
purpose-built *inference* engine, not a training framework. It is still ~7× the OpenNN
executable, and unlike OpenNN it cannot train: you ship a model trained elsewhere plus this
runtime, whereas OpenNN's 3.2 MB binary both trains and infers.

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

Any of the example targets gives ~3.2 MB; `iris_plant` is used here just as a concrete
command.

```bash
cmake -S . -B build-cpu -DOpenNN_DISABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DOpenNN_BUILD_EXAMPLES=ON
cmake --build build-cpu --config Release --target iris_plant -j
strip -o iris_plant.stripped build-cpu/examples/iris_plant/iris_plant   # or wherever the exe lands
ls -l iris_plant.stripped     # ~3.2 MB
ldd  build-cpu/.../iris_plant # standard Linux libs only — no CUDA, no BLAS
```
