<div align="center">
  <img src="http://www.opennn.net/images/opennn_git_logo.svg" alt="OpenNN logo" width="280">
</div>

# OpenNN

OpenNN is a high-performance C++ library for neural networks, deep learning, and advanced analytics.

> Fast, portable, and production-ready C++ neural network code with CPU and optional CUDA support.

## Why OpenNN?

- Written in modern **C++20** with a focus on predictable behavior and explicit control.
- Supports **CPU** and **CUDA** backends with cuDNN integration.
- Uses **CMake** for cross-platform builds and easy integration.
- Automatically fetches dependencies like **Eigen** and **googletest** during configure.
- Designed for numerical stability, memory efficiency, and real-world analytics.
- Easier to embed directly into native C++ applications than Python-first frameworks.
- Avoids a separate interpreter layer, so deployment and tooling stay closer to the system.
- Gives developers direct control over compilation, hardware targets, and runtime configuration.
- OpenNN is a good fit for projects where the neural network code must behave like any other C++ component.

## Features

- Feed-forward neural networks
- Convolutional and recurrent layers
- Transformers and attention mechanisms
- Runtime precision selection with FP32 and BF16 support on compatible CUDA GPUs
- Loss functions and optimization algorithms
- Model selection, data preprocessing, and training strategies
- Model export to standalone C, embedded C, Python, JavaScript, and PHP
- TinyML-oriented export checks for AVR and ARM Cortex-M targets
- Optional benchmark suite comparing OpenNN with PyTorch and TensorFlow
- Example applications for CPU and GPU

## Repository layout

- `opennn/` - core library sources, public headers, and CMake package export rules
- `examples/` - example applications and bundled small example datasets
- `tests/` - GoogleTest-based unit and validation tests, in folders mirroring
  the library so each test sits beside what it exercises
- `benchmarks/` - reproducible benchmark suite and benchmark methodology

The library itself is split by responsibility, and every include spells out the
folder it comes from — `#include "opennn/network/layers/dense_layer.h"`:

- `opennn/core/` - tensor types and operations, device backend, memory, generic
  utilities; `core/cuda/` holds the CUDA kernels
- `opennn/network/` - the network, its `layers/` and `operators/`,
  forward and back propagation, expression export
- `opennn/dataset/` - tabular, image, language, time series and YOLO datasets
- `opennn/training/` - losses and optimization algorithms
- `opennn/model_selection/` - inputs and neurons selection, genetic algorithm
- `opennn/evaluation/` - testing analysis

They depend on each other in that order, top to bottom: `core` knows nothing
about the rest, and `evaluation` may use everything above it. Datasets
sit above the network because the language datasets tokenize and the YOLO
dataset builds detection targets, while nothing in `network/` includes
a dataset.

## Quick start

### Requirements

- C++20 compiler with `std::format` support
  - GCC 13+
  - Clang 17+
  - MSVC 2022+
- CMake 3.24+
- Optional: CUDA Toolkit and **cuDNN 9.0+** for GPU builds

### Build CPU-only

Create a separate build directory outside the repository folder:

```bash
cmake -S . -B ../opennn-build -DCMAKE_BUILD_TYPE=Release -DOpenNN_DISABLE_CUDA=ON
cmake --build ../opennn-build --config Release
```

### Build with CUDA

```bash
cmake -S . -B ../opennn-build -DCMAKE_BUILD_TYPE=Release -DOpenNN_REQUIRE_CUDA=ON
cmake --build ../opennn-build --config Release
```

### Build examples and tests

```bash
cmake -S . -B ../opennn-build -DCMAKE_BUILD_TYPE=Release -DOpenNN_BUILD_EXAMPLES=ON -DOpenNN_BUILD_TESTS=ON
cmake --build ../opennn-build --config Release
```

### Run tests

```bash
../opennn-build/bin/opennn_tests
```

When using a multi-config generator such as Visual Studio, the test binary may be under `../opennn-build/bin/Release/`.

### Fast developer verification

The repository includes CPU/CUDA verification wrappers that keep incremental
builds outside the checkout and run focused tests during editing:

```powershell
.\tools\verify.ps1 quick -Filter 'Dense.*:DenseNoBiasTest.*'
.\tools\verify.ps1 full
```

```bash
./tools/verify.sh quick --filter 'Dense.*:DenseNoBiasTest.*'
./tools/verify.sh full
```

Use `quick` after edits and `full` once before completing a batch. Run either
wrapper with its help option for CUDA selection, cache locations and
compiler-cache support.

Full verification builds and runs both the unit tests and the response-optimization
integration scenarios on CPU and CUDA. CUDA verification requires a working GPU;
a CPU-only build cannot pass that gate.

On Linux (including WSL), put the intended CUDA toolkit's `bin` directory on
`PATH` before running the wrapper. It checks that `nvcc` and its host compiler
support C++20. For a custom cuDNN installation, set both
`OPENNN_CUDNN_INCLUDE_DIR` and `OPENNN_CUDNN_LIBRARY`. Set
`OPENNN_CUDA_ARCHITECTURES` when an explicit compute capability is needed.

GitHub runs CPU CI on hosted Linux and Windows machines and compiles the CUDA
library and test executables on a hosted Linux machine. The Linux CUDA runtime workflow
runs on pushes to `dev` and `master`, by manual dispatch, and nightly on the default
branch. It needs an online self-hosted runner labeled `linux` and `cuda` with the
GPU toolchain installed; otherwise the job remains queued.

For Python export execution tests, install both NumPy and pandas:
`python -m pip install numpy==2.4.4 pandas==2.3.3`.
JavaScript export execution tests require Node.js on `PATH`; CI uses Node 24.
Those tests report a skip when Node is unavailable locally. They compare
generated formulas and categorical controls with native network predictions.

Linux CI also runs AddressSanitizer (including leak detection) and
UndefinedBehaviorSanitizer. To reproduce that configuration with Clang 17:

```bash
CXX=clang++-17 cmake --preset verify-sanitizers -B ../opennn-sanitizers
cmake --build ../opennn-sanitizers --target opennn_tests opennn_response_tests --parallel 2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:allocator_may_return_null=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  OPENNN_THREADS=4 ctest --test-dir ../opennn-sanitizers --output-on-failure
```

After ordinary CUDA verification, run NVIDIA Compute Sanitizer against that
build with `bash tools/check_cuda_memory.sh /path/to/cuda-build`. The command
fails on detected memory errors or leaks. It excludes the process-exit death
test because child-process instrumentation hangs on the tested WSL setup;
ordinary CUDA verification includes that test. See [runner operations](tools/CI_RUNNER.md)
for the Linux GPU runner's requirements and availability.

## CMake options

| Option | Default | Description |
|---|---:|---|
| `OpenNN_DISABLE_CUDA` | `OFF` | Force a CPU-only build even when CUDA is available. |
| `OpenNN_REQUIRE_CUDA` | `OFF` | Fail configuration if CUDA cannot be enabled. |
| `OpenNN_BUILD_TESTS` | `ON` | Build the GoogleTest test suite. |
| `OpenNN_BUILD_EXAMPLES` | `ON` | Build example applications. |
| `OpenNN_BUILD_BENCHMARKS` | `OFF` | Build benchmark drivers from `benchmarks/`. |
| `OpenNN_BUILD_VISION` | `ON` | Build vision, sequence, transformer, and detection components. |
| `OpenNN_BUILD_SHARED` | `OFF` | Build OpenNN as a shared library instead of a static library. |
| `OpenNN_ENABLE_MKL` | `OFF` | Use Intel MKL as Eigen's BLAS/LAPACK backend. |
| `OpenNN_ENABLE_LTO` | platform-dependent | Enable interprocedural optimization for release builds. |

Clang static libraries built with LTO require a compatible Clang/LLVM toolchain
in their consumers. Their CMake target carries the required linker flags.
Set `OpenNN_ENABLE_LTO=OFF` when distributing native static objects across
compiler toolchains.

## CPU threads

OpenNN sizes one thread per CPU the process may run on (`sched_getaffinity`,
so `taskset` is honoured) and gives that count to Eigen, OpenMP and MKL alike.
`OPENNN_THREADS=n` overrides it. Every parallel region asks for the same team
on purpose: libgomp keeps a single pool sized to the last region, and a region
that wants fewer threads makes the surplus exit and the next full one recreate
them, which cost an LSTM forward pass 10% of its throughput before MKL was
pinned to the team. `OPENNN_OMP_DYNAMIC=1` re-enables dynamic teams.

One setting stays outside the library. GCC 14's libgomp detects hybrid Intel
CPUs (P- and E-cores) and stops spinning at barriers, so every fork/join
sleeps in the kernel; on such a machine set `GOMP_SPINCOUNT=300000` (libgomp's
own default elsewhere) or `OMP_WAIT_POLICY=active` before running anything
latency-sensitive. It is an environment variable the runtime reads before
`main`, which is why OpenNN cannot set it for you.

## Examples

The repository includes example apps for quick validation and experimentation.

- `examples/blank` - empty starter example for user experiments
- `examples/airfoil_self_noise` - approximation (regression) on tabular data
- `examples/iris_plant` - classification on tabular data and model export
- `examples/breast_cancer` - classification on tabular data
- `examples/amazon_reviews` - text classification
- `examples/emotion_analysis` - text classification
- `examples/bert` - BERT-style text classification
- `examples/translation` - sequence-to-sequence transformer
- `examples/gpt2` - character-level text generation
- `examples/forecasting_tinyml` - RNN/LSTM forecasting export for TinyML parity checks
- `examples/mnist` - image classification
- `examples/melanoma_cancer` - image classification
- `examples/yolo` - object detection

## Model export

OpenNN can export trained models as standalone source code through `ModelExpression`.
Supported targets include C, embedded C, Python, JavaScript, and PHP. The `iris_plant`
and `forecasting_tinyml` examples include parity checks for exported models, including
microcontroller-oriented AVR and ARM Cortex-M flows.

## Benchmarks

Reproducible benchmark recipes live in `benchmarks/`. They compare OpenNN with
reference engines across quality, throughput, capacity, energy and footprint
metrics. Large datasets and model files stay outside the repository; generated
results remain local under `benchmarks/results/` and are ignored by Git. See
`benchmarks/README.md` for usage, the general measurement protocol and the
reviewed project reports.

## Documentation

Full documentation and tutorials are available on the official website:

- http://opennn.net

Repository-local benchmark documentation is available in `benchmarks/`.

## Contributing

Contributions are welcome. If you want to help improve OpenNN, please follow these general steps:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes and add tests.
4. Submit a pull request with a clear description.

## License

OpenNN is distributed under the terms of the GNU Lesser General Public License. See [LICENSE.txt](LICENSE.txt) and the per-file license notices for details.

## 9.0 release preparation

This checkout prepares **9.0.0**; no final release is implied. See
[MIGRATION.md](MIGRATION.md) for 8.x source/model migration,
[DATASETS.md](DATASETS.md) for dataset attribution and unresolved permissions,
and [RELEASE_READINESS.md](RELEASE_READINESS.md) for publication gates.

[Reproduction recipes](tools/REPRODUCTION.md) rebuild verified datasets and
train repeatable reference models. [Packaging instructions](tools/PACKAGING.md)
describe candidate archives, checksums and validation from an extracted package.
