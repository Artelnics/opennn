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
- `tests/` - GoogleTest-based unit and validation tests
- `docs/benchmarks/` - reproducible benchmark suite and benchmark methodology

## Quick start

### Requirements

- C++20 compiler with `std::format` support
  - GCC 13+
  - Clang 17+
  - MSVC 2022+
- CMake 3.18+
- Optional: CUDA Toolkit and **cuDNN 9.0+** for GPU builds

### Build CPU-only

Create a separate build directory outside the repository folder:

```bash
cmake -S . -B ../opennn-build -DCMAKE_BUILD_TYPE=Release -DOpenNN_DISABLE_CUDA=ON
cmake --build ../opennn-build --config Release
```

### Build with CUDA

```bash
cmake -S . -B ../opennn-build -DCMAKE_BUILD_TYPE=Release
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

## CMake options

| Option | Default | Description |
|---|---:|---|
| `OpenNN_DISABLE_CUDA` | `OFF` | Force a CPU-only build even when CUDA is available. |
| `OpenNN_BUILD_TESTS` | `ON` | Build the GoogleTest test suite. |
| `OpenNN_BUILD_EXAMPLES` | `ON` | Build example applications. |
| `OpenNN_BUILD_BENCHMARKS` | `OFF` | Build benchmark drivers from `docs/benchmarks/`. |
| `OpenNN_BUILD_VISION` | `ON` | Build vision, sequence, transformer, and detection components. |
| `OpenNN_BUILD_SHARED` | `OFF` | Build OpenNN as a shared library instead of a static library. |
| `OpenNN_ENABLE_MKL` | `OFF` | Use Intel MKL as Eigen's BLAS/LAPACK backend. |
| `OpenNN_ENABLE_LTO` | platform-dependent | Enable interprocedural optimization for release builds. |

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
- `examples/text_generation` - character-level text generation
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

Reproducible benchmark recipes live in `docs/benchmarks/`. They compare OpenNN with
PyTorch and TensorFlow across quality, throughput, capacity, energy, and footprint
metrics. Large benchmark datasets and generated result artifacts are kept outside
the repository; see `docs/benchmarks/DATA_POLICY.md`.

## Documentation

Full documentation and tutorials are available on the official website:

- http://opennn.net

Repository-local benchmark documentation is available in `docs/benchmarks/`.

## Contributing

Contributions are welcome. If you want to help improve OpenNN, please follow these general steps:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes and add tests.
4. Submit a pull request with a clear description.

## License

OpenNN is distributed under the terms of the GNU Lesser General Public License. See [LICENSE.txt](LICENSE.txt) and the per-file license notices for details.

