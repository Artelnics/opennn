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
- I think OpenNN is a good fit for projects where the neural network code must behave like any other C++ component.

## Features

- Feed-forward neural networks
- Convolutional and recurrent layers
- Transformers and attention mechanisms
- Loss functions and optimization algorithms
- Model selection, data preprocessing, and training strategies
- Example applications for CPU and GPU

## Repository layout

- `opennn/` — core library sources
- `examples/` — example applications and demos
- `tests/` — unit tests and validation code
- `datasets/` — sample datasets used by examples
- `scripts/` — helper scripts for profiling and tooling

## Quick start

### Requirements

- C++20 compiler with `std::format` support
  - GCC 13+
  - Clang 17+
  - MSVC 2022+
- CMake 3.18+
- Optional: CUDA Toolkit and **cuDNN 9.0+** for GPU builds

### Build CPU-only

Create a separate build directory outside the repository folder, then configure from there:

```bash
mkdir -p ../opennn-build
cd ../opennn-build
cmake -DCMAKE_BUILD_TYPE=Release -DOpenNN_DISABLE_CUDA=ON ../opennn
cmake --build . --config Release
```

### Build with CUDA

```bash
mkdir -p ../opennn-build
cd ../opennn-build
cmake -DCMAKE_BUILD_TYPE=Release ../opennn
cmake --build . --config Release
```

### Build examples and tests

```bash
cmake -DOpenNN_BUILD_EXAMPLES=ON -DOpenNN_BUILD_TESTS=ON ..
cmake --build . --config Release
```

## Examples

The repository includes example apps for quick validation and experimentation.

- `examples/blank` — minimal CPU example and optional CUDA benchmark
- `examples/airfoil_self_noise` — approximation (regression) on tabular data
- `examples/iris_plant` — classification on tabular data
- `examples/breast_cancer` — classification on tabular data
- `examples/no2_forecasting` — time-series forecasting
- `examples/amazon_reviews` — text classification
- `examples/emotion_analysis` — text classification
- `examples/translation` — sequence-to-sequence transformer
- `examples/mnist` — image classification
- `examples/melanoma_cancer` — image classification
- `examples/yolo` — object detection

## Documentation

Full documentation and tutorials are available on the official website:

- http://opennn.net

## Contributing

Contributions are welcome. If you want to help improve OpenNN, please follow these general steps:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes and add tests.
4. Submit a pull request with a clear description.

## License

OpenNN is distributed under the terms of the [MIT License](LICENSE.txt).

