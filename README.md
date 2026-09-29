<div align="center">
  <img src="https://www.opennn.net/images/opennn_git_logo.svg" alt="OpenNN">
</div>

OpenNN is a C++20 library for building, training and running neural networks.
It supports tabular data, images, time series and text, with CPU and optional
CUDA execution. Models can be embedded in C++ applications or exported as
standalone source code for supported architectures.

| I want to… | Start here |
| --- | --- |
| Build and run my first network | [Quick start](#quick-start) |
| Choose CPU/CUDA backends and build options | [Build options](AGENTS.md#build-options) |
| Use OpenNN in my C++ application | [Install and link](#use-opennn-in-your-application) |
| Train, evaluate or export a model | [Examples](examples/README.md) |
| Update an 8.x application or model | [Migration](AGENTS.md#migrating-from-8x-to-90) |
| Understand the source folders | [Repository contents](#repository-contents) |
| Run or review performance comparisons | [Benchmarks](benchmarks/README.md) |
| Contribute or run the tests | [Contributing](#contributing-and-support) |

Tutorials are available on [opennn.net](https://www.opennn.net/).

## Quick start

You need Git, CMake 3.24+, and a C++20 compiler with `std::format` and OpenMP:
GCC 13+, Clang 17+, or Visual Studio 2022 C++ tools. On macOS, install the
OpenMP runtime (`libomp`). CMake retrieves missing dependencies during the
first configuration, so that step needs internet access.

Clone the repository and configure a CPU build:

```sh
git clone https://github.com/Artelnics/opennn.git
cd opennn
cmake -S . -B ../opennn-build -DCMAKE_BUILD_TYPE=Release -DOpenNN_DISABLE_CUDA=ON -DOpenNN_BUILD_TESTS=OFF -DOpenNN_BUILD_EXAMPLES=ON
cmake --build ../opennn-build --config Release --target blank --parallel
```

Run the blank example on Linux/macOS or with a single-configuration generator:

```sh
../opennn-build/bin/blank
```

With Visual Studio on Windows:

```powershell
../opennn-build/bin/Release/blank.exe
```

Next, follow the [Iris example](examples/README.md#train-your-first-model) to train
a classifier, evaluate it and export predictions as C or Python code.

### CUDA builds

Install a C++20-compatible CUDA toolkit and cuDNN 9+. Use a separate build
directory so the CPU and CUDA configurations do not overwrite each other:

```sh
cmake -S . -B ../opennn-build/cuda -DCMAKE_BUILD_TYPE=Release -DOpenNN_REQUIRE_CUDA=ON -DOpenNN_BUILD_TESTS=OFF -DOpenNN_BUILD_EXAMPLES=OFF
cmake --build ../opennn-build/cuda --config Release --target opennn --parallel
```

Require CUDA explicitly to make missing GPU build dependencies an error.

## Use OpenNN in your application

Install a completed build to a separate prefix:

```sh
cmake --install ../opennn-build --config Release --prefix ../opennn-install
```

In your application's `CMakeLists.txt`, consume the installed target:

```cmake
cmake_minimum_required(VERSION 3.24)
project(MyApplication LANGUAGES CXX)
find_package(OpenNN 9.0 CONFIG REQUIRED)
add_executable(my_application main.cpp)
target_link_libraries(my_application PRIVATE OpenNN::opennn)
target_compile_features(my_application PRIVATE cxx_std_20)
```

Configure the application with `-DCMAKE_PREFIX_PATH=/absolute/path/to/opennn-install`.
The exported target supplies OpenNN's include paths and dependency linkage, and
the package still works after moving the install prefix.
[The package consumer](tools/package_smoke/) is a complete working example.

Fetched Eigen and zlib are installed with the library. OpenMP, oneTBB, oneDNN,
MKL and NVIDIA runtimes are not copied into the installation; supply those
selected by the build. Use a compiler compatible with the one that built the
package. Installation packages exclude example datasets and models.

Include the public module headers directly, for example
`#include "opennn/network/network.h"`. Do not include the private precompiled
header `opennn/core/pch.h` in an application.

## Repository contents

| Folder | What it contains |
| --- | --- |
| [`opennn/`](opennn/) | The library: public headers, implementations, models and CPU/CUDA backends. See the [module map](AGENTS.md#library-modules-and-dependencies). |
| [`examples/`](examples/README.md) | Runnable applications and bundled data. |
| [`tests/`](tests/) | C++ unit tests of the library. |
| [`benchmarks/`](benchmarks/README.md) | Comparison drivers, input manifests, the [measurement protocol](benchmarks/PROTOCOL.md) and [reviewed results](benchmarks/reports/README.md). |
| [`tools/`](tools/) | Verification, packaging, asset reproduction and release checks; see [AGENTS.md](AGENTS.md#maintenance-tools). |
| [`.github/workflows/`](.github/workflows/) | Hosted CI and Linux CUDA verification. |

## Contributing and support

Base contributions on `dev`, include the validation you ran, and describe the
resulting behavior in the pull request. [AGENTS.md](AGENTS.md) holds the
engineering rules, verification commands and release procedure. For a bug
report, include the OpenNN commit, operating system, compiler, CPU/CUDA
configuration, reproduction command and error, and file it in
[GitHub Issues](https://github.com/Artelnics/opennn/issues).

## License

OpenNN is distributed under the GNU Lesser General Public License, version 2.1 or
(at your option) any later version; see [LICENSE.txt](LICENSE.txt) and the
per-file notices. Dependencies and downloadable pre-trained weights keep their own
terms, listed in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt). Example data
keeps its publishers' terms, recorded in the `SOURCE.md` notice of each data folder.
