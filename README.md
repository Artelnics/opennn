# OpenNN

OpenNN is a C++20 library for building, training and running neural networks.
It supports tabular data, images, time series and text, with CPU and optional
CUDA execution. Models can be embedded in C++ applications or exported as
standalone source code for supported architectures.

**This is the `dev` branch for the unreleased 9.0 candidate.** The published
8.x line is on `master`. See [migration](CHANGELOG.md#migrating-from-8x-to-90)
before updating an existing application or saved model.

| I want to… | Start here |
| --- | --- |
| Build and run my first network | [Quick start](#quick-start) |
| Use OpenNN in my C++ application | [Install and link](#use-opennn-in-your-application) |
| Train, evaluate or export a model | [Examples](examples/README.md) |
| Understand the source folders | [Repository contents](#repository-contents) |
| Run or review performance comparisons | [Benchmarks](benchmarks/README.md) |
| Contribute, test or prepare a release | [Development](DEVELOPMENT.md) |

## Quick start

You need Git, CMake 3.24+, and a C++20 compiler with `std::format` and OpenMP:
GCC 13+, Clang 17+, or Visual Studio 2022 C++ tools. On macOS, install the
OpenMP runtime (`libomp`). CMake retrieves missing dependencies during the
first configuration, so that step needs internet access.

Clone the development branch and configure a CPU build:

```sh
git clone --branch dev https://github.com/Artelnics/opennn.git
cd opennn
cmake -S . -B ../opennn-build -DCMAKE_BUILD_TYPE=Release -DOpenNN_DISABLE_CUDA=ON -DOpenNN_BUILD_TESTS=OFF -DOpenNN_BUILD_EXAMPLES=ON
cmake --build ../opennn-build --config Release --target blank --parallel
```

Run the minimal example on Linux/macOS or with a single-configuration generator:

```sh
../opennn-build/bin/blank
```

With Visual Studio on Windows:

```powershell
../opennn-build/bin/Release/blank.exe
```

It prints `Prediction: 2`. The [complete C++ example](examples/blank/main.cpp)
creates a two-input, one-output dense network, assigns fixed demonstration
weights and performs inference. It needs no dataset or model download.
The existing target name `blank` is retained for compatibility.

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
See [build configuration](DEVELOPMENT.md#build-configuration) for CPU backends,
portable binaries, shared libraries and custom dependency paths.

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
The exported target supplies OpenNN's include paths and dependency linkage.
[The package consumer](tools/package_smoke/) is a complete working example.
See [packaging](DEVELOPMENT.md#installation-packages) for external runtime
dependencies and compiler compatibility.

## Repository contents

| Folder | What it contains |
| --- | --- |
| [`opennn/`](opennn/) | The library: public headers, implementations, models and CPU/CUDA backends. See the [module map](DEVELOPMENT.md#source-map). |
| [`examples/`](examples/README.md) | Runnable applications, export checks and their bundled assets. `legacy_8/` is historical reference material. |
| [`tests/`](tests/) | C++ unit tests, response-optimization scenarios and Python benchmark tests. |
| [`benchmarks/`](benchmarks/README.md) | Comparison drivers, input manifests, measurement procedures and reviewed results. |
| [`tools/`](DEVELOPMENT.md#maintenance-tools) | Verification, packaging, asset reproduction and release checks. |
| [`.github/workflows/`](.github/workflows/) | Hosted CI and Linux CUDA verification. |

Build directories, dependency downloads, logs and raw benchmark results are
generated locally and are not repository contents. Most tracked files are
example images and data; the [data review](DATASETS.md) explains their sources
and reproduction status.

## Documentation

| Guide | Contents |
| --- | --- |
| [Examples](examples/README.md) | Choose an example, run training, check exports and identify download requirements |
| [Development](DEVELOPMENT.md) | Build settings, source map, tools, tests and the `dev` → `master` workflow |
| [Changelog and migration](CHANGELOG.md) | Release changes, renamed APIs and saved-model migration |
| [Example data](DATASETS.md) | Attribution, unresolved records and reconstruction recipes |
| [Benchmarks](benchmarks/README.md) | Commands, the [protocol](benchmarks/PROTOCOL.md) and [reviewed results](benchmarks/reports/README.md) |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | Dependency licences and attribution |

Tutorials are available on [opennn.net](https://www.opennn.net/).
The benchmark results describe their recorded models and computers; the current
review lists the measurements still required before website publication.

## Contributing and support

Base contributions on `dev` and follow [the development workflow](DEVELOPMENT.md)
and [engineering rules](AGENTS.md). For a bug report, include the OpenNN commit,
operating system, compiler, CPU/CUDA configuration, reproduction command and error.
Report issues in [GitHub Issues](https://github.com/Artelnics/opennn/issues).

Release promotion to `master` follows the owner's decision and the
[release checklist](DEVELOPMENT.md#release-checklist).
[RELEASE_SCOPE.json](RELEASE_SCOPE.json) records the current artifact scope.

## License

OpenNN is distributed under the GNU Lesser General Public License; see
[LICENSE.txt](LICENSE.txt) and the per-file notices. Dependencies and example
assets retain the separate terms recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and [DATASETS.md](DATASETS.md).
