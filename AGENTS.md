# OpenNN — guide for coding agents and maintainers

[README.md](README.md) is the user guide: first build, installed package and
example-data licences. This file covers where things are, build options, how to
verify changes, CI, example-data records, the 9.0 release notes and migration,
and releases. Keep rules here repository-wide; workstation paths belong in
environment variables, not in repository files.

## Working branches

- Use the existing OpenNN checkout on `dev` for normal development. Do not create
  additional clones or worktrees unless the user explicitly requests them.
- Keep development and release preparation on `dev`. Merge into `master` only
  when the user explicitly decides the work is ready for release.
- Preserve uncommitted work and saved stashes when switching or consolidating
  branches. A folder cleanup does not authorize publishing a release.
- Contributors base changes on `dev` and open a pull request describing the
  resulting behavior and the validation run.

## Compatibility and scope

- Preserve the public API and serialized model compatibility unless the task
  explicitly authorizes a breaking change.
- Neural Designer links against OpenNN and uses symbols that may appear unused
  inside this repository. Do not remove public or exported code based only on
  repository-local call sites.
- Preserve unrelated working-tree changes. Build products, downloaded models,
  generated data and raw benchmark results must remain outside Git.
- Keep `README.md` independent of the branch and release state, so it can be
  promoted to `master` unchanged. Put candidate status and pending work here.

## Where things are

| Path | Contents |
| --- | --- |
| `opennn/` | The library; module map below. |
| `tests/` | GoogleTest tests of the `opennn/` library only, one folder per library module (network layers and operators are separate). `tests/common/` holds the test `main`, the precompiled header and shared helpers. Do not add benchmark, example or tooling tests here, and keep no loose test files at the `tests/` root. |
| `examples/` | Runnable applications and bundled data. The catalog is `examples/README.md`. |
| `benchmarks/` | OpenNN versus PyTorch comparison drivers, input manifests, `README.md` (usage) and `PROTOCOL.md` (measurement rules). `benchmarks/reports/` is ignored by Git and holds each user's own reports. |
| `tools/` | Verification, checkers, packaging and reproduction; table below. |
| `.github/workflows/` | `ci.yml`, the hosted CI. |

`CMakePresets.json` holds the repeatable configure/build/test settings, and
`tools/CODE_QUALITY.json` the reviewed maintainability limits (ratchet).

`LICENSE.txt` (LGPL 2.1, matching the `LGPL-2.1-or-later` SPDX headers) and
`THIRD_PARTY_NOTICES.txt` are installed with every package.

### Library modules and dependencies

| Module | Responsibility and starting points |
| --- | --- |
| `opennn/core/` | Configuration, tensor and storage types, backend operations and persistence utilities. Start with `configuration.h` and `opennn_types.h`; CUDA kernels are under `core/cuda/`. `pch.h` is the library's private precompiled header. |
| `opennn/network/` | `Network` (`network.h`), propagation, save/load (`network_io.cpp`), chat and `ModelExpression` source export (`model_expression.cpp`). `layers/` holds network layers, `LayerType` and the layer factory (`layer_registry.h`); `operators/` holds reusable operators. |
| `opennn/models/` | Ready-made tabular, image, language and forecasting architectures declared in `models.h`; shared execution and tokenizer methods belong to `Network`. |
| `opennn/dataset/` | `Dataset` (`dataset.h`) and the four concrete classes below; I/O in `tabular_dataset_io.cpp` and `yolo_dataset_io.cpp`. |
| `opennn/training/` | `Training` (`training.h`), losses and optimizers such as `Adam` and `SGD`. |
| `opennn/evaluation/` | `Evaluation` (`evaluation.h`) for prediction-quality analysis. |
| `opennn/model_selection/` | Input and network-size selection. |
| `opennn/response_optimization/` | Optimization of inputs subject to model outputs and constraints (`response_optimization.h`); feasibility repair is internal to `FeasibilityRepairSystem`. |

`core` is the foundation. `network` and `models` define computation; datasets
feed training, and evaluation and selection use those interfaces. This is not a
strict linear chain: `tools/check_architecture.py` records the allowed directions
and a small list of existing exceptions. Keep public headers at their include
paths. The `core/cuda/flash_attention_shim/` hierarchy is required by the
optional FlashAttention backend.

### Choosing a dataset

`Dataset` is the shared interface for batches, sample roles and variables.
Choose one of the four concrete classes:

| Class | Data and configuration |
| --- | --- |
| `TabularDataset` | Numeric and categorical tables, including time series through `configure_forecasting(past, future, multi_target)`. |
| `TextDataset` | Labelled text, paired translation sequences or next-token prediction, selected with `TextDataset::Options::task`. Tokenizers and optional attention masks configure the input representation. |
| `ImageDataset` | Image classification with one label per image. |
| `YoloDataset` | Object detection with boxes, synchronized image/box augmentation and detection-head target layouts. |

### Maintenance tools

| Task | Entry point |
| --- | --- |
| Incremental CPU/CUDA verification | `tools/verify.sh`; shared logic in `tools/verify.cmake` |
| Size, complexity and dependency checks | `tools/check_code_quality.py`, `tools/check_architecture.py` |
| Coverage and public-header checks | `tools/check_coverage.py`, `tools/check_headers.sh` |
| JSON fuzzing | `tools/fuzz/` |
| Installed-package checks and C++ consumer | `tools/check_installed_package.py`, `tools/package_smoke/` |
| Full example/device matrix | `tools/run-opennn-examples/SKILL.md` |
| Benchmark preparation and execution | `benchmarks/prepare.py`, `benchmarks/run.py`, `benchmarks/compare.py`; procedure in `tools/run-opennn-benchmarks/SKILL.md` |

## Code organization

- Follow neighboring files for naming, include order and class layout.
- Keep reusable tensor and device primitives in `opennn/core/`; datasets,
  network code, training, model selection and evaluation must retain their
  existing dependency direction.
- Validate structural changes on both CPU and CUDA when they touch shared code.
  Some qualifications, includes and data-member ordering are intentionally
  significant even when a local edit suggests otherwise.

## Build options

Run CMake from the repository root and keep build and install directories
outside it. A default configuration builds examples and tests; select only the
targets you need. For a library-only build, set `OpenNN_BUILD_TESTS=OFF` and
`OpenNN_BUILD_EXAMPLES=OFF`, then build target `opennn`.

| Option | Default | Description |
| --- | ---: | --- |
| `OpenNN_DISABLE_CUDA` | `OFF` | Force a CPU-only build even when CUDA is available. |
| `OpenNN_REQUIRE_CUDA` | `OFF` | Fail configuration if CUDA cannot be enabled. |
| `OpenNN_BUILD_TESTS` | `ON` | Build the GoogleTest test suite. |
| `OpenNN_BUILD_EXAMPLES` | `ON` | Build example applications. |
| `OpenNN_BUILD_BLANK` | `ON` | Include the `blank` example template when examples are enabled. |
| `OpenNN_BUILD_BENCHMARKS` | `OFF` | Build benchmark drivers from `benchmarks/`. |
| `OpenNN_BUILD_FUZZERS` | `OFF` | Build Clang libFuzzer targets with ASan/UBSan. |
| `OpenNN_BUILD_VISION` | `ON` | Build vision, sequence, transformer and detection components. |
| `OpenNN_BUILD_SHARED` | `OFF` | Build OpenNN as a shared library instead of a static library. |
| `OpenNN_INSTALL` | `ON` | Generate library installation and package rules. |
| `OpenNN_CPU_TARGET` | `NATIVE` | `NATIVE` keeps host-specific ISA flags for local throughput; `PORTABLE` removes them for distributable binaries. |
| `OpenNN_ENABLE_LTO` | platform-dependent | Enable interprocedural optimization for release builds. |
| `OpenNN_ENABLE_MKL` | `OFF` | Use Intel MKL as Eigen's BLAS/LAPACK backend. |
| `OpenNN_ENABLE_ONEDNN` | `AUTO` | Use detected oneDNN CPU primitives; `ON` requires it and `OFF` disables the search. |
| `OpenNN_MKL_ROOT`, `OpenNN_ONEDNN_ROOT` | empty | Locate a nonstandard MKL or oneDNN installation. |
| `OpenNN_CHECK_HEADERS` | `OFF` | Compile public headers independently. |
| `OpenNN_UNIT_TEST_TIMEOUT` | `600` seconds | CTest limit for the unit suite; the sanitizer preset uses 1,800. |

OpenMP is required; configuration fails if the selected toolchain lacks it.
Eigen is found as a package or fetched; zlib and libjpeg-turbo are fetched by
CMake. oneTBB and oneDNN are optional detected CPU dependencies, and MKL is
explicitly opt-in. GoogleTest is needed only for tests. CUDA builds add NVIDIA
runtime dependencies and the fetched cuDNN frontend. Versions and licences are
listed in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).

`PORTABLE` removes only explicit host-specific ISA flags (`-march=native`,
`/arch:AVX2` or the Apple Silicon target). It does not change tensor layouts,
algorithms, thread counts or CUDA kernels. Clang static libraries built with
LTO require a compatible Clang/LLVM toolchain in their consumers; set
`OpenNN_ENABLE_LTO=OFF` when distributing static objects across toolchains.

### CPU threads

On Linux, OpenNN uses one thread per CPU the process may run on
(`sched_getaffinity`, so `taskset` is honoured). Other platforms use the reported
hardware concurrency, with OpenMP as a fallback. The same count is given to
Eigen, OpenMP and MKL. `OPENNN_THREADS=n` overrides it, and
`OPENNN_OMP_DYNAMIC=1` re-enables dynamic OpenMP teams.

On hybrid Intel CPUs (P- and E-cores), GCC 14's libgomp stops spinning at
barriers, so every fork/join sleeps in the kernel. Set `GOMP_SPINCOUNT=300000`
or `OMP_WAIT_POLICY=active` before running latency-sensitive work. The runtime
reads these variables before `main`, so OpenNN cannot set them for you.

## Build and verify

Use the repository wrapper for routine verification on Linux or WSL. It creates
persistent build trees outside the checkout and supports focused GoogleTest
filters:

```bash
./tools/verify.sh quick --filter 'Dense.*:DenseNoBiasTest.*'
./tools/verify.sh quick --backend cuda --filter '*Gpu*:*CUDA*'
./tools/verify.sh full
```

Use focused checks while editing and `full` as the final gate for a completed
batch. `full` builds and runs the unit tests on CPU and CUDA. A library change is not complete until the relevant
CPU and CUDA suites pass, or an unavailable backend is reported clearly; a CPU
fallback never counts as a CUDA pass. Run the wrapper with `--help` for CUDA
selection, cache locations and compiler-cache support.

For non-standard CUDA installations, configure the wrapper through
`OPENNN_CUDA_ARCHITECTURES`, `OPENNN_CUDNN_INCLUDE_DIR` and
`OPENNN_CUDNN_LIBRARY`. On Linux (including WSL), put the intended CUDA
toolkit's `bin` directory on `PATH`; the wrapper checks that `nvcc` and its host
compiler support C++20.

On Windows, use the CMake presets from a Visual Studio developer prompt. The
presets `verify-cpu`, `verify-cuda` and `verify-sanitizers` (Linux Clang)
require Ninja and build into `../opennn-build/<preset>`. Passing another
directory with `-B` requires the same directory in later `cmake --build` and
`ctest --test-dir` commands.

```sh
cmake --preset verify-cpu
cmake --build --preset verify-cpu --parallel
ctest --preset verify-cpu
```

### Python for checks and export tests

Use Python 3.12, the CI version; older interpreters cannot run the checkers.
In an isolated environment:

```sh
python -m pip install -r tools/test-requirements.txt -r tools/code-quality-requirements.txt
```

The C++ export execution tests need NumPy, pandas and onnxruntime for generated
Python and ONNX models, and Node.js on `PATH` for JavaScript (CI uses Node 24);
they report a skip when a runtime is missing. Use `python -B` so maintenance
checks leave no caches.

### Sanitizers

```bash
CXX=clang++-17 cmake --preset verify-sanitizers -B ../opennn-sanitizers
cmake --build ../opennn-sanitizers --target opennn_tests --parallel 2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:allocator_may_return_null=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  OPENNN_THREADS=4 ctest --test-dir ../opennn-sanitizers --output-on-failure
```

For longer JSON fuzzing than CI runs:

```sh
cmake --preset verify-sanitizers -B ../build-fuzz -DOpenNN_BUILD_FUZZERS=ON
cmake --build ../build-fuzz --target opennn_json_fuzz --parallel 2
mkdir -p ../build-fuzz/corpus ../build-fuzz/artifacts
../build-fuzz/bin/opennn_json_fuzz -max_total_time=3600 -artifact_prefix=../build-fuzz/artifacts/ ../build-fuzz/corpus tools/fuzz/corpus
```

### Before every commit

Run the source checks that gate CI:

```bash
python tools/check_code_quality.py
python tools/check_architecture.py
```

`check_code_quality.py` is a ratchet over first-party `.cpp`, `.h`, `.cu` and
`.cuh` files (excluding the vendored FlashAttention shim), and it also checks
SPDX identifiers. Any metric above `tools/CODE_QUALITY.json` fails CI. Unprefixed keys
cover C++, `cuda_` keys cover CUDA, and `combined_duplicate_percent` covers both.
When a commit legitimately grows the code (size metrics such as `nloc`,
`physical_lines`, `functions` or `files`), run
`python tools/check_code_quality.py --update` and include `tools/CODE_QUALITY.json` in
the same commit. Do not update it to absorb worse quality metrics (long or
complex functions, duplication); simplify the code instead.

`check_architecture.py` enforces the module dependency direction. Portable `.h`
interfaces may not expose CUDA vendor types; `.cuh` implementation headers may.

## CI and quality gates

`ci.yml` runs on hosted machines: Linux and Windows CPU builds, CUDA compilation
on Linux, package consumers (static and shared on Linux, portable AppleClang on
macOS, static GCC, Clang and MSVC), ASan/UBSan, 20,000 JSON fuzzing mutations
and the source checkers. It also runs:

- Static analysis that treats selected Clang analyzer ownership and
  use-after-move findings in the persistence, dataset and device modules as errors.
- Coverage of the CPU unit suite: at least 70% line and 35% branch coverage of
  non-CUDA sources, plus the per-module floors in `tools/check_coverage.py`.
  Raise thresholds as tests are added.

CI has no GPU: it compiles CUDA but never runs CUDA tests. Run
`./tools/verify.sh full` (or `quick --backend cuda`) on a machine with a GPU
before merging changes that touch CUDA or shared code, and report the result.

`python benchmarks/compare.py baseline.json candidate.json` gates performance
changes: it allows 5% throughput/memory variation and rejects lower confirmed
batch capacity.

## Examples and benchmarks

- When changing example targets or dependencies, update the catalog in
  `examples/README.md`. Run data-dependent examples from the executable directory.
- To run every example across the supported device/precision matrix, follow
  [tools/run-opennn-examples/SKILL.md](tools/run-opennn-examples/SKILL.md).
- Benchmark usage and the measurement contract live in
  [benchmarks/README.md](benchmarks/README.md) and
  [benchmarks/PROTOCOL.md](benchmarks/PROTOCOL.md). Approve only measurements
  that meet the protocol.
- To measure a benchmark cell or compare a change before and after, follow
  [tools/run-opennn-benchmarks/SKILL.md](tools/run-opennn-benchmarks/SKILL.md).
- Raw benchmark output belongs outside the checkout, in
  `../opennn-benchmark-results/` by default; `OPENNN_BENCH_RESULTS` overrides it.
  Benchmark results and reports are never committed.
- CI compiles every example and benchmark driver (targets `examples` and
  `benchmarks`) on Linux but runs neither.

## Example data

OpenNN's software licence does not license bundled datasets, images, text or
trained artifacts.

- Keep the bundled datasets until each affected example has a reproducible
  replacement, and keep each `SOURCE.md` notice with its data.
- Do not describe a dataset as cleared for redistribution without a confirmed
  source and licence.
- `examples/mnist/data/images.zip` and `examples/melanoma_cancer/data/images.zip`
  keep the original file names inside. CMake extracts only the selected
  example's archive into the build directory. Keep member names unique.

Cleared groups: `airfoil_self_noise`, `amazon_reviews`, `breast_cancer`, `concrete`,
`iris_plant`, `mnist` and `yacht_hydrodynamics`. Unresolved groups and what each needs:

| Group | Required resolution |
| --- | --- |
| `bert` | `sst2.txt` is consistent with SST-2 (Socher et al., EMNLP 2013). Verify the split and obtain redistribution terms. |
| `emotion_analysis` | Consistent with Saravia et al.'s corpus, whose README limits it to educational and research use. Resolve permitted use or replace it. |
| `ecg5000_anomaly_detection` | Matches the TensorFlow tutorial CSV and seed-21 test split; source PhysioNet `chf07` (ODC-By 1.0). Confirm terms for the UCR/UEA and TensorFlow derivative. |
| `melanoma_cancer` | 102 BMP images with unverified source; obtain source, attribution and permission. |
| `translation` | Spanish-English pairs with unrecorded authorship; confirm source and permission. |

The owner has been asked for the missing records. GitHub's automatic source
archives include every tracked file, so a tag publishes these assets.

## 9.0 release notes and migration

### What's new in 9.0

- A module layout under `opennn/<module>/` and shorter public names: `Network`,
  `Training`, `Evaluation`, `Adam`, `SGD`, `LSTM`, `LevenbergMarquardt`,
  `QuasiNewton`, `Autoencoder`, `Yolo` and `InputSelection`.
- Four dataset classes replace seven. `TabularDataset` includes forecasting
  windows, and `TextDataset` covers classification, translation, next-token
  prediction and token/attention-mask inputs.
- Stricter model loading: missing parameter files, parameter-count mismatches
  and nonfinite embedded weights are errors. The JSON parser is strict, bounded
  and fuzz-tested.
- Response optimization accepts original variable names in backticks and adds
  feasibility controls: `set_feasibility_evaluations()`,
  `set_feasibility_rounds()`, `set_sampling_budget_multiplier()` and
  `set_maximum_consecutive_failures()`.
- JavaScript export sanitizes feature identifiers, escapes HTML labels and uses
  a dropdown for categorical inputs.
- Configurable library logging.
- `NATIVE` and `PORTABLE` CPU targets, shared-library builds, a relocatable CMake
  package and a core package without the vision components.
- Many CPU/CUDA reliability fixes in buffer ordering, handle release,
  mixed-precision layouts and dataset device copies.

The itemized list, including internal refactoring, is in
[CHANGELOG.md at `af6ee7ef5`](https://github.com/Artelnics/opennn/blob/af6ee7ef5157a48b2aa23689136dbd2976124239/CHANGELOG.md).

### Migrating from 8.x to 9.0

9.0 is a major source and model-format migration. Recompile consumers;
replacing an 8.x shared library in place is not supported. The package version
file requires the same major version, so request `find_package(OpenNN 9.0 ...)`.
There are no old-name aliases or forwarding headers, and former JSON root and
factory names are rejected. The renames below do not change tensor layouts,
weights, numerical algorithms or memory ownership.

#### Headers

| Former header | Current header |
| --- | --- |
| `opennn/neural_network.h` | `opennn/network/network.h` |
| `opennn/dataset.h` | `opennn/dataset/dataset.h` and `opennn/dataset/tabular_dataset.h` |
| `opennn/standard_networks.h` | `opennn/models/models.h` |
| `opennn/training_strategy.h` | `opennn/training/training.h` |
| `opennn/dense_layer.h` | `opennn/network/layers/dense_layer.h` |
| `opennn/bounding_layer.h` | `opennn/network/layers/clamping_layer.h` |
| `opennn/response_optimization.h` | `opennn/response_optimization/response_optimization.h` |
| `opennn/variable.h` | `opennn/core/variable.h` |
| `opennn/registry.h` | `opennn/network/layers/layer_registry.h` (`LayerType`, `create_layer`); `create_optimizer` is in `opennn/training/optimizer.h` and `create_input_selection` in `opennn/model_selection/input_selection.h` |
| `opennn/pch.h` | `opennn/core/pch.h` (private precompiled header) |

Changing includes alone is insufficient: use `opennn::Dense` and `Clamping` for
the current layer classes and the dataset classes below.

#### Renamed classes and saved names

| Former | Current | Saved JSON and related names |
| --- | --- | --- |
| `NeuralNetwork` | `Network` | Root key `"Network"`. Generated Python models expose `module.Network()`; regenerate exports. |
| `TrainingStrategy` | `Training` | Root key `"Training"`. Model selection uses `get_training()` and `set_training()`. |
| `QuasiNewtonMethod` | `QuasiNewton` | `OptimizationMethod` value and nested object key `QuasiNewton`. |
| `LevenbergMarquardtAlgorithm` | `LevenbergMarquardt` | Factory and JSON name unchanged. |
| `AdaptiveMomentEstimation` | `Adam` | `OptimizationMethod` value and nested object key `Adam`. |
| `StochasticGradientDescent` | `SGD` | `OptimizationMethod` value and nested object key `SGD`. |
| `LongShortTermMemory` | `LSTM` | Layer key `LSTM` inside `Network.Layers.Items`; `LayerType::LSTM`; `LSTMOperator`. New layers default to the label `lstm_layer`. |
| `InputsSelection` | `InputSelection` | `InputSelection` object with `InputSelectionMethod`; `InputSelectionResult`; singular `input_selection` accessor and factory names. |
| `TestingAnalysis` | `Evaluation` | Header `opennn/evaluation/evaluation.h`. |
| `AutoencoderNetwork`, `YoloNetwork` | `Autoencoder`, `Yolo` | Declared in `opennn/models/models.h`. |

Optimizer headers are `opennn/training/adam.h`, `sgd.h`, `levenberg_marquardt.h`
and `quasi_newton.h`. Standalone optimizer and LSTM layer JSON use the new root
names too.

#### Datasets

| Removed class | Replacement |
| --- | --- |
| `TimeSeriesDataset` | `TabularDataset` with `configure_forecasting(past, future, multi_target)`; `get_sequence_data()` returns a three-dimensional window tensor. |
| `LanguageDataset` | `TextDataset`, choosing `Classification` or `SequenceToSequence` explicitly. |
| `TextGenerationDataset` | `TextDataset` with `Task::NextToken` and a positive sequence length. |
| `BertDataset` | `TextDataset` with `InputLayout::TokensAndMask` and a loaded `WordPieceTokenizer`. |

```cpp
TabularDataset series("readings.csv", ",", true);
series.set_variable_role("target", VariableRole::InputTarget);
series.configure_forecasting(24, 1);

TextDataset translation({.task = TextDataset::Task::SequenceToSequence});
translation.read_txt("parallel_text.tsv");

TextDataset corpus({.task = TextDataset::Task::NextToken, .sequence_length = 256});
corpus.read_txt("corpus.txt");
```

- Configure forecasting windows after loading the table and assigning roles. Use
  `InputTarget` when past values of the predicted variable are also inputs;
  `clear_forecasting()` restores ordinary table shapes.
- Text vocabulary and tokenizer access take an optional `VariableRole`. In
  `TokensAndMask` mode the model receives the attention mask as `Input` and token
  IDs as `Decoder`; install the tokenizer with
  `set_tokenizer(std::move(tokenizer), VariableRole::Decoder)`. See the
  [BERT example](examples/bert/main.cpp).
- Text dataset JSON stores tokenizer settings and sample splits. Loading it with a
  corpus whose token IDs or label order differ is an error; save fresh metadata.
- `YoloDataset` derives directly from `Dataset`, without image-classification methods.

#### Time-series windows

8.x treated sample `i` as the present instant: inputs were rows `[i-past, i-1]`
and targets began at `i+1`. 9.0 treats the sample index as the window start:
inputs are `[i, i+past-1]` and targets begin at `i+past`. With a single target
window the target is `i+past+future-1`; multi-target windows include every future
step. Regenerate chronological splits and compare windows explicitly; do not
reuse old sample indices or action-conditioned optimization constraints.

#### Models and parameters

9.0 uses JSON configuration with binary parameter storage. There is no general
8.x XML/NDM converter; migrate each production model as follows:

1. Keep the original artifacts and an 8.x environment that can load them.
2. Record the topology, activations, feature and category order, scaling
   descriptives, tensor shapes and output interpretation. Export representative
   inputs and expected outputs.
3. Rebuild the model with the current APIs. Transfer parameters only after
   checking each layer's layout; equal parameter counts do not imply equal order.
4. Compare predictions on the recorded cases, including missing values,
   categories, boundary values and forecasting windows.
5. Save with `Network::save` and `save_parameters_binary`, reload, and compare
   again. Keep the JSON and binary files together.

`Network::load(path)` requires the matching `.bin` file or embedded JSON
parameters and raises an error, leaving the existing network intact, when both
are absent. Embedded parameters must contain exactly the compiled buffer's
number of values, including alignment padding, and every value must be finite.
Raw snapshots use the compiled layout: `get_parameters_number()` reports the
logical count and `get_parameters_buffer_size()` the size `set_parameters`
expects. To load an architecture only, do it explicitly:

```cpp
Network network;
network.from_JSON(load_json_file(path));
// Initialize or load matching parameters before inference.
```

#### Other behavior changes

- Genetic algorithms default to `Random` initialization. Call
  `set_initialization_method` to keep `Correlations`; JSON records
  `InitializationMethod`.
- The optimizer prepares scaling for each training run, and input selection
  recompiles the model. The old convolution-specific Adam learning-rate override
  is gone; set learning rates explicitly.
- `ResponseOptimization::set()` clears objectives, constraints and cached bounds;
  configure expressions again after rebinding the network.
- An unavailable validation error is now NaN instead of zero. Model selection
  requires validation data, and small cross-validation folds are balanced.
- Custom WordLevel tokenizer framing is saved in JSON; existing tokenizer caches
  are rebuilt.
- Re-export HTML/JavaScript models to obtain the export fixes.
- Select CPU execution with `Configuration::instance().set(Device::CPU, Type::FP32)`.

#### Earlier 9.0 development snapshots

Code written against earlier 9.0 development snapshots must also move
`opennn/neural_network/`, `opennn/training_strategy/` and `opennn/testing_analysis/`
includes to `opennn/network/`, `opennn/training/` and `opennn/evaluation/`. JSON
saved by those snapshots only needs the root keys renamed (`"NeuralNetwork"` to
`"Network"`, `"TrainingStrategy"` to `"Training"`) and the optimizer names from the
table above; its parameter file stays paired and unchanged.

## Release

Installation packages exclude example data and can be published. GitHub's
source archives include every tracked file, so publishing them requires the
example data to be cleared first. Historical 8.x model compatibility is not
claimed, and Neural Designer compatibility was not evaluated, at the owner's
request. Engineering verification, data clearance and the release decision are
separate statuses.

Before promotion:

- Verify the exact candidate commit on every `ci.yml` job and run the CUDA
  suite locally on a GPU machine. Report skips and disabled tests separately;
  earlier passes do not certify a new candidate.
- If a complete production 8.x model becomes available, follow the
  [migration procedure](#models-and-parameters) and compare its reference predictions.
- Check the [release notes](#whats-new-in-90). The GitHub release notes can start from that summary and the itemized
  changelog linked there.
- After the owner's decision, merge `dev` into `master`, create the annotated
  `v9.0.0` tag on the reviewed commit, and publish the artifacts with their
  checksums and verification record.

### Installation packages

Configure with `OpenNN_INSTALL=ON` and `OpenNN_CPU_TARGET=PORTABLE`, build the
library, and write packages outside the checkout:

```sh
cpack --config /absolute/path/to/build/CPackConfig.cmake -C Release -G ZIP -B ../candidate-packages
cpack --config /absolute/path/to/build/CPackConfig.cmake -C Release -G TGZ -B ../candidate-packages
```

The archive name records the version, system, processor, CPU target and backend,
and each archive gets a SHA-256 file. `share/doc/OpenNN/build-info.json` records
the compiler, configuration, CUDA, shared-library and LTO settings. The package
installs `README.md`, `LICENSE.txt`, `THIRD_PARTY_NOTICES.txt` and the
libjpeg-turbo, zlib, Eigen, cuDNN frontend and FlashAttention notices that apply. Extract it into a new directory and check it:

```sh
python tools/check_installed_package.py /absolute/path/to/extracted-prefix
cmake -S tools/package_smoke -B ../archive-consumer -DCMAKE_PREFIX_PATH=/absolute/path/to/extracted-prefix -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
cmake --build ../archive-consumer --config Release
```

Then run `../archive-consumer/opennn_package_smoke` (under `Release/` with Visual
Studio). CPack does not create a Git tag or GitHub release.

### Verification records

Record the candidate commit, toolchain, backend, test passes and skips, package
checksums, consumer results and unresolved checks in the release or pull-request
description. Keep raw evidence outside Git. The earlier September 2026 release
audit, merge reconciliation and verification evidence are preserved at commit
[`a379ec5e6`](https://github.com/Artelnics/opennn/tree/a379ec5e634d65436b8b175fcd03c044bc98182b).

## Documentation

- Only `README.md` (users) and `AGENTS.md` (agents and maintainers) live at the
  root. Do not add Markdown files for individual tasks, audits or sessions;
  update these two or the guides below.
- Subfolder guides: `examples/README.md` (example catalog),
  `benchmarks/README.md`, `benchmarks/PROTOCOL.md`,
  `tools/run-opennn-examples/SKILL.md` and `tools/run-opennn-benchmarks/SKILL.md`.
- Preserve attribution notices (`examples/*/data/SOURCE.md`,
  `THIRD_PARTY_NOTICES.txt`) and skill entry points.
- Link to an immutable Git revision for superseded material.
- Agent-specific folders such as `.claude/` stay local and ignored by Git.
  Instructions that matter for the repository belong in this file.
