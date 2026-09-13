# Development and releases

Use this guide for verification, release preparation, packaging and CI operations.
The [README](README.md) covers building and using OpenNN; the
[changelog and migration guide](CHANGELOG.md) covers compatibility changes.

- [Working branches](#working-branches)
- [Build configuration](#build-configuration)
- [Source map](#source-map)
- [Maintenance tools](#maintenance-tools)
- [Verification](#verification)
- [Quality gates](#quality-gates)
- [Release checklist](#release-checklist)
- [Installation packages](#installation-packages)
- [Linux CUDA runner](#linux-cuda-runner)
- [Verification records](#verification-records)

## Working branches

Use one checkout. Everyday development and release preparation happen on `dev`.
Commit and push completed work to `dev`; promote it to `master` only when the
owner explicitly decides it is ready. Additional clones or worktrees require
an explicit request. Preserve pending changes and stashes when switching branches.

Contributors should base changes on `dev`, include relevant validation, and
submit a pull request with a description of the resulting behavior. Follow
[AGENTS.md](AGENTS.md) for repository engineering rules.

## Build configuration

Run commands from the repository root and keep build and install directories
outside it. A default configuration builds examples and tests; select only
the targets needed for your task. The README's first example builds `blank`.
For a library-only build, set `OpenNN_BUILD_TESTS=OFF` and
`OpenNN_BUILD_EXAMPLES=OFF`, then build target `opennn`.

### CMake options

| Option | Default | Description |
|---|---:|---|
| `OpenNN_DISABLE_CUDA` | `OFF` | Force a CPU-only build even when CUDA is available. |
| `OpenNN_REQUIRE_CUDA` | `OFF` | Fail configuration if CUDA cannot be enabled. |
| `OpenNN_BUILD_TESTS` | `ON` | Build the GoogleTest test suite. |
| `OpenNN_BUILD_EXAMPLES` | `ON` | Build example applications. |
| `OpenNN_BUILD_BENCHMARKS` | `OFF` | Build benchmark drivers from `benchmarks/`. |
| `OpenNN_BUILD_FUZZERS` | `OFF` | Build Clang libFuzzer targets with ASan/UBSan. |
| `OpenNN_BUILD_VISION` | `ON` | Build vision, sequence, transformer, and detection components. |
| `OpenNN_BUILD_SHARED` | `OFF` | Build OpenNN as a shared library instead of a static library. |
| `OpenNN_ENABLE_MKL` | `OFF` | Use Intel MKL as Eigen's BLAS/LAPACK backend. |
| `OpenNN_ENABLE_LTO` | platform-dependent | Enable interprocedural optimization for release builds. |
| `OpenNN_CPU_TARGET` | `NATIVE` | `NATIVE` preserves local throughput; `PORTABLE` removes host-specific ISA flags for distributable binaries. |

Clang static libraries built with LTO require a compatible Clang/LLVM toolchain
in their consumers. Their CMake target carries the required linker flags.
Set `OpenNN_ENABLE_LTO=OFF` when distributing native static objects across
compiler toolchains.

Additional controls:

| Option | Default | Purpose |
| --- | --- | --- |
| `OpenNN_INSTALL` | `ON` | Generate library installation and package rules. |
| `OpenNN_BUILD_BLANK` | `ON` | Include the minimal `blank` inference target when examples are enabled. |
| `OpenNN_ENABLE_ONEDNN` | `AUTO` | Use detected oneDNN CPU primitives; `ON` requires it and `OFF` disables the search. |
| `OpenNN_MKL_ROOT`, `OpenNN_ONEDNN_ROOT` | Empty | Locate a nonstandard MKL or oneDNN installation. |
| `OpenNN_CHECK_HEADERS` | `OFF` | Compile public headers independently. |
| `OpenNN_UNIT_TEST_TIMEOUT` | `600` seconds | CTest limit for the unit suite; the sanitizer preset uses 1,800. |

OpenMP is required. Eigen is found as a package or fetched; zlib and libjpeg-turbo
are fetched by CMake. oneTBB and oneDNN are optional detected CPU dependencies;
MKL is explicitly opt-in. GoogleTest is needed only for tests. CUDA adds NVIDIA
runtime dependencies and the fetched cuDNN frontend. Versions, licence notices
and optional CUDA kernels are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### CPU threads

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

### CMake presets

The verification presets require Ninja and use `../opennn-build/<preset>` by
default. Configure, build and test presets share the names `verify-cpu`,
`verify-cuda` and `verify-sanitizers` (Linux Clang). Each build includes both
test executables; CUDA testing explicitly requires an available GPU.

```sh
cmake --preset verify-cpu
cmake --build --preset verify-cpu --parallel
ctest --preset verify-cpu
```

Changing the preset's build directory with `-B` requires passing that same
directory to subsequent `cmake --build` and `ctest --test-dir` commands.
Use the wrappers below for automatic toolchain setup and persistent user-local
caches. They select their own external directories rather than the preset default.

## Source map

Use the public module headers directly; for example,
`#include "opennn/network/network.h"`. Do not include the private precompiled
header `pch.h` in an application.

| Module | Responsibility and starting points |
| --- | --- |
| `opennn/core/` | Configuration, tensor/storage types, backend operations, persistence utilities; CUDA kernels are under `core/cuda/`. Start with `configuration.h` and `opennn_types.h`. |
| `opennn/network/` | `Network`, layers, operators, propagation, save/load, chat and `ModelExpression` source export. |
| `opennn/models/` | Ready-made tabular, image, language and forecasting models declared in `models.h`. |
| `opennn/dataset/` | `TabularDataset` and specialized image, language and time-series data handling. |
| `opennn/training/` | `Training`, losses and optimizers such as `Adam` and `SGD`. |
| `opennn/evaluation/` | `Evaluation` for prediction-quality analysis. |
| `opennn/model_selection/` | Input and network-size selection. |
| `opennn/response_optimization/` | Optimize inputs subject to model outputs and constraints. |

`core` is the foundation. `network` and `models` define computation; datasets
feed training, and evaluation and selection use those interfaces. This is not
a strict linear dependency chain: [the architecture checker](tools/check_architecture.py)
records the allowed directions and specific existing exceptions. Tests generally
mirror these source modules. Public APIs may have consumers outside this
repository, including Neural Designer; local call counts are not removal criteria.

## Maintenance tools

Most users need the library and one example. These tools support development:

| Task | Entry point |
| --- | --- |
| Incremental CPU/CUDA verification | `tools/verify.ps1` or `tools/verify.sh`; shared orchestration in `verify.cmake` |
| Source-size, complexity and dependency checks | `tools/check_code_quality.py`, `tools/check_architecture.py` |
| Coverage and public-header checks | `tools/check_coverage.py`, `tools/check_headers.sh` |
| CUDA memory errors and JSON fuzzing | `tools/check_cuda_memory.sh`, `tools/fuzz/` |
| Installed-package checks and C++ consumer | `tools/check_installed_package.py`, `tools/package_smoke/` |
| Dataset inventory and release scope | `tools/check_dataset_manifest.py`, `tools/check_release_scope.py` |
| Reconstruct data and train reference models | `tools/reproduce_datasets.py`, `tools/reproduce_models/`; see [recipes](DATASETS.md#reproduction) |
| Full example/device matrix | `tools/run-opennn-examples/SKILL.md` |
| Benchmark preparation and execution | `benchmarks/prepare.py`, `benchmarks/run.py`; helpers stay in `benchmarks/tools/` |

The root JSON files are maintained inputs, not scratch results:

| File | Purpose |
| --- | --- |
| `CMakePresets.json` | Repeatable configure/build/test settings |
| `CODE_QUALITY.json` | Reviewed maintainability limits |
| `datasets.manifest.json` | Hashes and clearance state of indexed example assets |
| `RELEASE_SCOPE.json` | What the current candidate may claim and distribute |

Add usage or audit findings to these existing guides. Keep raw logs and generated
reports outside the checkout, and put superseded narrative reports in Git history.

## Verification

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
ordinary CUDA verification includes that test. See [runner operations](#linux-cuda-runner)
for the Linux GPU runner's requirements and availability.

The benchmark harness tests need only NumPy and SacreBLEU, not a PyTorch or
CUDA runtime. In an isolated Python environment:

```sh
python -m pip install numpy==2.4.4 sacrebleu==2.5.1
python -m unittest discover -s tests -p 'benchmark*_test.py'
```

Choose checks appropriate to the change. Documentation edits need working links
and, when installed documents change, package-install checks. Shared library
implementation changes need both CPU and CUDA verification.

## Quality gates

The configured CI workflows cover supported builds and runtime checks.
A candidate commit must pass the applicable jobs before promotion.

### Runtime-neutral build profiles

`OpenNN_CPU_TARGET=NATIVE` is the default and retains the existing local
throughput settings (`-march=native`, `/arch:AVX2`, or the Apple Silicon target).
Binary release packages use `OpenNN_CPU_TARGET=PORTABLE`; the selected value is
recorded in `build-info.json` and the archive name. The portable profile removes
only explicit host-specific ISA flags. It does not change tensor layouts,
allocations, algorithms, thread counts, CUDA kernels, or the native default.

CI installs and runs static and shared-library consumers on Linux, a portable
AppleClang consumer on macOS, and static consumers on GCC, Clang and MSVC. CUDA
is compiled on a hosted runner and executed on the registered GPU runner.

### Analysis, coverage and fuzzing

The static-analysis job treats selected Clang analyzer ownership defects and
use-after-move findings as errors in central persistence, dataset and device
modules. This focused blocking set avoids accepting a large unreviewed warning
baseline.

The coverage job executes the CPU unit suite with GCC instrumentation and
requires at least 70% line and 35% branch coverage across non-CUDA library
sources. It also enforces the reviewed per-module floors in
`tools/check_coverage.py`. Its JSON and HTML reports are retained as CI
artifacts; thresholds should rise as tests are added.

The sanitizer build also links `opennn_json_fuzz` with libFuzzer, ASan and
UBSan. CI runs 20,000 mutations from the checked-in corpus. Longer local runs
can use:

```sh
cmake --preset verify-sanitizers -B ../build-fuzz -DOpenNN_BUILD_FUZZERS=ON
cmake --build ../build-fuzz --target opennn_json_fuzz --parallel 2
../build-fuzz/bin/opennn_json_fuzz -max_total_time=3600 tools/fuzz/corpus
```

The JSON parser rejects non-standard numbers, unescaped control characters,
invalid surrogate pairs, nesting beyond 256 containers and inputs over 256 MiB.

### Maintainability ratchets

`python tools/check_code_quality.py` measures first-party C++ code while
excluding the vendored FlashAttention shim. `CODE_QUALITY.json` is the reviewed
upper bound for logical and physical lines, duplicate code, oversized functions
and cyclomatic complexity. A change must simplify a regression or update the
baseline explicitly during review; ordinary feature work cannot silently grow
these measures.

`python tools/check_architecture.py` enforces the dependency direction between
core, datasets, networks, training, evaluation, model selection and response
optimization. A small path-specific exception list records existing cycles so
they cannot spread to other files.

Coverage is gated per module as well as repository-wide. Performance changes
can be checked with `python benchmarks/compare.py baseline.json candidate.json`;
the default controlled-machine gate permits 5% measurement variation in
throughput and memory and rejects lower confirmed batch capacity.

Install the checker dependencies with
`python -m pip install -r tools/code-quality-requirements.txt` before running
the maintainability and architecture commands.

## Release checklist

OpenNN 9.0.0 is an unreleased candidate. Development includes master through
`efd566b38`; reconciliation does not publish a release. Public API and model
migration requirements are in [CHANGELOG.md](CHANGELOG.md#migrating-from-8x-to-90).
Neural Designer validation remains deferred at the owner's request and is not
claimed as verified compatibility.

Before promotion:

- Verify the actual candidate commit on hosted GCC, Clang and MSVC CPU builds,
  package consumers, shared and portable packages, macOS, CUDA compilation,
  static analysis, coverage, fuzzing and ASan/UBSan.
- Complete Linux GPU unit and response-optimization integration verification.
  Retain skips and disabled tests separately; historical passes do not certify
  the new candidate.
- Run `python tools/check_dataset_manifest.py`. Before distributing all tracked
  data, its `--release` check must also pass. The unresolved records in
  [DATASETS.md](DATASETS.md) still require review or reproducible replacements.
- Run `python tools/check_release_scope.py --package-kind binary` for an
  installation archive. [RELEASE_SCOPE.json](RELEASE_SCOPE.json) excludes example
  assets from binary packages and blocks full-source publication while their
  redistribution records remain unresolved. GitHub automatic source archives
  contain the tracked datasets.
- If a complete production 8.x model becomes available, follow the migration
  procedure and compare its reference predictions. No complete fixture was
  available for this review; current-format round trips and newly trained
  models do not establish historical compatibility.
- Check the final changelog and benchmark publication status. Approve only
  measurements that meet [the benchmark protocol](benchmarks/PROTOCOL.md).
- After the owner's release decision, merge `dev` into `master`, create the
  annotated `v9.0.0` tag on the reviewed master commit, and publish matching
  artifacts with their checksums and verification record.

Engineering verification, data clearance and the release decision are separate
statuses. Keep generated logs, packages and model files outside Git.

## Installation packages

CPack produces installation packages from a completed OpenNN build. Configure
with `OpenNN_INSTALL=ON` (the default), build the library, and keep package
output outside the checkout:

```sh
cpack --config /absolute/path/to/build/CPackConfig.cmake -C Release -G ZIP -B ../candidate-packages
cpack --config /absolute/path/to/build/CPackConfig.cmake -C Release -G TGZ -B ../candidate-packages
```

The archive name contains `9.0.0-candidate`, the system, target processor and
CPU target and CPU/CUDA backend. Release binaries must be configured with
`OpenNN_CPU_TARGET=PORTABLE`; native remains the default for local maximum
throughput. Each archive has a SHA-256 companion file. Its installed
`share/doc/OpenNN/build-info.json` records the library version, compiler/version,
configuration, CUDA, shared-library and LTO settings. The installation includes
OpenNN's LGPL/GPL licence texts, release/migration documentation, libjpeg-turbo
notices, zlib's licence, and fetched Eigen/cuDNN frontend notices when applicable.
Optional FlashAttention installations include its and its CUTLASS dependency's
licence notices as well.

Extract into a new directory, check notices, and configure a separate consumer:

```sh
python tools/check_installed_package.py /absolute/path/to/extracted-prefix
cmake -S tools/package_smoke -B ../archive-consumer -DCMAKE_PREFIX_PATH=/absolute/path/to/extracted-prefix -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
cmake --build ../archive-consumer --config Release
```

Run `../archive-consumer/opennn_package_smoke` (or the executable under
`Release/` on Visual Studio). Use a compatible compiler/toolchain for the binary
package. Fetched Eigen and zlib are installed with the library; an Eigen package
found externally during configuration remains an external consumer dependency.
OpenMP, oneTBB, oneDNN, MKL and NVIDIA runtimes are not automatically copied into
the archive. Supply any dependencies selected by that build. A package built
with LTO also requires the matching compiler/linker support.

These are binary installation archives. They exclude example datasets and model
assets. They do not certify the separate GitHub source archive's dataset rights,
historical model compatibility, or full-source publication approval. Run
`python tools/check_release_scope.py --package-kind binary` before packaging;
the full-source scope intentionally remains blocked while tracked assets have
unresolved redistribution records. Record the source commit,
archive checksums, toolchain and actual consumer results in the candidate verification record
before promotion. CPack does not create a Git tag or GitHub release.

## Linux CUDA runner

The `Linux CUDA` workflow runs only on pushes to `dev` and `master`, manual
dispatch, and the default branch's nightly schedule. Pull-request builds use
GitHub-hosted machines. A runner must have the `self-hosted`, `linux`, and
`cuda` labels, CMake 3.24+, Ninja, a C++20 compiler, CUDA, cuDNN 9, and a working
NVIDIA GPU. The job checks `nvidia-smi` and `nvcc` before full verification.

Configure the intended CUDA compiler through the runner's `PATH`. Nonstandard
cuDNN installations also require `OPENNN_CUDNN_INCLUDE_DIR` and
`OPENNN_CUDNN_LIBRARY`; `OPENNN_CUDA_ARCHITECTURES` can set the GPU architecture.
These paths belong in the runner environment, not in the repository workflow.

The runner `opennn-wsl-cuda` was registered on 2026-09-07. It uses a dedicated
Linux account, `opennn-ci`, and a systemd service. Its service restricts writes
to the runner directory and its own home, hides other Linux home directories
and the Windows C: mount, and prohibits gaining privileges. WSL DNS remains
accessible. This is a workstation runner; it can accept jobs only while its
Windows host is awake, online, and WSL is running.

On that workstation, the Windows scheduled task `OpenNN Linux CUDA runner`
starts a hidden WSL keepalive at sign-in. The Linux service starts with WSL.
The task does not wake the computer or make the runner available while it is
powered off. A dedicated always-on GPU host would remove this limitation.

Service operations in WSL:

```bash
sudo systemctl status actions.runner.Artelnics-opennn.opennn-wsl-cuda.service
sudo journalctl -u actions.runner.Artelnics-opennn.opennn-wsl-cuda.service -n 50
sudo systemctl restart actions.runner.Artelnics-opennn.opennn-wsl-cuda.service
```

To pause this runner, stop its Linux service. To disable startup permanently,
disable the service and the Windows scheduled task. Remove its registration in
GitHub repository Settings > Actions > Runners if retiring the host.

The implementation follows GitHub's [runner registration](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)
and [Linux service](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/configure-the-application)
documentation. Keep this runner off untrusted pull-request workflows; public
repository code must be reviewed before it runs on a persistent host.

## Verification records

Record the candidate commit, toolchain, backend, test passes/skips, package
checksums, consumer results and unresolved checks in its release or pull-request
record. Keep raw evidence outside the source tree. Do not add a new root Markdown
file for each verification session.

The earlier release preparation is preserved at commit `a379ec5e6`:

| Historical record | Scope |
| --- | --- |
| [Release plan](https://github.com/Artelnics/opennn/blob/a379ec5e634d65436b8b175fcd03c044bc98182b/RELEASE_PLAN.md) | September 7 audit and original roadmap; its counts and proposed work are historical. |
| [Merge reconciliation](https://github.com/Artelnics/opennn/blob/a379ec5e634d65436b8b175fcd03c044bc98182b/MERGE_RECONCILIATION.md) | Behavioral decisions and all 98 conflicted paths when joining development `5840396e9` with master `efd566b38`. |
| [Verification evidence](https://github.com/Artelnics/opennn/blob/a379ec5e634d65436b8b175fcd03c044bc98182b/RELEASE_VERIFICATION.md) | September 7–8 CPU/CUDA, sanitizers, model reproduction and candidate archives, with source identities and checksums. |

The September 8 functional commit `593c69c65` passed
[hosted CI](https://github.com/Artelnics/opennn/actions/runs/34198361151) and
[Linux CUDA verification](https://github.com/Artelnics/opennn/actions/runs/34198361146).
These links preserve dated evidence; check the new candidate's workflow results
before promotion. See [Actions](https://github.com/Artelnics/opennn/actions) for
run records.

To inspect any original document locally without another checkout:

```sh
git show a379ec5e6:RELEASE_VERIFICATION.md
```


## Repository audit follow-up

The September 11, 2026 navigation audit covered the tracked source tree, examples,
tests, benchmarks, build/package files, workflows and the GitHub landing branch.
The library already has distinct modules; no public symbols need moving merely
to make the repository easier to browse. The largest current content group is
bundled example assets. It contains 10,131 indexed files, including 10,000 MNIST
test images and 102 melanoma images. These assets remain subject to [their
provenance review](DATASETS.md); replacing them requires reproducible alternatives.

The first-use guide now leads to a real inference example and a catalog of all
16 current example targets. Build presets include both test executables, example
data follows the executable directory for multi-configuration builds, and CI
checks all three Python benchmark test modules. Historical 8.x projects are
explicitly marked in the example catalog rather than presented as current targets.

Remaining follow-up:

- GitHub's default branch is `master`, whose 8.x tree still contains the older
  layout, vendored dependencies, editor cache and a stale Travis badge. The
  `dev` improvements reach that landing page only through an approved release
  promotion; changing branches locally does not update it.
- The current guides are an entry point, not a complete generated API reference.
  Extend API documentation in the existing public headers as behavior is reviewed.
- Resolve outstanding asset records and real-data benchmark measurements before
  making the corresponding publication claims. Existing tests do not close those
  reviews. Source and benchmark history stay available without additional clones.
