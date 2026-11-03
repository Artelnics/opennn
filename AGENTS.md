# OpenNN — contributor and agent guide

[README.md](README.md) covers first use. Keep this file focused on repository-wide
working rules; task reports and verification records belong in PRs or releases.

## Working rules

- Work in the existing checkout on `dev`. Do not create clones or worktrees
  unless requested. Preserve unrelated changes and saved stashes.
- Keep development and release preparation on `dev`; merge into `master`, tag
  or publish a release only when the owner explicitly decides to release.
- Preserve public APIs and serialized formats unless a breaking change is
  authorized. Neural Designer uses exported symbols absent from local call sites.
- Keep builds, installs, downloaded models, generated data and raw results outside
  Git. Use environment variables for workstation paths.
- Follow neighboring code and retain the SPDX/copyright headers and licence notices.
  Portable `.h` interfaces must not expose CUDA vendor types; use `.cuh` internally.
- Keep README independent of branch/release state. Preserve attribution files,
  skill entry points and the public include paths. Agent-specific folders stay local.

## Library modules and dependencies

| Path | Responsibility |
| --- | --- |
| `opennn/core/` | Configuration, types, tensors, device primitives, persistence; CUDA under `core/cuda/`. |
| `opennn/network/` | Network, layers, operators, model I/O and exports. |
| `opennn/models/` | Ready-made architectures; shared execution belongs to Network. |
| `opennn/dataset/` | Dataset, TabularDataset (including forecasting), TextDataset, ImageDataset and YoloDataset. |
| `opennn/training/`, `evaluation/` | Losses, optimizers, training and quality analysis. |
| `opennn/model_selection/`, `response_optimization/` | Model selection and constrained input optimization. |
| `tests/` | Library GoogleTests, grouped by module; shared helpers under `tests/common/`. No loose root tests or tooling/example tests. |
| `examples/`, `benchmarks/`, `tools/` | Applications/data, comparison drivers and maintenance tools. |

Keep reusable tensor/device code in core and respect the dependency directions
checked by `tools/check_architecture.py`. Retain the optional FlashAttention shim.
For response repair, preserve float/double return-point feasibility checks and
constraint-container lifetimes; a failed repair does not prove infeasibility.

## Build options

Use [CMakePresets.json](CMakePresets.json) from the repository root. Ninja builds
live in `../opennn-build/<preset>`; Windows requires a Visual Studio developer
prompt. OpenMP is required. Dependency versions/licences are in
[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).

- `OpenNN_DISABLE_CUDA=ON` forces CPU; `OpenNN_REQUIRE_CUDA=ON` requires CUDA.
- For library-only builds, disable `OpenNN_BUILD_TESTS` and `OpenNN_BUILD_EXAMPLES`,
  then build target `opennn`. Benchmarks and fuzzers are opt-in.
- `OpenNN_BUILD_VISION`, `OpenNN_BUILD_SHARED` and `OpenNN_INSTALL` control scope,
  linkage and packaging. See [CMakeLists.txt](CMakeLists.txt) for all options.
- Use `OpenNN_CPU_TARGET=PORTABLE` for distribution; it removes host ISA flags,
  not algorithm or thread settings. Disable LTO across incompatible toolchains.
- MKL is opt-in; oneDNN is detected by default. Nonstandard installations use
  `OpenNN_MKL_ROOT`, `OpenNN_ONEDNN_ROOT`, `CMAKE_CUDA_COMPILER`,
  `CMAKE_CUDA_ARCHITECTURES`, `CUDNN_INCLUDE_DIR` and `CUDNN_LIBRARY`.
  CUDA/cuDNN runtime libraries must be on the runtime search path.

### CUDA batch preparation

`OPENNN_THREADS` overrides the CPU thread count (Linux otherwise uses affinity).
Dynamic OpenMP teams are opt-in through `OPENNN_OMP_DYNAMIC=1`. On hybrid Intel
CPUs with GCC/libgomp, benchmark `GOMP_SPINCOUNT=300000` or `OMP_WAIT_POLICY=active`
when barrier latency matters; set these before starting the process.

CUDA defaults to two batch workers (`OPENNN_BATCH_WORKERS`), one OpenMP thread
per fill (`OPENNN_BATCH_THREADS`, capped by the caller's limit),
`max(workers + 1, 3)` pinned host buffers and three eager device buffers.
Configure these independently with `set_workers_number`, `set_batch_pool_size`
and `set_device_batch_pool_size`; CUDA Graphs use their own pipeline. Preserve
buffer/event lifetimes, thread-setting restoration and memory-budget accounting.
More buffering consumes memory and cannot fix sustained slow preparation.
Binary classification images without augmentation can upload bytes and scale on
GPU; augmentation currently prepares floats on CPU. Measure end-to-end cost.

### Augmentation contract

Use only the active Training partition, including folds and Mosaic companions.
Validation/inference must not augment. Local RNG streams derive from global
`set_seed`, epoch and sample ID without consuming global RNG draws, independent
of worker scheduling and batch order. Adam/SGD reset epoch to zero per train call
and advance it; manual readers use `set_augmentation_epoch`. Full-batch optimizers
retain their prepared batch. Epoch is runtime state, not a resume checkpoint;
raw-buffer ImageDataset augmentation uses row IDs. Set configuration before
concurrent fills. This does not promise bit-identical training across backends.

YOLO applies source color, Mosaic, shared image/box jitter and flip, then Cutout.
Clip boxes and discard degenerate ones; Cutout preserves annotations. Keep old
JSON compatible: absent Cutout fields mean disabled. Setters and JSON loading
must reject invalid policies before changing policy/device residency, even disabled.

| Parameter | Accepted domain |
| --- | --- |
| Classification rotation | Finite ordered degrees; reduce modulo 360. |
| Classification translation | Finite ordered signed pixels strictly within ±2^(Index bits−1). |
| YOLO jitter / hue | Finite `[0, 0.5)` / `[0, 0.5]`. |
| Exposure / saturation | Positive finite factors with finite FP32 log-space samples/reciprocals; 1 neutral. |
| Cutout count / side ratio | Integer ≥ 0 / finite `[0, 1]`; either zero disables. Side capped per image axis. |

Domain references: [OpenCV geometry](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html),
[Darknet jitter](https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-%5Bnet%5D-section),
[Darknet scales](https://github.com/AlexeyAB/darknet/blob/master/src/utils.c),
[Torchvision hue](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html)
and [Cutout](https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py).
The numerical limits above are OpenNN's contract, not identical library APIs.

## Build and verify

```sh
cmake --preset verify-cpu
cmake --build --preset verify-cpu --parallel
ctest --preset verify-cpu

cmake --preset verify-cuda
cmake --build --preset verify-cuda --parallel
ctest --preset verify-cuda

python -B tools/check_code_quality.py
python -B tools/check_architecture.py
```

Use focused GoogleTest filters while editing, then full CPU and CUDA suites
without `GTEST_FILTER` for completed library changes. CUDA must run on a working
GPU; report unavailable backends and skips explicitly. Hosted CI compiles CUDA
but has no GPU. Shared-code changes require both backends. When overriding `-B`,
use that directory for subsequent build/CTest commands rather than the presets.

Use Python 3.12 and the dependencies in `tools/test-requirements.txt` and
`tools/code-quality-requirements.txt`. Export tests need NumPy, pandas,
onnxruntime and Node; missing runtimes are skips. Use `python -B` to avoid caches.
The Linux Clang `verify-sanitizers` preset runs ASan/UBSan; JSON fuzzing is under
`tools/fuzz/`. Coverage floors and static analysis are defined by
`tools/check_coverage.py` and [.github/workflows/ci.yml](.github/workflows/ci.yml).

Run quality/architecture checks before every commit. Update `tools/CODE_QUALITY.json`
with `check_code_quality.py --update` for legitimate size changes; never relax
complexity or duplication limits to hide a regression. Include the updated limits.
CI runs on PRs to dev/master and pushes to master; use workflow_dispatch for dev
without a PR. Markdown/benchmark-manifest-only changes skip automatic CI.

## Maintenance tools

| Task | Entry point |
| --- | --- |
| Examples | [Catalog](examples/README.md), [execution skill](tools/run-opennn-examples/SKILL.md) |
| Benchmarks | [Usage](benchmarks/README.md), [protocol](benchmarks/PROTOCOL.md), [execution skill](tools/run-opennn-benchmarks/SKILL.md) |
| Performance gate | `benchmarks/compare.py`: 5% throughput/memory allowance, no lower confirmed batch capacity. |
| Headers / coverage | `tools/check_headers.sh`, `tools/check_coverage.py` |
| Packages | `tools/check_installed_package.py`, `tools/package_smoke/` including dependency relocation. |

Follow the corresponding skill for full example matrices or benchmark cells.
Only approve protocol-compliant measurements. Raw benchmark output belongs in
`../opennn-benchmark-results/` or `OPENNN_BENCH_RESULTS`, never in commits.
Run data-dependent examples from their executable directory; update the example
catalog and data notices when changing examples, data, targets or dependencies.
CI compiles examples/benchmarks but does not run them.

## Data and releases

- Preserve `examples/*/data/SOURCE.md`, manifests and applicable licences. Keep
  archive member names unique; hash Git bytes, not CRLF-transformed checkouts.
- Keep Hugging Face model notices synchronized with their example copies. Preserve
  legacy GPT-2/BERT release assets still referenced by historical dev snapshots.
- Record YOLO provenance, including synthetic/external data and conversions. Obtain
  checkpoint-specific confirmation before redistributing Darknet weights.
- Packages exclude example data. Use `OpenNN_INSTALL=ON`, `OpenNN_CPU_TARGET=PORTABLE`
  and CPack outside the checkout; verify extracted packages, dependency relocation
  and a consumer. Retain licence files, build-info and SHA-256 checksums.
- Verify the exact release candidate with all CI jobs and local CUDA tests; record
  toolchain, passes, skips, downstream Neural Designer checks and unresolved work.
  Update release notes/migration requirements. Verification is not release approval.
- After the owner's release decision, promote dev to master and tag the reviewed,
  passing commit as `v<project version>`; publish artifacts and the verification record.

### Migrating from 8.x to 26.0

26.0 is a source/model-format break: rebuild consumers and migrate models with
reference-prediction checks. Parameter counts alone do not establish layout
compatibility. Keep paired JSON/binary files. Detailed release notes, renames and
the migration procedure remain in the
[archived guide](https://github.com/Artelnics/opennn/blob/54a45ae9f49c7d3e7cbb14fd46cdd8ad4fb5b613/AGENTS.md#260-release-notes-and-migration).
