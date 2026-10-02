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

## Code conventions

### Naming

| Category | Convention | Example |
| --- | --- | --- |
| Classes and enums | PascalCase; concrete layers are `final` | `class Convolutional final : public Layer` |
| Enum values | PascalCase | `Device::CPU`, `Type::FP32`, `ActivationFunction::LeakyReLU` |
| Member variables | `lowercase_with_underscores`, no prefix, initialized inline | `Index output_features = 0;` |
| Methods | `lowercase_with_underscores`; `get_`/`set_` for properties | `get_output_shape()`, `set_label()` |
| Files | `lowercase_with_underscores.h/.cpp`, paired | `dense_layer.h`, `dense_layer.cpp` |
| Type aliases | Suffix encodes content: `R`=`float`, `I`=`Index`, `B`=`bool`, `Map`=mapped view | `MatrixR`, `VectorI`, `TensorMapR<Rank>` |
| Constants | `UPPER_CASE` `constexpr` | `POS_INFINITY`, `ALIGN_BYTES` |
| Template rank params | `Rank` (semantic); generic types `T`, `TIn`, `TOut` | `template <int Rank>` |

### Tensor and matrix types

All owned storage uses row-major layout: `constexpr int Layout = Eigen::RowMajor`.

```cpp
// Matrix aliases  (opennn_types.h)
MatrixR   // Eigen::Matrix<float, Dynamic, Dynamic, RowMajor>
MatrixI   // …<Index, …>
VectorR   // Eigen::Matrix<float, Dynamic, 1>
VectorMap // Eigen::Map<VectorR, AlignedMax>

// Tensor aliases  (opennn_types.h)
Tensor0 … Tensor4           // Eigen::Tensor<float, N, Layout|AlignedMax>
TensorR<Rank>               // rank-generic owned tensor
TensorMapR<Rank>            // rank-generic non-owning map into host memory
```

`TensorView` (`tensor_types.h`) is the runtime handle: `void* data + Shape + Type + Device`.
Convert at call sites via `tv.as_matrix()`, `tv.as_vector()` or `tv.as_tensor<Rank>()`.
`Shape` holds up to four `Index` dims plus a rank field; grow it with `push_back` or `append`.
`Index` is `Eigen::Index` (64-bit signed).

Never use raw `Eigen::Tensor<float, N>` in new code without the `Layout|AlignedMax` flags.

### Layer class structure

All layers derive from `Layer` (`opennn/network/layers/layer.h`).

Key virtual methods to override:

```cpp
Shape get_output_shape() const override;          // pure virtual – always required
vector<TensorSpec> get_parameter_specs() const;   // learnable parameters
vector<TensorSpec> get_state_specs() const;       // persistent state (e.g. BN stats)
vector<TensorSpec> get_forward_specs(Index batch_size) const;   // intermediate slot shapes
vector<TensorSpec> get_backward_specs(Index batch_size) const;  // input-delta slot shapes
```

Propagation delegates to owned `Operator*` objects in order (forward) and reverse (backward):

```cpp
void forward_propagate(ForwardPropagation& fp, size_t layer, ForwardPropagationMode pass) override;
void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;
```

Operators are stored in `protected vector<Operator*> operators` and registered in the
constructor (e.g. `operators = {&combination, &batch_norm, &activation_operator, &dropout}`).
Use a private `enum Forward { Input, CombinationView, ..., Output }` to name slot indices;
never use magic numbers.

Constructors must pass `LayerType` to `Layer(type, trainability)`.

### ForwardPropagation and BackPropagation

`ForwardPropagation` (`network/forward_propagation.h`) owns a single `Buffer arena` for
all intermediate activations. Per-layer slot tensors are `TensorView` pointers into this
arena. Access layer inputs via `fp.inputs[layer][ordinal]` and intermediate slots via
`fp.slots[layer][slot_idx]`.

`BackPropagation` (`network/back_propagation.h`) owns a joint gradient `Buffer gradient`
and a `Buffer arena` for intermediate deltas. Use `bp.output_deltas[layer]` for the loss
gradient w.r.t. a layer's output; write input deltas into `bp.slots[layer][...]`.

Operators receive `(ForwardPropagation&, size_t layer_index, ...)` and route their I/O
via `input_slots` / `output_slots` index vectors declared on each operator.

### Tests

Tests live under `tests/`, organized by library module.  Every test file is named
`{feature}_test.cpp`.  Framework is GoogleTest; the `pch.h` precompiled header is in
`tests/common/`.

```cpp
#include "tests/common/pch.h"
#include "opennn/<module>/...h"
using namespace opennn;

TEST(SuiteName, CaseName) {
    // arrange
    // act
    // assert with EXPECT_* or ASSERT_*
    EXPECT_LT((result.array() - expected.array()).abs().maxCoeff(), type(1e-6));
}
```

### Numerical gradient testing

Use the helpers in `tests/common/numerical_derivatives.h` to validate analytical gradients:

```cpp
VectorR analytical  = calculate_gradient(loss);
VectorR numerical   = calculate_numerical_gradient(loss);
// Compare element-wise; typical tolerance 1e-3 relative.
```

`calculate_h(x)` computes the finite-difference step `1e-3 * (1 + |x|)`.
Always call `require_live_parameters(network)` before the gradient comparison to ensure
no dead parameters produce a spurious all-zero match.

### CMake

Sources are collected with `file(GLOB_RECURSE … CONFIGURE_DEPENDS "*.cpp")` in
`opennn/CMakeLists.txt`; add new `.cpp` files there automatically.  Vision/sequence
sources are excluded when `OpenNN_BUILD_VISION=OFF` via a regex filter — keep new layer
files in the right module folder so the filter works without modification.

CUDA availability is detected with `check_language(CUDA)` and guarded by
`OpenNN_DISABLE_CUDA`; wrap CUDA-specific code in `#ifdef OPENNN_HAS_CUDA`.
External dependencies (Eigen, GoogleTest, cuDNN frontend) are provided via
`FetchContent` with a `find_package` fast path; do not vendor them manually.

The minimum standard is C++20 (`cxx_std_20`); use it freely.

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
- MKL is opt-in (`OpenNN_ENABLE_MKL=ON`); given no installation, Linux x86-64
  downloads Intel's static libraries (`OpenNN_FETCH_MKL=OFF` disables it).
  oneDNN is detected by default and, when absent on x86-64 outside
  macOS, built from source (`OpenNN_FETCH_ONEDNN=OFF` disables the download);
  without it the CPU LSTM runs the much slower built-in path. Nonstandard installations use
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
CUDA image training transforms pixels and scales them on GPU, using kernels in
`core/cuda/kernel_image.cu`; classification supports FP32 matrix or byte inputs,
and YOLO uploads cached bytes. CPU workers read sources, sample deterministic
parameters and prepare matching boxes/targets. Mosaic shares repeated sources
within a batch. Allocate staging before CUDA Graph capture and account for it
in automatic batch sizing. Measure end-to-end cost.

### Augmentation contract

Augmentation is disabled by default. An enabled policy trains every Training
sample plus `factor` augmented copies per epoch, shuffled together: copy c of
sample s is index `s + c * samples`, copy 0 is the untouched original, and
`get_augmented_sample_indices` lists them. Copies are redrawn each epoch and
repeat neither the original nor each other while the policy can; setters reject
factors above its distinct transforms (Mosaic/Cutout layouts are unbounded), and
a policy without transforms adds no copies.

Use only the active Training partition, including folds and Mosaic companions.
Validation/inference must not augment. Local RNG streams derive from global
`set_seed`, epoch, sample ID and copy without consuming global RNG draws, independent
of worker scheduling and batch order. Adam/SGD reset epoch to zero per train call
and advance it; manual readers use `set_augmentation_epoch`. Full-batch optimizers
retain their prepared batch, copies included. Epoch is runtime state, not a resume
checkpoint; raw-buffer `augment_inputs` gives each row its first copy. Set
configuration before concurrent fills. This does not promise bit-identical
training across backends.

YOLO applies source color, Mosaic, shared image/box jitter and flip, then Cutout.
Clip boxes and discard degenerate ones; Cutout preserves annotations. Keep old
JSON compatible: absent Cutout and `AugEnabled` fields mean disabled, an absent
factor means 1. Setters and JSON loading
must reject invalid policies before changing policy/device residency, even disabled.
YOLOv8 sizes targets for the largest source box count (four sources with Mosaic),
keeping 100 slots as the compatibility minimum. Mosaic caches Training indices
until roles, folds or data change.

| Parameter | Accepted domain |
| --- | --- |
| Augmentation factor | Integer `[1, 10]`, at most the policy's distinct transforms. |
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
