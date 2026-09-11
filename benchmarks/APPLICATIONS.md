# Application startup and deployment

These experiments compare OpenNN C++ with the **PyTorch Python API**. They use
small applications, not the larger training-throughput models. Native source is
`families/application.cpp`; the Python counterpart is `families/application.py`.
Each native model is a separate `<model>_application_opennn` CMake target, so the
deployment measurement does not charge every application for all four models.

## Build and prepare

Linux/WSL, Python 3.12 with venv support, CMake, Ninja and a C++ compiler are
required. Build directories and Python installations must be outside the checkout.
For the CPU Eigen application:

```bash
python3.12 benchmarks/prepare.py applications --backends cpu-eigen \
  --work "$HOME/opennn-benchmark-data/applications"
```

For the complete matrix, choose `--backends cpu-eigen,cpu-mkl,cuda` and supply
`--mkl-root`, `--onednn-root`, `--cuda-root`, `--cudnn-include` and
`--cudnn-library`. All paths describe the computer being tested; none is built
into the code. `--gpu-arch` defaults to `native`. `--eigen-source` and
`--cudnn-frontend-source` optionally reuse dependency source directories.

Preparation builds the current checkout in Release with LTO and installs
separate CPU and CUDA Python environments. Their dependency versions are pinned
in `manifests/application-{cpu,cuda}-requirements.txt`. Python numerical work
runs in eager inference mode; startup does not prepay a compiler warm-up.
Do not install quality-scoring packages into these deployment environments.

The generated `applications.json` contains native binary patterns, Python
interpreters, runtime library search paths and build provenance. It can also
describe existing builds: each group has `engine`, `backend`, `device`,
`binary_pattern`, optional `env`, and Python `prefix_args` of
`["-I", "/absolute/path/to/benchmarks/families/application.py", "{family}"]`.
Every OpenNN device requires exactly one Python baseline for that device.

## Run

```bash
python benchmarks/run.py --family startup \
  --config "$HOME/opennn-benchmark-data/applications/applications.json" --smoke
python benchmarks/run.py --family startup \
  --config "$HOME/opennn-benchmark-data/applications/applications.json" --rounds 3 --repeats 5
python benchmarks/run.py --family deployment \
  --config "$HOME/opennn-benchmark-data/applications/applications.json"
```

The parent Python needs NumPy, as does the existing common runner. The isolated
application Python environments contain only their pinned runtime packages.
Output defaults to a new directory in `benchmarks/results/scratch/`.
An explicit `--out` must also be a new directory below that location.
Raw JSON, CSV, readable Markdown tables, loader traces and file inventories are
kept together. Failures remain visible and make the command fail.

## Workloads and timing

| Model | Small application |
|---|---|
| Dense | batch 2; 28 → 128 → 128 → 1; 20,353 parameters |
| LSTM | batch 2; 8 steps × 15 features; 128 units → 1; 73,857 trainable parameters |
| CNN | batch 2; RGB 32 × 32; convolutions 16/32 → dense 128 → 10; 268,650 parameters |
| Transformer | batch 2; sequence 8; vocabulary 128; d32/h4/ff64; one encoder and decoder; 33,792 parameters |

Startup spans the parent's process launch to the child's **first completed
prediction in host memory**, using Linux CLOCK_MONOTONIC in both processes.
It includes interpreter/imports, dynamic loading, context creation, model and
input construction, and the first forward pass. It excludes post-prediction
validation, printing and teardown. Process lifetime is recorded separately.
No model file is loaded. This differs from the older `footprint` process-lifetime
metric; do not mix their numbers.

CPU uses FP32. CUDA uses FP32 and BF16, each with saved and empty application
tuning caches. Filesystem caches stay warm. A separate process warms saved
caches; empty-cache samples receive fresh directories. Two threads and a fixed
0.2-second gap outside every timed region are used. Optional `--affinity` sets
the same Linux CPU IDs for parent and children. Case order rotates by a fixed
seed, and all repetitions are retained. The full matrix has 44 configurations;
three rounds of five samples produce 660 timed processes.

The runner records medians, ranges and CV. It does not enforce GPU clocks or
host idleness. These observations remain diagnostic; CV above 3% and unlocked
GPU clocks fail the publication contract. A smoke timing is never a speed claim.

## Deployment and dependencies

The primary OpenNN size includes its executable and exercised native runtime
dependency closure. The primary PyTorch size includes the application, Python
interpreter and standard library, its complete installed runtime packages, and
exercised external native dependencies. This compares an OpenNN native bundle
with a **standard Python installation**, not two maximally pruned bundles.

The artifact separately reports observed runtime files for both engines using
loader traces, memory maps, imported Python modules and recursive ELF dependency
resolution. That subset is diagnostic, not a proven portable minimal bundle.
Files are deduplicated by resolved path and counted as logical decimal bytes.
OS/driver files, model weights, datasets, installers and generated caches are
excluded. Missing native dependencies fail the measurement.

Each case records its Python package inventory and count. OpenNN's zero Python
packages does not mean zero native dependencies. CPU and CUDA Python
installations have different package closures; do not quote one count for both.

The September 10 measurements and original reproducibility archives imported
from the presentation folder remain in ignored `results/scratch/`, with their
original machine, source revisions and protocols. Older LibTorch comparisons
remain supplementary archived results; they are not the Python baseline.
