# Benchmark protocol

This document is the measurement contract for every benchmark family in this
directory. It is intentionally independent of a particular computer: results
from different machines are valid reproductions, but they are separate result
sets and must never be combined into a direct engine comparison.

Changing a rule that affects the workload, engine configuration, timed region,
instrumentation or validity gates creates a new protocol revision. Results
affected by such a change must be rerun; a note beside old numbers is not a
substitute for measuring them again.

Procedures: [standard families](#9-standard-family-procedure),
[startup and deployment](#12-application-startup-and-deployment),
[prediction quality](#13-prediction-quality).

## 1. What is being compared

A benchmark cell is defined by all of the following:

- source revision and whether the working tree is clean;
- benchmark family and mode;
- model topology, parameter count and numerical precision;
- dataset or model-weight revision and content hashes;
- device, backend, libraries, driver and runtime versions;
- batch size;
- engine options, warm-up policy, repeats and round count;
- environmental controls and measurement instruments.

Only results with the same cell definition and taken in the same session may be
used to calculate an engine ratio. Results from another GPU, CPU, operating
system, driver or library stack form a new session, even if the benchmark
arguments are identical.

The families compare OpenNN with PyTorch.

## 2. Repository and results

Benchmark source, small manifests and this protocol are versioned. Datasets,
model weights, converted models, third-party runtimes, build trees, logs and raw
results are not.

Set `OPENNN_BENCH_DATA` to a directory outside the repository. When it is not
set, the scripts choose a user-local default. Preparation must place every
download and generated asset below that directory and must not reuse a personal
framework cache or a globally running model server.

Committed manifests are inputs, not generated results. They pin revisions,
hashes and deterministic subsets so another machine can retrieve the same
material. If a downloaded or converted file does not match its manifest, stop
before measurement. Never update a manifest merely to accept an unexplained
hash difference.

Raw artifacts are written outside the checkout, below
`../opennn-benchmark-results/` by default. Set `OPENNN_BENCH_RESULTS` to an absolute
path to override the result store for every runner. It is outside Git and is
never committed. A run that is provisional or fails a
validity gate is written below `../opennn-benchmark-results/scratch/`; it is retained for
diagnosis but is not a valid result.

Reports are local to each user: keep them in `benchmarks/reports/`, which Git
ignores, or next to the raw results. Every reported value must be traceable to
an unedited raw artifact, state its machine and protocol, and show the relevant
gates.

## 3. Prepare the environment

Record enough provenance to reproduce the binary and explain a performance
change:

- operating system and architecture;
- CPU and logical-core count;
- GPU name, compute capability and VRAM;
- GPU driver, CUDA and cuDNN versions;
- compiler, build type and relevant CMake options;
- Python and framework versions;
- BLAS, attention, convolution and recurrent backends actually selected;
- every non-default environment variable and engine flag.

Use release builds. Do not compare a debug or sanitizer build with an optimized
runtime. Build all engines for the actual target architecture and verify that
GPU workloads really offload to the GPU. A fallback to CPU, PTX JIT when native
kernels were required, or an unintended reference kernel invalidates the cell.

Each engine should use its best stable production configuration for the stated
workload. Optimizations such as CUDA Graphs, Flash Attention, full GPU offload,
`torch.compile`, memory layouts or optimized BLAS are allowed when they are
available to normal users, deterministic enough for the test and recorded in
the artifact. Do not deliberately handicap one engine to make internal options
look symmetrical.

Configuration is fixed before a session begins. Exploratory tuning is a
separate scratch session; the chosen stable configuration is then rerun from
the start. Never select per-engine options after inspecting the final result.

## 4. Prepare and verify inputs

Run preparation once for each standard family:

```bash
python benchmarks/prepare.py dense
python benchmarks/prepare.py cnn
python benchmarks/prepare.py transformer
python benchmarks/prepare.py lstm
```

Multiple families or `all` may be passed together. Preparation must be
deterministic for a fixed manifest, seed and arguments. Record resolved paths,
file sizes and hashes in the artifact, but never copy input data into the
repository.

Before timing, verify equality of work between engines:

- identical train/test split and sample count;
- identical input and target shapes;
- identical model topology and logical parameter count;
- identical sequence length, vocabulary and special-token handling for text;
- identical precision policy or an explicitly reported unavoidable asymmetry;
- identical number of warm-up and timed iterations;
- identical stopping and error policy.

The shape gate is mandatory. The quality gate is mandatory wherever the family
reports a comparable quality metric. A speed figure from a failed gate may be
useful for debugging, but it is not a valid comparison.

## 5. Define the timed region

Timing boundaries must answer the named metric and must be identical in meaning
for every engine.

For standard training and inference cells, exclude process startup, dataset
preparation and warm-up. Include the complete repeated operation performed by
the family driver. Synchronize asynchronous devices at both boundaries so the
reported duration measures completed GPU work rather than command submission.

## 6. Warm-up, rounds and ordering

Warm up every engine before collecting timed samples. Warm-up should exercise
the same shapes and important code paths as measurement, including compilation,
kernel selection, graph capture and allocator initialization where applicable.
Warm-up output is diagnostic and is not included in the timing summary.

The default comparison uses three independent rounds. Rotate engine order each
round so startup temperature, boost state and background drift are not assigned
systematically to one engine. Keep every raw launch and sample in the artifact;
the summary never replaces the raw observations.

For the standard families, `--rounds` controls process launches and `--epochs`
or `--repeats` controls work inside a launch. A reduced smoke test proves only
that the pipeline works; it is never a performance result.

Use the median as the primary central estimate and report minimum and maximum.
Where the family computes coefficient of variation, a CV above 3% invalidates
that cell. Do not hide instability by discarding an inconvenient sample unless
there is an independently recorded machine or runtime failure.

## 7. Control the machine

Benchmark on an otherwise idle machine. Disable or pause scheduled work,
updates, indexers, synchronization clients and other GPU applications where
practical. Connect portable systems to power and select a stable performance
profile. Record controls that cannot be enforced.

The runner samples CPU activity before, during and after launches. Activity
above `OPENNN_BENCH_BUSY_THRESHOLD` sends the result to `scratch/`. CPU cells
may be pinned to an appropriate physical-core set; the chosen cores and thread
count must be the same for every engine and recorded.

Keep thread count, affinity and wait policy controlled across CPU engines.
Frameworks can link different OpenMP runtimes whose default spin/sleep policies
are not equivalent, and a model may also involve more than one thread pool.
Record settings such as `GOMP_SPINCOUNT`, BLAS thread count and selected core
set. If profiling shows idle workers competing with the pool doing useful work,
resolve that runtime configuration before comparing.

For CUDA comparisons, lock graphics and memory clocks when the platform and
driver permit it. Select supported, sustainable values for the machine under
test, document them with the result and restore default clocks even after a
failure. An unlocked-clock run is diagnostic and belongs in `scratch/`.

Linux provides [`tools/gpu_clocks.sh`](tools/gpu_clocks.sh) to lock and restore
the clocks.

If a machine cannot satisfy a gate, complete only diagnostic runs. Do not
weaken a threshold after seeing a result and then treat that same run as valid.

## 8. Memory, power and energy

GPU memory is whole-device used memory sampled during the launch, minus a
baseline read immediately before it. This includes contexts and allocator
caches because that memory is unavailable to other work. Framework-private
allocator counters may be stored as labeled diagnostics, but they are not used
for cross-engine comparison.

On WDDM, where per-process reporting is incomplete, use total device memory
minus the per-launch baseline.

CPU memory is peak anonymous resident memory. File-backed pages are recorded
separately so a memory-mapped dataset is not charged as if it were a private
heap allocation.

GPU energy is board power integrated over the engine's timed window. Prefer
driver-timestamped NVML power samples. If only a slower averaged instrument is
available, identify it explicitly. Energy is reported only when the timed
window contains at least fifty 20 ms samples; otherwise it is `null` with an
explanation, never zero.

CPU energy uses a readable package-level RAPL counter where available. It is a
different measurement domain from GPU board energy and must be labeled as such.
An unavailable counter produces an unmeasured field, not an estimate.

## 9. Standard family procedure

For `dense`, `cnn`, `transformer` and `lstm`:

1. Prepare the family data and verify its recorded identity.
2. Build the OpenNN benchmark targets in Release mode.
3. Start a session by setting a stable `OPENNN_BENCH_SESSION` value.
4. Stabilize the machine and lock GPU clocks for CUDA runs.
5. Run the desired family, mode, device, precision and batch cell.
6. Confirm both engines completed and the shape and quality gates passed.
7. Inspect raw launches for errors, background activity and instability.
8. Restore clocks and retain the generated JSON artifact locally.

Example:

```bash
python benchmarks/run.py --family dense --mode train --device cuda \
  --precision bf16 --batch 8192 --rounds 3
```

An explicit comma-separated batch list produces a throughput curve. A value
such as `1024:OOM` doubles the batch until a normal out-of-memory response. A
crash or signal is not an OOM frontier and must be reported as a failure.

The runner records `failure_kind` as `oom`, `error`, `crash`, or null on success.
Only a confirmed allocation failure after a successful batch validates the
frontier. Other failures leave `max_batch` as an observation, not a limit;
without a successful batch it is null. Unknown frontiers go to scratch and
the runner returns status 3. Generic C++ exceptions emit ERROR, not OOM.

## 10. Result validity

A valid artifact requires all applicable conditions:

- clean Git tree and recorded source commit;
- complete provenance and input hashes;
- successful process exits and requested work completed;
- matching shape/work gates;
- passing quality or deterministic-output gates where defined;
- acceptable environmental readings and locked GPU clocks;
- stable repeated measurements;
- no unexpected fallback, throttle, OOM or early termination.

An old passing flag is not a substitute for its underlying evidence. Missing
or nonfinite quality values mean that quality was not measured. Recalculate
variation from every retained launch, verify the input hashes and check the
actual workload before reusing an older result.

The raw artifact is the measurement record. A table or chart must be
derivable from raw samples and must name the hardware, operating system,
precision, workload, session and validity state. Ratios compare engines within
the same session. Never relabel a result as coming from another machine, and
never promote a scratch result to a primary number.

When a gate fails, retain the observation and its reason in `scratch/`, correct
the cause and rerun the complete affected cell. Do not edit generated JSON by
hand, move it into the valid directory, or average it with valid samples.

## 11. Final checklist

Before accepting a benchmark session, verify:

- [ ] source tree is clean and the intended commit is recorded;
- [ ] all inputs and external tools match committed manifests;
- [ ] builds are Release and use the intended device/backend;
- [ ] engine work, shapes, precision and stopping rules match;
- [ ] warm-up is excluded and asynchronous work is synchronized;
- [ ] engine order rotates and all raw repetitions are retained;
- [ ] the machine stayed within CPU, GPU, temperature and throttle gates;
- [ ] clocks were locked and then restored;
- [ ] memory and energy use comparable, labeled instruments;
- [ ] quality, determinism and variation gates passed;
- [ ] valid and diagnostic results landed in the correct local directories;
- [ ] any report traces back to the raw artifact and identifies the actual
      machine.

## 12. Application startup and deployment

These experiments compare OpenNN C++ with the **PyTorch Python API**. They use
small applications, not the larger training-throughput models. Native source is
`families/application.cpp`; the Python counterpart is `families/application.py`.
Each native model is a separate `<model>_application_opennn` CMake target, so the
deployment measurement does not charge every application for all four models.

### Build and prepare

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

### Run

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
Output defaults to a new directory in `../opennn-benchmark-results/scratch/`.
An explicit `--out` must also be a new directory below that location.
Raw JSON, CSV, readable Markdown tables, loader traces and file inventories are
kept together. Failures remain visible and make the command fail.

### Workloads and timing

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
No model file is loaded.

CPU uses FP32. CUDA uses FP32 and BF16, each with saved and empty application
tuning caches. Filesystem caches stay warm. A separate process warms saved
caches; empty-cache samples receive fresh directories. Two threads and a fixed
0.2-second gap outside every timed region are used. Optional `--affinity` sets
the same Linux CPU IDs for parent and children. Case order rotates by a fixed
seed, and all repetitions are retained. The full matrix has 44 configurations;
three rounds of five samples produce 660 timed processes.

The runner records medians, ranges and CV. It does not enforce GPU clocks or
host idleness. These observations remain diagnostic; CV above 3% and unlocked
GPU clocks fail the validity rules. A smoke timing is never a speed claim.

### Deployment and dependencies

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

## 13. Prediction quality

`run.py --family quality` trains OpenNN C++ and PyTorch Python, then scores their
held-out predictions in one neutral Python scorer. This is a separate experiment
from the short throughput runs. It does not infer accuracy from shape
agreement, untrained inference, or equal training loss.

| Model | Data | Primary metric |
|---|---|---|
| Dense | HIGGS | test accuracy, percent; higher is better |
| LSTM | Beijing PM2.5 | test RMSE in original target units; lower is better |
| CNN | ImageNet subset, ResNet-50 v1.5 | test top-1 accuracy, percent; higher is better |
| Transformer | WMT14 English–German, base encoder-decoder | generated-translation SacreBLEU; higher is better |

### Prepare

Use a separate Python environment with compatible torch/torchvision packages,
NumPy, pandas, Pillow and SacreBLEU. `manifests/quality-requirements.txt` lists the
versions/constraints. Install the CPU or CUDA wheel pair appropriate for the
machine. Do not add these packages to the application deployment environments.

Prepare the original datasets with the existing `prepare.py` commands, then:

```bash
python benchmarks/prepare.py quality --model all \
  --data-root "$OPENNN_BENCH_DATA" --out "$OPENNN_BENCH_DATA/quality"
```

The destination must be new and outside the checkout. Each model receives a
manifest and shared binary tensors. Input identity is checked before training.

- Dense retains HIGGS's prepared train/test files and training-fitted scaling.
- LSTM splits chronology 80/20 before forming windows, fits scaling on training
  rows only and scores in original target units. The prepared source retains the
  existing `prepare.py` missing-value interpolation; this is not a new raw-data
  cleaning study. The model uses a plain LSTM and head on prepared tensors.
- CNN splits each class 80/20 with a fixed seed, rejects identical image content
  across splits, and performs one shared RGB/bilinear resize. Both models divide
  the same uint8 inputs by 255. All classes remain present; no augmentation is used.
- Transformer deduplicates normalized pairs before a fixed 90/10 split. One
  vocabulary is fitted on training text and used by both engines. PAD/BOS/EOS
  handling, shifted decoder inputs and reference text are shared. Training uses
  causal decoder attention, dropout zero and cross-entropy averaged over non-PAD
  tokens. Scoring uses greedy autoregressive predictions through EOS or the
  fixed length, not teacher-forced token guesses. The SacreBLEU signature is saved.

These splits and preprocessing rules differ from speed-only datasets that were
used entirely for timing.

### Build and run

Enable `OpenNN_BUILD_BENCHMARKS=ON` in an external Release build, then build the
`quality_opennn` target. It supports CPU FP32 and CUDA FP32/BF16. Both engines
receive the same prepared tensors, batch size, epoch budget, learning rate and
seed list. Choose the budget before inspecting test scores.

```bash
python benchmarks/run.py --family quality --model dense \
  --manifest "$OPENNN_BENCH_DATA/quality" --epochs 100 --batch 256 \
  --seeds 42,43,44,45,46 --device cuda --precision fp32 \
  --opennn-binary /path/to/release/bin/quality_opennn
```

Repeat with `--model lstm`, `cnn` and `transformer`, choosing an adequate budget
for each task. `--model all` applies one explicitly supplied budget to all four.
`--python` selects a different PyTorch environment; by default it is the runner's
interpreter. `--threads` applies to both engines. Long runs have a configurable
`--timeout` per process (default 24 hours).
Use `--opennn-library-path` and `--pytorch-library-path` when Linux/WSL needs
different native library search paths for the two drivers. Their effective
values are recorded, so a CUDA 12 OpenNN build need not load the CUDA 13
Python wheel's cuDNN library.

Adam uses beta1 .9, beta2 .999 and float32 epsilon in both engines. The default
learning rate is .001, or .0001 for Transformer. There is no regularization,
gradient clipping, early stopping, best-checkpoint restoration, training warm-up
or CUDA graph replay. The prepared order is fixed and both engines train every
tail sample. Use a batch that avoids singleton CNN training batches because
training batch normalization needs more than one value per channel.

Initial weights are independently seeded, **not identical tensors**. Dense
uses Glorot initialization, LSTM follows each implementation's PyTorch-style
initialization, and CNN/Transformer retain their engine initialization policies.
Backend rounding and mixed-precision policies can also differ. These are
framework training-outcome comparisons, not a claim of identical optimization
trajectories. Review those differences before explaining a score gap.

### Results and smoke checks

Every run writes under `../opennn-benchmark-results/scratch/`: raw stdout/stderr,
per-epoch training histories, binary predictions, driver metadata and hashes,
plus an aggregate JSON, CSV and a Markdown table. Failed runs are retained and
make the command return nonzero. The scorer reports means and sample standard
deviations; one seed has no estimated standard deviation. Metrics are compared
within a model, never averaged across different tasks.

The quality gate is deliberately **unassessed**. Similar averages, overlapping
spread or successful execution alone do not establish equivalence. Review
achieved task quality and declare acceptable differences before a parity claim.

For a quick end-to-end check of all model paths, including a test tail batch:

```bash
python benchmarks/prepare.py quality --smoke --out "$OPENNN_BENCH_DATA/quality-smoke"
python benchmarks/run.py --family quality --manifest "$OPENNN_BENCH_DATA/quality-smoke" \
  --epochs 1 --batch 2 --seeds 42,43 --device cpu \
  --opennn-binary /path/to/cpu/bin/quality_opennn
```

Smoke fixtures use small topologies and synthetic data. Their numbers are
functional checks, not benchmark results.
