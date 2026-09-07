# Benchmark protocol

This document is the measurement contract for every benchmark family in this
directory. It is intentionally independent of a particular computer: results
from different machines are valid reproductions, but they are separate result
sets and must never be combined into a direct engine comparison.

Changing a rule that affects the workload, engine configuration, timed region,
instrumentation or validity gates creates a new protocol revision. Results
affected by such a change must be rerun; a note beside old numbers is not a
substitute for measuring them again.

## 1. What is being compared

A benchmark cell is defined by all of the following:

- source revision and whether the working tree is clean;
- benchmark family and mode;
- model topology, parameter count and numerical precision;
- dataset or model-weight revision and content hashes;
- device, backend, libraries, driver and runtime versions;
- batch size or prompt/output length;
- engine options, warm-up policy, repeats and round count;
- environmental controls and measurement instruments.

Only results with the same cell definition and taken in the same session may be
used to calculate an engine ratio. Results from another GPU, CPU, operating
system, driver or library stack form a new session, even if the benchmark
arguments are identical.

The standard families compare OpenNN and PyTorch. Qwen has two complementary
tracks:

- `core` compares the OpenNN CUDA engine with `llama-bench`, excluding
  tokenization, sampling, transport and text output;
- `runtime` compares OpenNN's `ChatSession`, `llama-server` and Ollama through
  their real user-facing paths, including tokenization, sampling and transport.

Core and runtime figures answer different questions and must not be placed in
the same performance column.

## 2. Repository, results and reports

Benchmark source, small manifests, the protocol and reviewed reports are
versioned. Datasets, model weights, converted models, third-party runtimes,
build trees, logs and raw results are not.

Set `OPENNN_BENCH_DATA` to a directory outside the repository. When it is not
set, the scripts choose a user-local default. Preparation must place every
download and generated asset below that directory and must not reuse a personal
framework cache or a globally running model server.

Committed manifests are inputs, not generated results. They pin revisions,
hashes and deterministic subsets so another machine can retrieve the same
material. If a downloaded or converted file does not match its manifest, stop
before measurement. Never update a manifest merely to accept an unexplained
hash difference.

Raw artifacts are written below `benchmarks/results/`. That directory is
ignored by Git and is never committed. A run that is provisional or fails a
validity gate is written below `benchmarks/results/scratch/`; it is retained for
diagnosis but is not a publishable result.

Reviewed project results live in `benchmarks/reports/`. These committed reports
are the source of truth for benchmarks officially performed by the project.
Every reported value must be traceable to an unedited raw artifact, state its
machine and protocol, show relevant gates, and distinguish measured facts from
analysis. Generating a local artifact does not automatically make it an
official report.

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

For footprint startup measurements, the process lifetime is itself the work;
do not amortize startup by placing several questions in one process.

For language-model inference, report the phases separately:

- **load/ready time:** runtime startup, model initialization and transfer until
  the engine can accept the measured request;
- **TTFT:** elapsed client time from request submission to the first generated
  token;
- **prefill throughput:** prompt tokens processed per second before generation;
- **decode throughput:** generated tokens after the first, divided by the time
  between the first and final generated token;
- **end-to-end throughput:** the complete request, including the components
  belonging to the selected core or runtime track.

Do not describe prefill throughput as decode throughput or combine the two into
one unlabeled tokens-per-second number. Prefill processes a prompt in parallel;
decode is autoregressive and has a different compute and memory profile.

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
or `--repeats` controls work inside a launch. For Qwen, use three rounds and
five timed repetitions per cell after warm-up. A reduced smoke test proves only
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
resolve that runtime configuration before publishing the comparison.

For CUDA comparisons, lock graphics and memory clocks when the platform and
driver permit it. Select supported, sustainable values for the machine under
test, document them with the result and restore default clocks even after a
failure. An unlocked-clock run is diagnostic and belongs in `scratch/`.

Linux provides [`tools/gpu_clocks.sh`](tools/gpu_clocks.sh). The Qwen Windows
wrapper performs its own lock and always attempts `-rgc` and `-rmc` in a
`finally` block. Clock values embedded in a platform wrapper are defaults for
that reference setup, not universal requirements; another GPU needs values it
supports and a separately identified result set.

Before a Qwen round, the strict environmental gate requires:

- CPU busy fraction at or below 3%;
- GPU utilization at or below 2%;
- GPU temperature at or below 45 °C;
- launch-baseline VRAM drift no greater than 64 MiB;
- successfully locked clocks;
- no reported thermal or power throttling.

If a machine cannot satisfy a gate, complete only diagnostic runs. Do not
weaken a threshold after seeing a result and then treat that same run as valid.

## 8. Memory, power and energy

GPU memory is whole-device used memory sampled during the launch, minus a
baseline read immediately before it. This includes contexts and allocator
caches because that memory is unavailable to other work. Framework-private
allocator counters may be stored as labeled diagnostics, but they are not used
for cross-engine comparison.

On WDDM, where per-process reporting is incomplete, use total device memory
minus the per-launch baseline. For Qwen, steady memory is the median over the
defined steady window after warm-up; peak memory is the maximum observed in the
measured window. State the window and metric in the artifact.

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

## 10. Qwen3 procedure

On the supported Windows path, run:

```powershell
.\benchmarks\tools\qwen_benchmark.ps1 prepare
.\benchmarks\tools\qwen_benchmark.ps1 build
.\benchmarks\tools\qwen_benchmark.ps1 smoke
.\benchmarks\tools\qwen_benchmark.ps1 run
```

Use `unlock` only to restore clocks explicitly after an interrupted external
session; normal `smoke` and `run` actions restore them automatically.

Preparation reads `manifests/qwen_manifest.json`, downloads pinned OpenNN and
canonical Qwen assets, builds one BF16 GGUF with the pinned converter, prepares
an isolated Ollama store and validates weights tensor by tensor. OpenNN and
GGUF logical tensors must match the canonical safetensors as BF16, allowing
only exact BF16-to-F32 expansion for tensors the GGUF format requires in F32.
Hash, shape, value, dtype, alias or layout mismatches stop the benchmark.

The default workload is batch 1, greedy generation, prompt lengths 128, 512,
2048 and 8192, and 256 output tokens. The primary cell is 2048+256. Fixtures
are generated deterministically and must produce exactly the requested token
IDs after applying the Qwen chat template in no-think mode. The token count and
ID hash must agree for all engines.

For the core track:

- keep weights and KV cache resident;
- exclude tokenization, sampling, HTTP and output formatting;
- use full GPU offload and the recorded attention/KV precision;
- run the same deterministic token sequence in OpenNN and `llama-bench`.

For the runtime track:

- use OpenNN `ChatSession`, streaming `llama-server` and streaming Ollama;
- use isolated free ports and stores;
- start only harness-owned processes and stop only their recorded PIDs;
- request greedy generation with the same prompt and output limit;
- reject early EOS instead of silently replacing the workload.

Record output hashes, within-engine repeat stability, common greedy prefix and
first cross-engine divergence. Cross-engine greedy divergence does not by
itself invalidate timing after input, weights, shapes and within-engine
determinism have passed, but it must remain visible in the report.

The full Qwen run emits immutable JSON, flat CSV, Markdown and SVG files under
the selected result directory. A cell is diagnostic if it encounters OOM,
partial CPU/GPU execution, early EOS, invalid weights, environmental failure,
unexpected backend fallback, failed clock control or excessive variation.
Other independent cells may remain valid.

## 11. Result validity and publication

A publishable artifact requires all applicable conditions:

- clean Git tree and recorded source commit;
- complete provenance and input hashes;
- successful process exits and requested work completed;
- matching shape/work gates;
- passing quality or deterministic-output gates where defined;
- acceptable environmental readings and locked GPU clocks;
- stable repeated measurements;
- no unexpected fallback, throttle, OOM or early termination.

The raw artifact is the measurement record; the reviewed document in
`reports/` is the project-level source of truth. A table or chart must be
derivable from raw samples and must name the hardware, operating system,
precision, workload, session and validity state. Ratios compare engines within
the same session. Never relabel a result as coming from another machine, and
never promote a scratch result to a primary number.

When a gate fails, retain the observation and its reason in `scratch/`, correct
the cause and rerun the complete affected cell. Do not edit generated JSON by
hand, move it into the valid directory, or average it with valid samples.

## 12. Final checklist

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
- [ ] any official report traces back to the raw artifact and identifies the
      actual machine.
