# OpenNN benchmarks

OpenNN against reference runtimes on model families, measuring throughput, peak
memory and energy **from the same execution**.

There are no numbers in this file. Results are generated locally under
`results/`, with each artifact naming the commit, machine and session it came
from. That directory is ignored completely by Git. Reviewed official results
are versioned under [`reports/`](reports/). This README is the quick entry
point; [`PROTOCOL.md`](PROTOCOL.md) contains the complete, machine-neutral
measurement contract. Changing a measurement rule means rerunning the affected
cells.

## Running one

```bash
python benchmarks/prepare.py dense                       # once per family
python benchmarks/run.py --family dense --mode train --batch 8192
```

That prints throughput, peak memory and energy for each engine, and writes an
artifact. `--family` is `dense`, `cnn`, `transformer` or `lstm`; `--mode` is
`train` or `infer`.

Capacity sweeps require an identified allocation failure after a successful
batch. Other failures leave capacity unknown and send the run to `scratch/`.
In that case `max_batch` is only the largest successful batch observed, or
`null` if none succeeded, not a confirmed memory limit.

`footprint` is the exception: it takes no batch and no dataset, and its modes
are `memory`, `startup` and `export` — one process each, since a startup cost
is already paid by anything sharing a process with it.

Footprint memory/startup use CPU FP32 in both engines. Its baseline is current
Linux RSS or Windows working set (MiB), not private commit or a peak. Missing
readings are `null`. Each launch has a typed `footprint` metrics object.
Internal prediction times remain diagnostic: OpenNN starts at `main`, whereas
PyTorch includes its import but excludes interpreter startup. Compare the
parent's `process_lifetime_seconds` for a common external boundary: process
creation through exit and output collection, including teardown. This is not
time-to-first-prediction. `wall_seconds` remains its rounded legacy alias.

Qwen is the other deliberate exception. It compares the current OpenNN
Qwen3-4B implementation with llama.cpp at the engine level and with
llama.cpp/Ollama as complete runtimes. On Windows, use the pinned wrapper:

```powershell
.\benchmarks\tools\qwen_benchmark.ps1 prepare
.\benchmarks\tools\qwen_benchmark.ps1 build
.\benchmarks\tools\qwen_benchmark.ps1 smoke
.\benchmarks\tools\qwen_benchmark.ps1 run
```

The Qwen runner fixes BF16 logical weights, greedy generation, batch 1, 256
generated tokens and prompt lengths 128/512/2048/8192. Its primary cell is
2048+256. Models, Python, llama.cpp and Ollama all live below
`OPENNN_BENCH_DATA`; the wrapper does not use a globally installed Ollama.
Qwen uses the same `prepare.py` / `run.py --family qwen` entry points, shared
provenance, monitoring and result-directory helpers as the other families.
The PowerShell wrapper provides Windows setup and clock restoration; it does
not define a reference machine. Each result records the detected GPU, CPU and
operating system. Historical measurements remain in `reports/` with their
original hardware identity.

Before a measured run, set `OPENNN_BENCH_SM_CLOCK_MHZ` and
`OPENNN_BENCH_MEMORY_CLOCK_MHZ` to supported, sustainable integer MHz values
for the GPU under test. There are no default clock targets: missing targets or
failed locks make the run diagnostic-only in `results/scratch/`. The current
instrumentation uses NVIDIA device 0; use a single-GPU setup for comparisons.
Builds target the detected CUDA architecture (`native`); override with
`OPENNN_CUDA_ARCHITECTURES` when needed. Non-standard cuDNN installations use
`OPENNN_CUDNN_INCLUDE_DIR` and `OPENNN_CUDNN_LIBRARY`, as in the verification
wrappers. The pinned portable Ollama download is Windows x64-specific; this
setup wrapper does not provision other operating systems.

Greedy output is bit-reproducible across processes only if every process runs
the same cuBLASLt kernel for every shape, and the tuner in
`opennn/core/device_backend.cpp` picks kernels by timing them. OpenNN persists
each winner below `%TEMP%\opennn-lt-plans\<card>-sm<cc>-cublaslt<version>`
(`OPENNN_LT_PLAN_CACHE_DIR` moves it, `OPENNN_LT_PLAN_CACHE=0` disables it), so
the first process on a card tunes and every later one loads. Warm that cache
with one throwaway launch before a timed round; a cold first process may pick a
different kernel than the rest. `OPENNN_LT_DETERMINISTIC=1` skips the tuner
altogether and takes the heuristic's first candidate. Record this setting:
it changes the algorithm-selection policy and can affect performance.
Incompatible plan-cache records require re-tuning before measurement.

`--batch` is the only sweep axis:

| | |
|---|---|
| `--batch 8192` | one rung: the speed cell |
| `--batch 1024,8192,65536` | several rungs: the throughput curve |
| `--batch 1024:OOM` | double until a launch fails: the capacity frontier |

## The families

| family | model | data | why this one |
|---|---|---|---|
| `dense` | 28 → 1024 × 2 → 1 classifier | HIGGS | the shape a tabular workload actually has |
| `cnn` | ResNet-50 v1.5 | ImageNet subset, 1000 classes × 50 | the citable convolution benchmark |
| `transformer` | d512 · h8 · ff2048 · 6L | WMT14 English-German | the *Attention Is All You Need* base model, on its own corpus |
| `lstm` | LSTM(15→128) → Linear | Beijing PM2.5, hourly | both engines reach the same cuDNN kernel here |
| `footprint` | — | — | what a framework costs *before* it runs anything |
| `qwen` | Qwen3-4B BF16 | pinned Hugging Face weights | engine and end-user runtime comparison |

Each family keeps its C++ and Python implementation in
[`families/`](families/). The standard families expose training and inference
modes; `footprint` and Qwen use the specialized modes described above.

## What every run checks before it reports

A throughput number means nothing if the two engines were not doing the same
work, so two gates run first and the artifact records both.

**The shape gate** compares what each engine reports about the work: sample
count, sequence length, vocabulary, and **parameter count**. These must agree
before throughput can be compared.

**The quality gate** compares test accuracy across engines at each batch
wherever a driver reports one — today that is dense training only; the other
families are held to the shape gate. A speed win bought by computing something
different is not a speed win.

Qwen additionally validates the OpenNN and GGUF tensors against the pinned
canonical weights, verifies exact prompt-token counts and records whether each
engine generated the requested number of tokens. Its `core` track excludes
tokenization and sampling; its `runtime` track includes the complete serving
path.

**The attention-backend gate** applies to OpenNN's `runtime` track. The
attention layer counts which backend each launch took while the profiler is
on, and the driver turns the profiler on for its one unreported warm request
only. The result carries them as `attention_backends` — `prefill_sdpa`,
`prefill_fallback`, `sdpa_batched`, `gemm`, `kernel`, `decode_split` — and a
run whose prefill did not go entirely through the cuDNN SDPA graph is marked
invalid: a number from the materialized or generic path is not the number the
engine is meant to publish. Decode launches are CUDA-graph replays, which the
counters never see, so `decode_split` only reflects the capture.

## Reading a result

Energy is reported only when the timed window held a second of the driver's
20 ms power samples — a short run says so rather than reporting `0.0000 Wh`,
which is a claim and not a measurement. Peak memory is whole-device, minus the
idle reading;
`torch.cuda.max_memory_allocated()` never appears, because it excludes the CUDA
context and cached blocks and so flatters PyTorch by construction.

A dirty tree writes to `results/scratch/`, never to the valid-results area.
That is enforced in code. Neither location is committed.

OpenNN Qwen results also include `inference_memory`: reserved KV capacity and
bytes, the decode arena **view** size (not the complete shared prefill arena),
and graph workspace bytes. These diagnostics are not substitutes for measured
device memory. On Windows, process launches record sampled private commit as
`peak_private_mib`, separately from VRAM and working set. Raw process-memory,
GPU telemetry and timestamped power samples accompany these launches. Windows
uses `nvml.dll` when available; the labeled `nvidia-smi` fallback remains in
place when NVML is unavailable.

The OpenNN runtime currently computes `output_token_hash` by retokenizing the
decoded response. It is not a trace of the original sampler IDs; strict
token-by-token optimization acceptance needs that additional validation.

## Files

| | |
|---|---|
| [`run.py`](run.py) | common benchmark runner and Qwen dispatcher |
| [`prepare.py`](prepare.py) | dataset, model and external-runtime preparation by family |
| [`families/`](families/) | C++ and Python implementations for each benchmark family |
| [`PROTOCOL.md`](PROTOCOL.md) | detailed, machine-neutral measurement contract |
| [`reports/`](reports/) | reviewed, versioned source of truth for official results |
| [`tools/common.py`](tools/common.py) | provenance, binaries, sampling and metrics |
| [`tools/gpu_clocks.sh`](tools/gpu_clocks.sh) | lock the GPU clock on Linux |
| [`tools/qwen_benchmark.ps1`](tools/qwen_benchmark.ps1) | prepare, build, smoke-test and run Qwen on Windows |
| [`tools/qwen_support.py`](tools/qwen_support.py) | parsing and aggregation helpers used by Qwen |
| [`tools/validate_qwen.py`](tools/validate_qwen.py) | tensor-by-tensor OpenNN/GGUF weight gate |
| [`tools/llama-bench-fixed-context.patch`](tools/llama-bench-fixed-context.patch) | fixed-context instrumentation for the pinned llama-bench |
| [`manifests/imagenet_subset.manifest`](manifests/imagenet_subset.manifest) | exact, hashed CNN image subset |
| [`manifests/qwen_manifest.json`](manifests/qwen_manifest.json) | Qwen revisions, asset hashes and protocol defaults |
| [`CMakeLists.txt`](CMakeLists.txt) | builds one `<family>_opennn` per family |
| `results/` | generated raw local artifacts; ignored completely by Git |

Datasets, model weights and external runtimes never enter the repository. The
committed manifests pin the exact CNN image subset and every Qwen asset needed
to reproduce the comparison.
