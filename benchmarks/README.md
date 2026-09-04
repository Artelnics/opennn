# OpenNN benchmarks

OpenNN against reference runtimes on model families, measuring throughput, peak
memory and energy **from the same execution**.

There are no numbers in this file. Results are generated locally under
`results/`, with each artifact naming the commit, machine and session it came
from. That directory is ignored completely by Git. This README is the complete
usage and measurement contract; changing a measurement rule means rerunning
the affected cells.

## Running one

```bash
python benchmarks/prepare.py dense                       # once per family
python benchmarks/run.py --family dense --mode train --batch 8192
```

That prints throughput, peak memory and energy for each engine, and writes an
artifact. `--family` is `dense`, `cnn`, `transformer` or `lstm`; `--mode` is
`train` or `infer`.

`footprint` is the exception: it takes no batch and no dataset, and its modes
are `memory`, `startup` and `export` — one process each, since a startup cost
is already paid by anything sharing a process with it.

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
Current Qwen results are internal RTX 4080/Windows measurements and must not be
presented as RTX 5070 Ti results without rerunning the protocol on that card.

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
| `lstm` | LSTM(14→128) → Linear | Beijing PM2.5, hourly | both engines reach the same cuDNN kernel here |
| `footprint` | — | — | what a framework costs *before* it runs anything |
| `qwen` | Qwen3-4B BF16 | pinned Hugging Face weights | engine and end-user runtime comparison |

Each family keeps its C++ and Python implementation in
[`families/`](families/). The standard families expose training and inference
modes; `footprint` and Qwen use the specialized modes described above.

## What every run checks before it reports

A throughput number means nothing if the two engines were not doing the same
work, so two gates run first and the artifact records both.

**The shape gate** compares what each engine reports about the work: sample
count, sequence length, vocabulary, and **parameter count**. It exists because
it kept finding real problems — OpenNN's tokeniser produced 158-position
sequences against PyTorch's 128, and `nn.Transformer` carried 2,048 parameters
of final `LayerNorm` that OpenNN's had no counterpart for. Neither is visible
in a samples-per-second figure.

**The quality gate** compares accuracy across engines at each batch. A speed
win bought by computing something different is not a speed win.

Qwen additionally validates the OpenNN and GGUF tensors against the pinned
canonical weights, verifies exact prompt-token counts and records whether each
engine generated the requested number of tokens. Its `core` track excludes
tokenization and sampling; its `runtime` track includes the complete serving
path.

## Reading a result

Energy is reported only when the timed window was long enough to sample — a
short run says so rather than reporting `0.0000 Wh`, which is a claim and not a
measurement. Peak memory is whole-device, minus the idle reading;
`torch.cuda.max_memory_allocated()` never appears, because it excludes the CUDA
context and cached blocks and so flatters PyTorch by construction.

A dirty tree writes to `results/scratch/`, never to the valid-results area.
That is enforced in code. Neither location is committed.

## Files

| | |
|---|---|
| [`run.py`](run.py) | common benchmark runner and Qwen dispatcher |
| [`prepare.py`](prepare.py) | dataset, model and external-runtime preparation by family |
| [`families/`](families/) | C++ and Python implementations for each benchmark family |
| [`tools/common.py`](tools/common.py) | provenance, binaries, sampling and metrics |
| [`tools/gpu_clocks.sh`](tools/gpu_clocks.sh) | lock the GPU clock on Linux |
| [`tools/qwen_benchmark.ps1`](tools/qwen_benchmark.ps1) | prepare, build, smoke-test and run Qwen on Windows |
| [`tools/qwen_support.py`](tools/qwen_support.py) | parsing and aggregation helpers used by Qwen |
| [`tools/validate_qwen.py`](tools/validate_qwen.py) | tensor-by-tensor OpenNN/GGUF weight gate |
| [`tools/llama-bench-fixed-context.patch`](tools/llama-bench-fixed-context.patch) | fixed-context instrumentation for the pinned llama-bench |
| [`manifests/imagenet_subset.manifest`](manifests/imagenet_subset.manifest) | exact, hashed CNN image subset |
| [`manifests/qwen_manifest.json`](manifests/qwen_manifest.json) | Qwen revisions, asset hashes and protocol defaults |
| [`CMakeLists.txt`](CMakeLists.txt) | builds one `<family>_opennn` per family |
| `results/` | generated local artifacts; ignored completely by Git |

Datasets, model weights and external runtimes never enter the repository. The
committed manifests pin the exact CNN image subset and every Qwen asset needed
to reproduce the comparison.
