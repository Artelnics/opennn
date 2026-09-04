# OpenNN against PyTorch: the results, and why

One document per family, each answering the same three questions: what was
measured, what the numbers are, and *why* the margin is what it is — with the
evidence for the why, so that a reader who distrusts a number can see what
would have to be wrong for it to be wrong.

These reviewed, versioned documents are the project's source of truth for its
official benchmark results. Raw generated artifacts remain local under
`../results/` and are never committed.

| document | cells |
|---|---|
| [`dense.md`](dense.md) | HIGGS classifier: CUDA and CPU, training and inference |
| [`cnn.md`](cnn.md) | ResNet-50 v1.5 on the ImageNet subset: CUDA training and inference |
| [`transformer.md`](transformer.md) | the *Attention Is All You Need* base model on WMT14: CUDA training and inference |
| [`lstm.md`](lstm.md) | LSTM forecasting on Beijing PM2.5: CUDA and CPU, training and inference |
| [`footprint.md`](footprint.md) | what each framework costs before it does any work |

The contract the numbers were taken under is [`../PROTOCOL.md`](../PROTOCOL.md);
this page summarises only what a reader needs to interpret the table.

## The results

| cell | batch | precision | OpenNN samples/s | PyTorch samples/s | OpenNN / PyTorch | peak memory MiB (OpenNN / PyTorch) | energy Wh (OpenNN / PyTorch) |
|---|---|---|---|---|---|---|---|
| `cuda-dense-train` | 8,192 | bf16 | 10,768,761 | 10,115,756 | **1.065×** | 492 / 632 | 0.1201 / 0.1518 |
| `cuda-dense-infer` | 8,192 | bf16 | 37,317,959 | 37,178,529 | **1.004×** | 368 / 412 | 0.1389 / 0.1860 |
| `cpu-dense-train` | 4,096 | fp32 | 70,025 | 55,177 | **1.27×** | 339 / 787 | 0.0871 / 0.0969 |
| `cpu-dense-infer` | 4,096 | fp32 | 220,962 | 170,916 | **1.29×** | 314 / 575 | 0.0984 / 0.1066 |
| `cuda-cnn-train` | 64 | bf16 | 1,650 | 1,406 | **1.17×** | 3,608 / 4,208 | 3.8627 / 4.2546 |
| `cuda-cnn-infer` | 64 | bf16 | 6,942 | 5,833 | **1.19×** | 718 / 868 | 2.4425 / 2.9421 |
| `cuda-transformer-infer` | 32 | bf16 | 5,302 | 4,707 | **1.13×** | 844 / 1,210 | 11.9895 / 14.9908 |
| `cuda-lstm-train` | 256 | bf16 | 297,615 | 99,631 | **2.99×** | 380 / 512 | 0.0560 / 0.1248 |
| `cuda-lstm-infer` | 256 | bf16 | 917,344 | 523,041 | **1.75×** | 294 / 442 | 0.0460 / 0.0636 |
| `cpu-lstm-train` | 256 | fp32 | 22,995 | 13,286 | **1.73×** | 242 / 592 | 0.0433 / 0.0608 |
| `cpu-lstm-infer` | 256 | fp32 | 80,566 | 69,329 | **1.16×** | 189 / 459 | 0.0215 / 0.0234 |

| footprint question | OpenNN | PyTorch | PyTorch / OpenNN |
|---|---|---|---|
| memory | 0.124 s, 117 MiB | 3.241 s, 449 MiB | **26.1×** the time, 3.8× the memory |
| startup | 0.568 s, 325 MiB | 1.926 s, 375 MiB | **3.4×** the time, 1.2× the memory |
| export | 0.448 s, 156 MiB | 1.928 s, 376 MiB | **4.3×** the time, 2.4× the memory |

## The state of this round

This is a first version, and the gaps in it are marked rather than papered
over. Anything reading *[pending the final measurement round]* is a number
that has not been taken yet, not a number that was inconvenient.

**One cell is missing from the table above.** `cuda-transformer-train` reads
1.130× on throughput, 1.45× on memory and 1.38× on energy across three
attempts, but every attempt tripped the runner's foreign-activity gate — the
editor session on the measuring machine put a busy second inside a timed
window, against a 3% threshold — so all three are filed under
`results/scratch/` and none is evidence. The transformer document reports the
figures with that warning attached. Eleven of the twelve cells are
evidence-grade.

**Kernel-level evidence is complete for five cells** — `cuda-dense-infer`,
`cuda-dense-train`, `cuda-lstm-infer`, `cuda-lstm-train` and
`cuda-cnn-infer`, both engines each. `cuda-cnn-train` has OpenNN's profile
only; the four CPU cells and the two transformer cells have none in this
round. Where a document explains a margin without a kernel table behind it,
it says so, and the explanation rests on the artifacts, the drivers and the
power decomposition instead.

**Two controlled comparisons the dense document refers to are not yet
re-measured** at this commit: the batch sweep and the
`OPENNN_SMALL_K_LINEAR=0` variant. The tile-selection A/B
(`OPENNN_LT_TILE_TOLERANCE=0` against the default), which is the one that
carries this round's energy result, *has* been measured and is quoted in
full.

Every row is the last run of that cell in session `2026-09-03-publish`, the
median of three rounds, each round launching both engines in alternating order
in fresh processes. All gates passed in every row: the tree was clean, the
machine was quiet before and after, the GPU clock was locked, both engines
reported the same shapes and parameter counts, and — where the family defines
one — the same accuracy within tolerance. Memory and energy are different
quantities on the two devices and are labelled as such.

## What "same work" means here

A cell compares one network definition, driven by two engines with the same
positional arguments and the same `key=value` output. Before any throughput is
reported, the runner checks:

- **shape** — sample count, sequence length, vocabulary, parameter count, as
  printed by each engine; a mismatch is not a comparison;
- **quality** — where a driver reports a test accuracy, which today is dense
  training only, both engines must agree within 2%; the other families are
  held to the shape gate, and each document says what that leaves unchecked;
- **whole batches only** — both engines drop the tail of the epoch;
- **warmup excluded** — training runs untimed epochs first (allocation, graph
  capture, autotuning, `torch.compile`: two for dense and LSTM, one for the
  transformer, two for OpenNN's CNN against one for PyTorch's) and inference
  an untimed pass, before the clock starts; the family documents give the
  exact counts.

Throughput is samples per second inside the engine's own timed window, which
begins after warmup and ends after a device synchronisation, so the clock stops
when the work is done and not when it was queued.

## Each engine at its best

An engine measured below its own ceiling makes the other look good for the
wrong reason, so both are configured the way their users would configure them:

| | GPU | CPU |
|---|---|---|
| OpenNN | captured CUDA graph (training, and CNN and transformer inference; the dense inference path runs three launches on device views, and the LSTM runs eager because the cuDNN RNN path refuses capture), device-resident split, MKL for CPU-side work | 16 threads on the P-cores, MKL, oneDNN for recurrent layers |
| PyTorch 2.13 | `torch.compile` in the mode measured best per cell (`reduce-overhead` for dense training, `max-autotune-no-cudagraphs` for dense inference, the transformer and CNN training, default for CNN inference), bf16 weights for inference, `channels_last`, TF32; eager for the LSTM, where compiling measured slower | eager (compiling measured slower), MKL, oneDNN for recurrent layers |

Every one of those choices was measured before it was adopted, and the driver
docstrings carry the numbers for every mode tried. `PT_COMPILE_MODE`,
`PT_INFER_CAST` and the `OPENNN_*` variables override them, so the choices stay
measurable rather than baked in.

## The machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti (`sm_120`), driver 610.43.02, 300 W limit, SM clock locked at 2,692 MHz, persistence mode on |
| CPU | Intel Core i7-14700F: 8 P-cores (16 threads) + 12 E-cores; CPU cells pinned to the P-cores, CUDA cells given the whole CPU; governor `performance`, turbo disabled (2.1 GHz) |
| RAM | 32 GB |
| OS | Ubuntu 24.04, Linux 7.0, native (not WSL) |
| CUDA / cuDNN | 13.3 / 9.25.1 (OpenNN); PyTorch's wheel bundles cuDNN 9.23.2 |
| PyTorch | 2.13.0+cu130, Python 3.12.3 |
| MKL | 2026.0.1 (OpenNN); PyTorch's wheel bundles its own |
| oneDNN | 3.11 (OpenNN, OpenMP runtime); PyTorch's wheel bundles 3.12 |
| OpenNN | commit `21ab64c08` (library built from `d3acd71b5`; the later commit changes only the PyTorch drivers), Release, LTO, GCC 13.3 |

The GPU clock is locked below its boost ceiling because a floating clock drifts
by more per session than the margins being measured (PROTOCOL §7). Turbo is
disabled on the CPU for the same reason. Both are recorded in every artifact.

Two cells have short timed windows and are read with that in mind:
`cuda-dense-train` finishes its three timed epochs in about 0.07 s and
`cuda-lstm-train` in about 0.5 s, so their energy column is unreliable (the
power sampler reads once a second and some launches record `--`) and their
round-to-round spread is wider than the other cells'. Their documents show the
spread, and the dense one a longer-window variant.

## How to read the "why"

Each family document attributes its margin with two kinds of evidence:

- **A profile of the published launch command.** GPU cells were traced with
  Nsight Systems (`nsys --trace=cuda --cuda-graph-trace=node`) and the trace
  reduced to the GPU-busy fraction inside the engine's timed window, the kernel
  launch rate, the gaps between kernels, and the kernels that took the time.
  CPU cells were sampled with `perf record` at 499 Hz and split by shared
  object and symbol inside the same window. A profiled launch is a separate run
  from the published one; both throughputs are quoted so the profiler's
  overhead is visible.
- **A controlled variant.** Where a single mechanism is claimed to explain a
  margin, the document shows the cell measured with that mechanism removed or
  swapped (`OPENNN_GEMM_MODE`, `PT_COMPILE_MODE`, `GOMP_SPINCOUNT`, …), so the
  attribution is a measurement rather than an inference from a profile.

Where the margin owes something to an asymmetry that is not the framework —
a different cuDNN or oneDNN build, a pre-decoded image cache against a JPEG
decoder — the document says so and, where it could be measured, says how much.

## Reproducing

```bash
sudo ./tools/gpu_clocks.sh lock 2700             # PROTOCOL §7; the artifact records the clock
export OPENNN_BENCH_SESSION=$(date +%F)-mine
python run.py --family dense --mode train --device cuda --batch 8192 --precision bf16 --epochs 3 --rounds 3
python run.py --family dense --mode infer --device cuda --batch 8192 --precision bf16 --repeats 5 --rounds 3
python run.py --family cnn   --mode train --device cuda --batch 128  --precision bf16 --epochs 2 --rounds 3
python run.py --family cnn   --mode infer --device cuda --batch 128  --precision bf16 --repeats 5 --rounds 3
python run.py --family transformer --mode train --device cuda --batch 32 --precision bf16 --epochs 2 --rounds 3
python run.py --family transformer --mode infer --device cuda --batch 32 --precision bf16 --repeats 5 --rounds 3
python run.py --family lstm  --mode train --device cuda --batch 256  --precision bf16 --epochs 3 --rounds 3
python run.py --family lstm  --mode infer --device cuda --batch 256  --precision bf16 --repeats 5 --rounds 3
python run.py --family dense --mode train --device cpu  --batch 4096 --precision fp32 --epochs 3 --rounds 3
python run.py --family dense --mode infer --device cpu  --batch 4096 --precision fp32 --repeats 5 --rounds 3
python run.py --family lstm  --mode train --device cpu  --batch 256  --precision fp32 --epochs 3 --rounds 3
python run.py --family lstm  --mode infer --device cpu  --batch 256  --precision fp32 --repeats 5 --rounds 3
python run.py --family footprint
```

Run them from a shell whose CPU affinity is the whole machine (`taskset -pc $$`
should list every CPU): the runner pins CPU cells itself, but CUDA launches
inherit the shell's mask, and a shell confined to the E-cores halves every CUDA
number on both engines without failing any gate.

The artifacts behind this page are on the reference machine under
`benchmarks/results/`, one JSON per cell, named by `run_id`; the documents
quote every launch, not only the median, so the spread is visible without them.
