# CPU HIGGS dense inference: OpenNN vs PyTorch vs TensorFlow

This benchmark measures repeated CPU inference passes for the same dense binary
classifier shape on the prepared UCI HIGGS test split.

**Status:** full 500k-row publication test split, native Linux, i9-12900K
(8 P-core threads), MKL-linked OpenNN. Last updated 2026-08-10, commit
52e21e15d. Artifact: `results/cpu-dense-higgs-infer-20260810T150333Z.json`.

## Result

Higher is better.

| Framework | Inference speed (samples/s) | Median pass (s) |
|---|---:|---:|
| OpenNN | **416,825** | **1.199** |
| PyTorch | 340,279 | 1.469 |
| TensorFlow | 356,244 | 1.403 |

OpenNN delivered **1.23x PyTorch speed** and **1.17x TensorFlow speed** on this
CPU inference run (median of 5 runs, 10 passes each), using the MKL-linked
OpenNN binary. A smaller batch-256 probe
was overhead-sensitive and put PyTorch ahead; the current row uses batch 1024 and
five timed runs.

## Re-measured with every engine at its best (2026-08-20)

*Laptop, WSL2, i7-12700H, MKL-linked OpenNN, full 500k-row test split, batch
1024, 10 passes per run, 12 threads, single runs (about 10% drift).*

| Engine | Best setting | Inference (samples/s) | Also measured |
|---|---|---:|---|
| TensorFlow | XLA | **197,852** | 153,108 with XLA off |
| PyTorch | eager, `inference_mode` | **148,298** | 143,809 on a repeat run |
| OpenNN (MKL) | - | **97,752** | median pass 5.11 s |

**At best settings OpenNN infers at 0.49x TensorFlow and 0.66x PyTorch here**,
again the opposite of the table above.

### How much of this machine to believe

The single runs above were taken on a laptop that turned out to swing far more
than the 10% they were reported with: the same binary, the same arguments and
the same thread count measured 56,173 and 101,182 samples/s back to back, and a
standalone MKL sgemm at one shape measured 128, 233 and 359 GFLOP/s across three
runs. Only **alternated** pairs survive that, and those are stable:

| round | OpenNN | PyTorch | ratio |
|---|---:|---:|---:|
| 1 | 102,215 | 143,106 | 1.40x |
| 2 | 100,466 | 143,830 | 1.43x |

So take **PyTorch at 1.40-1.43x OpenNN on CPU inference** as the result, and
treat the single-run TensorFlow figures as indicative of a large gap rather than
as a measured ratio. Anything smaller than about 2x cannot be attributed on this
machine from single runs.

## Where OpenNN's CPU inference time goes

Profiled over 5,856 layer calls, and isolated by adding and removing hidden
layers:

* `op:combination_fwd` (the GEMM and its bias) is **96.5%** of the time;
  `op:activation_fwd` is 3.4%.
* One 1024x1024 hidden layer costs about **7.4 ms per batch of 1024**.
* The two thin layers (28->1024 and 1024->1) cost about **2.9 ms per batch
  between them** - for 3% of the arithmetic. MKL runs those same two shapes
  standalone in **0.21 ms**, so roughly 2.7 ms per batch, a quarter of the whole
  step, is spent around the thin GEMMs rather than inside them.
* Throughput rises with batch size - 28,977 at 1,024, 72,277 at 4,096, 84,722
  at 16,384 samples/s in one back-to-back sweep - which is the signature of a
  fixed per-call cost rather than of the arithmetic.

Three explanations were tested and rejected:

* **"MKL is not being used."** It is: an instrumented counter records 7,320
  calls into `cblas_sgemm` with zero refusals for a five-pass run, exactly three
  per batch. (`MKL_VERBOSE` prints nothing for these calls, which is misleading.)
* **"The OpenMP teams churn."** They do - `add_bias` opens a parallel region per
  layer per batch, and one run creates about 10,000 threads - but pinning the
  team (`OPENNN_OMP_DYNAMIC=0`) measured **2.5x slower** (39,812 against
  100,426), and `MKL_THREADING_LAYER=GNU` measured neutral.
* **"MKL's GEMM is slower than PyTorch's."** Not supported: alternated at
  1024x1024x1024, MKL measured 359/279/128 GFLOP/s against PyTorch's
  221/233/343. Same class.

What is left, and what the next attempt should attack, is the fixed cost of a
forward call outside the GEMM.

## The bias pass, and what fixing it was worth (2026-08-20)

Instrumenting the dense forward by part found the cost, and it was not the
GEMM:

| scope | ms per call | calls per batch |
|---|---:|---:|
| `cpu:sgemm_wide` (1024 cube) | 5.634 | 1 |
| **`cpu:add_bias`** | **1.361** | **3** |
| `cpu:sgemm_thin_k` (28->1024) | 0.180 | 1 |
| `cpu:sgemm_thin_n` (1024->1) | 0.056 | 1 |
| `op:activation_fwd` | 0.128 | 3 |

Adding a bias to a 4 MB output is one pass over memory and should cost a
fraction of a millisecond; at 1.36 ms per call it was 40% of the forward, more
than every GEMM combined. The cause was the OpenMP region the pass opened per
layer per batch - about ten thousand thread creations in one inference run.
Hoisting the fused ReLU out of the inner loop changed nothing (97,672 against a
96,996 baseline), which ruled out vectorisation and left the threading. Three
ways of spreading the rows, measured in one binary:

| bias parallelism | inference (samples/s) | `add_bias` ms/call |
|---|---:|---:|
| OpenMP region per call (what it did) | 96,589 | 1.550 |
| the library's persistent thread pool | 143,090 | 0.173 |
| **none at all** | **154,219** | **0.121** |

The work is memory bound and short enough that every way of spreading it costs
more than it saves, so the pass is serial now, with `OPENNN_BIAS_MODE=pool|omp`
kept for the A/B.

### Result

Alternated against PyTorch at its own best setting, three rounds:

| round | OpenNN | PyTorch | ratio |
|---|---:|---:|---:|
| 1 | 157,897 | 152,673 | **1.03x** |
| 2 | 153,276 | 150,506 | **1.02x** |
| 3 | 153,238 | 149,510 | **1.02x** |

**OpenNN is now ahead of PyTorch on CPU inference in every round**, where before
this change it ran at 0.65x - a 58% gain on the same machine, same build, same
protocol. Training gains too, 34,209 to 43,047 samples/s (+26%), with the
held-out metrics unchanged to every digit printed (accuracy 0.73236, ROC AUC
0.815341), which is what one expects from a change that only reorders how the
same additions are spread over cores.

TensorFlow with XLA remains ahead of both on this benchmark and is the next
target.

### What the harness was doing wrong

The table above came from a harness that did not let the other two engines run
their own fast paths:

* `tensorflow_higgs_cpu.py` pinned `@tf.function(jit_compile=False)` in both
  training and inference - XLA off - while the GPU family has always measured
  TensorFlow with XLA on. XLA is worth **+32% training / +29% inference** to it
  here, so the published rows measured TensorFlow below itself.
* `pytorch_higgs_cpu.py` had no `torch.compile` at all and ran inference under
  `no_grad` rather than `inference_mode`.
* `run_higgs_cpu.py` applied `OMP_PLACES=cores OMP_PROC_BIND=close` to OpenNN
  and PyTorch but **not** to TensorFlow, which on a hybrid CPU is not a neutral
  omission: unpinned threads land on efficiency cores.

All three are fixed. Every engine now runs its best configuration by default,
and `TF_PLAIN=1` / `PYTORCH_PLAIN=1` restore the old behaviour for an A/B.
One surprise worth recording: on this CPU `torch.compile` is a *pessimisation*
for this model - 29,449 samples/s against eager's 41,523, inductor's CPU codegen
losing to eager on a three-GEMM MLP - so compilation is opt-in (`PT_COMPILE=1`)
and eager is what the driver measures.

The i9 rows above were measured under the old harness and cannot be cited as
they stand; they need re-running on that machine with the corrected drivers.

## Setup

| Item | Value |
|---|---|
| Dataset | UCI HIGGS, prepared with `docs/benchmarks/throughput/higgs/prepare_higgs.py` |
| Test split used here | 20,000 rows |
| Layout | `feature_0,...,feature_27,label`, normalized from train-set statistics |
| Model | `28 -> 1024 -> 1024 -> 1` |
| Activation | ReLU hidden layers, sigmoid output |
| Batch | 1024 |
| Repetitions | 20 timed passes after warmup |
| Timed runs | 5 |
| Device | CPU only |
| Environment | WSL2, Intel Core i7-12700H host, CPU path |
| OpenNN CPU backend | MKL-linked binary (`libmkl_rt`) |
| Thread environment | `MKL_NUM_THREADS=20`, `OMP_NUM_THREADS=20`, dynamic threading disabled |

CPU joules are **not measured** in WSL on this machine; the result JSON includes
only a normalized inverse-throughput energy proxy.

## Artifact

Local result JSON:
`docs/benchmarks/results/cpu-dense-higgs-infer-20260704T-higgs100k-infer-b1024-mkl-threads20.json`

Runner:
[`higgs/run_higgs_cpu.py`](higgs/run_higgs_cpu.py)

OpenNN binary source:
[`higgs/opennn_higgs_cpu.cpp`](higgs/opennn_higgs_cpu.cpp)
