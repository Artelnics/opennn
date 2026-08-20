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
