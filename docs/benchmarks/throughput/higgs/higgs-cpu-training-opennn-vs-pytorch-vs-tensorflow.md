# CPU HIGGS dense training: OpenNN vs PyTorch vs TensorFlow

This benchmark measures one CPU training epoch for the same dense binary
classifier on a prepared UCI HIGGS subset.

**Status:** full 10.5M-row publication split, native Linux, i9-12900K
(8 P-core threads, `OMP_PLACES=cores OMP_PROC_BIND=close`), MKL-linked OpenNN.
Last updated 2026-08-10, commit 52e21e15d. Artifact:
`results/cpu-dense-higgs-train-20260810T145227Z.json`.

## Result

Higher is better.

| Framework | Training speed (samples/s) | Median epoch (s) | Accuracy | Log loss | ROC AUC |
|---|---:|---:|---:|---:|---:|
| OpenNN | **107,121** | **98.02** | 0.7715 | 0.4677 | 0.8565 |
| PyTorch | 99,923 | 105.08 | 0.7695 | 0.4714 | 0.8539 |
| TensorFlow | 102,040 | 102.90 | 0.7720 | 0.4674 | 0.8566 |

OpenNN trained at **1.07x PyTorch speed** and **1.05x TensorFlow speed** on
this full-split CPU run, using the MKL-linked OpenNN binary, with held-out
quality in the same band as the best engine.

## Re-measured with every engine at its best (2026-08-20)

*Laptop, WSL2, i7-12700H (6 P-cores / 20 threads), MKL-linked OpenNN, 1M-row
training subset, batch 1024, one epoch, single runs on a machine that drifts
about 10%. Absolute numbers are not comparable with the i9 table above - the
machine, the OS layer and the split all differ - but the engines here ran on the
same machine, the same data and the same protocol as each other.*

| Engine | Best setting | Training (samples/s) | Also measured |
|---|---|---:|---|
| TensorFlow | XLA, 20 threads | **68,752** | 66,461 at 12 threads; 50,460 with XLA off |
| PyTorch | eager, 12 threads | **41,523** | 29,449 compiled at 12; 29,624 compiled at 20 |
| OpenNN (MKL) | 12 threads | **34,209** | 32,575 at 6; 32,289 at 14; 33,772 at 20 |

**With each engine at its best, OpenNN trains at 0.50x TensorFlow and 0.82x
PyTorch here** - the opposite of the ordering above. Note that TensorFlow beats
OpenNN on this machine even with XLA off (50,460 against 34,209), so the gap is
not only the protocol: OpenNN's dense CPU training path is behind on this
hardware.

Best thread count is 12 for every engine that was swept, i.e. the six P-cores'
worth of threads rather than all 20 logical CPUs.

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
| Split used here | 100,000 train rows / 20,000 test rows |
| Layout | `feature_0,...,feature_27,label`, normalized from train-set statistics |
| Model | `28 -> 1024 -> 1024 -> 1` |
| Activation | ReLU hidden layers, sigmoid output |
| Objective | Binary cross entropy |
| Optimizer | Adam |
| Epochs | 1 |
| Batch | 1024 |
| Timed runs | 1 |
| Device | CPU only |
| Environment | WSL2, Intel Core i7-12700H host, CPU path |
| OpenNN CPU backend | MKL-linked binary (`libmkl_rt`) |
| Thread environment | `MKL_NUM_THREADS=20`, `OMP_NUM_THREADS=20`, dynamic threading disabled |

CPU joules are **not measured** in WSL on this machine; the result JSON includes
only a normalized inverse-throughput energy proxy.

## Artifact

Local result JSON:
`docs/benchmarks/results/cpu-dense-higgs-train-20260704T-higgs100k-train-mkl.json`

Runner:
[`higgs/run_higgs_cpu.py`](higgs/run_higgs_cpu.py)

OpenNN binary source:
[`higgs/opennn_higgs_cpu.cpp`](higgs/opennn_higgs_cpu.cpp)
