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
