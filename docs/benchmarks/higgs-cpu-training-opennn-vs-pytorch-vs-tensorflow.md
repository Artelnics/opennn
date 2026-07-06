# CPU HIGGS dense training: OpenNN vs PyTorch vs TensorFlow

This benchmark measures one CPU training epoch for the same dense binary
classifier on a prepared UCI HIGGS subset.

**Status:** internal WSL2 subset result. Use it for migration checks and
presentation drafts, not as a public headline. It is not the full 10.5M-row
publication split.

## Result

Higher is better.

| Framework | Training speed (samples/s) | Median epoch (s) | Accuracy | Log loss | ROC AUC |
|---|---:|---:|---:|---:|---:|
| OpenNN | 59,372 | 1.6843 | 0.6903 | 0.5874 | 0.7547 |
| PyTorch | 25,651 | 3.8985 | 0.6729 | 0.6003 | 0.7396 |
| TensorFlow | 49,101 | 2.0366 | 0.6803 | 0.5952 | 0.7443 |

OpenNN trained at **2.32x PyTorch speed** and **1.21x TensorFlow speed** on
this CPU run, using the MKL-linked OpenNN binary. OpenNN also reached the
highest one-epoch quality metrics in this single run.

## Setup

| Item | Value |
|---|---|
| Dataset | UCI HIGGS, prepared with `docs/benchmarks/higgs/prepare_higgs.py` |
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
