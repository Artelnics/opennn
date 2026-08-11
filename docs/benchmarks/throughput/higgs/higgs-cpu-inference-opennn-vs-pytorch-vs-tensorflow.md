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
