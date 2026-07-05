# CPU HIGGS dense inference: OpenNN vs PyTorch vs TensorFlow

This benchmark measures repeated CPU inference passes for the same dense binary
classifier shape on the prepared UCI HIGGS test split.

**Status:** internal WSL2 subset result. Use it for migration checks and
presentation drafts, not as a public headline. It is not the full 500k-row
publication test split.

## Result

Higher is better.

| Framework | Inference speed (samples/s) | Median pass (s) |
|---|---:|---:|
| OpenNN | **209,774** | **0.0927** |
| PyTorch | 166,003 | 0.1172 |
| TensorFlow | 153,281 | 0.1269 |

OpenNN delivered **1.26x PyTorch speed** and **1.37x TensorFlow speed** on this
CPU inference run, using the MKL-linked OpenNN binary. A smaller batch-256 probe
was overhead-sensitive and put PyTorch ahead; the current row uses batch 1024 and
five timed runs.

## Setup

| Item | Value |
|---|---|
| Dataset | UCI HIGGS, prepared with `docs/benchmarks/higgs/prepare_higgs.py` |
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
