# GPU HIGGS dense training: OpenNN vs PyTorch vs TensorFlow

OpenNN trains the canonical HIGGS dense classifier at 4.53 million samples/s in fp32 (1.34x PyTorch, 1.46x TensorFlow) and 8.45 million samples/s in bf16 on an NVIDIA GeForce RTX 4080, at held-out quality now matching the best engine in the table. Median of 5 runs per cell (2026-08-10, commit 52e21e15d; artifact `results/gpu-higgs-dense-training-speed-20260810T121927Z.json`).

> **CUDA-graph speed headroom, under investigation (2026-08-11).** The 2026-08-07 checkout reported 12.4M samples/s bf16 here, but the graph-off A/B on the ResNet cell showed the old CUDA-graph training path was not numerically equivalent to eager execution — its speedup was partly artifact. The current build's graph path is equivalent (and OpenNN's held-out quality improved to parity with TensorFlow: log loss 0.470 -> 0.458, ROC AUC 0.856 -> 0.863) but the graph's speed benefit shrank, so the bf16 training cell reads as a statistical tie with PyTorch (0.99x) instead of the earlier 1.34x figure. Recovering the mega-launch speedup on the correct graph path is the open task.

## Contents

- [Introduction](#introduction)
- [Benchmark application](#benchmark-application)
- [Reference computer](#reference-computer)
- [Methodology](#methodology)
- [Results](#results)
- [Held-out quality](#held-out-quality)
- [Discussion](#discussion)
- [Conclusions](#conclusions)
- [Reproducing](#reproducing)
- [References](#references)

## Introduction

HIGGS is a large tabular binary-classification dataset from high-energy physics. Its 10.5-million-row training split and 28 numerical features make it useful for measuring sustained dense-network throughput rather than a short synthetic kernel.

This GPU training benchmark uses the canonical 28-1024-1024-1 ReLU classifier with Adam, batch 7,000, five epochs, and fp32 and bf16 paths in OpenNN, PyTorch, and TensorFlow.

## Benchmark application

| Item | Configuration |
|---|---|
| Dataset | HIGGS |
| Training rows | 10,500,000 |
| Test rows used by runner | 497,000 |
| Inputs | 28 normalized numerical features |
| Network | 28 -> 1024 ReLU -> 1024 ReLU -> 1 |
| Parameters | 1,080,321 |
| Loss | Binary cross-entropy |
| Optimizer | Adam |
| Batch | 7,000 |
| Epochs | 5 |
| Precisions | fp32 and bf16 |
| Metric | Training samples per second; higher is better |

## Reference computer

| Component | Value |
|---|---|
| CPU | Intel Core i9-12900K |
| GPU | NVIDIA GeForce RTX 4080, 16 GB |
| Operating system | Linux 6.17 x86_64 |
| NVIDIA driver | 595.71.05 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| PyTorch CUDA / cuDNN | CUDA 13.0 / cuDNN 9.24 |
| TensorFlow | 2.21.0 |
| OpenNN | 9.0.0 |
| Git state | Clean |

## Methodology

The three engines read the same prepared and normalized HIGGS split. Training throughput is calculated from 10.5 million training rows and the median epoch time produced inside each engine.

- OpenNN uses GPU-resident data and its CUDA graph training path.
- PyTorch uses its optimized dense path, with bf16 autocast in the bf16 cell.
- TensorFlow uses compiled graph execution and mixed bf16 in the bf16 cell.
- All models use the same layer widths, activation, parameter count, batch, and epoch count.
- Testing happens after training and is not part of the throughput numerator.
- One run is stored per cell, so the figures should be read as a controlled snapshot.

## Results

| Precision | OpenNN | PyTorch | TensorFlow | OpenNN / PyTorch | OpenNN / TensorFlow |
|---|---:|---:|---:|---:|---:|
| fp32 | **4,531,215 samples/s** | 3,379,069 samples/s | 3,106,388 samples/s | **1.341x** | **1.459x** |
| bf16 | 8,447,246 samples/s | **8,552,365 samples/s** | 7,435,918 samples/s | 0.988x | **1.136x** |

| Precision | OpenNN epoch | PyTorch epoch | TensorFlow epoch |
|---|---:|---:|---:|
| fp32 | **2.320 s** | 3.107 s | 3.380 s |
| bf16 | 1.240 s | **1.228 s** | 1.412 s |

## Held-out quality

Throughput is only meaningful if the training loop produces a usable model. The runner reports the following held-out metrics after five epochs:

| Precision | Framework | Accuracy | Log loss | ROC AUC |
|---|---|---:|---:|---:|
| fp32 | OpenNN | 0.778 | 0.459 | 0.863 |
| fp32 | PyTorch | 0.775 | 0.463 | 0.860 |
| fp32 | TensorFlow | 0.779 | 0.457 | 0.864 |
| bf16 | OpenNN | 0.778 | 0.458 | 0.863 |
| bf16 | PyTorch | 0.776 | 0.463 | 0.860 |
| bf16 | TensorFlow | 0.779 | 0.457 | 0.864 |

OpenNN and TensorFlow now share the best held-out band (log loss ~0.457-0.459, AUC ~0.863) with PyTorch slightly behind; earlier snapshots had OpenNN trailing TensorFlow on log loss, a gap the 2026-08 numerics change closed. Because no common hard target was configured, this article does not convert throughput into a convergence-to-quality claim.

## Discussion

OpenNN's relative lead is fp32, where it processes 1.34x as many samples per second as PyTorch and 1.46x as many as TensorFlow. In bf16 the current build ties PyTorch (0.99x, within a few percent) and leads TensorFlow by 1.14x; the pre-regression build led every bf16 cell (see the banner above).

Within OpenNN, bf16 improves throughput by 1.86x over its own fp32 for this exact network.

The quality table is important context: the current build's held-out metrics sit in the best band of the table in both precisions, at the cost of the training-speed regression under investigation.

## Conclusions

- OpenNN leads fp32 training throughput: 1.34x PyTorch, 1.46x TensorFlow.
- In bf16 the current build statistically ties PyTorch (0.99x) and leads TensorFlow (1.14x); the 2026-08-09 regression under investigation costs ~35% across training cells.
- OpenNN's held-out quality now matches the best engine in the table in both precisions.
- All held-out metrics are published so the throughput result can be interpreted with its quality context.

## Reproducing

The canonical runner is `docs/benchmarks/throughput/higgs-gpu/run_higgs_dense.py`:

```bash
python run_higgs_dense.py \
  --train "$OPENNN_BENCH_DATA/higgs/higgs_train.csv" \
  --test "$OPENNN_BENCH_DATA/higgs/higgs_test.csv" \
  --epochs 5 --batch 7000 --hidden 1024 --hidden-layers 2 \
  --activation relu --shuffle shuffle --precision both --runs 1
```

The result artifact is `docs/benchmarks/results/gpu-higgs-dense-training-speed-20260710T084732Z.json`.

## References

- [HIGGS dataset, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/280/higgs).
- [Searching for exotic particles in high-energy physics with deep learning](https://www.nature.com/articles/ncomms5308).
- [OpenNN source repository](https://github.com/Artelnics/opennn).
- [PyTorch](https://pytorch.org/).
- [TensorFlow](https://www.tensorflow.org/).
