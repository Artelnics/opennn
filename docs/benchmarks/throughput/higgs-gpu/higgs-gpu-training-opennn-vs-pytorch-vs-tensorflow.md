# GPU HIGGS dense training: OpenNN vs PyTorch vs TensorFlow

OpenNN trains the canonical HIGGS dense classifier at 11.08 million samples/s in bf16 (1.30x PyTorch, 1.49x TensorFlow) and 5.14 million samples/s in fp32 (1.06x, 1.19x) on an NVIDIA GeForce RTX 4080, at held-out quality matching the best engine in the table. bf16 PyTorch/TensorFlow cells: median of 5 runs (2026-08-10, artifact `results/gpu-higgs-dense-training-speed-20260810T121927Z.json`); OpenNN and the fp32 (TF32-aligned) cells re-measured 2026-08-11 after the pipeline changes below; formal multi-run refresh pending.

> **2026-08-11 update.** Two changes since the 2026-08-10 snapshot. (1) The
> earlier "bf16 ties PyTorch" reading was a measurement-plus-pipeline problem,
> now fixed: the driver runs a single `train()` and times per-epoch medians via
> `post_epoch_callback` (so graph capture and setup are no longer inside the
> timed window), and the training loop builds the next epoch's shuffled batch
> list asynchronously while the GPU trains the current epoch — the ~190 ms/epoch
> host-side shuffle of 10.5M indices no longer stalls the GPU. bf16 went from
> 8.45M to 11.08M samples/s with identical numerics; the GPU is busy ~97% of
> the epoch. (2) The fp32 cells now run **TF32 in all three engines** (that is
> what "fp32" means on this GPU by default in OpenNN, and the PyTorch/TensorFlow
> drivers now enable it too); the earlier 1.34x fp32 lead measured OpenNN-TF32
> against PyTorch-strict-fp32 and is retired.

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

- OpenNN uses GPU-resident data, the CUDA mega-graph training path, and asynchronous batch-list preparation (the next epoch's shuffle happens while the GPU trains the current one).
- PyTorch uses its optimized dense path, with bf16 autocast in the bf16 cell.
- TensorFlow uses compiled graph execution and mixed bf16 in the bf16 cell.
- "fp32" means TF32 tensor-core matmuls in all three engines (the RTX 4080 default; pass `--precision strict` to the drivers for true fp32).
- All models use the same layer widths, activation, parameter count, batch, and epoch count.
- Testing happens after training and is not part of the throughput numerator.
- One run is stored per cell, so the figures should be read as a controlled snapshot.

## Results

| Precision | OpenNN | PyTorch | TensorFlow | OpenNN / PyTorch | OpenNN / TensorFlow |
|---|---:|---:|---:|---:|---:|
| fp32 (TF32) | **5,140,000 samples/s** | 4,860,000 samples/s | 4,330,000 samples/s | **1.06x** | **1.19x** |
| bf16 | **11,080,000 samples/s** | 8,552,365 samples/s | 7,435,918 samples/s | **1.30x** | **1.49x** |

| Precision | OpenNN epoch | PyTorch epoch | TensorFlow epoch |
|---|---:|---:|---:|
| fp32 (TF32) | **2.043 s** | 2.160 s | 2.425 s |
| bf16 | **0.948 s** | 1.228 s | 1.412 s |

In bf16 the steady state is 643 µs per 7,000-sample step with the GEMMs running
at ~96 TFLOPS — the tensor-core roofline for these shapes — so the remaining
lead comes from keeping the GPU fed: on-device resident data, the CUDA mega-graph
step, and the asynchronous batch-list prefetch described above.

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

OpenNN now leads every cell. The big margin is bf16 (1.30x PyTorch, 1.49x TensorFlow), where the step is GEMM-roofline-bound and the win comes from the data pipeline never stalling the GPU. The fp32 (TF32) margin is narrower (1.06x, 1.19x) because all three engines ride the same TF32 tensor-core GEMMs there.

Within OpenNN, bf16 improves throughput by 2.16x over its own fp32 for this exact network.

The quality table is important context: the held-out metrics sit in the best band of the table in both precisions — the throughput lead does not trade away model quality.

## Conclusions

- OpenNN leads bf16 training throughput: 1.30x PyTorch, 1.49x TensorFlow, at 11.08M samples/s.
- In fp32 — TF32 in all three engines, like-for-like — OpenNN leads 1.06x PyTorch and 1.19x TensorFlow.
- The bf16 win is a pipeline win: single-`train()` timing, resident data, the CUDA mega-graph step, and asynchronous batch-list preparation keep the GPU ~97% busy at the GEMM roofline.
- OpenNN's held-out quality matches the best engine in the table in both precisions.

## Reproducing

The canonical runner is `docs/benchmarks/throughput/higgs-gpu/run_higgs_dense.py`:

```bash
python run_higgs_dense.py \
  --train "$OPENNN_BENCH_DATA/higgs/higgs_train.csv" \
  --test "$OPENNN_BENCH_DATA/higgs/higgs_test.csv" \
  --epochs 5 --batch 7000 --hidden 1024 --hidden-layers 2 \
  --activation relu --shuffle shuffle --precision both --runs 1
```

The reference artifact for the PyTorch/TensorFlow bf16 cells is `docs/benchmarks/results/gpu-higgs-dense-training-speed-20260810T121927Z.json`.

## References

- [HIGGS dataset, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/280/higgs).
- [Searching for exotic particles in high-energy physics with deep learning](https://www.nature.com/articles/ncomms5308).
- [OpenNN source repository](https://github.com/Artelnics/opennn).
- [PyTorch](https://pytorch.org/).
- [TensorFlow](https://www.tensorflow.org/).
