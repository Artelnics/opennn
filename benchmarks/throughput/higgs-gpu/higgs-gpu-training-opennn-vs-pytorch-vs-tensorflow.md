# GPU HIGGS dense training: OpenNN vs PyTorch vs TensorFlow

On an NVIDIA GeForce RTX 5070 Ti, OpenNN trains the canonical HIGGS dense classifier at 11.23 million samples/s in bf16 and 5.95 million in fp32, at held-out quality matching the best engine in the table. Against PyTorch that is 1.29x in both precisions. Against TensorFlow it is 1.07x in fp32 and a tie in bf16. Medians of five alternated rounds, 2026-08-20, artifact `results/gpu-higgs-dense-training-speed-20260820T091507Z.json`.

> **2026-08-20 TensorFlow dispatch and protocol correction.** The TensorFlow
> figures this note used to carry -- 1.49x bf16 and 1.19x fp32 -- were wrong, and
> by large factors. Two independent causes, both ours.
>
> First, the driver's epoch loop shuffled by gathering per batch:
> `tf.random.shuffle` plus two eager `tf.gather` calls dispatched from Python for
> every step, on top of the step call itself. TensorFlow enqueues
> asynchronously, so that host cost is hidden only while it stays under the
> GPU's, and here it did not -- at batch 7,000 enqueueing an epoch cost
> 0.9167 ms/batch against 0.9295 ms to enqueue *and* run it, meaning the GPU was
> idle waiting on Python. Shuffling once per epoch into a permuted copy and
> slicing from it -- identical shuffling, identical loss curve -- moved
> TensorFlow from 7.41M to 11.25M samples/s. We had been measuring their
> dispatch, not their training.
>
> Second, the runner measured engines in blocks: all N runs of one, then the
> next. The GPU's state drifts between blocks by more than the margins being
> compared -- bf16 read 0.987x blocked and 1.019x alternated, a three-point
> swing on a two-point effect. Engines now alternate within a round and the
> starting engine rotates.
>
> A caveat on resolution. This card boosts from 405 MHz to 2835 MHz over the
> first ~2.5 s of load and its sustained clock drifts with ambient temperature
> across a session; OpenNN's bf16 reading moved 8% across one day while
> TensorFlow's held. Three same-session interleaved measurements of the bf16
> cell gave 1.019x, 0.993x and 0.997x. Margins under about 2% are therefore not
> resolvable here without locked clocks, which is why the bf16 cell is reported
> as a tie rather than a number.

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
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| Operating system | Linux 7.0 x86_64 |
| NVIDIA driver | 610.43.02 |
| CUDA | 13.3 |
| PyTorch | 2.13.0+cu130 |
| TensorFlow | 2.21.0 |
| OpenNN commit | `4ccb88dea` |
| Run ID | 20260820T091507Z |

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

Five alternated rounds per precision, medians.

| Precision | OpenNN | PyTorch | TensorFlow | OpenNN / PyTorch | OpenNN / TensorFlow |
|---|---:|---:|---:|---:|---:|
| fp32 (TF32) | **5,945,352 samples/s** | 4,613,420 samples/s | 5,564,210 samples/s | **1.29x** | **1.07x** |
| bf16 | 11,227,978 samples/s | 8,728,218 samples/s | **11,307,621 samples/s** | **1.29x** | tie |

The fp32 margin over TensorFlow is paired 5/5 and comfortably outside the
session drift. The bf16 cell is not: OpenNN won 11 of 12 paired rounds in one
session and lost 5 of 5 and 4 of 5 in two others, so it is reported as a tie.
Both PyTorch cells are paired 5/5 and far outside the noise.

In bf16 the steady state is 642 µs per 7,000-sample step. An Nsight trace with
graph-node tracing puts three GEMMs -- forward, dgrad and wgrad of the
1024x1024 layer -- at 0.508 ms of that, 79% of the step, running at 85-91
TFLOPS against the 92.6 TFLOPS a standalone cuBLAS probe measures as the best
achievable for this shape. That is 92-98% of ceiling, and TensorFlow runs the
same shapes on the same silicon, which is why bf16 is a tie: neither engine can
pull away on the math. The remaining 21% is the fusion tail -- Adam 0.025 ms,
activation backward 0.016, the 28-wide first layer 0.046, batch gathers 0.011 --
and that is the only place a bf16 margin could come from.

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

OpenNN leads PyTorch clearly in both precisions (1.29x, paired 5/5 in each) and
TensorFlow in fp32 (1.07x, paired 5/5). In bf16 it is level with TensorFlow.

The asymmetry is worth stating plainly rather than glossing: bf16 is where the
step is closest to the tensor-core roofline, so it is the precision where a
pipeline advantage has the least room to show. fp32 leaves more of the step
outside the GEMMs, and that is where OpenNN's margin survives.

Within OpenNN, bf16 improves throughput by 1.89x over its own fp32 for this
exact network.

The quality table is important context: the held-out metrics sit in the best band of the table in both precisions — the throughput lead does not trade away model quality.

## Conclusions

- OpenNN leads PyTorch by 1.29x in both precisions, paired 5/5 in each.
- In fp32 — TF32 in all three engines, like-for-like — OpenNN leads TensorFlow
  1.07x, paired 5/5.
- In bf16 OpenNN and TensorFlow are level. The step is 79% GEMMs at 92-98% of
  the cuBLAS ceiling for these shapes, so there is no headroom on the math for
  either engine.
- OpenNN's held-out quality matches the best engine in the table in both precisions.
- Margins under ~2% are not resolvable on this machine without locked GPU clocks;
  see the 2026-08-20 correction.

## Reproducing

The canonical runner is `benchmarks/throughput/higgs-gpu/run_higgs_dense.py`:

```bash
python run_higgs_dense.py \
  --train "$OPENNN_BENCH_DATA/higgs/higgs_train.csv" \
  --test "$OPENNN_BENCH_DATA/higgs/higgs_test.csv" \
  --epochs 5 --batch 7000 --hidden 1024 --hidden-layers 2 \
  --activation relu --shuffle shuffle --precision both --runs 1
```

The reference artifact for the PyTorch/TensorFlow bf16 cells is `benchmarks/results/gpu-higgs-dense-training-speed-20260810T121927Z.json`.

## References

- [HIGGS dataset, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/280/higgs).
- [Searching for exotic particles in high-energy physics with deep learning](https://www.nature.com/articles/ncomms5308).
- [OpenNN source repository](https://github.com/Artelnics/opennn).
- [PyTorch](https://pytorch.org/).
- [TensorFlow](https://www.tensorflow.org/).
