# Peak-batch training throughput: OpenNN vs PyTorch vs TensorFlow

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). First execution 2026-08-11. Linux x86_64, NVIDIA GeForce RTX 4080 (16 GB), driver 595.84, CUDA 13.3, cuDNN 9.23.1; OpenNN commit `6a721ddc8`, PyTorch 2.13.0+cu130, TensorFlow 2.21.0. Artifacts: [`results/gpu-higgs-peak-batch-speed-20260811T153230Z.json`](../../results/), [`results/gpu-resnet50-peak-batch-speed-20260811T145117Z.json`](../../results/), [`results/gpu-transformer-peak-batch-speed-20260811T144150Z.json`](../../results/).*

The fixed-batch throughput notes compare the engines at one common batch; the
capacity notes showed their memory ceilings differ by up to 2.8×. This
benchmark asks the combined question: **how fast can each engine train when it
picks its own best batch?** Every (engine, precision) sweeps the batch
geometrically from the standard size to the training-set size with the same
speed drivers as the fixed-batch notes, one fresh process per point; the peak
of each curve is the engine's score ([protocol](README.md)).

## The result

| Family | Precision | OpenNN peak | PyTorch peak | TensorFlow peak | OpenNN / PT | OpenNN / TF |
|---|---|---:|---:|---:|---|---|
| HIGGS dense | bf16 | **11.13M/s @ 14k** | 9.08M/s @ 14k | 9.78M/s @ 896k | **1.23×** | 1.14× |
| HIGGS dense | fp32 | **5.14M/s @ 7k** | 4.84M/s @ 7k | 5.19M/s @ 448k | 1.06× | ⚖️ 0.99× |
| ResNet-50 | bf16 | 57,533/s @ 1,024 | 61,380/s @ 2,048 | **65,752/s @ 2,048** | ✗ 0.94× | ✗ 0.88× |
| ResNet-50 | fp32 | **38,159/s @ 512** | 31,204/s @ 2,048 | 28,214/s @ 1,024 | **1.22×** | 1.35× |
| Transformer | bf16 | **6,048/s @ 64** | 4,935/s @ 32 | 4,460/s @ 32 | **1.23×** | 1.36× |
| Transformer | fp32 | **3,558/s @ 32** | 2,004/s @ 32 | 1,876/s @ 64 | **1.78×** | 1.90× |

OpenNN takes four of the six cells outright; HIGGS fp32 is a statistical tie
(TF +1% at a 64× larger batch), and **ResNet-50 bf16 is an honest loss** —
discussed below.

## What the curves say

**The engines' best batches are wildly different.** OpenNN's peaks sit at or
near the standard batch in every family — its fixed-batch pipeline (resident
data, CUDA mega-graph, async batch preparation) already saturates the GPU
there, so more batch buys little. TensorFlow is the opposite: XLA amortizes
poorly at small batches and its HIGGS curves *rise monotonically* to peaks at
batch 448k–896k. Comparing engines at one common batch therefore understates
TensorFlow on HIGGS — this benchmark gives every engine its best case, and
OpenNN still leads bf16 (1.23×/1.14×).

**ResNet-50 bf16 is where OpenNN's curve dies too early.** OpenNN peaks at
batch 1,024 (57.5k/s) and then *collapses* — 2,048: 52.3k, 4,096: 10.5k,
8,192: 6.8k, OOM at 16,384 — while PyTorch and TensorFlow hold ~57–66k/s up to
batch 16,384 and OOM at 32,768. The artifact records the mechanism: past its
peak, the resident-dataset + batch-pool + graph speed path leaves too little
VRAM for cuDNN convolution workspaces ("cudnn-frontend path unavailable —
cudaMalloc(1024 MiB)" at the frontier), so convolutions fall to a slow path
long before the memory truly runs out. The capacity trial proves batch 27,306
*fits* monolithically (pool 1, no residency — see
[resnet50-max-batch](../../capacity/resnet50-max-batch/resnet50-max-batch-gpu-opennn-vs-pytorch-vs-tensorflow.md));
closing the gap between that ceiling and the fast path's usable range is the
identified engineering follow-up from this benchmark's first run.

**fp32 is TF32 everywhere** (suite policy). The transformer fp32 margin
(1.78×/1.90×) carries the fused-attention design advantage described in the
[training note](../attention-speed/transformer-training-gpu-opennn-vs-pytorch.md).

## Frontier classification

Every ascent ended at a genuine memory frontier: the raw output stored with
each frontier shows `cudaMalloc` failures (OpenNN) or allocator OOMs (PyTorch
"CUDA out of memory", TensorFlow RESOURCE_EXHAUSTED / internal abort). In the
transformer and ResNet artifacts OpenNN's frontier is labeled `error` rather
than `oom` — its "CUDA Error: 2" message was not yet in the runner's OOM
markers (added right after; the HIGGS artifact, run last, classifies it
correctly). The label does not affect the curves or peaks.

## Caveats

* Saturation benchmark, not time-to-quality: giant batches change convergence
  at a fixed learning rate ([quality/convergence](../../quality/convergence/README.md)
  is the quality-gated comparison). The peak batches here are *throughput*
  optima, not training recommendations.
* Single consumer GPU (16 GB); peaks and frontiers scale with VRAM.
* One full run per cell (fresh process per point); the curves are smooth and
  the peaks consistent with the fixed-batch five-run medians, but this first
  execution has no run-to-run dispersion estimate yet.
