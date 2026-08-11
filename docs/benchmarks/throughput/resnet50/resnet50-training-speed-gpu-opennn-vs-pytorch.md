# GPU ResNet-50 training speed: OpenNN vs PyTorch vs TensorFlow (CIFAR-10)

*Last updated 2026-08-10. Linux x86_64, NVIDIA GeForce RTX 4080 (16 GB), driver 595.84, CUDA 13.3, cuDNN 9.23.1. Artifact: [`results/gpu-resnet50-training-speed-cifar10-20260810T093239Z.json`](../../results/).*

This note measures a **real architecture**: ResNet-50 v1.5 — 53 convolutions,
53 batch normalizations, residual connections, 23.5M parameters — trained on
CIFAR-10 with identical configuration in all three frameworks.

## The result

Training throughput on 50,000 CIFAR-10 images (32×32×3), batch 128,
cross-entropy + Adam, 5 timed epochs after 2 warmup epochs, median of 5 runs.
Fair paths: OpenNN GPU-resident data + CUDA graph, PyTorch channels_last +
`torch.compile` + TF32, TensorFlow XLA:

| Precision | OpenNN | PyTorch | TensorFlow | OpenNN / PyTorch | OpenNN / TF |
|---|---:|---:|---:|---|---|
| bf16 | **27,092 ± 47** | 21,635 ± 74 | 20,618 ± 184 | **1.25×** | 1.31× |
| fp32 | **21,396 ± 60** | 16,813 ± 11 | 15,179 ± 77 | **1.27×** | 1.41× |

**OpenNN trains ResNet-50 1.25–1.41× faster than both engines in both
precisions** with the same architecture (matched to torchvision's resnet50
v1.5; parameter counts agree to the dense-bias rounding) and the same data
residency.

> ⚠️ **CUDA-graph speed headroom, under investigation (2026-08-11).** The
> 2026-08-07 checkout reported ~42,000 samples/s bf16 here — but a graph-off
> A/B shows that number was not clean: the old CUDA-graph path converged
> differently from its own eager training (loss 1.41 vs 2.44 at identical
> work), i.e. its speedup was partly an artifact of non-equivalent execution.
> The current build's graph path **is** numerically equivalent to eager
> (2.43 ≈ 2.46) but only ~14% faster than eager, where the old (incorrect)
> path was ~90% faster. Eager speed is identical between the two builds, CPU
> training and every inference/capacity cell are unaffected, and the dense
> HIGGS graph path shows the same pattern. The open task is recovering the
> mega-launch speedup on the *correct* graph path.

## How OpenNN got here

The work fell into two phases. The first moved the math onto fast kernels;
the second removed the launch overhead around them.

**Phase 1 — get the kernels right (2,912 → ~5,200 samples/s).** The first run
took 31 s/epoch. `OPENNN_PROFILE=1` found the epoch dominated not by
convolutions but by **batch normalization through the legacy
`cudnnBatchNormalization*` API** (~10× what the kernels should cost on NHWC
activations). Moving batch norm — and convolution forward/weight/data
gradients — onto the **cudnn-frontend graph API** (the same engine interface
PyTorch uses), then autotuning each graph's plan on first execution
(`cudnn.benchmark`-equivalent), then fusing the residual add + block-end ReLU
into the batch-norm graph and dropping the redundant convolution biases under
batch norm, took it to ~5,200 samples/s. At that point OpenNN was already
1.9× eager PyTorch but still behind `torch.compile`.

**Phase 2 — find where the last gap actually was.** Per-kernel CUDA-event
timing (`OPENNN_GRAPH_TIMING=1`) of the captured step settled it: OpenNN's
**convolution kernels alone cost 13.6 ms/step versus PyTorch's entire
53-convolution budget of 17.1 ms** — the compute was already faster. Two
kernel-rewrite ideas (hand-written batch-norm CUDA kernels; a strided-view
trick for the 1×1 stride-2 projections) were implemented, measured, and
**discarded** — cuDNN's batch norm was already near roofline, and the strided
view regressed the projection gradients 2–4×. The gap was not in the math. It
was **launch overhead**: ~150 kernel launches per step, each paying WSL's
expensive CUDA-API issue latency.

The fix that won was the **resident CUDA-graph mega-launch**. OpenNN already
captured-and-replayed steps as CUDA graphs, but the GPU-resident data path
replayed only *one* step per launch. Extending it to bundle **8 steps into a
single captured graph** — issuing the 8 device-side batch gathers on the
transfer stream outside the graph, then capturing only the 8 compute steps —
amortizes the per-step launch and cross-stream waits eightfold. That single
change took 5,200 → 8,433 samples/s (CUDA graph on, the benchmark default;
`opennn_resnet50_speed.cpp` enables it in code via `set_cuda_graph`).

A third small fix made the architecture buildable at all: **`Same`-padded
convolutions on small feature maps** — the layer rejected kernels larger than
the input even when padding makes the shape valid (ResNet's stage-4 3×3
convolutions on 2×2 maps); the check now applies only to unpadded
convolutions. Everything else was in place from the MNIST work: the
cudnn-frontend convolution graphs, fused conv+bias(+ReLU) forward graphs, and
the GPU-resident `ImageDataset` mode (enabled in code via the
`GPUPersistantData` storage mode) that stages the 614 MB dataset once and
gathers batches device-side.

## Setup

| | Value |
|---|---|
| Data | CIFAR-10 / CIFAR-100 train split: 50,000 BMPs, 32×32×3, 10 / 100 classes |
| Network | ResNet-50 v1.5: conv 7×7/2 → maxpool 3×3/2 → bottleneck stages [3,4,6,3] → Dense 10/100 (softmax) |
| Loss / optimizer | cross-entropy, Adam (lr 0.001), no regularization |
| Protocol | shuffled epochs, 2 warmup epochs, timed epochs after |
| Precision | fp32, framework-default TF32 policy |
| Residency | dataset GPU-resident in both engines |

On 32×32 inputs the standard ImageNet stem reduces the final feature map to
1×1×2048, so the global average pool is the identity and is omitted on the
OpenNN side; the PyTorch model keeps its (no-op) `AdaptiveAvgPool2d(1)`. Both
models drop the convolution biases under batch normalization (its β absorbs
them, matching torchvision's `bias=False`), so the parameter counts agree to
the dense-bias rounding. Softmax + cross-entropy is fused on both sides (the
gradient is the collapsed `softmax_output − target`), so neither engine
materializes a softmax-Jacobian — the 10→100 head change is free at the
gradient.

Hardware/software: NVIDIA GeForce RTX 4080 (16 GB, driver 595.84), Intel Core
i9-12900K, Linux x86_64. OpenNN built with g++ 13.3 + CUDA 13.3 + cuDNN 9.23.1;
PyTorch 2.13.0+cu130 and TensorFlow 2.21.0 on CPython 3.12.3.

## Caveats

* All engines are timed in the same session with a GPU-resident dataset.
  OpenNN's number uses the CUDA graph; PyTorch runs `torch.compile` in its
  default mode (no CUDA Graphs, plain Adam). A 2026-08-11 probe measured
  PyTorch at ~31,100 samples/s bf16 with `mode="reduce-overhead"` + fused Adam
  (+44% over the table). The driver will move to that path together with the
  OpenNN CUDA-graph speed-recovery work, so both engines' graph-replay modes
  are re-measured in the same pass.
* Single consumer desktop GPU; ratios shift with hardware and input size — at
  224×224 the workload becomes conv-FLOP-bound and launch overhead is a
  smaller share, so the engines should converge toward the same cuDNN-kernel
  floor.
* Batch-norm numerics differ slightly between the frontend engines and the
  legacy API (reduction order), so loss trajectories track in band rather
  than bit-for-bit; the library's GPU test-suite failure set is unchanged
  versus the pre-change baseline.

## Reproducing

The data prep, all engine drivers, and the canonical runner are in
[`docs/benchmarks/throughput/resnet50/`](.): 

```bash
python prepare_cifar10.py "$OPENNN_BENCH_DATA/cifar10"   # downloads CIFAR-10, writes BMPs + npy
python run_resnet50.py --precision both --runs 5          # 3-way harness, writes the results/ artifact
# or individually (CUDA graph on by default + CIFAR data GPU-resident, both in code):
./opennn_resnet50_speed "$OPENNN_BENCH_DATA/cifar10/train" [epochs] [batch] [fp32|bf16]
python pytorch_resnet50_speed.py [epochs] [batch] "$OPENNN_BENCH_DATA/cifar10"
```

The Python drivers read the class count from the labels; the OpenNN program
reads it from the dataset shape.
