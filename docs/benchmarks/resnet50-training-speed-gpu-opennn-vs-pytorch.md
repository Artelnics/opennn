# GPU ResNet-50 training speed: OpenNN vs PyTorch (CIFAR-10)

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-13. Linux x86_64 (WSL2), NVIDIA RTX 3060 Laptop GPU, CUDA 12.9, cuDNN 9.23.*

The [MNIST CNN note](cnn-training-speed-gpu-opennn-vs-pytorch-vs-tensorflow.md)
measures a minimal convolutional network. This note scales the same question
to a **real architecture**: ResNet-50 — 53 convolutions, 53 batch
normalizations, residual connections, 23.5M parameters — trained on CIFAR-10
with identical configuration in both frameworks.

## The result

Training throughput on 50,000 CIFAR-10 images (32×32×3), batch 128, fp32,
cross-entropy + Adam, timed after warmup (medians of three timed runs on one
session):

| | Epoch time | Samples/s |
|---|---:|---:|
| **OpenNN** (CUDA graph, fp32, GPU-resident data) | **5.9 s** | **8,433** |
| PyTorch `torch.compile` (fp32) | 9.5 s | 5,268 |
| PyTorch (eager fp32) | 12.6 s | 3,960 |

**OpenNN trains ResNet-50 1.6× faster than `torch.compile` and 2.1× faster
than eager PyTorch** on the same GPU, with the same architecture (the PyTorch
model is written out to match torchvision's resnet50 v1.5 exactly; parameter
counts agree at 23.5M) and the same data residency. Training is real:
cross-entropy descends from 2.3 at initialization to ≈1.0 within three epochs
in both engines.

This is the headline number after a full optimization pass. The first run of
this benchmark was 2,912 samples/s; the section below traces how it got from
there to 8,433 — and, importantly, where the real bottleneck turned out *not*
to be.

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
change took 5,200 → 8,433 samples/s (`OPENNN_CUDA_GRAPH=1`, the benchmark
default).

A third small fix made the architecture buildable at all: **`Same`-padded
convolutions on small feature maps** — the layer rejected kernels larger than
the input even when padding makes the shape valid (ResNet's stage-4 3×3
convolutions on 2×2 maps); the check now applies only to unpadded
convolutions. Everything else was in place from the MNIST work: the
cudnn-frontend convolution graphs, fused conv+bias(+ReLU) forward graphs, and
the GPU-resident `ImageDataset` mode (`OPENNN_GPU_RESIDENT_DATA=1`) that stages
the 614 MB dataset once and gathers batches device-side.

## Setup

| | Value |
|---|---|
| Data | CIFAR-10 train split: 50,000 BMPs, 32×32×3, 10 classes |
| Network | ResNet-50 v1.5: conv 7×7/2 → maxpool 3×3/2 → bottleneck stages [3,4,6,3] → Dense 10 (softmax) |
| Loss / optimizer | cross-entropy, Adam (lr 0.001), no regularization |
| Protocol | shuffled epochs, 2 warmup epochs, timed epochs after |
| Precision | fp32, framework-default TF32 policy |
| Residency | dataset GPU-resident in both engines |

On 32×32 inputs the standard ImageNet stem reduces the final feature map to
1×1×2048, so the global average pool is the identity and is omitted on the
OpenNN side; the PyTorch model keeps its (no-op) `AdaptiveAvgPool2d(1)`.
OpenNN's convolutions carry biases (23,555,088 parameters vs PyTorch's
23,528,522 bias-free convs) — extra work OpenNN does, not an advantage.

Hardware/software: NVIDIA GeForce RTX 3060 Laptop GPU (driver 555.85) under
WSL2 Ubuntu 24.04 on Windows 11 (i7-12700H). OpenNN built with g++ 13.3 +
CUDA 12.9.86 + cuDNN 9.23; PyTorch 2.6.0 (cu124 wheels) on CPython 3.12.

## Caveats

* The comparison is fp32 on all three engines, all timed on the same session.
  OpenNN's number requires `OPENNN_CUDA_GRAPH=1` and a GPU-resident dataset —
  the CUDA-graph mega-launch is the headline result, not the eager loop (which
  lands ~5,200 samples/s, still 1.3× eager PyTorch). `torch.compile` is the
  fair opponent for a graph-replaying engine, and OpenNN is 1.6× ahead of it
  here ([`pt_compile_probe.py`](resnet50-training-speed/pt_compile_probe.py)).
* The mega-launch's leverage is largest under WSL2, where CUDA-API issue
  latency is high; on native Linux the per-launch cost is lower and the margin
  over `torch.compile` would narrow. The conv/BN kernels themselves are cuDNN
  on both sides.
* Single consumer laptop GPU; ratios shift with hardware and input size — at
  224×224 the workload becomes conv-FLOP-bound and launch overhead is a
  smaller share, so both engines should converge toward the same cuDNN-kernel
  floor.
* Batch-norm numerics differ slightly between the frontend engines and the
  legacy API (reduction order), so loss trajectories track in band rather
  than bit-for-bit; the library's GPU test-suite failure set is unchanged
  versus the pre-change baseline.

## Reproducing

The data prep, both training programs, the compile probe, and the runner are
in [`docs/benchmarks/resnet50-training-speed/`](resnet50-training-speed/):

```bash
python prepare_cifar10.py cifar10        # downloads CIFAR-10, writes BMPs + npy
./run_resnet50.sh 5 128                  # both engines + summary
# or individually:
OPENNN_GPU_RESIDENT_DATA=1 ./opennn_resnet50_speed cifar10/train [epochs] [batch] [fp32|bf16]
python pytorch_resnet50_speed.py [epochs] [batch] cifar10
```
