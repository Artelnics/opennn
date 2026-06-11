# GPU ResNet-50 training speed: OpenNN vs PyTorch (CIFAR-10)

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-12. Linux x86_64 (WSL2), NVIDIA RTX 3060 Laptop GPU, CUDA 12.9, cuDNN 9.23.*

The [MNIST CNN note](cnn-training-speed-gpu-opennn-vs-pytorch-vs-tensorflow.md)
measures a minimal convolutional network. This note scales the same question
to a **real architecture**: ResNet-50 — 53 convolutions, 53 batch
normalizations, residual connections, 23.5M parameters — trained on CIFAR-10
with identical configuration in both frameworks.

## The result

Training throughput on 50,000 CIFAR-10 images (32×32×3), batch 128, fp32,
cross-entropy + Adam, timed after warmup:

| | Epoch time | Samples/s |
|---|---:|---:|
| **OpenNN** (CUDA, fp32, GPU-resident data) | **17.2 s** | **2,912** |
| **PyTorch** (eager fp32, GPU-resident data) | 24.0 s | 2,080 |

**OpenNN trains ResNet-50 1.4× faster than eager PyTorch** on the same GPU,
with the same architecture (the PyTorch model is written out to match
torchvision's resnet50 v1.5 exactly; parameter counts agree at 23.5M) and the
same data residency. Training is real: cross-entropy descends from 2.3 at
initialization to ≈1.3 within three epochs in both engines.

## How OpenNN got here

The first run of this benchmark took 31 s/epoch — 1.3× *slower* than
PyTorch. Profiling (`OPENNN_PROFILE=1`) found the epoch dominated not by the
convolutions but by **batch normalization through the legacy
`cudnnBatchNormalization*` API**: 20 s of the 31 s epoch (0.4–0.6 ms per call
on NHWC activations, ~10× what the kernels should cost). The same pattern the
[MNIST note](cnn-training-speed-gpu-opennn-vs-pytorch-vs-tensorflow.md)
found for convolutions and their bias gradients.

Two library changes closed the gap:

1. **Batch normalization on the cudnn-frontend graph API** — forward
   (with running-stat updates fused) and backward now run through the same
   engine interface PyTorch uses, with the legacy path as automatic fallback.
   Epoch time: 31 s → 17.2 s.
2. **`Same`-padded convolutions on small feature maps** — the layer rejected
   kernels larger than the input even when padding makes the shape valid
   (ResNet's stage-4 3×3 convolutions on 2×2 maps); the check now applies
   only to unpadded convolutions. This is what made the architecture
   buildable at all.

Everything else was already in place from the MNIST work: convolutions and
their bias gradients on the cudnn-frontend API, fused conv+bias(+ReLU)
forward graphs, and the GPU-resident `ImageDataset` mode
(`OPENNN_GPU_RESIDENT_DATA=1`) that stages the 614 MB dataset once and
gathers batches device-side.

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

* The headline compares the **plain eager fp32 training loops** — what each
  framework executes out of the box. PyTorch's optimizing compiler changes
  the picture: `torch.compile` reaches **5,772 samples/s** (8.7 s/epoch) on
  this model — 2.8× its eager mode and 2× OpenNN
  ([`pt_compile_probe.py`](resnet50-training-speed/pt_compile_probe.py)
  measures it). OpenNN has no whole-graph fusion compiler; closing that gap
  would mean fusing BN+ReLU+add chains into the conv graphs, which the
  cudnn-frontend API supports but OpenNN does not yet exploit.
* Single consumer laptop GPU; ratios shift with hardware and input size — at
  224×224 the workload becomes conv-FLOP-bound, where both engines run the
  same cuDNN kernels and should converge.
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
