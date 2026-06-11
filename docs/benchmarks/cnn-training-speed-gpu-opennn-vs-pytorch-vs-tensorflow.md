# GPU CNN training speed: OpenNN vs PyTorch vs TensorFlow (MNIST)

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-11. Linux x86_64 (WSL2), NVIDIA RTX 3060 Laptop GPU, CUDA 12.9, cuDNN 9.23.*

The other notes in this series measure footprint and CPU behavior. This one
measures **GPU training throughput on a convolutional network**: the simplest
CNN that exercises the convolution, pooling, and dense paths — one
convolutional and one pooling layer — trained on the repository's MNIST
images with the same configuration in all three frameworks.

## The result

Training throughput (samples/second, higher is better; epoch wall time in
parentheses), 40 timed epochs after a 2-epoch warmup:

| | Batch 128 | Batch 1,024 |
|---|---:|---:|
| **OpenNN** (CUDA, fp32, GPU-resident data) | **195,000–212,000** (0.047 s) | **397,259** (0.025 s) |
| **PyTorch** (eager, fp32, GPU-resident data) | 111,241 (0.090 s) | 298,947 (0.034 s) |
| **TensorFlow** (Keras `fit`, fp32) | 55,339 (0.181 s) | 134,307 (0.075 s) |

OpenNN's batch-128 figure is a range over repeated runs (laptop GPU thermal
variance); the other rows showed < ±5% spread.

* OpenNN trains this CNN **1.9× faster than PyTorch** at the everyday batch
  size (128) and **1.3× faster** at batch 1,024 — and 2.9–3.8× faster than
  TensorFlow.
* The training is real, not a no-op loop: the runs end at cross-entropy
  ≈ 0.10 after 12 epochs (from 1.14 at initialization), identical between
  OpenNN's fast path and its legacy path to four decimals.

A model this small (≈31k parameters, 28×28 inputs) measures the framework's
**per-step pipeline cost** — batch assembly, host↔device traffic, kernel
launches, optimizer update — more than raw conv FLOPS. That is the regime
real small-CNN workloads live in, and it is where a native C++ loop can beat
the Python frameworks outright.

## How OpenNN got here

The first run of this benchmark had OpenNN at 103k/132k samples/s — behind
PyTorch at both sizes. Profiling (`OPENNN_PROFILE=1`) attributed the gap to
three specific costs, each now fixed in the library:

1. **NHWC bias gradients.** `cudnnConvolutionBackwardBias` took 2.2 ms/step
   on NHWC deltas — 69% of the whole backward pass, for a 16-float reduction.
   The bias gradient is now a cudnn-frontend reduction graph (~0.1 ms).
2. **Legacy cuDNN v7 convolution API.** The convolution forward/wgrad/dgrad
   now run through the cudnn-frontend graph API (the engine interface
   PyTorch uses), with conv + bias + ReLU fused into a single forward graph
   when the layer requests it. The v7 path is kept as an automatic fallback
   (`OPENNN_CONV_LEGACY=1` forces it).
3. **Host-side batch assembly.** Each batch was assembled by re-reading the
   image cache (one `pread` per image), casting, scaling, and copying to the
   GPU — 1.7 ms per 128-image batch, more than the GPU math it fed.
   `ImageDataset` now supports the library's GPU-resident dataset mode
   (`OPENNN_GPU_RESIDENT_DATA=1`): the scaled pixels and one-hot targets are
   staged once and batches are gathered device-side, the same arrangement the
   PyTorch script uses. MNIST resident is 31 MB.

With those fixed, the per-step pipeline that remains (native loop, device
gather, cuDNN engines, fused Adam) is simply leaner than eager PyTorch's
Python dispatch — which is the whole-step overhead this benchmark measures.

## Setup

Identical in all three frameworks:

| | Value |
|---|---|
| Data | the repo's MNIST subset: 10,000 BMPs, 28×28×1, 10 classes (`examples/mnist/data`) |
| Network | Conv 16@3×3 (stride 1, Same, ReLU) → MaxPool 2×2 (stride 2) → Flatten → Dense 10 (softmax) |
| Loss / optimizer | cross-entropy, Adam (lr 0.001), no regularization |
| Pixels | scaled to [0, 1] once before the timed loop |
| Data residency | GPU-resident for OpenNN and PyTorch; TensorFlow streams from host NumPy via `fit` |
| Protocol | shuffled epochs, 2 warmup epochs (cuDNN autotune, allocator), 40 timed epochs |
| Precision | fp32 end-to-end, each framework's default TF32 policy |
| Metric | OpenNN: mean epoch time over the timed run; PyTorch/TensorFlow: median epoch time |

Hardware/software: NVIDIA GeForce RTX 3060 Laptop GPU (driver 555.85) under
WSL2 Ubuntu 24.04 on Windows 11 (i7-12700H). OpenNN built with g++ 13.3 +
CUDA 12.9.86 + cuDNN 9.23; PyTorch 2.6.0 (cu124 wheels), TensorFlow 2.21.0
on CPython 3.12. The Windows driver caps the stack at the CUDA 12.x line;
newer 12.x runtimes work through CUDA minor-version compatibility.

## Caveats

* Single consumer laptop GPU; ratios will differ on datacenter parts and at
  other model sizes.
* OpenNN's GPU-resident mode requires the dataset to fit in VRAM (it stages
  scaled inputs + one-hot targets as one float matrix) and is off by default;
  without it OpenNN measures 114k/165k samples/s — still ahead of
  host-streamed PyTorch (92k at batch 128).
* `torch.compile`, XLA, and OpenNN's bf16 path would all raise their
  respective numbers; this note deliberately compares the plain fp32 training
  loops.
* OpenNN's 40-epoch mean includes its one-time residency staging (~0.15 s)
  amortized across the run; PyTorch's tensors are staged outside its timed
  region.

## Reproducing

The data prep, the three training programs, the probe used in the analysis,
and the runner are in [`docs/benchmarks/cnn-training-speed/`](cnn-training-speed/):

```bash
python prepare_mnist.py ../../../examples/mnist/data   # BMP folders -> npy
./run_cnn_speed.sh 40 128                              # all three engines + summary
./run_cnn_speed.sh 40 1024
# or individually:
OPENNN_GPU_RESIDENT_DATA=1 ./opennn_cnn_speed <data_path> [epochs] [batch] [fp32|bf16]
python pytorch_cnn_speed.py [epochs] [batch]
python tensorflow_cnn_speed.py [epochs] [batch]
```
