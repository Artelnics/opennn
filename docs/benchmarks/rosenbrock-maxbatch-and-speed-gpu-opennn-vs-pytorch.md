# GPU max batch and speed on a dense MLP: OpenNN vs PyTorch (Rosenbrock)

**Important measurement caveat (WSL2 vs native Windows).** These numbers were
taken under WSL2. WSL2 was found to *specifically* degrade OpenNN's bf16
tensor-core GEMM path while leaving the framework baselines unaffected — so the
WSL bf16 dense numbers **understate OpenNN**. On native Windows (same RTX 3060,
CUDA 12.5 / cuDNN 9.20) OpenNN's bf16 dense inference runs **1.8× faster than its
own fp32** (8.3 M samples/s at batch 8000, hidden 1000) and matches TensorFlow's
bf16 — confirmed with Nsight Systems, which shows the work running in a
`cutlass_80_tensorop_bf16_s16816gemm_relu` tensor-core kernel with a fused ReLU
epilogue and a one-time input cast. **Re-measure on native Windows before using
any dense GPU number as an investor headline.** The fp32 figures below are less
affected; the bf16 gap is the WSL artifact.

The [ResNet-50 note](resnet50-training-speed-gpu-opennn-vs-pytorch.md) measures a
convolutional architecture, where the cost is cuDNN kernels and launch overhead.
This note asks the same head-to-head question on the opposite workload: a
**purely dense network** where the cost is cuBLAS GEMMs, and where a small batch
makes per-step overhead, not arithmetic, the thing under the microscope. It also
adds a second axis the convolutional notes do not: **how large a batch each
engine can fit on the same 6 GB card** — for inference and for training.

The network is the Rosenbrock regressor scaled up to stress the GPU: **1000
inputs → 1000 (tanh) → 1**, MSE loss, Adam, fp32 — the same protocol as the
[training-precision note](precision-opennn-vs-pytorch-vs-tensorflow.md), widened
from 10 inputs to 1000.

## The result

Four measurements on one RTX 3060 Laptop (6 GB), fp32, each engine run alone
with the GPU idle in between:

| Axis | OpenNN | PyTorch | OpenNN vs PyTorch |
|---|---:|---:|---|
| **Inference throughput** (batch ≥ 2000) | **3.8–4.4 M/s** | 2.5–2.8 M/s | **1.43–1.56×** |
| **Training throughput** (few steps/epoch) | **1.77 M/s** | 1.31 M/s | **1.35×** |
| **Max training batch** (VRAM-bound) | **482,344** | 399,507 | **1.21×** |
| **Max inference batch** (VRAM-bound) | 524,288 | 534,773 | 0.98× (tie) |

**OpenNN is faster at inference and at training, and fits a larger training
batch, on the same card.** The two speed numbers and the training-batch number
are wins; the inference-batch number is a tie. Each came from understanding a
specific piece of the GPU step — below is where each one was, because the
headline hides a subtlety on the training axis that is worth stating plainly.

## Inference: keep everything on the device (1.43–1.56×)

The naive way to time inference is to call the prediction entry point in a loop.
On OpenNN that entry point (`NeuralNetwork::calculate_outputs`) is built for a
one-shot prediction, not a hot loop: **every call** re-uploads the parameters to
the device, allocates a fresh activation workspace, copies the input
host→device, runs, and copies the output device→host. For a 1000→1000→1 forward
whose arithmetic is microseconds, those copies and allocations dominate — and
PyTorch, whose input tensor already lives on the GPU, pays none of them.

The fix is a device-resident inference path
(`NeuralNetwork::calculate_outputs_resident`): the caller owns a persistent
activation workspace, the input is already on the GPU, the parameters are
uploaded once, and the output is left on the GPU. The repeated forward then pays
only the kernels. That is **4–6.5× faster than the one-shot path** and lands
**1.43–1.56× ahead of PyTorch** for batches of 2000 and up (a tie at batch 512,
where launch latency dominates both):

| Batch | OpenNN one-shot | OpenNN resident | PyTorch | resident vs PyTorch |
|---|---:|---:|---:|---|
| 512 | 339 K | 2.19 M | 2.24 M | 0.98× |
| 2,000 | 742 K | 3.83 M | 2.49 M | **1.54×** |
| 8,000 | 870 K | 3.99 M | 2.80 M | **1.43×** |
| 32,000 | 1.02 M | 4.36 M | 2.79 M | **1.56×** |

## Training: the compute wins; the pipeline is the cost

The training axis carries the subtlety. Measured at a full 50 mini-batches per
epoch, OpenNN looks *slower* than PyTorch (≈0.6×). Decomposing the step shows
that is **not** an arithmetic deficit:

* OpenNN's three GEMMs per step (forward, weight-gradient, input-gradient) cost
  **3.69 ms** at batch 8000; PyTorch's same three cost **4.98 ms**. OpenNN's
  matmuls are *faster*.
* Yet OpenNN's whole step is 8.29 ms versus PyTorch's 6.06 ms. The ≈3.9 ms
  difference is **host-side per-step pipeline coordination** — the device-side
  batch gather and the cross-stream events that hand each batch from the
  prefetch worker to the compute stream — not GPU math.

That overhead is *per step*, so it compounds with the number of mini-batches.
Sweeping the batches-per-epoch makes it unmistakable:

| Mini-batches / epoch | OpenNN | vs PyTorch (1.31 M/s) |
|---|---:|---|
| 1 | 1.77 M/s | **1.35×** |
| 4 | 1.71 M/s | **1.30×** |
| 12 | 1.39 M/s | 1.06× |
| 50 | 0.90 M/s | 0.69× |

**When the per-step pipeline cost is amortized over real work, OpenNN's faster
kernels win by 1.30–1.35×.** The resident CUDA-graph mega-launch
(`OPENNN_CUDA_GRAPH=1`, the same mechanism that carried the ResNet note) recovers
about **+23 %** of the multi-batch case by bundling eight steps per launch, but
does not fully close it: the residual is the cross-stream gather coordination
that runs even inside the graph's host loop. Three candidate explanations were
implemented or probed and **ruled out** by measurement — cuBLASLt algorithm
selection (autotuning 16 timed candidates per shape: no change; the heuristic is
already near-optimal and both engines use TF32 tensor cores), graph group size
(8 → 25, eliminating all group boundaries and leftover steps: no change), and the
gather kernel itself (contiguous vs shuffled batches: identical). Fully removing
the residual would mean gathering *inside* the captured graph, reading the
resident dataset directly — a re-architecture left for future work.

## Max batch: measure against VRAM, not against system RAM

"Largest batch that fits" is a real axis on a 6 GB card, but it has a trap under
WSL2: the NVIDIA driver silently spills GPU allocations into **system RAM** once
VRAM fills, so a naive probe reports a batch far larger than the card and
measures host memory, not the GPU. Both engines were therefore capped to
physical VRAM (PyTorch with `set_per_process_memory_fraction`; OpenNN's spill was
confirmed and excluded by watching the VRAM plateau). Against the true 6 GB
ceiling, the largest batch that completes one training step (forward + backward +
Adam) is:

* **PyTorch: 399,507.** Holds one copy of the batch on the device.
* **OpenNN: 306,708 by default → 482,344 with `OPENNN_BATCH_POOL=1`.** OpenNN's
  prefetch pool holds three `Batch` objects by default — three device copies of
  the input — to overlap loading with compute. Dropping the pool to one copy
  fits **57 % more samples** (and beats PyTorch by 1.21×) at a ≈6 % throughput
  cost on GPU-resident data, where the "prefetch" is a cheap index gather. The
  default stays at three, which matters for disk-streamed pipelines (e.g. the
  ResNet note) where the overlap hides real I/O latency.

For inference-only (forward pass, no gradients or optimizer state) the two land
in a statistical tie at ≈525–535 K — both are then bounded by the same activation
footprint.

## Setup

| | Value |
|---|---|
| Network | 1000 → 1000 (tanh) → 1, dense |
| Loss / optimizer | MSE, Adam (lr 0.001), no regularization |
| Data | synthetic Rosenbrock-shaped tensors, GPU-resident in both engines |
| Precision | fp32, framework-default TF32 policy (TF32 tensor cores on both) |
| Protocol | warmup excluded; steady-state samples/s; max batch = largest one-step fit capped to physical VRAM |

Hardware/software: NVIDIA GeForce RTX 3060 Laptop GPU (6 GB, driver 555.85) under
WSL2 Ubuntu 24.04 on Windows 11 (i7-12700H). OpenNN built with g++ 13.3 + CUDA
12.9.86 + cuDNN 9.23; PyTorch 2.6.0 (cu124 wheels) on CPython 3.12.

## Caveats

* The training-throughput headline (1.30–1.35×) is at low mini-batches per
  epoch, where OpenNN's faster kernels are not masked by its per-step pipeline
  coordination; at many mini-batches per epoch the coordination overhead
  compounds and PyTorch leads until the mega-graph (`OPENNN_CUDA_GRAPH=1`) closes
  part of the gap. The honest statement is that OpenNN's *compute* is faster and
  its *data pipeline* is the cost — the article reports both and the sweep that
  separates them.
* OpenNN's inference win requires the device-resident path
  (`calculate_outputs_resident`); the convenience one-shot entry point is 4–6.5×
  slower because of per-call host↔device copies, and is the wrong thing to time
  in a loop.
* Max-batch numbers are VRAM-capped on purpose. Uncapped under WSL2 both engines
  spill into system RAM and report larger, meaningless batches; OpenNN's
  spill-on-overflow was observed directly. On native Linux without the spill the
  ceiling is the same physics.
* OpenNN's default training max batch (306,708) is below PyTorch's; the win
  requires `OPENNN_BATCH_POOL=1`, which trades ≈6 % throughput for the larger
  batch and is appropriate only when the data is GPU-resident.
* Single consumer laptop GPU under WSL2; the pipeline-coordination share is
  largest exactly here (high CUDA-API issue latency). On native Linux the
  multi-batch gap narrows. The library's GPU test-suite failure set is unchanged
  versus the pre-change baseline.

## Reproducing

The two OpenNN programs, the PyTorch counterparts, the max-batch search driver,
and hand-link build scripts are in
[`docs/benchmarks/rosenbrock-max-batch/`](rosenbrock-max-batch/) (see its
[`README.md`](rosenbrock-max-batch/README.md); the `build_*.sh` paths are
machine-specific — edit them for your tree):

```bash
# Inference speed (device-resident path) — args: batch iters inputs hidden
./build_resident.sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib ./opennn_rosenbrock_resident_infer 8000 500 1000 1000
python pytorch_rosenbrock_throughput.py inference 8000 500 1000 1000

# Training speed — args: mode samples batch iters inputs hidden
./build_tput.sh
OPENNN_GPU_RESIDENT_DATA=1 ./opennn_rosenbrock_throughput train 8000 8000 200 1000 1000   # 1 batch/epoch
OPENNN_GPU_RESIDENT_DATA=1 OPENNN_CUDA_GRAPH=1 ./opennn_rosenbrock_throughput train 400000 8000 20 1000 1000
python pytorch_rosenbrock_throughput.py train 8000 200 1000 1000

# Max batch (fresh process per trial; auto VRAM-bound) — args: inputs hidden
./build_trial.sh
./run_maxbatch.sh 1000 1000                    # OpenNN; OPENNN_BATCH_POOL=1 for the larger train batch
python pytorch_rosenbrock_maxbatch.py 1000 1000   # PyTorch, capped to physical VRAM
```
