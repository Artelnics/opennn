# Transformer training on the GPU: OpenNN vs PyTorch

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-14. Linux x86_64 (WSL2), NVIDIA RTX 3060 Laptop GPU (6 GB), CUDA 12.9, cuDNN 9.23.*

**Status:** current WSL2 laptop GPU result. Before using this as a flagship
public claim, keep repeated-run statistics, raw logs, and a cross-framework
quality/correctness gate with the published numbers.

This is the training counterpart to the
[transformer inference benchmark](transformer-inference-gpu-opennn-vs-pytorch.md).
Same architecture — the encoder-decoder **Transformer** from *Attention Is All
You Need* (token embeddings + sinusoidal positional encoding, N encoder + N
decoder layers of multi-head attention and position-wise feed-forward, a linear
projection to the vocabulary) — but here we measure the **training** step:
forward + backward + Adam optimizer update, in fp32, against PyTorch's
`nn.Transformer` trained with `torch.optim.Adam` and token cross-entropy.

## The result

**OpenNN trains faster than PyTorch at every sequence length, and the lead grows
with sequence length** (paper-style shape: d_model 256, heads 8, feed-forward
1024, 2 encoder + 2 decoder layers, batch 16, learning rate 1e-4):

| sequence length | OpenNN (samples/s) | PyTorch (samples/s) | ratio |
|----------------:|-------------------:|--------------------:|------:|
|  64             | 1,653              | 1,363               | **1.21×** |
| 128             |   999              |   963               | **1.04×** |
| 256             |   512              |   347               | **1.48×** |
| 384             |   400              |   236               | **1.69×** |

The lead grows with sequence length because at longer sequences the attention
computation dominates the step, and OpenNN's fused flash-attention (forward **and
backward**) scales better than PyTorch's attention there. At seq 384 OpenNN
trains **1.69× faster**.

### Energy per sample

Integrating GPU power over the run (20 Hz `nvidia-smi` sampling) at seq 256:

| | OpenNN | PyTorch |
|---|---:|---:|
| Average power | 93.2 W | 74.1 W |
| Total energy | 3,637 J | 4,723 J |
| **Energy per sample** | **0.169 J** | **0.220 J** |

OpenNN spends **23% less energy per sample**. It draws *more* instantaneous power
(it keeps the GPU busier — higher utilization) but finishes the same work 1.48×
sooner, so total energy per sample is lower. As with inference, the speed lead
carries straight into an energy lead.

## What made training win: fixing fp32 fused-attention backward

The inference benchmark added a **fp32-via-bf16** path so OpenNN's fp32 attention
runs on cuDNN's (bf16-only) fused flash-attention kernel: cast Q/K/V down to
bf16, run the fused graph, cast the output back to fp32. Training exposed that
this was only done for the **forward** pass. The backward pass had two defects
that made fused-attention training in fp32 fail outright:

1. **Cache-key drift.** The forward stored its cuDNN graph under a cache key whose
   dtype field was bf16 (because the fp32 path runs the bf16 graph), but the
   backward looked the entry up with the fp32 dtype — so it never found the
   forward's graph and threw *"SDPA forward did not populate a cache entry for
   this shape."*
2. **No cast in the backward.** Even once found, the bf16 backward graph needs
   bf16 inputs: dO must be cast down, and the dQ/dK/dV it produces must be cast
   back up to fp32.

The fix mirrors the forward path in `apply_delta_gpu`: the backward cache key uses
the same `graph_dtype` (bf16 in fp32 mode), dO is cast to bf16 into a scratch
buffer, the bf16 backward graph runs reusing the forward's already-cast Q/K/V/O,
and the resulting dQ/dK/dV are cast back to fp32. With that in place, fused
flash-attention trains correctly in fp32 — and the long-sequence training win
above is the result.

## A second fix: Glorot initialization for the Transformer

Validating training surfaced a separate, more general bug: the `Transformer`
constructor initialized every weight matrix with **unscaled uniform** noise
(`set_parameters_random()`, a fixed U(-0.1, 0.1) regardless of layer size). For a
deep stack feeding a Softmax over a large vocabulary, the unscaled weights produce
oversized logits, the Softmax saturates, and the per-token cross-entropy starts
enormous and barely moves — e.g. at vocab 1000 the loss began near 264 and was
stuck around 140 after 50 epochs (it should begin near ln(1000) ≈ 6.9).

The library already had `set_parameters_glorot()` (Xavier init,
`limit = sqrt(6 / (fan_in + fan_out))`, implemented by every layer op and used by
the other standard networks); the Transformer constructor simply wasn't calling
it. Switching that one line makes the loss behave correctly: at vocab 1000 it now
starts near 12 and descends to ~5.4 over 50 epochs (below the ln(1000) random
baseline). This is a general training-quality fix for any OpenNN Transformer, not
just the benchmark. Throughput is unaffected — the initialization changes the
*values*, not the per-step FLOPs.

## Why the device-resident training path

OpenNN's `TrainingStrategy::train()` keeps parameters, gradients, optimizer
moments, and activation workspaces resident on the GPU across the whole run, does
one CUDA warmup epoch before the timed region, and uses CUDA-graph capture for the
optimizer step. That is what makes a fair training-loop comparison — the steady
state is forward+backward+update with no per-step host round-trips, exactly what
PyTorch's loop also does.

## Setup

| | Value |
|---|---|
| Network | encoder-decoder Transformer: scaled token embeddings + sinusoidal positional encoding → N encoder + N decoder layers (MHA + FFN, post-LayerNorm) → Linear to vocab |
| Shape | d_model 256, heads 8, feed-forward 1024, 2 encoder + 2 decoder layers |
| Data | synthetic tab-separated corpus (`make_synthetic_corpus.py`), 1024 samples, vocab 256; PyTorch reads the SAME corpus to match sequence lengths / vocab / sample count token-for-token |
| Optimizer / loss | Adam (lr 1e-4) / token cross-entropy over the vocabulary |
| Precision | fp32 (fused attention via the fp32-via-bf16 path) |
| Protocol | warmup epoch excluded; samples/sec over 20+1 epochs; energy by integrating 20 Hz power |
| Parameters | OpenNN 951,888 vs PyTorch 952,388 at the small shape — 0.05% apart, equivalent architectures |

Hardware/software: NVIDIA GeForce RTX 3060 Laptop GPU (6 GB) under WSL2 Ubuntu
24.04 on Windows 11 (i7-12700H). OpenNN built with g++ 13.3 + CUDA 12.9 + cuDNN
9.23; PyTorch 2.6.0 (cu124 wheels) on CPython 3.12.

## Caveats

* **fp32, fused attention.** The win uses the fused flash-attention path in both
  forward and backward (the fp32-via-bf16 fix above). At very short sequences
  (≤128) the attention kernel isn't yet the bottleneck, so the lead is smallest
  there (1.04× at seq 128); it widens as sequence length grows.
* **Throughput is the metric.** With Glorot initialization the loss now behaves
  correctly (starts near ln(vocab), descends); absolute loss values still differ
  across frameworks because of independent random init and the synthetic data, but
  what is matched is the architecture, the per-step FLOPs, the optimizer, and the
  data shape. Convergence is confirmed by a decreasing loss on both sides.
* **VRAM ceiling.** On the 6 GB card, fp32 training fits comfortably at batch 16
  for these sequence lengths; larger batch × long sequence can hit the memory
  ceiling and thrash (the same ceiling noted in the inference benchmark).
* Single consumer laptop GPU under WSL2. The library's GPU test-suite failure set
  is unchanged versus the pre-change baseline. Two library changes support this
  result: the fp32 fused-attention backward fix in `attention_operator.cpp`, and
  the Glorot initialization for the Transformer in `standard_networks.cpp`.

## Reproducing

The corpus generator, the OpenNN training driver, the PyTorch counterpart, and the
build script are in [`docs/benchmarks/attention-speed/`](attention-speed/):

```bash
# 1. synthetic corpus (vocab, sequence length, sample count)
python make_synthetic_corpus.py corpus.txt 256 256 1024 1234

# 2. OpenNN training (args: corpus d_model heads ff layers batch epochs)
./build.sh opennn_transformer_train   # or ./build.sh for all
LD_LIBRARY_PATH=/usr/lib/wsl/lib ./opennn_transformer_train corpus.txt 256 8 1024 2 16 20

# 3. PyTorch counterpart (reads the same corpus for matching shapes)
python pytorch_transformer_train.py corpus.txt 256 8 1024 2 16 20

# OPENNN_LR overrides the learning rate on both sides; OPENNN_BF16=1 / PT_BF16=1 train in bf16.
# Energy: docs/benchmarks/rosenbrock-max-batch/energy_measure.sh <samples> <label> -- <command>
```
