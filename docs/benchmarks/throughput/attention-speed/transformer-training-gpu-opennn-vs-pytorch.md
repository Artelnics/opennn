# Transformer training on the GPU: OpenNN vs PyTorch vs TensorFlow

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-08-10. Linux x86_64, NVIDIA GeForce RTX 4080 (16 GB), driver 595.84, CUDA 13.3, cuDNN 9.23.1. Artifact: [`results/gpu-transformer-training-speed-20260810T075000Z.json`](../../results/).*

**Status:** current desktop-GPU result on commit `52e21e15d`. Supersedes the
2026-06 WSL2 laptop numbers and refreshes the 2026-07-10 blog figures (all
three engines measure faster on the current driver; the ratios hold).

This is the training counterpart to the
[transformer inference benchmark](transformer-inference-gpu-opennn-vs-pytorch.md).
Same architecture — the encoder-decoder **Transformer** from *Attention Is All
You Need* (token embeddings + sinusoidal positional encoding, N encoder + N
decoder layers of multi-head attention and position-wise feed-forward, a linear
projection to the vocabulary) — but here we measure the **training** step:
forward + backward + Adam optimizer update, against PyTorch's `nn.Transformer`
(autocast bf16 / TF32 fp32, fused Adam) and TensorFlow (mixed_bfloat16, XLA).

## The result

**OpenNN trains faster than both PyTorch and TensorFlow in both precisions**
(d_model 256, heads 8, feed-forward 1024, 2 encoder + 2 decoder layers,
vocab 256, seq 256, 4,096 samples, batch 32, 9 epochs):

| Precision | OpenNN (tok/s) | PyTorch (tok/s) | TensorFlow (tok/s) | OpenNN / PyTorch | OpenNN / TF |
|---|---:|---:|---:|---|---|
| bf16 | **2,827,190** | 2,528,316 | 2,287,020 | **1.12×** | 1.24× |
| fp32 | **1,724,350** | 1,019,754 |   962,684 | **1.69×** | 1.79× |

In samples/s: bf16 5,521.9 vs 4,938.1 vs 4,466.8; fp32 3,367.9 vs 1,991.7 vs
1,880.2. The fp32 gap is the striking one — OpenNN's fp32 path routes attention
through the same fused flash-attention kernel as bf16 (cast-down/cast-back),
while PyTorch and TensorFlow fall back to slower fp32 attention. Energy for
this family is measured separately with the fixed-work protocol in
[`energy/transformer-energy/`](../../energy/transformer-energy/).

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
moments, and activation workspaces resident on the GPU across the whole run and
does one untimed warmup `train()` pass before the timed region. **CUDA-graph
capture is deliberately OFF in this benchmark** (the driver never calls
`set_cuda_graph`): for the transformer's large GEMMs the launch overhead a
graph amortizes is already negligible (<1%), and the graph-epoch training path
has a known convergence caveat under investigation. That keeps the comparison
eager-fair — the steady state is forward+backward+update with no per-step host
round-trips, exactly what PyTorch's eager loop also does.

## Setup

| | Value |
|---|---|
| Network | encoder-decoder Transformer: scaled token embeddings + sinusoidal positional encoding → N encoder + N decoder layers (MHA + FFN, post-LayerNorm) → Linear to vocab |
| Shape | d_model 256, heads 8, feed-forward 1024, 2 encoder + 2 decoder layers, vocab 256, seq 256 |
| Data | synthetic tab-separated corpus (`make_synthetic_corpus.py`), 4,096 samples; PyTorch and TensorFlow read the SAME corpus to match sequence lengths / vocab / sample count token-for-token |
| Optimizer / loss | Adam (lr 1e-4) / token cross-entropy over the vocabulary |
| Precision | bf16 and fp32 (fused attention in both; fp32 via the fp32-via-bf16 path) |
| Protocol | warmup excluded; median samples/sec over 9 timed epochs, batch 32 |

Hardware/software: NVIDIA GeForce RTX 4080 (16 GB, driver 595.84), Intel Core
i9-12900K, Linux x86_64. OpenNN built with g++ 13.3 + CUDA 13.3 + cuDNN 9.23.1;
PyTorch 2.13.0+cu130 and TensorFlow 2.21.0 on CPython 3.12.3.

## Caveats

* **Fused attention in both precisions.** The fp32 win uses the fused
  flash-attention path in both forward and backward (the fp32-via-bf16 fix
  above) — that is where the 1.69×/1.79× fp32 lead comes from.
* **Throughput is the metric.** With Glorot initialization the loss behaves
  correctly (starts near ln(vocab), descends); absolute loss values still differ
  across frameworks because of independent random init and the synthetic data, but
  what is matched is the architecture, the per-step FLOPs, the optimizer, and the
  data shape. Convergence is confirmed by a decreasing loss on every engine.
* **Training runs at a fixed sequence length** (256). The inference note sweeps
  128/256/512; a training seq sweep was considered and skipped to keep the cell's
  runtime bounded.
* Single consumer desktop GPU; the honest comparison is the three engines
  measured back-to-back in the same session, which is what the harness does.

## Reproducing

The corpus generator, the OpenNN training driver, the PyTorch counterpart, and the
build script are in [`docs/benchmarks/throughput/attention-speed/`](attention-speed/):

```bash
# Full 3-way harness (generates the corpus if missing, writes the results/ artifact):
python run_transformer_train.py --batch 32 --epochs 9 --runs 1 --precision both

# Or one engine by hand (args: corpus d_model heads ff layers batch epochs):
./build.sh opennn_transformer_train
./opennn_transformer_train synthetic_corpus.txt 256 8 1024 2 32 9
python pytorch_transformer_train.py synthetic_corpus.txt 256 8 1024 2 32 9

# OPENNN_LR overrides the learning rate; OPENNN_BF16=1 / PT_BF16=1 / TF_BF16=1 train in bf16.
```
