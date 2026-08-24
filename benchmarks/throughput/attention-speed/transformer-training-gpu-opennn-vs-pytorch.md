# Transformer training on the GPU: OpenNN vs PyTorch vs TensorFlow

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-08-11. Linux x86_64, NVIDIA GeForce RTX 4080 (16 GB), driver 595.84, CUDA 13.3, cuDNN 9.23.1. PyTorch/TensorFlow cells: [`results/gpu-transformer-training-speed-20260810T075000Z.json`](../../results/). OpenNN cells re-measured 2026-08-11 (single-train driver + padding-scan kernel fix, below); formal multi-run refresh pending.*

**Status:** current desktop-GPU result (OpenNN cells at the 2026-08-11 checkout,
after the padding-scan kernel fix). Supersedes the
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
| bf16 | **3,025,608** | 2,528,316 | 2,287,020 | **1.20×** | 1.32× |
| fp32 | **1,819,822** | 1,019,754 |   962,684 | **1.78×** | 1.89× |

In samples/s: bf16 5,909.4 vs 4,938.1 vs 4,466.8; fp32 3,554.3 vs 1,991.7 vs
1,880.2. The fp32 gap is the striking one — OpenNN's fp32 path routes attention
through the same fused flash-attention kernel as bf16 (cast-down/cast-back),
while PyTorch and TensorFlow fall back to slower fp32 attention. Energy for
this family is measured separately with the fixed-work protocol in
[`energy/transformer-energy/`](../../energy/transformer-energy/).

## Why the bf16 margin is smaller than the fp32 margin

A natural question: if OpenNN wins fp32 by 1.78×, why "only" 1.20× in bf16?
Because the two margins measure different things. An nsys kernel profile of the
OpenNN step shows GEMMs (2.4 ms) plus fused flash attention (1.3 ms) account
for two thirds of the 6 ms step — both already on tensor cores, both at the
same kernels PyTorch uses. In **bf16 every engine runs fused attention**, so
the comparison is pure like-for-like efficiency and the honest margin is 1.20×.
In **fp32 only OpenNN routes attention through the fused bf16 kernel**
(cast-down/cast-back); PyTorch and TensorFlow execute unfused fp32 attention,
which roughly doubles their step time. The extra fp32 margin is OpenNN's
fp32-via-bf16 design win, not extra kernel efficiency. The profile confirms the
hardware side: OpenNN's fp32 (TF32) GEMM time is 1.95× its bf16 GEMM time —
exactly the tensor-core throughput ratio of the RTX 4080.

## A kernel fix found by the profile: the padding scan

The same profile flagged one non-GEMM hotspot: the padding-length scan that
feeds cuDNN's ragged-sequence flash attention. Each attention op re-derives
per-sample sequence lengths by scanning its source stream for the first
all-zero (padded) token — 6 scans per step. The old kernel walked tokens
**serially** (one `__syncthreads` round per token, one thread block per
sample), costing ~53 µs per scan, ~5% of the whole training step. The rewrite
computes the same quantity — the index of the first all-padding token — as a
parallel min-reduction (warps stride over tokens, lanes over the embedding),
bit-identical semantics, ~10× faster. That single kernel is worth +5% bf16 and
+3% fp32 end-to-end in this benchmark.

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
moments, and activation workspaces resident on the GPU across the whole run.
The driver runs a **single `train()`** of one untimed warmup epoch plus nine
timed epochs and reports the median per-epoch throughput via
`post_epoch_callback` — so graph capture and optimizer setup are paid outside
the timed window, the same place PyTorch and TensorFlow pay their `compile`/XLA
warmup. **CUDA-graph capture is ON by default** (set
`OPENNN_TRANSFORMER_TRAIN_NO_GRAPH=1` to disable): it contributes ~5% at these
step sizes, and its numerics are equivalent to eager (final loss within the
run-to-run band). PyTorch runs autocast bf16 with fused Adam; TensorFlow runs
`mixed_bfloat16` with XLA.

## Setup

| | Value |
|---|---|
| Network | encoder-decoder Transformer: scaled token embeddings + sinusoidal positional encoding → N encoder + N decoder layers (MHA + FFN, post-LayerNorm) → Linear to vocab |
| Shape | d_model 256, heads 8, feed-forward 1024, 2 encoder + 2 decoder layers, vocab 256, seq 256 |
| Data | synthetic tab-separated corpus (`make_synthetic_corpus.py`), 4,096 samples; PyTorch and TensorFlow read the SAME corpus to match sequence lengths / vocab / sample count token-for-token |
| Optimizer / loss | Adam (lr 1e-4) / token cross-entropy over the vocabulary |
| Precision | bf16 and fp32 (fused attention in both; fp32 via the fp32-via-bf16 path) |
| Protocol | single `train()`, 1 warmup epoch excluded; median per-epoch samples/sec over 9 timed epochs, batch 32; CUDA graph ON (`OPENNN_TRANSFORMER_TRAIN_NO_GRAPH=1` disables) |

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
build script are in [`benchmarks/throughput/attention-speed/`](attention-speed/):

```bash
# Full 3-way harness (generates the corpus if missing, writes the results/ artifact):
python run_transformer_train.py --batch 32 --epochs 9 --runs 1 --precision both

# Or one engine by hand (args: corpus d_model heads ff layers batch epochs):
./build.sh opennn_transformer_train
./opennn_transformer_train synthetic_corpus.txt 256 8 1024 2 32 9
python pytorch_transformer_train.py synthetic_corpus.txt 256 8 1024 2 32 9

# OPENNN_LR overrides the learning rate; OPENNN_BF16=1 / PT_BF16=1 / TF_BF16=1 train in bf16.
```
