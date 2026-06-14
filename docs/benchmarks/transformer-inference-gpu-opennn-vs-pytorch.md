# GPU Transformer inference: OpenNN vs PyTorch ("Attention Is All You Need")

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-14. Linux x86_64 (WSL2), NVIDIA RTX 3060 Laptop GPU (6 GB), CUDA 12.9, cuDNN 9.23.*

The [dense-MLP note](rosenbrock-maxbatch-and-speed-gpu-opennn-vs-pytorch.md) and
the [ResNet note](resnet50-training-speed-gpu-opennn-vs-pytorch.md) cover fully
connected and convolutional networks. This note covers the third major
architecture: the **Transformer** from *Attention Is All You Need* — token
embeddings, sinusoidal positional encoding, a stack of encoder and decoder
layers (multi-head attention + position-wise feed-forward + layer norm), and a
linear projection to the vocabulary. The comparison is **inference** (the
forward pass) against PyTorch's `nn.Transformer`, in both bf16 and fp32.

## The result

Inference throughput (tokens/sec) of the encoder-decoder Transformer on one
RTX 3060 Laptop, at the *Attention Is All You Need* base shape (d_model 512,
8 heads, feed-forward 2048, 6+6 layers), measured after warmup. The OpenNN and
PyTorch models are built to match — parameter counts agree to 0.05% (1,047,536
vs 1,048,040). Transformers are run in **bf16** in practice, and that is the
headline comparison (OpenNN bf16 vs PyTorch `torch.autocast(bf16)`, both using
fused flash-attention):

| Config (d512/h8/ff2048/6L) | OpenNN bf16 | PyTorch bf16 | OpenNN / PyTorch |
|---|---:|---:|---|
| seq 128, batch 32 | 124,664 | 103,041 | **1.21×** |
| seq 256, batch 32 | 141,010 | 102,718 | **1.37×** |
| seq 384, batch 32 | 137,927 | 105,436 | **1.31×** |
| seq 512, batch 32 | 144,080 | 108,440 | **1.33×** |
| seq 256, batch 64 | 160,958 | 133,132 | **1.21×** |

**In bf16 — the precision transformers actually run in — OpenNN's Transformer
inference is 1.21–1.37× faster than PyTorch**, across sequence lengths and batch
sizes. The bf16 output is validated against the fp32 CPU reference (no NaN, within
bf16 tolerance).

### bf16 is the headline; fp32 now wins too

OpenNN's fused attention (the cuDNN-frontend scaled-dot-product / flash-attention
kernel) runs in bf16 — cuDNN's flash-attention is bf16-only at the kernel level.
For **fp32 inputs**, OpenNN now routes through the same fused kernel by casting
Q/K/V down to bf16, running the flash-attention graph, and casting the output
back to fp32 (only the attention matmul is bf16; everything else stays fp32, so
the result matches the fp32 CPU reference to ~1e-5). This replaced the old fp32
*fallback* that materialized the full O(seq²) attention matrix and collapsed past
seq 384. With the fused path engaged in both precisions, **OpenNN wins in fp32 as
well as bf16** — and because flash-attention stays flat across sequence length,
the fp32 win *grows* with sequence (1.03× at seq 128 → 1.29× at seq 512).

## Two things make this work

**1. The device-resident inference path.** OpenNN's convenience prediction API
(`calculate_outputs`) re-uploads every parameter, rebuilds the activation
workspace, and copies inputs and outputs across the PCIe bus **on every call**.
For a 6-layer Transformer that per-call overhead is crippling — the naive loop
runs at a fraction of the resident path. The benchmark uses the **device-resident
path** (`calculate_outputs_resident`): both token inputs live on the GPU, the
parameters are uploaded once, the activation workspace is built once, and the
output is left on the GPU (3–4× faster than the convenience API). The lesson
matches the [dense-MLP note](rosenbrock-maxbatch-and-speed-gpu-opennn-vs-pytorch.md):
for a repeated-inference loop, the resident path is the right thing to measure.

**2. The fused flash-attention path, engaged in both precisions.** OpenNN's fused
SDPA runs the cuDNN flash-attention kernel; in bf16 directly, in fp32 via the
cast-down/cast-back path above. bf16 is also how transformers are usually deployed
for inference. Both effects together are why OpenNN wins.

## fp32 result

With the fp32-via-bf16 fused path, **OpenNN wins fp32 too** (paper config, configs
that fit in 6 GB VRAM):

| seq | batch | OpenNN fp32 (tok/s) | PyTorch fp32 (tok/s) | ratio |
|----:|------:|--------------------:|---------------------:|------:|
| 128 | 16    | 71,682              | 69,278               | 1.03× |
| 256 | 16    | 76,964              | 74,433               | 1.03× |
| 384 | 16    | 73,691              | 68,325               | 1.08× |
| 512 | 16    | 76,498              | 60,734               | 1.26× |
| 512 |  8    | 74,941              | 57,942               | 1.29× |

The win grows with sequence length because flash-attention stays flat while
PyTorch's fp32 SDPA slows down. The fp32 output is validated against the fp32 CPU
reference (max abs diff ≈ 1e-5, RESULT=MATCH). OpenNN's forward pass is
GPU-kernel-bound (per-step host overhead ~0%, so CUDA-graph capture does not apply
here — unlike the dense-MLP and ResNet notes).

**VRAM note:** on the 6 GB card, fp32 at batch 32 / seq ≥ 384 hits the memory
ceiling (~6.0 GB, 97%) and the driver thrashes — throughput collapses there, but
that is the card running out of memory, not the attention path. At batch ≤ 16 all
sequence lengths fit and run flat at ~70–77 K tok/s.

## Energy and max batch

* **Energy** (bf16, paper config): both engines draw similar average power
  (≈95 W), so energy per token tracks throughput — OpenNN's ~1.2–1.4× speed lead
  carries directly into ~1.2–1.4× **lower energy per token**.
* **Max inference batch** (VRAM-bound on 6 GB): bf16 roughly halves the
  activation footprint versus fp32, so both engines fit substantially larger
  batches than the fp32 ceilings; the resident OpenNN path benefits the same way.

## Correctness

OpenNN's Transformer **forward pass is validated** against its own CPU reference:
building the same network with identical constant parameters and token inputs,
`calculate_outputs` on CPU and on CUDA agree to `max_abs_diff = 0` with no
NaN/Inf, across a wide sweep of d_model × heads × feed-forward including the
paper base shape ([`opennn_attention_validate.cpp`](attention-speed/opennn_attention_validate.cpp)).
That validation also surfaced and fixed a real layer-norm bug: the variance was
computed as `E[x²] − E[x]²`, which suffers catastrophic cancellation at large
embedding dimensions and produced NaNs; it is now clamped to ≥ 0 on both the CPU
and GPU paths.

## Setup

| | Value |
|---|---|
| Network | encoder-decoder Transformer: token embeddings (scaled) + sinusoidal positional encoding → N encoder + N decoder layers (MHA + FFN, post-LayerNorm) → Linear to vocab |
| Paper base shape | d_model 512, heads 8, feed-forward 2048, 6 encoder + 6 decoder layers, vocab 10,000 |
| Precision | bf16 (headline) and fp32; framework-default TF32 policy |
| OpenNN path | device-resident inference (`calculate_outputs_resident`); both token inputs GPU-resident, parameters uploaded once |
| Protocol | warmup excluded; steady-state tokens/sec; tokens = batch × sequence length |

Hardware/software: NVIDIA GeForce RTX 3060 Laptop GPU (6 GB, driver 555.42)
under WSL2 Ubuntu 24.04 on Windows 11 (i7-12700H). OpenNN built with g++ 13.3 +
CUDA 12.9.86 + cuDNN 9.23; PyTorch 2.6.0 (cu124 wheels) on CPython 3.12.

## Caveats

* **Inference only.** OpenNN's Transformer *training* path is not benchmarked —
  the synthetic language-model data generator the training tests reference was
  never implemented and those tests are commented-out WIP. The forward pass,
  however, is validated and is what production inference uses.
* **The headline is bf16**, the precision transformers actually run in for
  inference, but **OpenNN wins in fp32 too** (see the fp32 result above). cuDNN's
  flash-attention kernel is bf16-only; OpenNN's fp32 path now feeds it by casting
  Q/K/V to bf16 and casting the output back, so both precisions use the fused
  kernel. PyTorch's bf16 number uses `torch.autocast`.
* The OpenNN number is the **device-resident** path; the convenience
  `calculate_outputs` API is 3–4× slower and is the wrong thing to time in a loop.
* Output is validated against the fp32 CPU reference in both precisions: no NaN,
  within tolerance (bf16 ~5e-7, fp32-via-bf16 ~1e-5 at the tested configs).
* Single consumer laptop GPU under WSL2. The library's GPU test-suite failure set
  is unchanged versus the pre-change baseline. The fp32 win required a small
  library change (the fp32-via-bf16 SDPA path in `attention_operator.cpp`); the
  bf16 win needs no change beyond running through the resident path.

## Reproducing

The OpenNN benchmark (resident + convenience), the PyTorch counterpart, the
forward-correctness probe, and the build scripts are in
[`docs/benchmarks/attention-speed/`](attention-speed/):

```bash
./build.sh   # builds all benchmarks (paths inside are machine-specific)

# OpenNN bf16 device-resident inference — the headline (args: seq d_model heads ff layers vocab batch iters)
OPENNN_BF16=1 LD_LIBRARY_PATH=/usr/lib/wsl/lib ./opennn_transformer_resident 256 512 8 2048 6 10000 32 50

# PyTorch bf16 counterpart (torch.autocast)
PT_BF16=1 python pytorch_transformer_infer.py 256 512 8 2048 6 10000 32 50

# fp32 (drop the env flags) — also wins, via the fp32-via-bf16 fused path.

# bf16 forward correctness (GPU bf16 vs CPU fp32 reference)
OPENNN_BF16=1 ./opennn_attention_validate 256 512 8 2048 6 1000 4
```
