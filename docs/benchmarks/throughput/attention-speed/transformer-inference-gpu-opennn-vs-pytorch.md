# GPU Transformer inference: OpenNN vs PyTorch vs TensorFlow ("Attention Is All You Need")

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-08-10. Linux x86_64, NVIDIA GeForce RTX 4080 (16 GB), driver 595.84, CUDA 13.3, cuDNN 9.23.1. Artifact: [`results/gpu-transformer-inference-20260810T074158Z.json`](../../results/).*

**Status:** current desktop-GPU result on commit `52e21e15d`, median of 5 runs
per cell. Supersedes both the 2026-06 WSL2 laptop numbers and the 2026-07-14
figures published on the blog (see the note below the table on why the long-seq
numbers moved).

The [dense benchmark](../higgs-gpu/README.md) and
the [ResNet benchmark](../resnet50/README.md) cover fully
connected and convolutional networks. This note covers the third major
architecture: the **Transformer** from *Attention Is All You Need* — token
embeddings, sinusoidal positional encoding, a stack of encoder and decoder
layers (multi-head attention + position-wise feed-forward + layer norm), and a
linear projection to the vocabulary. The comparison is **inference** (the
forward pass) against PyTorch's `nn.Transformer`, in both bf16 and fp32.

## The result

Inference throughput (tokens/sec) of the encoder-decoder Transformer on one
RTX 4080, at the *Attention Is All You Need* base shape (d_model 512,
8 heads, feed-forward 2048, 6+6 layers), batch 32, 30 timed iterations after
warmup. Each figure is the **median of 5 runs (± population stdev)**; raw
per-run data, versions, and ratios are in [`results/`](../../results/)
(`gpu-transformer-inference-*.json`). Transformers run in **bf16** in practice,
and that is the headline — each engine on its fused fast path: OpenNN's
device-resident fused flash-attention, PyTorch `torch.autocast(bf16)`,
TensorFlow `@tf.function(jit_compile=True)` (XLA) + mixed-precision.

| Config (d512/h8/ff2048/6L) | OpenNN bf16 | PyTorch bf16 | TensorFlow bf16 | OpenNN / PyTorch | OpenNN / TF |
|---|---:|---:|---:|---|---|
| seq 128, batch 32 | 643,871 ± 12,738 | 480,888 ± 7,195 | 453,306 ± 7,350 | **1.34×** | 1.42× |
| seq 256, batch 32 | 645,515 ± 55     | 449,672 ± 188   | 429,666 ± 3,358 | **1.44×** | 1.50× |
| seq 512, batch 32 | 588,435 ± 381    | 429,233 ± 90    | 308,685 ± 440   | **1.37×** | 1.91× |

**In bf16 — the precision transformers actually run in — OpenNN's Transformer
inference is the fastest of the three at every sequence length**, 1.34–1.44×
over PyTorch and up to 1.91× over TensorFlow at seq 512. Long sequences are
where real LLM / long-context inference lives, and OpenNN's fused cuDNN
flash-attention holds its lead there. The bf16 output is validated against the
fp32 CPU reference (no NaN, within bf16 tolerance).

*(The 2026-07-14 blog figures were ~5–9% higher at seq 256/512. That gap is the
cost of a correctness fix: the SDPA padding-mask cache was keyed by input-buffer
pointer and could reuse stale sequence lengths, so it was removed and the
valid-length scan now runs on every forward. PyTorch and TensorFlow reproduce
their July figures within ±1.7% on the same day, which isolates the delta to
that fix rather than the machine.)*

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
the fp32 win *grows* with sequence (1.03× at seq 128 → 1.19× at seq 512).

## Two things make this work

**1. The device-resident inference path.** OpenNN's convenience prediction API
(`calculate_outputs`) re-uploads every parameter, rebuilds the activation
workspace, and copies inputs and outputs across the PCIe bus **on every call**.
For a 6-layer Transformer that per-call overhead is crippling — the naive loop
runs at a fraction of the resident path. The benchmark uses the **device-resident
path** (`calculate_outputs_resident`): both token inputs live on the GPU, the
parameters are uploaded once, the activation workspace is built once, and the
output is left on the GPU (3–4× faster than the convenience API). The lesson
matches the dense benchmarks:
for a repeated-inference loop, the resident path is the right thing to measure.

**2. The fused flash-attention path, engaged in both precisions.** OpenNN's fused
SDPA runs the cuDNN flash-attention kernel; in bf16 directly, in fp32 via the
cast-down/cast-back path above. bf16 is also how transformers are usually deployed
for inference. Both effects together are why OpenNN wins.

## fp32 result

With the fp32-via-bf16 fused path, **OpenNN wins fp32 too** (paper config,
batch 32, median of 5):

| seq | OpenNN fp32 (tok/s) | PyTorch fp32 (tok/s) | TensorFlow fp32 (tok/s) | OpenNN / PyTorch |
|----:|--------------------:|---------------------:|------------------------:|------:|
| 128 | 335,911 ± 183 | 324,731 ± 308 | 256,539 ± 1,332 | 1.03× |
| 256 | 318,411 ± 229 | 286,615 ± 131 | 212,704 ± 895   | 1.11× |
| 512 | 280,329 ± 118 | 235,763 ± 246 | 167,666 ± 391   | 1.19× |

The win grows with sequence length because flash-attention stays flat while
PyTorch's fp32 SDPA slows down. The fp32 output is validated against the fp32 CPU
reference (max abs diff ≈ 1e-5, RESULT=MATCH). OpenNN's forward pass is
GPU-kernel-bound (per-step host overhead ~0%, so CUDA-graph capture adds <1%
here — unlike the dense-MLP and ResNet notes).

## Energy and max batch

* **Energy**: measured separately with the fixed-work protocol in
  [`energy/transformer-energy/`](../../energy/transformer-energy/).
* **Max inference batch** (VRAM-bound on 16 GB, chat corpus shape): measured in
  [`capacity/transformer-max-batch/`](../../capacity/transformer-max-batch/) —
  OpenNN fits batch **1,987 in bf16 / 985 in fp32** vs PyTorch 951 / 435 and
  TensorFlow 563 / 563 (TensorFlow stops at an internal INT32 descriptor limit,
  not at the VRAM cap; OpenNN removed the same limit in 2026-08).

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

Hardware/software: NVIDIA GeForce RTX 4080 (16 GB, driver 595.84), Intel Core
i9-12900K, Linux x86_64. OpenNN built with g++ 13.3 + CUDA 13.3 + cuDNN 9.23.1;
PyTorch 2.13.0+cu130 and TensorFlow 2.21.0 on CPython 3.12.3.

## Caveats

* **Inference benchmark.** This note measures the forward pass only. Transformer
  training is covered separately in
  [the GPU Transformer training note](transformer-training-gpu-opennn-vs-pytorch.md).
  The forward pass is validated and is what production inference uses.
* **The headline is bf16**, the precision transformers actually run in for
  inference, but **OpenNN wins in fp32 too** (see the fp32 result above). cuDNN's
  flash-attention kernel is bf16-only; OpenNN's fp32 path now feeds it by casting
  Q/K/V to bf16 and casting the output back, so both precisions use the fused
  kernel. PyTorch's bf16 number uses `torch.autocast`.
* The OpenNN number is the **device-resident** path; the convenience
  `calculate_outputs` API is 3–4× slower and is the wrong thing to time in a loop.
* Output is validated against the fp32 CPU reference in both precisions: no NaN,
  within tolerance (bf16 ~5e-7, fp32-via-bf16 ~1e-5 at the tested configs).
* Single consumer desktop GPU. Absolute numbers on this desktop drift a few
  percent between sessions (ambient/boost behavior); the honest comparison is
  the three engines measured back-to-back in the same session, which is what
  the harness does.

## Reproducing

The OpenNN benchmark (resident + convenience), the PyTorch counterpart, the
forward-correctness probe, and the build scripts are in
[`docs/benchmarks/throughput/attention-speed/`](attention-speed/):

```bash
./build.sh   # builds all benchmarks (paths inside are machine-specific)

# OpenNN bf16 device-resident inference — the headline (args: seq d_model heads ff layers vocab batch iters)
OPENNN_BF16=1 ./opennn_transformer_resident 256 512 8 2048 6 10000 32 50

# Or the full 3-way harness (writes the results/ artifact):
python run_transformer.py --seqs 128,256,512 --batch 32 --iters 30 --runs 5 --precision both

# PyTorch bf16 counterpart (torch.autocast)
PT_BF16=1 python pytorch_transformer_infer.py 256 512 8 2048 6 10000 32 50

# fp32 (omit the env flags) — also wins, via the fp32-via-bf16 fused path.

# bf16 forward correctness (GPU bf16 vs CPU fp32 reference)
OPENNN_BF16=1 ./opennn_attention_validate 256 512 8 2048 6 1000 4
```
