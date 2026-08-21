# Transformer training: throughput at every batch size (working note)

*Working note for the transformer family of the peak-batch benchmark
(`docs/benchmarks/throughput/peak-batch-speed/run_peak_batch_speed.py --family transformer`),
2026-08-17, on the laptop (RTX 3060 Laptop 6 GB, WSL2, CUDA 12.9, cuDNN 9.24,
PyTorch 2.13.0+cu130, TensorFlow 2.21.0). Same method as
[`../resnet50/resnet50-large-batch-plan.md`](../resnet50/resnet50-large-batch-plan.md):
the three engines under each one's best protocol, the OpenNN step profiled,
one lever at a time, only what measures is kept. The desktop-GPU product note
stays [`transformer-training-gpu-opennn-vs-pytorch.md`](transformer-training-gpu-opennn-vs-pytorch.md);
this one is the trajectory and the reasoning.*

## Setup

Same network as the product note: encoder-decoder Transformer, d_model 256,
8 heads, feed-forward 1024, 2 encoder + 2 decoder layers, vocab 256 (260 with
the specials), sequence 256/256, 4,096 synthetic samples, Adam 1e-4, token
cross-entropy; ladder 32, 64, ..., 4096, 5 timed epochs per point, one fresh
process per point, the first OOM/error ends an engine's ascent. Fused
attention in both precisions (bf16 native; fp32 through the fp32-via-bf16
path).

Three protocol points fixed first:

* **The corpus.** `synthetic_corpus.txt` is generated locally and gitignored;
  the copy on this machine (Jul 25) was vocab 4,000 / sequence 128, not the
  documented configuration. Regenerated with
  `make_synthetic_corpus.py synthetic_corpus.txt 256 256 4096` (what the
  training harness generates when the file is missing). Everything below is
  on that corpus; the first look on the old one is in the appendix.
* **Where the corpus lives.** OpenNN reads its int32 token cache per batch on
  the prefetch workers; with the repository on OneDrive under `/mnt/c` every
  read is a 9p round trip and a 256-sample batch took ~30 ms to fill, which
  the epoch-start pipeline drain turns into a 9-15% loss (bf16 128: 1,771 vs
  1,928 samples/s with the same corpus on the WSL ext4 disk). PyTorch and
  TensorFlow hold random tokens on the device and read nothing. The runner
  takes `TRANSFORMER_CORPUS=<absolute path>` for that; the numbers below use
  `~/opennn-benchmark-data/transformer/synthetic_corpus.txt`. (Native-disk
  machines such as the RTX 4080 box are unaffected; the durable fix is a
  device-resident token dataset, noted under Next.)
* **PyTorch's best configuration.** As for ResNet, the sweep runs PyTorch with
  `torch.compile(mode="reduce-overhead")` (CUDA graphs) and `Adam(fused=True)`
  (`PT_COMPILE_MODE`, `PT_FUSED_ADAM` in `pytorch_transformer_train.py`;
  `PYTORCH_PLAIN=1` reverts). On this GPU that is worth 1.67x to PyTorch at
  batch 32 (eager 1,518 -> 2,538 samples/s on the old corpus). TensorFlow
  already runs the whole step under XLA.

## Finding 1: the OpenNN step was not being captured into a CUDA graph

The training driver requests CUDA graphs and reported nothing, but an nsys API
trace of the baseline binary (19f8287c8) showed one `cudaStreamBeginCapture`,
one `cudaStreamEndCapture` returning 901, and 215 `cudaLaunchKernel` per step
for the whole run: the capture failed on the first step and the run trained
eagerly. Cause: since b2d5cf34d (2026-08-14) the Transformer's embeddings
export per-sample valid lengths, and `compute_token_valid_lengths` produced
them on the host - a device-to-host copy plus `cudaStreamSynchronize` inside
the step (`embedding_lookup_operator.cpp`), which is illegal during capture
(error 900) and, graphs or not, a hard host sync twice per step. The SDPA
graphs then took the lengths from a pinned staging ring, which under a graph
would have replayed whatever the ring held at capture time.

Fix: on CUDA the record is device-resident. `token_valid_lengths_cuda`
(`kernel_embedding.cu`, one warp per sample) writes one int32 per sample into
a `ForwardPropagation` buffer owned per layer (`device_valid_lengths`,
`input_device_valid_lengths`; the `SequenceLengths` record carries the host
vector for CPU runs and the device pointer for CUDA runs); the SDPA graphs
fill their two length tensors from it with `attention_sdpa_lengths_cuda`
(query side full, key side clamped to [1, S] as before), the unfused
length-masked softmax and the sequence pooling take the pointer directly, and
`SequenceLengthStaging` is gone. Guarded by
`AdaptiveMomentEstimationTest.CudaGraphCapturesTransformerStep{SdpaBf16,UnfusedFp32}`
(a small encoder-decoder trained through one grouped graph step, asserting
`Optimizer::get_cuda_graph_capture_failed()` is false); the driver prints
`cuda_graph=captured|failed|off`. Padding semantics unchanged
(`SdpaAttentionMatchesUnfusedOnExportedValidLengths`,
`SdpaMatchesUnfusedThroughAveragePoolingOnPaddedBatches`,
`SdpaAttentionRefreshesPaddingBetweenBatches`, the Pool3d tests).

What the graph is worth here: little. Same binary, graph on vs off, bf16
batch 128: 1,616 / 1,527 vs 1,672 / 1,630; batch 32: 1,430 / 1,340 vs
1,385 / 1,331 - inside the laptop's thermal noise. At these shapes the step is
GPU-bound already at batch 32 (~20 ms for ~200 launches), so the host was
never the limiter; the value of the fix is correctness (no per-step host sync,
no stale-length hazard) and the graph itself (2,480 nodes per 8-step group at
batch 128).

## Profile and levers

nsys cannot trace kernels in this WSL setup (API only), so the breakdown is
`OPENNN_PROFILE=1` with graphs off (per-layer and per-op scopes synchronised
on both sides). bf16, batch 128, per step, before the levers (81.5 ms):

| part | ms/step | share |
|---|---:|---:|
| MultiHeadAttention backward (6 layers) | 30.6 | 38% |
| Dense backward (9 layers) | 20.1 | 25% |
| MultiHeadAttention forward | 12.2 | 15% |
| Dense forward | 6.9 | 8% |
| LayerNorm forward + backward (10) | 7.4 | 9% |
| ReLU backward (unfused, 4) | 2.7 | 3% |
| residual delta copies + adds | ~2.4 | 3% |
| embedding, Adam, loss | ~1.8 | 2% |

From that: the GEMMs run at 17-22 TFLOPS (cuBLASLt heuristics; PyTorch's
`a @ w` on the same shapes on this GPU: 22-27 TFLOPS), the head transposes
(`swap_heads`, 8 per attention layer) cost ~6 ms, LayerNorm ran at 100-150
GB/s against ~330 GB/s of bandwidth.

Levers, each measured against the previous binary at bf16 128 (5 timed
epochs; corpus still on `/mnt/c` for these pairs, so the absolute numbers are
~8% under the ladder below):

| lever | what | bf16 128 | note |
|---|---|---:|---|
| baseline 19f8287c8 (eager) | | 1,655 | capture failed |
| A. device-resident valid lengths | the step captures | 1,601 | +3.7% at 32, -3% at 128: neutral within noise |
| B. interleaved heads (BSHD) | projections write (B,S,H,D) straight into the head slots, the SDPA graphs read and write it through strides; no `swap_heads` in the SDPA path (8 transposes per layer per step). `MultiHeadAttention::apply_sdpa_choice` keeps the attention and projection operators in step; the unfused path keeps (B,H,S,D) | 1,733 | **+8%** |
| C. warp-per-row LayerNorm | forward one warp per row with 16-byte vectors and the residual add fused; backward one pass producing dX and the dgamma/dbeta partials ([blocks][2][D], deterministic) plus a finalize kernel; block kernels kept for other widths (`NormPartials` workspace) | 1,786 | **+3%**; LN now at bandwidth (0.23 / 0.20 ms per call) |
| D. LN backward stores dX twice | the fused add + norm's second input delta written by the kernel instead of a copy pass | 1,780 | neutral (0.5 ms of a 74 ms step); kept, it is fewer passes |
| E. DReLU epilogue in bf16 | `OPENNN_DRELU_FUSION=1` allowed for BF16 (was FP32-only); its mask validation compared against the wrong tensor (fixed - `DenseDreluFusedGradient` passes again) | 1,650 vs 1,780 | **-7%**: RELU_AUX_BIAS / DRELU make cuBLASLt pick slower kernels; stays opt-in and off |
| F. SDPA autotune | `OPENNN_SDPA_AUTOTUNE=1`: A+B heuristic lists, top-K timed on real tensors | 1,810 / 1,754 vs 1,778 / 1,778 | neutral: cuDNN 9.24 has one engine for this shape here; kept opt-in for GPUs with more |
| G. no padding mask (experiment, removed) | | 1,778 / 1,679 vs 1,755 / 1,720 | neutral: the mask costs nothing measurable |

After B-D the step at bf16 128 is ~74 ms: attention (cuDNN SDPA) 22 ms
(forward 0.86 ms and backward 2.8 ms per layer), projections 13 ms, the other
GEMMs 28 ms, LayerNorm 4.4, ReLU backward 2.7, the rest ~3.

Second round (commit after 82bcfb204), each pair alternated on the ext4
corpus, bf16 128 / 32:

| lever | what | bf16 128 | bf16 32 |
|---|---|---:|---:|
| H. cuBLASLt autotune | the heuristics' top-8 candidates timed on the first real call of each GEMM shape (`autotune_lt_plan`, `OPENNN_LT_AUTOTUNE_CANDIDATES`, 1 = heuristic choice) | 1,802 / 1,768 vs 1,749 / 1,742: **+2.2%** | 1,619 / 1,596 vs 1,570 / 1,566: **+2.5%** |
| I. residual add folded into the sublayer's dgrad | the planner's `input_delta_addend` (a fan-out's other consumer delta) is cuBLASLt's C with beta = 1 in the sublayer's input-delta GEMM (`linear_backward(..., addend)`; Dense and the attention layer's first-writing projection fold it, `folds_input_delta_addend`); the accumulate pass over those edges is gone | 1,824 / 1,790 vs 1,772 / 1,758: **+2.3%** | +1% (noisy) |
| J. cross-entropy 3d metrics in one pass | one warp per token, coalesced argmax, loss / active / hit sums block-reduced into the metrics buffer (`cross_entropy_3d_metrics_cuda`); replaces the per-token arrays and three `cublasSasum` | neutral (~0.3 ms) | neutral |
| single Adam kernel | not done: one ~3 us launch per step | | |

I found a real bug on the way: the fold's first version also fed the addend to
the attention layer's *output* projection (a `CombinationOperator` too), which
the transformer's loss trajectory exposed (0.079 vs 0.054 at 5 epochs) and the
unit tests did not - there was no end-to-end transformer gradient test. There
is now: `GpuComparison.TransformerTrainingGradient` (CPU vs the numerical
gradient, then GPU vs CPU) plus `ActivationsTest.DenseFanoutFoldedResidualGradient{CPU,GPU}`
and the memory-pool test updated to the folded semantics.

Paired against PyTorch's best config after H-J (alternated O P P O per point,
machine hot after eight hours of benchmarks, so absolute numbers are ~8% under
the cool ladders): bf16 128 OpenNN 1,819 / 1,788 vs PyTorch 1,795 / 1,791
(+0.6%); 32: 1,628 / 1,515 vs 1,586 / 1,569 (-0.4%); 64: 1,488 / 1,511 vs
1,536 / 1,535 (-2.4%); 256: 1,636 / 1,606 vs 1,708 / 1,716 (-5.3%). Still
parity within the thermal band, PyTorch ahead at 256; the levers below the
attention kernel are used up.

## Where the bf16 gap to PyTorch is

A microbenchmark of the attention shape (B 128, 8 heads, S 256, d 32, bf16;
[`sdpa_probe.py`](sdpa_probe.py)) on this GPU:

| kernel | fwd | bwd | fwd+bwd |
|---|---:|---:|---:|
| PyTorch flash-attention 2 (its default here) | 0.44 ms | 2.27 ms | 2.71 ms |
| PyTorch cuDNN backend | 0.76 ms | 3.23 ms | 3.99 ms |
| OpenNN cuDNN SDPA graph (BSHD, padding mask, stats) | 0.86 ms | 2.80 ms | 3.65 ms |
| causal: FA2 / cuDNN (PyTorch) | 0.34 / 0.60 | 1.81 / 2.52 | 2.15 / 3.13 |

FlashAttention-2 is ~1.35x faster than cuDNN's Ampere SDPA at head dim 32:
6 attention layers x ~1 ms = 6-8 ms per step, the one structural item left
between the two stacks in bf16 (the GEMMs, LayerNorm and the elementwise
passes are at the same speed or at bandwidth on both sides). Nothing on the
cuDNN side moves it (F, G above); the OpenNN-side items left (residual add
folded into the sublayer's dgrad, softmax + cross-entropy in one pass, one
Adam kernel) add up to ~2 ms. A clear bf16 lead on Ampere therefore needs a
FlashAttention-class kernel for the fused attention path: FA2's kernels
(BSD-3, CUTLASS 3.x headers; for this benchmark the hdim32 bf16 causal and
non-causal forward/backward instantiations, ~4 .cu files) take exactly the
(B, S, H, D) layout the interleaved-heads path now uses; padded batches
would need the varlen entry (packed sequences) or the cuDNN graph as the
fallback. That is a dependency decision, noted below.

**Done, 2026-08-19**, and padded batches turned out to need neither: see
[`flash-attention-integration.md`](flash-attention-integration.md). Measured
here, alternated against the cuDNN rung: bf16 +13-22%, fp32 +7-9%, same loss,
CUDA graph still captured. Decoder self-attention keeps cuDNN, because FA2
anchors a causal mask to the other corner when the batch is padded.

## The ladder (RTX 3060 Laptop, 5 timed epochs, samples/s)

PyTorch = compile(reduce-overhead) + fused Adam; TensorFlow = XLA; OpenNN with
the corpus on the ext4 disk. Points past the 6 GB of VRAM measure WSL2 paging,
not kernels (PyTorch 512 and 1024, OpenNN 1024, fp32 512), and are marked.

| batch | OpenNN 19f8287c8 (eager) | **OpenNN after A-D** | PyTorch best | TensorFlow |
|---|---:|---:|---:|---:|
| bf16 32 | 1,531 | **1,776** | 1,797 | 1,166 |
| bf16 64 | 1,591 | **1,822** | 1,865 | 1,216 |
| bf16 128 | 1,678 | **1,917** | 1,992 | 1,315 |
| bf16 256 | 1,681 | **1,908** | 2,030 | OOM |
| bf16 512 | 1,412 | **1,619** | 357 (paging) | |
| bf16 1024 | 182 (paging) | 305 (paging) | 118 (paging) | |
| bf16 2048 | OOM | OOM | OOM | |
| fp32 32 | 734 | **825** | 619 | 642 |
| fp32 64 | 823 | **947** | 617 | 691 |
| fp32 128 | 880 | **989** | 618 | OOM |
| fp32 256 | 802 | **926** | 603 | |
| fp32 512 | 67 (paging) | 98 (paging) | 68 (paging) | |

Against the baseline the levers are worth +13-16% in bf16 and +12-15% in fp32
at every point that fits in memory.

The rows above come from separate ladders (PyTorch/TensorFlow at ~02:00 on a
cool machine, OpenNN at ~04:00). The results artifact is the single
back-to-back sweep of the three engines,
[`results/gpu-transformer-peak-batch-speed-20260817T020628Z.json`](../../results/gpu-transformer-peak-batch-speed-20260817T020628Z.json)
(commit 82bcfb204, OpenNN first within each precision, machine already warm):

| batch | OpenNN | PyTorch best | TensorFlow | OpenNN / PyTorch |
|---|---:|---:|---:|---:|
| bf16 32 | **1,780** | 1,739 | 1,201 | 1.02 |
| bf16 64 | **1,820** | 1,775 | 1,211 | 1.03 |
| bf16 128 | **1,909** | 1,887 | 1,375 | 1.01 |
| bf16 256 | 1,894 | **1,912** | OOM | 0.99 |
| bf16 512 | **1,619** | 520 (paging) | | 3.1 |
| bf16 1024 | 312 (paging) | 137 (paging) | | |
| fp32 32 | **898** | 561 | 609 | 1.60 |
| fp32 64 | **1,037** | 584 | 674 | 1.78 |
| fp32 128 | **1,013** | 599 | OOM | 1.69 |
| fp32 256 | **941** | 585 | | 1.61 |
| fp32 512 | 90 (paging) | 69 (paging) | | |

Peak-to-peak: bf16 1,909 vs 1,912 (0.999); fp32 1,037 vs 599 (1.73x).

So, on this laptop and with the thermal band it has (a run 90 minutes later
on the same binary moves any engine by +-5%):

* **fp32: OpenNN leads at every batch, 1.6-1.8x in the artifact** (1.33-1.60x
  in the cool-machine ladders) - the fp32-via-bf16 fused attention against
  PyTorch's fp32 attention.
* **bf16: parity.** In the artifact OpenNN is +1-3% at 32-128 and -1% at
  256; in the cool-machine ladders PyTorch was +1-6%. The structural
  difference is the one measured below (FlashAttention-2 vs cuDNN SDPA,
  ~5-8 ms of a ~70 ms step in PyTorch's favour); it is what separates
  "parity within noise" from a clear OpenNN lead. From 512 up OpenNN keeps
  1,619 while PyTorch pages (its step needs more memory: OpenNN uses 3.2 GB
  at batch 256 of the 6 GB).
* TensorFlow is well behind both and out of memory from 256 (bf16) / 128 (fp32).

## Next

1. RTX 4080 (the product GPU): the same sweep with this protocol; the
   cuDNN-vs-FA2 ratio there decides how much of the bf16 gap survives.
2. Decide on FlashAttention-2 (or an equivalent kernel) for the fused
   attention path: on this GPU it is the difference between parity within
   noise and a clear bf16 lead.
3. Device-resident token dataset (`Dataset::can_device_gather` excludes
   batches with a decoder section), which removes the host prefetch path and
   its epoch-start drain regardless of disk.
4. fp32: keep the SDPA Q/K/V/O bf16 pack pooled so the backward does not
   re-cast (4 casts per layer, ~2.5% of the fp32 step). (The residual fold,
   the fused cross-entropy metrics and the cuBLASLt autotune are done, H-J.)

## Appendix: first look on the old corpus (vocab 4000 / seq 128), 5 epochs

OpenNN 19f8287c8: bf16 32 2,310, bf16 128 2,659, fp32 32 1,262. PyTorch eager
bf16 32 1,518; PyTorch best bf16 32 2,538, 128 2,852, fp32 32 1,265. Not
comparable with the tables above (half the tokens per sample, 15x the vocab).
