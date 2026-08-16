# Plan: beating PyTorch and TensorFlow at large batch (ResNet-50 GPU training)

*2026-08-15. Grounded in per-step measurements on the audit machine (RTX 3060
Laptop 6 GB, WSL2, cuDNN 9.10.2) — OpenNN `OPENNN_GRAPH_TIMING` /
`OPENNN_PROFILE` versus PyTorch `torch.profiler`, both at batch 2048, plus a
per-shape census of the batch-norm backward path. Companion to
`resnet50-training-kernel-fusion-audit.md`.*

## 1. Where the deficit is — measured, per training step at batch 2048

**bf16** (OpenNN autotuned; PyTorch `cudnn.benchmark` + `torch.compile`):

| component | OpenNN ms | PyTorch ms | verdict |
|---|---:|---:|---|
| convolutions (fwd + wgrad + dgrad, cuDNN graphs) | **55** | 64 | we are faster |
| batch-norm graphs (fwd 11 + bwd 18) | 29 | 31 | parity |
| **BN-backward work *around* the graph** (fp32 staging casts, standalone dReLU, residual-delta copies) | **~15–20** | 0 | **the leak** |
| **residual-join adds** (`accumulate_output_deltas`, 16/step) | **5.6** | 0 (fused into Triton BN-bwd) | **the leak** |
| Adam | 2.3 | 5.3 | we are faster |
| pooling | 2.8 | ~5 (Inductor maxpool bwd) | we are faster |
| **step** | **~117** | **~100** | 0.85× |

**fp32**: same shape, larger numbers — conv ≈ 137–148 vs 137 (parity once the
timing mode's per-graph sync overhead, ~10 ms/step, is discounted; the
workspace budget is *not* a factor, see §6), BN 72 vs 57 (the 16 residual
layers' separate dReLU + delta copy ≈ 15–19 ms at 4-byte tensors),
residual-join adds 11 vs 0, Adam/pooling in our favour; step ≈ 245 vs 199
(0.81×).

*Measurement caveat:* GRAPH_TIMING event-times each cuDNN graph with a sync
(~160 syncs/step), OPENNN_PROFILE syncs per scope, and this laptop drifts
thermally by ±7% between runs, so single components carry ±10% slop. The
conclusions below rest on the components that are large and reproduced across
runs: BN-backward degradation, the residual adds, conv parity-or-better.

**The census** (`BatchNormalizationOperator backward … rung` lines, now printed
once per shape): at bf16/2048 on cuDNN 9.10, **49 of 53 BN layers run
degraded** — 33 through **FP32 staging** (three full-tensor casts + fp32 math at
2× bytes; includes the largest tensor in the network, the 67 MB stem output),
and all **16 residual layers on the un-fused bf16 engine** with a separate dReLU
kernel and a residual-delta copy. Only 4 layers get the fully fused native
engine. In fp32 nothing stages, but the same 16 residual layers run un-fused
(+ dReLU + copy). PyTorch runs all 53 as two fused Triton passes.

## 2. Why this only bites at large batch

Our structural advantage — CUDA graphs bundling 8 steps, GPU-resident gather,
~zero host work — is a **fixed cost per step**: worth 2× at batch 128 (PyTorch
spends ~13 ms/step in dispatch + WSL launch latency), ~1% at 2048. What is left
at large batch is bytes moved per sample. The extra passes above scale
linearly with the tensors, and at 128 they partly live near L2 (4 MB
activations) while at 2048 every pass hits DRAM (64 MB). Same code, opposite
regime: the fixed advantage shrinks exactly as the linear disadvantage grows.
Nothing about large batches is wrong in our code any more (that was the
workspace cliff, fixed) — our kernels simply move more bytes per sample than
PyTorch's.

## 3. The plan

Ordered by (throughput won) / (effort × risk). Each phase has a validation gate;
none is merged without it.

### Phase 0 — configuration and cheap wins (days)

- **0a. Autotune everywhere it counts, and make it cheap — DONE.** Autotune
  cuts our wgrad time 39% at 2048 versus the heuristic (conv 69 → 55 ms/step)
  but building every candidate engine cost minutes of warmup per point.
  `finalize` now builds only the heuristic's first K viable candidates
  (default 8, `OPENNN_AUTOTUNE_CANDIDATES`; 0 = all) and `autotune()` skips
  the unbuilt slots. Measured, bf16, fresh plan cache:

  | | tune + warmup | samples/s |
  |---|---:|---:|
  | 1024, top-8 | **70 s** | **18,482** |
  | 1024, all | 196 s | 17,589 |
  | 2048, top-8 | **47 s** | **19,214** |
  | 2048, all | 215 s | 17,316 |

  3–4.5× less warmup and *higher* throughput: the exotic engines deep in the
  list win the tuning micro-benchmark and lose in the real step. Top-8 is the
  default; the plan cache key includes K.
- **0b. Workspace budget under autotune — measured, dropped.** fp32/2048 with a
  1 GiB budget: conv 155 ms/step, 7,799 samples/s; with the auto 256 MiB
  budget: 148 ms/step, 8,569 samples/s. A larger budget does not help; the
  auto ceiling stays. (Kept here so nobody re-runs it.)
- **0c. Census on the benchmarks machine** (RTX 4080, cuDNN 9.23): run the same
  binary and read the rung lines. The in-source note says 16/24 shapes stage
  there; this decides how much of Phase 1 the canonical numbers get.
- **0d. Gradient check of the un-fused bf16 engine — DONE, and it changed
  Phase 1.** `GpuComparison.ResidualBlockGradientBf16PerBackwardRung` pins each
  rung (`device::set_batch_norm_backward_rung`) on a ResNet-style residual
  block: fp32 GPU vs CPU reference 7e-7 on both rungs; **bf16 plain vs bf16
  FP32-staged 1.6e-8** — the un-fused bf16 `batchnorm_backward` computes the
  same gradient as the staged path on cuDNN 9.10; bf16 vs fp32 reference 9%,
  the network's bf16 rounding, identical across rungs. The "bad math" note was
  not reproducible here. Run the test on any new cuDNN before trusting it.

### Phase 1 — batch-norm backward off the FP32-staged path

**1a — DONE (no kernel needed on cuDNN 9.10).** With 0d in hand, the ladder now
tries the plain native bf16 engine (ReLU mask as its own kernel) *before* FP32
staging on the non-residual layers. Census at bf16/2048: 33 staged → **0**.
Measured, autotuned top-8, fresh plan cache, bf16:

| batch | before | **after** | PyTorch | TensorFlow |
|---:|---:|---:|---:|---:|
| 128 | 9,895–11,341 | **12,813** | 5,495 | 7,963 |
| 1024 | 18,482 | **20,899** | 19,098 | 18,819 |
| 2048 | 19,214 | **22,065** | 21,405 | 21,641 |

+13–15% at 1024/2048, losses in band. **OpenNN bf16 now leads both engines at
every batch on this GPU**, by 2–3% at the peak (inside noise) and decisively
below it. fp32 is untouched by this (it never staged).

**1b — the fused kernel — DONE.** `batchnorm_backward_fused_cuda`
(`kernel_normalization.cu`): NHWC, bf16/fp32 IO, fp32 math, dReLU and the
residual fork fused, channel-pair vector loads, and for non-residual fp32
layers the reduce pass rebuilds x̂ from Y and skips X. Auto now takes cuDNN's
fully fused engine where the shape has one and this kernel otherwise (49 of
53 layers here); the plain and staged cuDNN graphs remain pinnable rungs
(`device::set_batch_norm_backward_rung`) and the gradient test compares all of
them: fp32 own vs CPU 2.5e-4, bf16 own vs staged 6.6e-8. Measured A/B against
the cuDNN plain path, autotuned top-8, fresh cache: fp32 2048 → **11,731**
(PyTorch 10,363, **1.13×**), fp32 1024 10,842 → 10,989, bf16 2048 23,627 →
24,030, bf16 1024 21,913 → 21,604 (noise). The gradient test caught a wrong
first version of the x̂-from-Y trick (the apply pass needs x̂ on masked
elements too) — keep it in the loop for every kernel change. Design notes:

1. **Reduce**: per channel `Σ dy'` and `Σ dy'·x̂` with `dy' = dy ⊙ [y > 0]`
   (residual layers read the saved Y; non-residual ones can rebuild the mask
   from X, mean, invvar, scale, bias). NHWC → coalesced along C; block-level
   partials over N·H·W with a two-level (per-block → global) sum, deterministic
   order. Reads DY, Y|X (+X for x̂): 2–3 passes.
2. **Apply**: `dx = scale·invvar·(dy' − mean(dy') − x̂·mean(dy'·x̂))` written
   in place over DY, and `dPre = dy'` written to the residual branch **with
   `+=` into the block-input delta when it already holds conv1's dgrad** (see
   Phase 2). Reads DY, Y|X, X; writes DX, DPre: 3–4 passes.

≈ 5–6 tensor passes versus today's staged path (~11: three casts × 2 passes,
fp32 BN at 2× bytes, dReLU 3, copy 2) or un-fused path (~8). Expected at 2048:
**bf16 −15…20 ms/step, fp32 −15…19 ms/step**. That alone takes bf16 from 0.85×
to ≈1.0× and fp32 to ≈0.9×. The 2026-08-11 note discarded hand BN kernels
because cuDNN was near roofline — true for the *native fused* path, which we
keep; this replaces the *staged/un-fused* paths, which are far from it.
Gate: per-layer gradient match to the fp32 reference (rel. 1e-2 in bf16, 1e-4
fp32) in `gpu_comparison_test`; loss trajectory in-band; the cuDNN native rung
kept and A/B'd per shape; census shows 0 staged / 0 copy shapes.

### Phase 2 — fuse the residual-join add (days)

The block-input delta is `dgrad(conv1) + dPre` (16 adds/step, 3 passes each:
5.6 ms bf16, 11 ms fp32). In backward order dPre is produced first (conv3's BN
backward), conv1's dgrad last. Two equivalent fixes: (a) Phase 1's apply kernel
writes dPre straight into the block-input delta and conv1's `conv_dgrad` graph
gets a `pointwise ADD` of that buffer as its epilogue (runtime fusion, in-place
same-index aliasing) — no extra pass; or (b) dgrad writes first and the fork
`+=`. (a) needs no ordering change. Also make `allows_input_delta_alias()` true
for `Convolutional` where the planner can prove it. Expected: **−5.6 ms bf16,
−11 ms fp32** at 2048. Gate: loss identical (pure reordering of adds), the
`bwd:accumulate_output_deltas` scope at ~0.

**Status after 0a + 0d + 1a + 1b + Phase 2 (all measured, 2026-08-16) at 2048
on this GPU:** bf16 **24,030** samples/s vs PyTorch 21,405 / TensorFlow 21,641
(**1.12× / 1.11×**); fp32 **11,731** vs PyTorch 10,363 (**1.13×**; TensorFlow
does not fit 2048 in fp32). bf16 4096: 22,834 vs PyTorch 19,898 (**1.15×**),
TensorFlow OOM. OpenNN leads both engines at every measured batch in both
precisions on this GPU. On the RTX 4080 the same fixes should clear
TensorFlow's 65,752 bf16 peak (the in-source +21% measurement of the un-staged
path is this same win) — to be measured there (§ runbook in the audit thread).

### Phase 3 — the MLPerf fusion architecture: measured, and NOT the way forward on this stack

The design rested on one premise: that cuDNN's fused engines (SBRCS: BN-apply
+ ReLU in the conv prologue and genstats in its epilogue; DBAR: dgrad + dReLU
+ dbn_weight) run near plain-convolution speed on our shapes. NVIDIA's MLPerf
work at 224x224 never had to establish that for 8x8-to-1x1 spatial at batch
2048, so `cudnn_fusion_probe` (this folder; builds with the benchmarks) times
every pattern per real ResNet-50/CIFAR shape through the library's own
autotune path. On the audit machine (sm_86, cuDNN 9.10.2), batch 2048:

| pattern | bf16 | fp32 |
|---|---|---|
| conv + genstats | **no engine** on any of 9 shapes | no engine |
| SBRCS (scale-bias-ReLU prologue + conv + genstats) | **no engine** | no engine |
| DBAR (dgrad + dReLU + dbn_weight) | **no engine** | no engine |
| dgrad + dReLU (the one fused form that exists) | **1.1–14× slower** than plain dgrad | 1.3–4.3× slower |

The plain cutlass/xmma engines cuDNN picks for our convolutions do not offer
these prologue/epilogue fusions; where a runtime-fusion engine exists it is far
slower on these shapes. (Our own `dgrad + ADD` fold measured *faster* because a
source-add is a native cutlass epilogue; a masked ReLU with a second full
tensor is not.)

**Re-run on cuDNN 9.24.0 (2026-08-16), same GPU: identical.** genstats, SBRCS
and DBAR still have no engine on any of the nine shapes; `dgrad + dReLU` is
1.5–19× slower than plain. The verdict is architecture-bound (Ampere), not a
library-version artefact; the RTX 4080 is the same generation and likely the
same, Blackwell may differ - the probe answers in two minutes wherever it runs.
So on this GPU generation Phase 3 as designed cannot deliver, and the weeks it
would cost are better spent elsewhere. One caveat kept on record: the cuDNN
samples exercise these patterns in FP16, a training type the library does not
offer and one that brings loss scaling with it.

**Library parity.** The 3-way sweeps until 2026-08-16 ran OpenNN on the WSL
system cuDNN 9.10.2 while PyTorch's wheel bundles cuDNN 9.24. A 9.24 build
(`nvidia-cudnn-cu12` wheel, no sudo needed; RUNPATH carries the wheel's lib
dir) measures equal within noise at top-8 - bf16 2048 23,987 → 24,421, bf16
1024 21,472 → 21,310, fp32 2048 11,089 - with all gradient tests unchanged
(plain-bf16 rung and own kernel exact on 9.24 too) and top-8 still ≥ all
candidates. Benchmarks and the gate on this machine now run on 9.24 (its own
gate key: `... | cudnn 92400`).

### Phase 3' — what replaces it (own kernels, no dependence on cuDNN fusion engines)

The step at 2048 is now conv ~60% / batch-norm ~35% / rest ~5%. Without
cross-operator fusion the batch-norm passes are close to their floor: forward
is cuDNN's fused two-pass graph, backward our own two-launch kernel at 7-8
passes. What remains:

- **3'a. Own batch-norm forward emitting a 1-bit ReLU mask — done for BF16:
  +6% at bf16 2048 over the shipped code, parity at 128; FP32 measured slower
  and keeps its previous path.** `batchnorm_forward_fused_cuda` (stats reduce;
  finalize with the running-statistics update; apply + residual add + ReLU)
  packs (y > 0) eight channels per byte into a new `ReluMask` forward slot of
  `Convolutional`; `batchnorm_backward_fused_cuda` gates dY by that byte in
  both passes instead of re-reading Y, on an eight-channel (16-byte) vector
  layout. `device::BatchNormForwardRung {Auto, CudnnGraph, OwnKernel}` pins it
  like the backward rung (harness: `OPENNN_BN_FORWARD_RUNG`,
  `OPENNN_BN_BACKWARD_RUNG`); Auto takes the own forward for BF16 ReLU outputs
  with channels % 8 == 0 (every BN of a ResNet) and cuDNN's fused graph
  elsewhere; cuDNN's fully fused backward, where a shape has one, still wins
  and still runs. Correctness: `ResidualBlockBatchNormForwardRungParity` (own
  vs cuDNN forward: gradient 6e-6 fp32 / 1.5e-4 bf16, inference after a
  training step 1e-5 vs the CPU reference) and the two gradient tests.

  Measured on the RTX 3060 / cuDNN 9.24. The laptop drifts ~10% with
  temperature after an hour of benchmarking, so the numbers that count are
  cooled, order-alternated pairs (own : cuDNN forward : yesterday's binary):

  | point | own forward + mask | cuDNN forward, new backward | shipped (yesterday) |
  |---|---:|---:|---:|
  | bf16 2048 | **24,035 / 23,355** | 22,963 / 23,267 | 22,663 / 22,232 |
  | bf16 128 | 13,063 / 12,020 | 12,977 / 12,022 | 12,792 / 12,246 |
  | fp32 2048 (8-channel fp32 backward, since reverted) | 11,342 / 11,434 | — | 11,618 / 11,828 |
  | fp32 128 | 6,733 / 5,966 | 6,373 / 6,053 | 6,564 / 6,237 |

  Final check of the committed binary (c1b7be414) against its direct
  predecessor HEAD (48a43e9ce) and the morning binary, fresh plan cache, cooled
  pairs: bf16 2048 **24,876** vs HEAD 22,451; fp32 2048 11,940 / 11,938 vs
  HEAD 11,923 / 11,774 / 11,685 vs morning 12,118 / 12,017 / 11,945 - fp32 at
  parity (the -3..-8% seen in earlier fp32 pairs was a stale autotune cache
  keyed differently by the two binaries, plus WSL memory pressure from build
  trees: two runs paged at ~1,600 samples/s until the trees were deleted).

  A first cut lost 5-9% at batch 128: CIFAR's late 1x1/2x2 stages give a BN
  128-512 rows there, and the reduce still spawned 128 nearly idle row blocks
  whose partials one thread per channel then summed serially. Row blocks now
  follow the rows (at least four per lane) and the finalize sums a warp of
  channels with eight lanes over the row blocks - that fix applies to the
  backward the shipped code already had, which is why 128 ends slightly ahead.
  In FP32 the eight-channel reduce kernel sits at ~90 registers (half
  occupancy; the 64-channel layers run a second, mostly empty wave) and the
  whole step measured ~3% slower, so FP32 keeps channel pairs, Y and the
  x_hat-from-Y trick, and Auto leaves its forward on cuDNN; the FP32 own path
  stays pinnable (`OPENNN_BN_FORWARD_RUNG=own`) for the RTX 4080, where the
  register budget is the same but the wave arithmetic is not.

  Less than the ~5% first estimated for BF16 too: the mask removes ~27% of the
  batch-norm backward's traffic and that backward is ~10% of the step. Side
  finding, fixed: `NeuralNetwork::compile` left the batch-norm running variance
  at zero (states are zeroed; only `set_parameters_random` set the defaults),
  so a fresh network's inference was scaled by 1/sqrt(epsilon) until training
  moved it; operators now initialize their states on compile
  (`Operator::initialize_states`).
- **3'b. Convolution engine choice — measured on cuDNN 9.24, nothing beats the
  defaults here.** Three knobs now live in `finalize` (defaults unchanged, all
  environment-controlled, all folded into the plan-cache key): heuristic mode
  `OPENNN_CUDNN_HEURISTICS=A|B|AB`, per-kind candidate count
  `OPENNN_AUTOTUNE_CANDIDATES_{FORWARD,WGRAD,DGRAD}`, and a numeric-note
  restriction for convolution graphs `OPENNN_CONV_ENGINE_NOTES=WINOGRAD,FFT`
  (falls back to the unrestricted list for shapes without such engines).
  Measured at batch 2048, autotuned, fresh plan cache, RTX 3060 / cuDNN 9.24
  (repeat runs of the default itself differ by ~1.5–4%):

  | knob | bf16 | fp32 |
  |---|---:|---:|
  | default (A, K=8, auto budget) | 22,804 / 23,138 | 11,500 / 11,481 / 11,966 |
  | heuristics B | 23,268 | 11,117 |
  | heuristics A+B | 23,474 | 11,391 |
  | wgrad K=16 | 23,410 | 11,112 |
  | Winograd/FFT + 1 GiB budget | — | 11,489 (2048), 11,090 vs 10,680 (1024) |

  All inside noise: the top-8 autotune already reaches the good engines on this
  GPU. The knobs stay as levers for other GPUs (the RTX 4080 may rank
  differently); the gate protects the defaults.
- **3'c. Pooling backward from a forward argmax mask — done, +3-5% bf16 at
  2048, fp32 and small batch at parity.** `pooling_probe` (raw cuDNN, in the
  benchmark tree) put the numbers on it first: on the RTX 3060 cuDNN's max-pool
  forward runs at the copy roofline (260-317 GB/s) but its backward at 59 GB/s
  in bf16 (2.82 ms for the CIFAR stem pool at batch 2048, 7.6 ms at the
  ImageNet stem shape) against ~0.3 ms for the traffic a mask path needs. Now
  `max_pooling_forward_cuda` writes Y plus one byte per output (the window
  position of the maximum; windows up to 255 elements) into the layer's
  `MaximalIndices` slot - already there for the CPU path, INT8 on CUDA,
  training-only - and `max_pooling_backward_cuda` is a gather: each input
  element visits the at most ceil(pool/stride)^2 outputs whose window covers
  it and sums the dY whose argmax lands on it. No atomics, no zero fill,
  deterministic, X and Y not read again. `device::MaxPoolingRung {Auto,
  Cudnn, OwnKernel}` (harness `OPENNN_POOLING_RUNG`); Auto = own kernels in
  training where the mask slot exists, cuDNN at inference and for average
  pooling. `MaxPoolingGradientPerRung`: both rungs equal the CPU reference to
  4e-9 in fp32 and each other exactly in bf16 (max pooling selects, it does
  not round). Measured, same binary, cooled pairs (own : cuDNN):

  | point | own | cuDNN |
  |---|---:|---:|
  | bf16 2048 | **25,333** / 24,167 | 24,058 / 23,839 |
  | bf16 128 | 13,753 | 13,572 |
  | fp32 2048 | 11,691 | 11,752 |

  ResNet-50 has one max pool, so this is the whole of it: bf16 peak on this
  GPU 25.3k, fp32 inside noise (its cuDNN backward was 2.7 ms of a ~175 ms
  step). Memory: 1 byte per pool output, ~8 MB at batch 2048 here.

Realistic remaining upside on this stack after 3'a and 3'c: **~1-3%** (the
Phase 4 items). Beyond that the
architecture is at the floor of what cuDNN's convolution engines allow on
these shapes; more would need custom convolution kernels (weeks, high risk) or
a GPU/cuDNN generation where the fused engines exist - which the probe tells
in two minutes.

### Phase 4 — parallel small items (measured 2026-08-16)

- **Resident gather inside the graph: already the case.** The host profile at
  batch 128 (`OPENNN_PROFILE=1`, bf16) shows the host in `step:group_sync`
  68% of the epoch and in `cudaGraphLaunch` 25% (~19 ms per 8-step graph of
  ~5,000 nodes, well under the ~74 ms the GPU takes for the group);
  `step:gather_issue` is 0.6%. The GPU is the bottleneck at 128; the gather
  kernel already runs inside the captured step and only the 1 KB index upload
  is outside. Nothing to move.
- **`cast_bf16_to_fp32` vectorized** (four elements per thread, the widening
  twin of `cast_fp32_to_bf16`): the 53 per-step conv weight-gradient casts
  measured neutral (bf16 128 14,146 / 14,136 vs 14,129 / 14,080; 512 20,725
  vs 20,658; 2048 25,556 vs 25,603) - launch-bound small tensors, not
  bandwidth. Kept for symmetry; no claim.
- **PyTorch protocol - the one that matters.** The sweep runs PyTorch on
  `PT_FAST` (channels_last + `torch.compile` default mode + TF32, foreach
  Adam). Its strongest one-line options, `torch.compile(mode="reduce-overhead")`
  (CUDA graphs) and `Adam(fused=True)`, now behind `PT_COMPILE_MODE` /
  `PT_FUSED_ADAM` in the driver and `PYTORCH_BEST=1` in the sweep runner,
  measured on the RTX 3060 / cuDNN 9.24, bf16, same session (hot GPU, so the
  ratios are the robust part):

  | batch | PT fast (sweep protocol) | PT fast + reduce-overhead + fused Adam | OpenNN (same session) |
  |---:|---:|---:|---:|
  | 128 | 5,342 (7,965 in the cool sweep) | **13,211** (fused Adam alone 10,710; graphs alone 6,468) | 14,1xx |
  | 512 | 13,819 | **19,093** | 20,7xx |
  | 2048 | 20,138 | **22,661** | 25,556 |

  The published protocol understates PyTorch by 2.5x at 128, 1.4x at 512,
  1.13x at 2048. OpenNN still leads at every point against PyTorch's best
  (+7% at 128, +8% at 512, +13% at 2048 in this session), but the small-batch
  margin is a few percent, not the 1.75x the sweep table shows. Recommendation:
  make `PYTORCH_BEST=1` the sweep's PyTorch protocol (a PyTorch user gets both
  options for free), re-run the three-way table with it, and state the margin
  against that. The same question is open for TensorFlow (XLA `jit_compile`).

## 4. Projected trajectory (batch 2048, ms/step; lower is better)

| | bf16 OpenNN | bf16 PyTorch | fp32 OpenNN | fp32 PyTorch |
|---|---:|---:|---:|---:|
| 2026-08-15 morning (autotuned) | 117 | 100 | 245 | 199 |
| + 0a top-K autotune (measured) | 107 | | | |
| + 1a plain-bf16 rung (measured) | 93 | | 245 | |
| + Phase 2 fused join (measured) | 89 | | 218 | |
| + 1b own BN-backward kernel (measured) | **85** | 100 | **175** | 199 |
| + Phase 3' (own BN fwd + mask, conv/pool tweaks) | ~78–80 | | ~160–165 | |
| + Phase 3 (MLPerf fusion) | 75–80 | 100 | 165–175 | 199 |

## 5. Validation gates (all phases)

Per-layer gradient tests against the fp32 reference; 5-epoch loss trajectory
in-band with the current path; the BN census printing 0 degraded shapes on the
target GPU; the peak-batch sweep on both machines (this one and the RTX 4080)
with the corrected sample counts; no regression at batch 128/256.

## 6. Closed measurement: workspace budget under autotune

fp32/2048, autotune on, RTX 3060: 1 GiB budget → conv 155 ms/step
(wgrad 68.5, fwd 46.2, dgrad 40.2), 7,799 samples/s; auto 256 MiB budget →
conv 148 ms/step (63.0 / 45.0 / 40.5), 8,569 samples/s. Not a lever.

## 7. Diagnostics added for this plan (in the tree)

- `BatchNormalizationOperator backward c<C> r<rows> batch <N>: rung <k> (...)`
  — printed once per shape whenever the backward is not on the fully fused
  native rung. Run any training and count: the target of Phase 1 is zero lines.
- `ConvolutionOperator wgrad ...: no FP32-store engine` — once per shape when
  the BF16-in/FP32-out wgrad store is unavailable (it is, on cuDNN 9.10).
- `PT_PROFILE=1 [PT_PROFILE_STEPS=n]` in `pytorch_resnet50_speed.py` — kernel
  time by category and top kernels for the same configuration, so the two
  engines can be compared component by component.
