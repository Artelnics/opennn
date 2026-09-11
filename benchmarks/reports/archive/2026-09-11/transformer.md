# Transformer: the base model on WMT14

OpenNN against PyTorch 2.13 on the *Attention Is All You Need* base
configuration — 6 encoder and 6 decoder layers, d_model 512, 8 heads,
feed-forward 2,048, a 20,000-token vocabulary on each side, 74,878,496
parameters — over 199,575 English–German sentence pairs from WMT14 News
Commentary v9 at 130 tokens per sequence, batch 32, bf16, on the RTX 5070 Ti.
Session `2026-09-06-energy-publish`, commit `520f0f2f8`:

| cell | OpenNN | PyTorch | OpenNN / PyTorch |
|---|---|---|---|
| `cuda-transformer-infer` | 5,413 sequences/s | 4,694 | **1.153×** |
| `cuda-transformer-train` | 1,352 sequences/s | 1,145 | **1.181×** |

Both cells now have an `nsys` kernel trace, and it changes what this document
can claim. The matrix products dominate both engines — 90.9% of OpenNN's
kernel time on inference against 77.3% of PyTorch's — and the margin is in
those kernels, not around them. The launch-cost explanation the previous
version of this document offered is **withdrawn**: over each engine's own
timed window the GPU is 99.4% busy on both engines on inference, and 98.7% on
OpenNN against 99.1% on PyTorch on training. PyTorch is, if anything, the
busier of the two; neither engine is host-starved. What separates them on
training is that PyTorch launches four times as many standalone pointwise and
reduction kernels as OpenNN — 20.8% of its kernel time against 8.1%. On
inference the larger part of the margin is one kernel: PyTorch's attention
takes 56.24 µs per launch against OpenNN's 17.63, and that gap is a backend
default, a library version and a mask asymmetry rather than engine work.

## What is measured

**Network.** The "Attention Is All You Need" base model: encoder–decoder,
d_model 512, 8 heads, feed-forward 2048, 6 layers each side, post-layer-norm,
scaled token embeddings with sinusoidal positions, and a final projection to
the vocabulary. Heads and feed-forward width follow d_model by the paper's
ratios (d_model/64 and 4·d_model) in both drivers. PyTorch assembles it from
`nn.TransformerEncoderLayer`/`nn.TransformerDecoderLayer` with `norm=None`
on the stacks — `nn.Transformer` would append a final LayerNorm to each stack,
2,048 parameters and two normalisations per pass that OpenNN's `Transformer`
does not have. Both engines report **74,878,496 parameters** (2 × 20,000 × 512
embeddings, 6 × 3,152,384 encoder, 6 × 4,204,032 decoder, 10,260,000 output
projection), which the runner checks. Training is cross-entropy over the
vocabulary with Adam at learning rate 1e-4.

There are **18 attention blocks** in a forward pass — 6 encoder
self-attention, 6 decoder self-attention, 6 decoder cross-attention — and the
profiles confirm it: in the timed window of the inference profile OpenNN
launches its fused SDPA kernel and PyTorch launches
`pytorch_flash::flash_fwd_kernel` 336,744 times each over 18,708 batches,
exactly 18 per batch on both sides. (Earlier versions of this document said
12. That was wrong.)

Three things differ inside that identical shape — the masks, the dropout
PyTorch's layers apply by default, and the loss denominator. All three run in
OpenNN's favour to some degree and all three are set out in full under
"Asymmetries and caveats" below, with what the profile and the variant runs
price them at.

**Data.** WMT14 English–German, the corpus the paper reports on, through its
News Commentary v9 training file. `prepare.py transformer` tokenises both
sides with exactly the rule OpenNN's `WordLevelTokenizer` applies
(ASCII-lowercase, alphanumeric runs and single punctuation characters, other
bytes dropped), truncates each side to 128 tokens and keeps the first 200,000
well-formed pairs: **199,575 pairs**. Writing the corpus pre-split is what
makes the two tokenisers agree — OpenNN's is then idempotent over it and a
whitespace split gives PyTorch the same tokens — so both engines build the
same padded length, **130** (128 + START + END), and the same capped
vocabulary, **20,000** including the four reserved ids. Token identities are
irrelevant to a speed measurement and are not compared; the shapes are. Whole
batches only: at batch 32 an epoch or a pass covers 6,236 batches =
**199,552 sequences**, 25.9 M padded tokens. For training both engines keep the id
tensors resident on the device; PyTorch slices them sequentially, OpenNN
gathers each batch by index. For inference both fill the first batch of 32
pairs once and replay it 6,236 times per pass, so the inference cell times
the resident forward pass alone.

**Cells.**

| cell | device | batch | precision | timed window |
|---|---|---|---|---|
| `cuda-transformer-train` | RTX 5070 Ti | 32 | bf16 | 2 epochs after 1 untimed |
| `cuda-transformer-infer` | RTX 5070 Ti | 32 | bf16 | 5 passes after 1 untimed |

Throughput is compared per sequence; both drivers also print
`tokens_per_sec` (× 130), the figure the literature quotes. The traces
corroborate the untimed epoch: the profiled inference run's timed window
covers 3 passes, 18,708 batches, inside a trace that runs longer; the
profiled training run's covers 1 epoch, 6,236 steps.

**Each engine at its best.** bf16 autocast on both sides. PyTorch runs
`torch.compile(mode="max-autotune-no-cudagraphs")` for the training step and
the inference forward — Inductor with its GEMM and pointwise templates
benchmarked per shape and no CUDA graphs. That mode was chosen by measurement
in the `2026-09-02-variants` session: training read 1,150 sequences/s at
`8e47e7662` against 1,132 under `reduce-overhead` at `918805ce1`; inference
read 4,657 at `d3acd71b5` against 4,574 under `reduce-overhead` and 4,499 in
the default mode, both at `8e47e7662`, all three with the weights stored in
bf16 once (`PT_INFER_CAST=weights`). Under autocast the same best mode read
3,641, at `8e47e7662`. Only the 4,574/4,499 pair is same-commit; the ranking
rests on three commits, and the runs establish that ranking rather than
today's levels — PyTorch's inference number at `93cc90e07` is 4,695, above
the best of them. Attention goes through `scaled_dot_product_attention`, and
the trace shows which backend it selects: `pytorch_flash::flash_fwd_kernel`,
the FlashAttention path, 14.9% of its inference kernel time. The driver does
not choose that backend, and does not choose against cuDNN's; it sets only
`allow_tf32`. The *Why* section prices what that default costs PyTorch.

OpenNN captures the Adam step and the inference pass as CUDA graphs and, on
cuDNN ≥ 9.25 with bf16, uses cuDNN's fused scaled-dot-product attention for
sequences of 128 tokens and more. The training driver prints
`sdpa_min_sequence_length=128`; the inference driver does not print the
field, but the trace settles it: the kernel
`cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16` runs 336,744
times in the timed window, 18 per batch, so the fused path is what ran on
both cells. The
driver's comment records the measurement behind that threshold: fused beat
the materialised attention by 28% at 128 tokens over five launches each way,
while the library-wide default stays at 192 because a single pass cannot
amortise the 0.3–2 s of plan construction. Adam runs over a joint gradient
arena (`set_joint_gradient_arena(true)`): each layer's gradient is planned
into the forward arena by lifetime, beside the deltas, so the 300 MB
gradient of a 75 M-parameter model reuses memory whose lifetime has ended;
the update is then one streaming launch per parameterised layer, which the
trace confirms as 467,774 `adam_update_kernel` launches over the 6,236 steps
of the timed window — 75 per step.

Since `84790b7da` the attention plan itself is chosen by measurement: cuDNN
offers several engines for the fused graph, the frontend's heuristic
returns one, and the library now times the candidates under the workspace
cap described in `cudnn_frontend_utilities.h` and keeps the fastest
(`OPENNN_SDPA_AUTOTUNE`, on by default; `=0` restores the heuristic's
pick). The knob existed before and was off pending this measurement. In the
direct A/B it is worth 5,335 → 5,413 sequences/s on inference and 1,330 →
1,353 on training, repeatable to the unit; the publish rows above carry it.
It costs plan workspace — 9 MiB on inference, 88 MiB on training — which is
where the memory rows moved from 618 and 2,233 MiB at `e76425bd3`. The
kernel tables in the *Why* section were traced at `93cc90e07`, before the
change, and name the heuristic's engine; the mechanism they describe is the
same, the numbers are a few percent older.

**Gates.** Samples (199,575), sequence (130), input and target vocabulary
(20,000) and parameters (74,878,496) must agree between the engines. There is
**no loss gate** in this family; two epochs of a base transformer are a speed
measurement, not a translation result.

## Results

Session `2026-09-06-energy-publish`, commit `520f0f2f8`, clean tree, clocks locked at
2692/810 MHz, turbo off, governor `performance`. Both cells are
evidence-grade: the training run that the previous version of this document
had to file under `results/scratch/` for foreign CPU activity has been
replaced by a quiet one.

| cell | batch | precision | OpenNN samples/s | PyTorch samples/s | OpenNN / PyTorch | peak memory MiB (OpenNN / PyTorch) | energy Wh (OpenNN / PyTorch) |
|---|---|---|---|---|---|---|---|
| `cuda-transformer-infer` | 32 | bf16 | 5,413 | 4,694 | **1.153×** | 627 / 1,161 | 11.4501 / 15.0000 |
| `cuda-transformer-train` | 32 | bf16 | 1,352 | 1,145 | **1.181×** | 2,321 / 3,350 | 16.2770 / 22.7057 |

Both cells win all three axes, as every cell in the twelve-cell matrix now
does.

`cuda-transformer-train` — batch 32, bf16, epochs 2 per launch, 3 rounds. Artifact `cuda-transformer-train-publish-20260906T180409Z.json`, commit `520f0f2f8`, clean tree, quiet True (busy 0.4% before, 0.2% after, 1.7% max during, threshold 3%), clocks locked True, shape gate True.

| engine | median samples/s | min | max | peak device MiB | Wh (board) | mean W |
|---|---|---|---|---|---|---|
| OpenNN | 1,352 | 1,352 | 1,353 | 2,321.2 | 16.27703 | 198.7 |
| PyTorch | 1,145 | 1,145 | 1,145 | 3,349.6 | 22.70570 | 234.7 |
| **ratio** | **1.181×** | | | 1.44× less | 1.395× less | 1.18× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 1,352 | 1,145 |
| 2 | pytorch → opennn | 1,352 | 1,145 |
| 3 | opennn → pytorch | 1,353 | 1,145 |

OpenNN's median epoch is 150.0 s and PyTorch's 174.24 s; in tokens, 172,867
against 148,883 per second.

`cuda-transformer-infer` — batch 32, bf16, passes 5 per launch, 3 rounds. Artifact `cuda-transformer-infer-publish-20260906T173620Z.json`, commit `520f0f2f8`, clean tree, quiet True (busy 0.1% before, 0.1% after, 1.6% max during, threshold 3%), clocks locked True, shape gate True.

| engine | median samples/s | min | max | peak device MiB | Wh (board) | mean W |
|---|---|---|---|---|---|---|
| OpenNN | 5,413 | 5,413 | 5,413 | 627.2 | 11.45014 | 223.9 |
| PyTorch | 4,694 | 4,693 | 4,696 | 1,161.2 | 14.99995 | 254.6 |
| **ratio** | **1.153×** | | | 1.85× less | 1.310× less | 1.14× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 5,413 | 4,693 |
| 2 | pytorch → opennn | 5,413 | 4,696 |
| 3 | opennn → pytorch | 5,413 | 4,694 |

Per-pass times are 37.4 s on OpenNN against 42.51 s on PyTorch; in tokens,
693,600 against 610,321 per second.

## Why

Both cells were profiled with `nsys` in this session. The traces are **not
committed** — the four SQLite exports total 3.8 GB — and the `Reproduce`
section gives the command that regenerates them. Every figure below is taken
over each engine's own timed window, the interval its driver prints as
`TIMED_START_UNIX`–`TIMED_END_UNIX`, which excludes the untimed warmup and
with it, on PyTorch's side, Inductor's max-autotune benchmarking; whole-trace
averages are contaminated by that phase and are not used. The profiled runs
reproduce the throughputs published at `93cc90e07` to within a count — 1,328 against 1,329
for OpenNN on training and 1,144 against 1,145 for PyTorch, 5,331 against
5,335 and 4,691 against 4,695 on inference — so the profiler is not
distorting what it measures here.

### The step, by kernel

Share of each engine's kernel time inside its own timed window: one epoch,
6,236 steps, on training; three passes, 18,708 batches, on inference. Launch
counts are for the same window.

`cuda-transformer-infer`, OpenNN:

| kernel | share | launches |
|---|---|---|
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_nn` | 35.11% | 1,346,976 |
| `nvjet_sm120_tst_mma_128x256x32_4_64x64x32` | 20.83% | 224,496 |
| `cudnn_generated_matMul_pointwise_pointwise_cutlass_sm80_knob_41` | 19.51% | 224,496 |
| `cudnn_generated_matMul_pointwise_cutlass_sm80_knob_41` | 15.41% | 18,707 |
| `cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16` | 5.32% | 336,744 |
| `norm_forward_warp_kernel` | 3.14% | 561,240 |
| `embedding_forward_kernel` | 0.44% | 37,416 |
| masking (`attention_sdpa_lengths`, `token_valid_lengths`) | 0.24% | 374,160 |

`cuda-transformer-infer`, PyTorch, aggregated by kind:

| kind | share | launches |
|---|---|---|
| Inductor GEMM templates (`triton_tem_*`) | 40.89% | 692,196 |
| CUTLASS GEMMs | 36.44% | 561,240 |
| `pytorch_flash::flash_fwd_kernel` | 14.93% | 336,744 |
| Inductor pointwise/reduction (`triton_poi/per/red_*`) | 4.88% | 1,047,648 |
| ATen elementwise and layer-norm | 2.86% | 448,992 |

Matrix products are 90.9% of OpenNN's inference kernel time and 77.3% of
PyTorch's. One honest note on that: the family
`cutlass_80_tensorop_bf16_s16816gemm_relu_bf16` appears on *both* sides — it
is 35.11% of OpenNN's step in one instantiation, `256x128_32x3_nn`, and
36.44% of PyTorch's in two, `128x256_32x3_tn` and `64x64_32x6_tn`. Both
libraries select from the same CUTLASS family and land on different tiles and
different layouts. About a third of each engine's inference step is the same
vendor kernel, and where that kernel is the work neither engine has an
advantage.

`cuda-transformer-train`, OpenNN:

| kernel | share | launches |
|---|---|---|
| `cutlass_80_tensorop_s16816gemm_bgrada_bf16_64x64_32x6_nt` | 12.52% | 455,265 |
| `adam_update_kernel<__nv_bfloat16>` | 10.52% | 467,774 |
| `cudnn_generated_fort_native_sdpa_sm80_flash_bprop_wmma_f16` | 9.08% | 112,266 |
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_nn` | 8.90% | 448,992 |
| `cutlass_80_tensorop_s16816gemm_bgrada_bf16_128x128_64x3_nt` | 8.03% | 112,260 |
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn` | 6.29% | 299,328 |

`cuda-transformer-train`, PyTorch, aggregated by kind:

| kind | share | launches |
|---|---|---|
| Inductor GEMM templates (`triton_tem_*`, `triton_mm`) | 42.26% | 910,456 |
| Inductor pointwise/reduction (`triton_poi/per/red_*`) | 19.83% | 3,336,260 |
| CUTLASS GEMMs | 14.83% | 342,980 |
| `pytorch_flash` forward and backward | 12.28% | 448,992 |
| Inductor *foreach* Adam (`triton_for_*`) | 9.85% | 81,066 |
| ATen elementwise, plus the RNG kernel | 0.96% | 236,968 |

The same buckets on OpenNN's training step: matrix products 69.44%,
attention including its cuDNN helpers 11.97%, Adam 10.52%, everything else
8.07% in 841,995 launches.

### `cuda-transformer-train`, 1.181×: what still spills into its own kernel

Per batch: OpenNN 24,078 µs against PyTorch's 27,948 µs, from the
throughputs above. In the timed window OpenNN issues **573 kernels per step**
taking 23.77 ms of GPU time; PyTorch issues **859** taking 27.71 ms. The
kernel-time ratio, 1.166×, accounts for the 1.161× measured at `93cc90e07` on
its own (1.181× now, the difference being the autotuned attention plan), and
neither figure exceeds its own engine's wall clock.

The mechanism is what those kernels are. With the optimiser and the
attention-internal helpers excluded on both sides, PyTorch spends **20.8%**
of its training step in standalone pointwise and reduction kernels, in
3,573,228 launches over the window, against OpenNN's **8.1%** in 841,995 —
normalisation 4.41%, cross-entropy and generic tensor 2.52%, ReLU backward
0.74%, embedding 0.31%, masking 0.07%. OpenNN's bias and ReLU ride in the
epilogue of the GEMM that produced the tensor: `combination_operator.cpp` passes
`CUBLASLT_EPILOGUE_RELU_BIAS` and `RELU_AUX_BIAS`, and no standalone bias or
forward-activation kernel appears anywhere in the inference trace. The
residual add does *not* ride there — it is fused into the normalisation
kernel instead, which the trace names as
`norm_forward_warp_kernel<__nv_bfloat16, (bool)1, ...>`, the first template
parameter being `FuseResidual`.

Inductor fuses into its templates as well, and the trace is unambiguous about
it: 37.2% of PyTorch's training kernel time is in `triton_tem_*` GEMM
templates whose fused names carry dropout, layer norm or threshold. The
difference is not fusion against no fusion; it is how much still spills into
kernels of its own.

Issue cost is real on this cell and it is small. OpenNN's step is
graph-captured, and evidence for that is in the trace: the 74 gaps between
its 75 per-step Adam launches are 160 or 192 ns for 99.8% of them and never
above 800 ns. Over the timed windows the GPU is busy 98.7% of the time on
OpenNN and 99.1% on PyTorch — PyTorch is marginally the *busier* of the two,
so it is not host-starved and issue cost cannot be the explanation. The
direct isolation agrees: `OPENNN_NO_CUDA_GRAPH=1` read 1,293 sequences/s at
commit `918805ce1` against 1,297 in the nearest publish round at `38ad27e16`,
and 5,254 against 5,274 on inference — graph replay is worth well under 1%,
on a cross-commit pair. The previous version of this document was wrong to
lead with launch cost.

Three further things belong here, and the first two cut against the result:

*Adam is not a differentiator.* OpenNN's Adam is 10.52% of its step, in 75
launches per step. PyTorch's compiled *foreach* Adam appears as twelve
`triton_for_fused_*` kernels — `for` is Inductor's foreach codegen — ten of
which carry the time, totalling 9.85%. Slightly cheaper. OpenNN's update was
examined this session and deliberately left alone: it runs at
778.9 GB/s, 86.9% of the pin bandwidth, and the launch-count hypothesis died
on the graph capture. There is no margin to take here.

*Dropout.* `nn.TransformerEncoderLayer` and `DecoderLayer` default to dropout
0.1 in every sub-layer and the driver keeps that; OpenNN's model has none.
Inductor fuses the mask into its GEMM templates rather than emitting separate
kernels for it — four of the five largest `triton_tem_*` kernels carry
`native_dropout_backward` in their names — but those same kernels also do the
matrix products, so no share read off the profile is what dropout costs. The
RNG itself is negligible: `distribution_elementwise_grid_stride_kernel`,
0.006% of the step, one launch per step. What prices it is the variant:
`PT_DROPOUT=0` read 1,140 sequences/s at commit `918805ce1` against 1,124
with dropout on at commit `38ad27e16`, a 1.4% effect — but both of those runs
are `compile:default`, not the max-autotune mode the cell publishes and in
which the fusion above happens. Cross-commit and out of mode: read it as an
order of magnitude, not a measurement. It is recorded rather than removed,
because dropout 0.1 is what the library's layer does by default.

*Memory.* 2,321 MiB against 3,350 — 2,233 at `e76425bd3`, the 88 MiB being
the workspace of the autotuned attention plans. The library's own attribution
was taken for the cell (`OPENNN_MEMORY_DEBUG=1`, one epoch at batch 32,
device figures): the forward arena is 940 MiB, and that is its planner's
lower bound — the 238 gradient entries of the backward pass are co-planned
into the same arena by lifetime, so no separate gradient arena exists;
Adam's state is 428 MiB, a bf16 first moment and an fp32 second moment over
74.9 M parameters; the fp32 master is 286 MiB and its bf16 mirror 143; the
transient pool 73 MiB and the shared scratch 17. That is 1,887 MiB of
buffers under the 2,233 MiB reading at `e76425bd3`, and the remainder is the CUDA context
and the kernel images cuDNN and cuBLASLt load, which the metric charges to
both engines alike. Nothing in the list shrinks without changing what the
optimiser computes: the arena is the saved-activation set in bf16, the same
set PyTorch keeps, and the second moment is fp32 on both sides. PyTorch's
extra 1,140 MiB is not decomposed here — no allocator trace was taken on its
side — so the 1.51× is measured, and only OpenNN's half of it is explained.

### `cuda-transformer-infer`, 1.153×: mostly one attention kernel

Per batch: OpenNN 5,912 µs, PyTorch 6,817 µs, and 627 MiB of device memory
against 1,161.

The launch-count story does not apply to this cell at all. In the timed
window OpenNN issues **167 kernels per batch** and PyTorch **165** — PyTorch
issues slightly fewer, which cuts against us — and both engines keep the GPU
99.4% busy. OpenNN replays the whole forward pass as one CUDA graph and
PyTorch issues it from Inductor's generated code, and on this workload that
makes almost no difference: the kernels are large enough that the host keeps
ahead either way. The margin is 5.97 ms of OpenNN kernel time per batch
against 6.78 ms of PyTorch's, a ratio of 1.136 that lands on the measured
1.136.

Most of that 0.81 ms is one kernel. Both engines run 18 attention launches
per batch on identical grids — 2 × 8 × 32 blocks of 128 threads — and
OpenNN's `cudnn_generated_fort_native_sdpa_sm80_flash_fprop` takes **17.63 µs**
against **56.24 µs** for PyTorch's `pytorch_flash::flash_fwd_kernel`. That is
0.317 ms per batch against 1.012 ms: **0.695 ms of the 0.814 ms kernel-time
gap, 85% of this cell's whole margin.**

Little of that is to our credit, and it belongs here rather than buried in
the caveats. Two things could produce it and the trace separates neither.
First, a backend default: PyTorch exposes a cuDNN attention backend and the
driver never selects it — `families/transformer.py` sets only `allow_tf32` —
so `scaled_dot_product_attention` falls to its own FlashAttention path, and
that path is built against the cuDNN 9.23.2 the artifact reports where OpenNN
runs 9.25.1. Second, a mask asymmetry: OpenNN's attention honours the padding
lengths and the decoder's causal mask, PyTorch's attends over every position,
so OpenNN's blocks may be exiting early on work PyTorch performs in full. The
grids are identical, so the profile cannot tell the two apart, and no run
exists with PyTorch's cuDNN backend forced or with OpenNN's masks removed.
Either way the largest single component of this cell's margin is a backend
default, a library version or a model asymmetry rather than engine work.

Training is not this. There OpenNN's attention and its cuDNN helpers are
11.97% of the step against PyTorch's 12.28% — near parity, because the
backward dominates and the two backward kernels cost 119.94 µs and 127.02 µs
per launch. In milliseconds that is 2.85 against 3.40 per step, 14% of the
training margin against 85% of the inference one.

The other thing that moved this cell is not work done on it either. The two
`cudnn_generated_matMul_pointwise*` kernels in the table above, **19.51% +
15.41% = 34.9% of the step**, are the cuDNN matmul plan that was found for
`cuda-dense-infer` by enumerating cuDNN's engine set, where all 13,460 valid
cuBLASLt configurations, five CUTLASS 3.8 tile shapes and six hand-written
mma.sync kernels had failed. OpenNN's matmul dispatcher selects it here on
shape, without anyone having considered this cell. Scaled to what it moved:
between the `2026-09-03-publish` round (commit `6b7179dde`) and this one,
OpenNN's inference throughput went 5,302 → 5,335 samples/s, **+0.62%**, while
PyTorch's went 4,707 → 4,695; peak memory went the other way, 844 → 863 MiB
(and to 618 this round, for the reason given under *Memory* below).

Energy over the same two rounds fell from 11.9895 Wh to 11.5286 and mean
board power from 229.4 W to 221.9, while PyTorch's barely moved (14.9908 →
14.9475 Wh, 254.7 → 253.2 W), and the energy ratio went from 1.250× to
1.297×. That fall is **not** attributed to the plan. On the dense cell the
same plan is the fast, *hot* option — 227.7 W against 169 W for the lean
cuBLASLt nvjet tile it displaces — so a 7.5 W drop is the opposite of what it
predicts. There `cuda-dense-infer` publishes 1.019× throughput and 1.058×
energy where the variant with the plan disabled reads 0.966× and 1.315×:
winning its throughput axis cost it energy, deliberately, its energy ratio
falling from 1.339× to 1.058×, and the throughput it now wins there is
measured against a faster PyTorch than the previously published 1.004× faced,
whose autotune cache a reboot had cleared and which re-tuned. `dense.md`
carries both. No `OPENNN_CUDNN_MATMUL=0` run exists for *this* cell, so what
lowered its power between those two commits is not established by any
measurement available.

*Memory.* 627 MiB against 1,161 — 618 at `e76425bd3`, plus the autotuned
attention plan's 9 MiB — where the table before that read 863. The
245 MiB that left is the fp32 master copy of the parameters, which the
inference driver was keeping on the device beside the bf16 mirror the
forward pass actually reads. The library has always had a path that
releases it — `upload_parameters_bf16_inference()`, which casts each
parameter slot on the host, uploads the bf16 (and the few fp32) slots into
compact device storage and drops the master — but it was reached only
through the model-loading functions, and the benchmark driver uploaded
through `copy_parameters_device()`, which migrates the master and builds
the mirror beside it. The driver now deploys through the public API before
its warm-up pass (`families/transformer.cpp`), which is the same step the
PyTorch driver takes with `model.to(torch.bfloat16)` under the published
`PT_INFER_CAST=weights` mode: both engines now hold the 74.9 M parameters
once, in bf16. Throughput and energy did not move with the release (5,335 →
5,335 samples/s; 11.53 → 11.55 Wh at `e76425bd3`, inside the launch-to-launch
spread; the 5,413 in the table is the attention autotune described under
*What is measured*) and the quality gate agrees, as it should — the same
bf16 bits are read either way.

What the 627 MiB is: the library's attribution rows for the cell give a
161.5 MiB inference arena (its planner's lower bound; the output projection
onto the 20,000-word vocabulary is 157 MiB of it), the 143 MiB bf16
mirror, and under 1 MiB of batch buffers; the remaining ~310 MiB is the
CUDA context with cuDNN's and cuBLASLt's kernel images, including the fused
attention plan. PyTorch's 1,162 MiB is not decomposed here.

The bf16-weights choice matters on PyTorch's side more than in any other
family — 4,657 sequences/s at `d3acd71b5` against 3,641 for the same compile
mode under autocast at `8e47e7662` — because autocast reads the 74.9 M
parameters as 300 MB of fp32 weights and writes a 150 MB bf16 copy on every
call; storing them in bf16 once removes that traffic, and it is the mode the
table uses. OpenNN's forward pass reads its bf16 mirror from the start; what
changed this round is that the fp32 master no longer sits beside it.

### Where the energy goes

Both cells win energy on **power**: OpenNN draws 223.9 W against PyTorch's
254.6 W on inference and 198.7 W against 234.7 W on training. Combined with
margins of 1.153× and 1.181× on time, that compounds to 1.310× and 1.395× on
energy — the widest energy margins in the matrix outside the LSTM family.

Part of that is the GEMM tile-selection rule described in the dense document,
which prefers a tile shape that moves less memory when the throughput cost is
within tolerance. On training the rule was isolated in the
`2026-09-02-gemmenergy` session at commit `bc6f4c2d0`: tolerance 0 read
17.338 Wh at 1,305 sequences/s, the rule read 16.405 Wh at 1,298 — **5.4% of
the cell's energy for 0.5% of its throughput.** The same caution that
withdraws the inference split applies here at smaller scale: the cuDNN plan
now serves 8.8% of the training step too, on shapes the tile rule would
otherwise govern, and the cell has moved from that commit's 1,298 sequences/s
at 16.405 Wh to 1,329 at 16.343. The 5.4%-for-0.5% split is the best
available measurement of the rule, not a measurement of today's cell.

The same isolation on inference is **not carried forward**. The paired run
there was taken before the cuDNN plan existed, and the run with the rule
enabled did not pass the quiet gate. Since the plan now covers a third of the
inference step, the old split no longer describes the current cell, and no
tolerance-0 rerun exists at `93cc90e07`. The figure the previous version of
this document quoted for inference has been removed rather than restated.

Part of the power gap is that PyTorch is doing extra element-wise work — on
training the 20.8% against 8.1% above, of which the dropout its layers apply
by default is one component; on inference the same split is narrower, 7.7%
against 3.8%, and dropout does not run at all under `model.eval()`. The
element-wise ledger does not run entirely our way, though: OpenNN's masking
kernels are extra work PyTorch does not do, 0.24% of its inference kernel
time, while the mask they compute may be what makes OpenNN's attention kernel
cheap. Neither effect is isolated.

## Asymmetries and caveats

Three things differ in what the two networks compute, all recorded here
because none of them is something the runner can equalise without changing
one engine's model:

- **Masks.** OpenNN's decoder self-attention is causal and every attention
  block sees the padding mask that the embedding layer exports with its valid
  lengths; the PyTorch model applies no mask at all — no causal mask in the
  decoder, no key-padding mask anywhere. Same tensor shapes; OpenNN pays
  0.24% of its inference kernel time and
  0.07% of its training step to build the masks, and trains the correct
  model, where PyTorch trains a model that can see the future. But the mask
  is also a candidate explanation for OpenNN's attention kernel running
  17.63 µs against PyTorch's 56.24 — blocks that can exit early on padding
  and causality. A masked PyTorch model is the fairer comparison, and on this
  evidence it might well be *faster*, not slower.
- **Dropout.** `nn.TransformerEncoderLayer` and `DecoderLayer` default to
  dropout 0.1 and the driver does not override it; OpenNN's dropout operator
  defaults to 0 and the model builder leaves it there. Dropout in training
  is extra element-wise work (a random mask, a multiply) on every attention
  and feed-forward output on PyTorch's side, fused by Inductor into its GEMM
  templates rather than emitted separately. `PT_DROPOUT=0` runs the PyTorch
  model without it, and the *Why* section gives what it is worth (1,140
  against 1,124 sequences/s, across two commits and both under
  `compile:default` rather than the published mode). Inference is unaffected
  (`model.eval()`).
- **The loss denominator.** PyTorch's `CrossEntropyLoss` averages over every
  position, padding included; OpenNN's `CrossEntropyError3d` averages over
  the valid tokens. Identical work, different scale of the gradient — the
  learning rate is the same on both sides, so the two models do not follow
  the same trajectory; there is no accuracy gate in this family to be
  affected by it.

And the ones that are about the build rather than the model:

- **PyTorch's attention backend is a default we did not override.** The
  driver never calls `torch.backends.cuda.enable_cudnn_sdp` or selects a
  backend at all, so `scaled_dot_product_attention` runs its FlashAttention
  path while OpenNN runs cuDNN's fused SDPA. That single kernel is 85% of the
  inference margin. No run exists with PyTorch's cuDNN backend forced, so how
  much of the cell would survive one is not known.
- **A shared vendor kernel.** The GEMM family
  `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16` runs on both sides — 35.1%
  of OpenNN's inference step in one instantiation, 36.4% of PyTorch's in two.
  Where that kernel is the work, neither engine has an advantage; the margin
  comes from the shapes each library routes to it and from what surrounds it.
- **The cuDNN plan is not isolated on this cell.** The 34.9% attribution
  above rests on the profile and on two rounds' worth of dates, not on a
  variant run with the plan disabled. Only `cuda-dense-infer` has that run,
  and the energy half of the attribution is withdrawn above.
- **Different cuDNN builds**, 9.25.1 against 9.23.2 (`torch_built_cudnn`
  reads 92302 in the artifact), as in the CNN family. Some of OpenNN's
  attention and matmul advantage may be the newer library rather than the
  engine, and nothing here separates the two.
- **Different cuBLAS builds.** `ldd` on `transformer_opennn` resolves
  `libcublasLt.so.13` to CUDA 13.3; the benchmark environment ships
  `nvidia_cublas 13.1.1.3` for PyTorch. This is visible, not theoretical:
  `nvjet_sm120_*` kernels, cuBLASLt's sm120-native path, are 20.8% of
  OpenNN's inference kernel time and 17.3% of its training step, and a query
  for `nvjet` or `sm120` over either PyTorch trace returns nothing — its
  GEMMs land on sm80 CUTLASS kernels and Triton templates throughout. How
  much of the GEMM margin is the newer library rather than the dispatcher is
  not separated here.
- **The memory column is occupancy, not live footprint.** The artifacts
  record `memory_metric = device_used_minus_idle`: NVML device-used less a
  per-launch idle baseline (251 MiB for OpenNN, 241 for PyTorch on the
  inference run). That counts PyTorch's caching allocator's reserved pool,
  including blocks it holds but is not using, so part of the 1.35× and 1.51×
  is allocator policy rather than a smaller working set.
- **Compile-mode and cast variants are dated.** The 1,150/1,132 and
  4,657/4,574/4,499/3,641 figures come from the `2026-09-02-variants` session
  at commits `918805ce1`, `8e47e7662` and `d3acd71b5`; only the 4,574/4,499
  pair is same-commit. They establish which mode is fastest, which is what
  "each engine at its best" needs; they are not current absolute levels, and
  PyTorch's inference number has since risen above the best of them.
- **Both cells' PyTorch figures are a draw from an autotune cache.**
  `max-autotune-no-cudagraphs` benchmarks 22 kernel choices per GEMM and
  caches the winner on disk. That cache is not fixed: on `cuda-dense-infer` a
  reboot cleared it, PyTorch re-tuned and found a kernel about 4% faster than
  the one behind the previously published number. This family's PyTorch
  readings have held to 4,695–4,708 on inference and 1,145–1,151 on training
  across the last three commits' rounds, so no such shift is visible here,
  but a re-draw could
  move PyTorch's side of these two cells by a percent or more in either
  direction.
- **No accuracy gate.** Both drivers print tokens per second alongside
  samples per second and the runner compares the sample count, sequence
  length, vocabulary and parameter count (74,878,496); neither reports a
  translation metric. The shape gate is what holds this family.
- **Warmup is one epoch on both sides,** with the graph capture, the
  `torch.compile` trace and the SDPA kernel selection inside it.
- **An earlier version of this table timed OpenNN inference over zeros.**
  Until commit `4338506c8` the OpenNN inference drivers filled each batch
  (`Batch::fill()`) and never uploaded it: `fill()` only stages the rows on
  the host, and in the library the transfer is issued by the optimizer, which
  an inference driver does not run. The forward pass ran over all-zero inputs
  for every published inference cell before that commit. Nothing in the gates
  could see it — the parameter count, the sample count and the input file are
  the same whatever the batch holds, and the arithmetic of a forward pass does
  not depend on the values — and it was found by printing outputs (every one
  read `sigmoid(0) = 0.5`). The transformer is the one family where it
  mattered: the embedding exports the number of non-zero token ids as each
  sequence's valid length (`compute_token_valid_lengths`), so every sequence
  reached the attention kernels as all padding — no valid keys instead of 130.
  Attention is a small share of this model's time (5.3% on inference in the
  profile), so the cell moved by about 1%. The rows above are from the fixed
  drivers, which upload the batch (`upload_to_device_batch_async()`) after
  filling it and, where the split is resident, take the batch as a device
  view; the session before the fix read 5,274 sequences/s for OpenNN
  (`cuda-transformer-infer-publish-20260902T041233Z.json`).

## Reproduce

```bash
export OPENNN_BENCH_SESSION=$(date +%F)-mine
python run.py --family transformer --mode train --device cuda --batch 32 --precision bf16 --epochs 2 --rounds 3
python run.py --family transformer --mode infer --device cuda --batch 32 --precision bf16 --repeats 5 --rounds 3
```

`prepare.py transformer` downloads WMT14 News Commentary v9, tokenises it
with the same rule both engines read (`opennn_tokens`: ASCII-lowercased
alphanumeric runs and single punctuation marks, `--max-tokens 128`,
`--max-pairs 200000`) and writes the 199,575 pairs both engines load.
`PT_DROPOUT=0`, `PT_COMPILE_MODE=default|reduce-overhead|max-autotune-no-cudagraphs|eager`
and `PT_INFER_CAST=autocast` are the PyTorch knobs; `OPENNN_NO_CUDA_GRAPH=1`
runs OpenNN without graph replay, and `OPENNN_LT_TILE_TOLERANCE=0` disables
the GEMM tile-selection rule.

The four traces the *Why* section rests on are not committed — 3.8 GB of
SQLite between them. They were taken this session with

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --force-overwrite true -o <name> <driver>
nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true <name>
```

run against the family drivers directly, one epoch of training and three
passes of inference. Every share above is computed over the interval the
driver prints as `TIMED_START_UNIX`–`TIMED_END_UNIX`, so a re-profiled run
is comparable to these numbers without matching the trace length.
