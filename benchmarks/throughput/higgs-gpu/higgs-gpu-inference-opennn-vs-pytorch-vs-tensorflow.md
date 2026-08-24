# GPU HIGGS dense inference: OpenNN vs PyTorch vs TensorFlow

This benchmark is measured across a **ladder of batch sizes** (256 to 65,536), not
at one batch, and with each engine at the configuration it is fastest in *at that
batch size*. Both of those changed the answer.

On an NVIDIA GeForce RTX 5070 Ti, medians of six rotated rounds, 2026-08-21,
artifact `results/gpu-higgs-infer-sweep-headtohead-cutlass-20260821T111358Z.json`:

* **fp32 (TF32): OpenNN leads both engines at every batch size** - 1.12x to 1.20x
  PyTorch and 1.04x to 1.15x TensorFlow, ahead in every one of the six rounds.
* **bf16: OpenNN leads TensorFlow at every batch size** (1.10x to 1.42x, ahead in
  every round) and leads PyTorch at every batch size up to 16,384 - 1.25x at 256,
  1.45x at 1,024, 1.06x at 4,096, 1.02x at 8,192, 1.005x at 16,384, each of them
  in all six rounds. **65,536 is the one cell PyTorch still holds**, by 1.6%.

The bf16 large-batch gap has a mechanism, and it is not the one this note used to
give: OpenNN fuses *more* than PyTorch does - three CUDA graph nodes a batch
against PyTorch's four - and what PyTorch has is better GEMM kernels.
`torch.compile`'s Triton templates beat cuBLASLt's best-of-eight by 2% on the
1024x1024 layer and by up to 32% on the 28->1024 one, and no cuBLASLt
configuration reaches them (three were measured, none kept).

> **2026-08-11 TF32 correction.** The previously published fp32 lead (1.54x
> PyTorch) compared OpenNN running TF32 against PyTorch/TensorFlow running
> strict fp32. OpenNN's GPU fp32 GEMMs always use TF32 tensor cores; the
> PyTorch and TensorFlow drivers now enable TF32 for their fp32 cells too, and
> at that point all three engines saturate the same GEMM roofline: OpenNN
> 17.13M, PyTorch 16.97M, TensorFlow 17.22M samples/s — a tie within ±1%.
> The honest fp32 story here is parity, and bf16 is where OpenNN's margin is.

> **2026-08-20 protocol correction.** Two changes to how these are measured, and
> both moved numbers. The runner measured engines in blocks -- all five runs of
> one, then the next -- so GPU state drifted between blocks by more than the
> margins being compared; on the training benchmark that was worth three points
> on a two-point effect. Engines now alternate within a round with the starting
> engine rotating. And the GPU clock is pinned for the measurement
> (`benchmarks/tools/gpu_clocks.sh`): this card idles near 400 MHz, takes
> ~2.5 s of load to reach boost, and its sustained clock drifts with ambient
> across a session, which alone moved one engine's reading 8% across a day.
> Pinning costs ~6% of absolute throughput -- these figures are lower than the
> 2026-08-19 run for that reason -- and every engine pays it equally, which is
> what makes the ratios comparable.
>
> The PyTorch cells also moved because PyTorch is now measured at its best
> configuration (`PT_COMPILE_MODE`, `PT_BF16_WEIGHTS`). Its fp32 margin was
> previously reported as 1.308x and is 1.113x once it is given max-autotune.

> **2026-08-19 TensorFlow dispatch correction.** The TensorFlow driver called
> its XLA-compiled step once per batch from Python. That costs ~0.23 ms of eager
> dispatch per batch, and TensorFlow enqueues asynchronously, so the cost is
> hidden whenever the GPU work per batch is longer than it and is paid in full
> when it is not. At batch 8192 the bf16 step is ~0.22 ms, just under the
> threshold: enqueueing a pass took as long as enqueueing *and executing* it, so
> the GPU was idle waiting on Python. OpenNN and PyTorch were not exposed to
> this, because a captured-graph replay costs one cheap launch. The driver now
> also offers the batch loop compiled inside a single `tf.function`, times both
> paths, and reports the faster with `tf_path` naming the winner. Which one wins
> depends on precision: compiling the loop is worth +11% in bf16 and -5% in
> fp32. The per-batch results are bit-identical between the two paths.
>
> The reference machine moved to the RTX 5070 Ti with this correction. The RTX
> 4080 numbers this note used to lead with were taken on the old driver, on a
> machine no longer available, and their artifact
> `gpu-higgs-dense-inference-speed-20260810T123521Z.json` was never committed --
> `results/` is ignored by default and reviewed evidence is promoted with
> `git add -f`, which 29 artifacts are, none of them for this benchmark. They
> are kept below for provenance and should not be cited.

> bf16 results are medians across five runs. CUDA Graphs are active in the OpenNN and PyTorch paths, and both stage each batch through the same device-to-device copy pattern.

## The batch sweep (2026-08-21)

Everything above this section was measured at **one** batch size, 8,192, with each
engine pinned to one configuration. Both of those turned out to be load-bearing.
This section replaces the Results table below; that table is kept for provenance
and should not be cited.

### The instrument

The same protocol the CPU family converged on, ported here:

* Each engine sweeps the whole batch ladder **inside one process**, so the rungs
  of a row share one load and one thermal window. All three drivers take a
  comma-separated batch list.
* The **engine order rotates** every round and the **batch order rotates** with
  it. The first round is a **soak** and is discarded. Six rounds are kept.
* Every driver prints `batch_<B>_pass_times=` in temporal order before the
  median, so drift stays in the artifact instead of being averaged away.

The runner is `run_higgs_infer_sweep.py`. It also runs A/Bs: two arms of the same
engine differing in one environment variable alternate against each other under
exactly this protocol, which is how every lever below was decided.

Why it matters here, with the clock already pinned: absolute readings still move
3-15% round to round, but the **paired ratio does not**. At bf16 batch 65,536 the
six OpenNN/PyTorch ratios are 0.933 0.933 0.942 0.934 0.924 0.938 - a 2% spread
on cells whose absolute values move 7%. Rotation is what converts an unusable
absolute into a decidable ratio.

### Giving PyTorch its best configuration, which is not one configuration

The runner used to hand PyTorch `torch.compile(mode="reduce-overhead")` for bf16
and `max-autotune` for fp32, one setting for every batch size. That is not where
PyTorch is fastest, and it is not fastest at any single setting. Medians of three
rotated rounds, bf16, samples/s:

| batch | eager + hand-captured CUDA graph | compile reduce-overhead | compile max-autotune |
|---|---:|---:|---:|
| 256 | **12,491,364** | 4,555,869 | 4,350,403 |
| 1,024 | **19,210,673** | 17,976,775 | 17,544,645 |
| 8,192 | 29,370,503 | 28,603,963 | **37,018,545** |
| 65,536 | 25,480,267 | 25,464,720 | **36,726,327** |

The crossover sits between 1,024 and 8,192 and the penalty for being on the wrong
side of it is large in both directions: **2.9x** at 256 for taking inductor,
**1.44x** at 65,536 for taking the graph. The old pinned setting measured PyTorch
**2.7x under itself** at batch 256 and **29% under itself** at 8,192, which is
most of the margin this note used to report.

`pytorch_higgs_infer.py` now times its candidate paths at every rung and reports
the faster, naming it in `pt_path` - the same contract `tensorflow_higgs_infer.py`
has had for its two dispatch paths since 2026-08-19. fp32 behaves the same way
(max-autotune wins from 1,024 up, the graph wins at 256).

TensorFlow was checked for the same failure and one candidate was found and
rejected: holding the resident rows as bfloat16, which is what the other two
engines do. Alternated pairs at 256 / 8,192 / 65,536 measured 10.92 / 32.68 /
24.57 M samples/s with it against 11.96 / 32.66 / 24.44 M without - **-9% at 256
and a tie above**, so `TF_BF16_INPUT` defaults off. XLA already folds that cast
into the first layer's operand read.

### Results

Medians of six rotated rounds, GPU clock pinned to 2692 MHz, each engine at its
own best path per rung. Artifact
`results/gpu-higgs-infer-sweep-headtohead-cutlass-20260821T111358Z.json`. The
ratio is the **median of the six paired per-round ratios**, not the ratio of the
medians - at a cell this tight the two differ, and the paired one is the honest
statistic. "rounds ahead" is how many of the six put OpenNN in front.

**bf16**, samples/s:

| batch | OpenNN | PyTorch | TensorFlow | vs PyTorch | rounds ahead | vs TensorFlow | rounds ahead |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | **15,630,336** | 12,492,117 | 12,329,697 | **1.25x** | 6/6 | **1.27x** | 6/6 |
| 1,024 | **27,778,160** | 19,214,902 | 24,159,787 | **1.45x** | 6/6 | **1.15x** | 6/6 |
| 4,096 | **35,718,674** | 33,725,753 | 30,957,634 | **1.06x** | 6/6 | **1.15x** | 6/6 |
| 8,192 | **37,705,596** | 36,943,350 | 34,365,975 | **1.02x** | 6/6 | **1.10x** | 6/6 |
| 16,384 | **37,705,824** | 37,519,636 | 32,230,600 | **1.005x** | 6/6 | **1.17x** | 6/6 |
| 65,536 | 36,051,926 | **36,641,306** | 25,456,413 | 0.98x | 0/6 | **1.42x** | 6/6 |

**fp32 (TF32)**, samples/s:

| batch | OpenNN | PyTorch | TensorFlow | vs PyTorch | rounds ahead | vs TensorFlow | rounds ahead |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | **11,357,390** | 9,586,884 | 10,318,808 | **1.19x** | 6/6 | **1.10x** | 6/6 |
| 1,024 | **15,953,970** | 13,857,177 | 13,857,905 | **1.15x** | 6/6 | **1.15x** | 6/6 |
| 4,096 | **18,665,322** | 16,483,464 | 17,075,574 | **1.13x** | 6/6 | **1.09x** | 6/6 |
| 8,192 | **18,560,815** | 16,540,616 | 17,901,600 | **1.12x** | 6/6 | **1.04x** | 6/6 |
| 16,384 | **18,304,760** | 15,272,418 | 17,268,688 | **1.20x** | 6/6 | **1.06x** | 6/6 |
| 65,536 | **18,014,210** | 15,221,426 | 17,227,321 | **1.18x** | 6/6 | **1.05x** | 6/6 |

The one cell PyTorch holds is bf16 at 65,536, where the six per-round ratios are
0.981-0.986. It is a real loss and a consistent one, and what remains inside it
is described under [what is left](#what-is-left): two thirds of it is the square
GEMM, not the narrow layer.

### Where the batch goes, kernel by kernel, against PyTorch

`nsys profile --cuda-graph-trace=node` on both engines, bf16, nanoseconds per
batch. This is the whole of each step: the columns sum to the measured batch time
within 2%.

| step | 256 OpenNN | 256 PyTorch | 8,192 OpenNN | 8,192 PyTorch | 65,536 OpenNN | 65,536 PyTorch |
|---|---:|---:|---:|---:|---:|---:|
| 1024x1024 GEMM + bias + ReLU | 9,402 | 8,589 | 202,632 | **197,682** | 1,618,758 | **1,582,779** |
| 28->1024 GEMM + bias + ReLU | 1,911 | **1,242** | 21,504 | **14,726** | 179,991 | **150,762** |
| 1024->1 head (+ bias) | 1,133 | 3,769 | **7,442** | 6,659 | **151,275** | 155,153 |
| sigmoid | 980 | fused | 978 | fused | 1,355 | 1,173 |
| fp32 -> bf16 cast of the input | 670 | none | 1,022 | none | 3,490 | none |
| per-batch staging copy | 1,900 | 1,267 | **2,054** | 5,986 | 10,847 | **6,327** |

Three things fall out of that table, and only one of them was expected:

1. **The activation fusion argument was backwards.** This note used to attribute
   the PyTorch margin to OpenNN fusing ReLU where PyTorch does not. Inductor's
   `max-autotune` emits Triton matmul templates that carry bias and ReLU in the
   epilogue exactly as cuBLASLt does, *and* carries the final sigmoid too. It was
   OpenNN that had the unfused activation.
2. **What is actually left is kernel quality.** On the square GEMM - 87% of the
   batch - PyTorch is 2.5% faster; on the thin first layer, 46% faster at 8,192
   and 19% at 65,536. Everything else is a rounding error beside those two.
3. **OpenNN wins the head and the staging.** Its warp-per-row reduction beats
   cuBLASLt's GEMV by 1.6x (0.0118 ms best-of-8 against 0.00744 at 8,192), and
   its device-to-device memcpy beats PyTorch's copy *kernel* at 8,192.

### What changed, and what it was worth

Two levers, both from that table, both kept:

* **The output layer's activation rides in the reduction kernel.** A layer with
  one output produces one value per row, so its activation was a whole kernel
  launch to read and write one number per row.
  `linear_forward_single_output_cuda` now takes an ActivationFunction and applies
  it to the value already in the register.
  `OPENNN_SINGLE_OUTPUT_ACTIVATION=0` keeps the separate pass.
* **The resident rows are staged in the network's compute type.** They were fp32
  against bf16 weights, so every batch ran a cast kernel inside the graph and
  moved twice the bytes it needed to. PyTorch's driver already did this
  (`PT_BF16_WEIGHTS`); OpenNN's did not. `OPENNN_INFER_STAGE=fp32` restores it.

Together they take the inference graph from **five nodes a batch to three**.
Four arms alternated under the protocol above, medians of four rotated rounds:

| batch | bf16 both / neither | bf16 fusion alone | fp32 both / neither |
|---|---:|---:|---:|
| 256 | **1.125x** | 1.008x | **1.090x** |
| 1,024 | **1.058x** | 1.057x | **1.023x** |
| 4,096 | 1.019x | 1.010x | 1.003x |
| 8,192 | 1.018x | 1.005x | 1.002x |
| 16,384 | 1.006x | 1.001x | 1.001x |
| 65,536 | 1.006x | 1.001x | 1.000x |

The shape is what a fixed per-batch cost looks like: worth 13% where a batch
costs 18 microseconds and nothing where it costs two milliseconds. In bf16 at 256
the two levers are **not additive** - removing either one alone leaves the pass
at the same 16.4 microseconds, and only removing both costs 11% - which says the
last node removed at that rung was already hidden behind the others.

The fp32 column doubles as a control on the instrument: `OPENNN_INFER_STAGE` is a
no-op in fp32, because the staged type and the compute type are the same. That
arm ran identical code to the default arm and measured **1.000x at every one of
the six rungs**. An A/B that cannot tell identical code apart from itself is
worth exactly what it says.

`SingleOutputActivationFusion` covers the numerics, including a head of 1,022
features - a width the reduction refuses - which is how the test reaches the
fallback where `linear_forward` has to run the activation itself. Asking for the
fusion is only ever a speed decision, never a correctness one, and that is the
half of the change worth testing.

### The forward GEMM is issued in chunks of rows above 16,384

A third lever, and the only one that touches the item worth 86% of an fp32 batch.
cuBLASLt loses throughput on a very tall operand. The hidden layer alone, top-8
heuristics timed for each shape (`gemm_chunk_probe.cu`), TFLOP/s:

| rows | fp32 one call | fp32 best chunked | bf16 one call | bf16 best chunked |
|---|---:|---:|---:|---:|
| 8,192 | 43.7 | 44.6 | 87.5 | 88.0 |
| 16,384 | 44.7 | 44.7 | 86.8 | 87.8 |
| 32,768 | 41.5 | **44.5** | 81.1 | **86.6** |
| 65,536 | 41.5 | **44.3** | 82.3 | **88.5** |

So `linear_forward` now issues chunks of 16,384 rows above 16,384 rows, one
`cublasLtMatmul` each, sharing one plan. The chunks re-read the four-megabyte
weight panel, which stays in L2, and the extra launches are inside every number
above. Splitting a GEMM by rows is exact - an output row depends only on its own
input row - and the outputs are bit-identical in both precisions.

In the application, four arms alternated, medians of four rotated rounds:

| batch | fp32 off -> chunked | bf16 off -> chunked |
|---|---:|---:|
| 8,192 | 1.000x (gated off) | 1.000x (gated off) |
| 16,384 | 1.000x (gated off) | 1.000x (gated off) |
| 32,768 | **+4.7%** | **+5.0%** |
| 65,536 | **+3.6%** | **+3.8%** |

The gate was swept from both sides. Chunks of 32,768 are *worse* than one call at
65,536 (-1.6% fp32, -0.8% bf16), and below the gate every chunk size loses:
at 4,096 / 8,192 / 16,384 a chunk of 2,048 costs 7.7% / 7.9% / 9.4% in fp32 and
a chunk of 4,096 costs 0.6% / 0.8% / 2.5%. One call is already best there, which
is exactly what the probe's first two rows say. `OPENNN_GEMM_ROW_CHUNK` sets the
chunk and 0 disables it.

What it is worth end to end: **the thinnest fp32 cell against TensorFlow, batch
65,536, goes from 1.014x to 1.048x** on the per-round median, and against PyTorch
in bf16 the same rung goes from 0.934x to 0.971x - cutting that deficit from 6.6%
to 2.9%. Nothing below 16,384 rows moves, by construction.

`GemmRowChunk` covers it. Nothing else in the suite runs a batch tall enough to
take this path, and what a chunked GEMM gets wrong is not arithmetic but offset
arithmetic - a stride applied to the wrong index leaves every row past the first
chunk reading from the wrong place, which a finite-output check would not notice.
The test runs 20,000 rows so that it crosses two chunk boundaries, gives every
row a distinct value so a row served from the wrong offset cannot pass by
coincidence, and reports the chunk index of the first row that disagrees.

### Two rewrites of the head kernel, neither kept

The single-output head is 5.5% of an fp32 batch at 8,192 (24.1 us) and the
profile says why it should be less: it moves 33 MB of input but issues 67 MB of
loads, because every warp re-reads the whole 4 KB weight vector for its one row.

* **One accumulator per vector slot instead of one per lane**, on the theory that
  the multiply-add chain was the limit: 23,424 ns against 23,456. No change, and
  that is the useful part - it says the kernel is not latency-bound.
* **Load the weight slice once per warp and reuse it across a group of rows**,
  which halves the load traffic: fp32 24.1 -> 29.8 us at a group of four and
  28.6 at a group of two. It backfires because at 8,192 rows a group of four
  leaves 64 blocks for 70 SMs, and the occupancy costs more than the traffic
  saves. It helps bf16 (7.44 -> 6.62 us) and a kernel whose sign flips with the
  dtype is not worth having.

The traffic argument is still right and the fix would have to keep the block
count up - more rows per warp only above some row count, or the weight vector in
shared memory per block. It is 9 us of a 442 us batch at 8,192 and nothing at
65,536, where the head is already at DRAM bandwidth.

### Measured and not kept

The thin first layer is OpenNN's largest single deficit, so cuBLASLt was pushed
at it three ways. None of them reached inductor's Triton template.

* **Padding the contraction to k=32** (`l1_align_probe.cu`, top-8 heuristics
  timed the way OpenNN's autotune times them, milliseconds): 0.0027 -> 0.0023 at
  256, 0.0113 -> **0.0163** at 4,096, 0.0217 -> 0.0189 at 8,192, 0.0414 ->
  0.0352 at 16,384, 0.1860 -> 0.1864 at 65,536. It helps at some rungs and
  costs 31% at one of them, and even its best is 29% short of PyTorch's 0.0147.
* **Padding only the leading dimension to 32**, which costs no arithmetic at all
  and was the more promising idea - cuBLASLt's alignment promise is about `ldb`
  and the pointer, not about `k`. Measured 1.000x, 1.004x and 1.003x at 4,096,
  8,192 and 16,384: it changes nothing, so the kernel choice is not gated on
  alignment here.
* **The same padding end to end**, using the `higgs_test_pad32.csv` split so the
  padded input costs nothing to produce: -2.9% at 4,096, +0.9% at 8,192, +1.3%
  at 16,384, -3.3% at 65,536. The standalone gain does not transfer, and a
  library implementation would additionally have to materialise the padded input
  on every call.
* **A wider cuBLASLt autotune** was refuted from the probe rather than built:
  `gemm_probe.cu` puts the square layer's top-1 heuristic and its best-of-8 at
  the same 0.2021 ms. There is nothing for a wider search to find.

### What is left

* **The square layer's chunking gate is per-shape, not universal.** 16,384 rows
  is where it belongs for a 1024x1024 layer on this card, and the chunk that wins
  is the same 16,384 at both 32,768 and 65,536 rows, but nothing here establishes
  that for another width or another card. The probe is the thing to re-run.
### The narrow layer through CUTLASS (2026-08-21)

cuBLASLt's kernels *are* CUTLASS kernels - the profiles name them,
`cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_...` - drawn from a catalogue
compiled for sm_80 and reused here. For a contraction of 28 it can only promise
two-element alignment on the input, so it dispatches an `align2` kernel and
reaches 23.8 TFLOP/s. So the fix is not a different library but the same one,
instantiated for this shape: alignment 4, which 28 does divide, and a
threadblock tile chosen by row count rather than by heuristic. The bias arrives
as a C operand whose row stride is zero, which broadcasts one vector down the
output and lets the ReLU ride in the same epilogue - the fusion the cuBLASLt
path gets from `CUBLASLT_EPILOGUE_RELU_BIAS`.

Milliseconds, `l1_cutlass_probe.cu`, against cuBLASLt's best of eight heuristics,
with **bit-identical output at every point**:

| rows | 256 | 1,024 | 4,096 | 8,192 | 16,384 | 65,536 |
|---|---:|---:|---:|---:|---:|---:|
| cuBLASLt | 0.0022 | 0.0042 | 0.0114 | 0.0216 | 0.0415 | 0.1831 |
| CUTLASS | 0.0022 | **0.0034** | **0.0090** | **0.0156** | **0.0281** | **0.1730** |
| | 1.03x | 1.25x | 1.27x | 1.39x | 1.48x | 1.06x |

Five tiles were swept and the winner moves with the row count, so it is chosen
rather than fixed: 64x64 below 512 rows, 64x128 to 2,048 and above 32,768, and
128x128 between. **The ordering against the row chunking matters.** CUTLASS's
margin is widest at 16,384 rows and narrowest at 65,536, so a large batch must
meet this layer *after* chunking, not before: wired the other way round it took
the 1.06x instead of the 1.48x and measured +1.0% at 65,536 where the chunked
order measures +1.3%.

Two arms alternated, medians of five rotated rounds, bf16:

| batch | 256 | 1,024 | 4,096 | 8,192 | 16,384 | 65,536 |
|---|---:|---:|---:|---:|---:|---:|
| CUTLASS / cuBLASLt | 1.000x | 1.001x | **1.018x** | **1.029x** | **1.028x** | **1.013x** |

**Small batch is untouched and the middle of the ladder gains 2-3%**, which is
what turned 8,192 and 16,384 from ties into wins in all six rounds. The path
declines every shape it does not cover - fp32, a contraction over 32 or not a
multiple of 4, an output width not a multiple of 8, an unaligned pointer, and
every build without CUTLASS, where it compiles to a stub - so it is a fast path
and never a behaviour change. `OPENNN_CUTLASS_NARROW_K=0` keeps cuBLASLt.

Split-K is the only thing CUTLASS would want a workspace for and this never asks
for it, which is what makes the call safe inside a captured CUDA graph: a
steady-state forward must not allocate.

**The dependency.** CUTLASS is header-only and compile-time - it adds nothing to
the shipped binary but the kernels actually instantiated, and the build grew by
about 40 seconds. It is nevertheless a dependency this project did not have, so
it is **opt-in**: `-DOpenNN_CUTLASS_INCLUDE_DIR=<path>/include` enables it and
without it everything compiles and behaves as before. `CutlassNarrowGemm` covers
the three tile boundaries, the chunk gate at 20,000 rows, and the fp32 decline,
and its assertions hold either way - a test that only passed with CUTLASS present
would be testing the wrong promise.

### A hand-written kernel for the same layer, written and not kept (2026-08-21)

The 28->1024 layer is the largest per-step deficit against PyTorch in bf16, so it
was attacked directly rather than through cuBLASLt. `l1_kernel_probe.cu` holds
the kernel and the measurement.

The shape suggests its own design: k is small enough that a thread can keep the
whole weight column it owns in registers - 28 values for two adjacent output
columns - leaving x as the only thing read per row, from shared memory and at the
same address for every thread in the block, which the hardware broadcasts.
Accumulation stays fp32 and f ascends, so it is deterministic; against cuBLASLt's
own output the bf16 results are **bit-identical** at every row count.

It is not fast enough. Milliseconds, best of a swept row tile, against cuBLASLt's
best-of-eight:

| rows | bf16 cuBLASLt | bf16 narrow-k | fp32 cuBLASLt | fp32 narrow-k |
|---|---:|---:|---:|---:|
| 256 | **0.0022** | 0.0030 | **0.0026** | 0.0029 |
| 1,024 | **0.0043** | 0.0056 | 0.0058 | **0.0055** |
| 4,096 | **0.0114** | 0.0153 | 0.0168 | **0.0154** |
| 8,192 | **0.0217** | 0.0266 | 0.0332 | **0.0278** |
| 65,536 | **0.1830** | 0.2344 | **0.3695** | 0.3890 |

**It wins fp32 between 1,024 and 8,192 rows by up to 1.19x and loses bf16
everywhere**, which is the wrong way round: fp32 already leads PyTorch by 19% at
every batch size and bf16 is the cell that needs help. It is not kept.

Two things were learned that are worth more than the kernel:

* **A runtime loop bound over a register array is a fourteenfold loss.** The
  first version took k as an argument, so the compiler could not unroll and put
  the weight arrays in local memory - which is DRAM. It measured 58 GB/s of
  output against the 1,700 the store floor reaches. Templating k on a
  compile-time constant moved it to 630 GB/s with zero spill bytes.
* **Scalar arithmetic is the wrong tool for the bf16 cell.** Each output needs k
  multiply-adds and nothing reduces that: 8.4 M outputs at 28 each is 235 M FFMA,
  which at this card's issue rate is a 9.8 us floor - the same as the floor for
  writing the 16 MB of output. A scalar kernel therefore tops out around 13 us
  however well it is tiled, which is where PyTorch's Triton template already is,
  and it gets there with tensor cores. cuBLASLt reaches 23.8 TFLOP/s on this
  shape precisely because it is using them.

So the remaining path for the bf16 cell was an MMA kernel - which is what the
CUTLASS section above is, reached by instantiating one rather than writing one.
The estimate there was right about the size and the shape of the outcome: 8,192
and 16,384 turned into wins, 65,536 did not.

### The square layer is exhausted inside cuBLASLt

At 65,536 the square GEMM is **66% of the deficit** against PyTorch (45 us of 68),
not the narrow layer. Five levers were alternated against the shipping
configuration, bf16, medians of four rotated rounds, and none is kept:

| lever | 16,384 | 32,768 | 65,536 |
|---|---:|---:|---:|
| 32 autotune candidates instead of 8 | 0.998x | 1.007x | 1.009x |
| row chunk 12,288 | 1.011x | 1.005x | 1.013x |
| row chunk 20,480 | 1.002x | 0.998x | 1.007x |
| row chunk 8,192 + 32 candidates | 1.018x | 1.016x | 1.028x |

(Above 1.000x means the shipping configuration is faster.) Chunks of 16,384 with
the default eight candidates is the best cuBLASLt offers here. OpenNN reaches
91.7 TFLOP/s on that layer and inductor's Triton template reaches 94.6, and the
remaining 3.1% belongs to the kernel, not to how it is called.

* **The 28->1024 layer is now CUTLASS's, and there is still a little left in
  it.** CUTLASS reaches 0.0156 ms at 8,192 against cuBLASLt's 0.0216, inductor's
  Triton template's 0.0147, and a pure-store floor of 0.0099 for the 16 MB it
  writes. Tuning the tile further, or an epilogue that writes wider, is worth at
  most another 6% of this layer.
* **bf16 at 65,536 is the one cell left, and it is mostly the square GEMM.**
  Two thirds of that deficit is the 1024x1024 layer, where OpenNN reaches 91.7
  TFLOP/s and inductor 94.6, and every cuBLASLt lever for it has been measured
  and rejected. Closing it means instantiating that layer through CUTLASS too -
  a much larger surface than the narrow one, because it is the shape every dense
  network spends its time in.
* **The square layer has no headroom inside cuBLASLt.** 85.0 TFLOP/s measured
  against a bf16-with-fp32-accumulate ceiling of about 88 on this consumer
  silicon, where fp32 accumulation runs at half rate. Inductor's 2.5% over it is
  the whole margin available, and reaching it means not using cuBLASLt.
* fp32 needs none of this: OpenNN leads both engines at every rung already.

## Contents

- [The batch sweep (2026-08-21)](#the-batch-sweep-2026-08-21)
- [Introduction](#introduction)
- [Benchmark application](#benchmark-application)
- [Reference computer](#reference-computer)
- [Methodology](#methodology)
- [Results](#results)
- [Discussion](#discussion)
- [Conclusions](#conclusions)
- [Reproducing](#reproducing)
- [References](#references)

## Introduction

Inference removes gradient and optimizer work from the HIGGS dense network and measures the forward path alone. This exposes device residency, kernel launch overhead, dense GEMM efficiency, activation fusion, and the cost of the selected precision.

The comparison uses the same 28-1024-1024-1 ReLU network in OpenNN, PyTorch, and
TensorFlow, over a ladder of batch sizes from 256 to 65,536. Batch size decides
which of those costs dominates - at 256 a batch is 18 microseconds and mostly
launch, at 65,536 it is two milliseconds and entirely GEMM - so a single batch
size answers only one of the questions the benchmark exists to ask. See
[The batch sweep](#the-batch-sweep-2026-08-21) for the current result; the
sections below describe the single-batch measurement it replaced.

## Benchmark application

| Item | Configuration |
|---|---|
| Dataset | HIGGS held-out split |
| Samples processed | 499,712 |
| Inputs | 28 normalized numerical features |
| Network | 28 -> 1024 ReLU -> 1024 ReLU -> 1 |
| Parameters | 1,080,321 |
| Mode | Forward-only inference |
| Batch | ladder: 256, 1,024, 4,096, 8,192, 16,384, 65,536 (8,192 in the superseded single-batch tables) |
| Precisions | fp32 and bf16 |
| Metrics | Samples/s and milliseconds/batch |

## Reference computer

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB, SM clock pinned to 2692 MHz |
| Operating system | Linux 7.0 x86_64 |
| NVIDIA driver | 610.43.02 |
| CUDA | 13.3 |
| PyTorch | 2.13.0+cu130 (cuDNN 9.23) |
| TensorFlow | 2.21.0 |
| OpenNN commit | `e37d6f711` |
| Run ID | 20260820T102740Z |

## Methodology

Each engine processes the same held-out rows with the same network, batch, activation, parameter count, and precision. Labels are ignored by the timed inference path.

- OpenNN uses `calculate_outputs_resident`, keeps parameters and activations on the GPU, and replays a captured CUDA Graph.
- PyTorch uses inference mode, bf16 autocast for the bf16 cell, and a manually captured `torch.cuda.CUDAGraph`. The eager path remains available with `PT_NOGRAPH=1` for diagnostic comparisons.
- OpenNN and PyTorch copy each resident batch into a fixed capture buffer with one device-to-device copy before graph replay, so both graph paths use stable pointers under the same staging contract.
- TensorFlow uses compiled graph execution and mixed bf16 for the bf16 cell.
  Since 2026-08-19 it times two dispatch paths -- the XLA step called per
  batch from Python, and the batch loop compiled inside one `tf.function` --
  and reports the faster, naming it in `tf_path`. `TF_NOLOOP=1` forces the
  per-batch path, mirroring `PT_NOGRAPH` on the PyTorch side.
- Since 2026-08-21 the PyTorch driver does the same for its own two paths - the
  hand-captured CUDA graph over eager modules and `torch.compile`, reported in
  `pt_path` - because which one is faster depends on the batch size.
- All three drivers take a comma-separated batch list and sweep it inside one
  process, and `run_higgs_infer_sweep.py` rotates the engine order and the batch
  order every round, discards a soak round, and reports medians of six.
- Framework warmup and TensorFlow XLA compilation occur before the timed passes.
- Samples per second and milliseconds per batch are reported from the same measured pass.
- The bf16 artifact contains five successful runs per engine; the table reports medians across those runs. The fp32 (TF32) cells are 2026-08-11 single-run alignment measurements.

Dataset loading, process startup, model construction, the initial host-to-device upload, graph capture, and warmup are outside the measured region. The per-batch device-to-device staging copy is included.

## 2026-08-19 protocol correction: PyTorch's best configuration

The PyTorch cell of this benchmark was never eager - the driver builds a CUDA
graph by hand and replays it per batch - but it was missing two things a
deployment would use: `torch.compile` over the model, and bf16 *weights*
instead of weights cast inside `autocast` on every replay. Both are now knobs
(`PT_COMPILE_MODE`, `PT_BF16_WEIGHTS`) and the runner sets PyTorch's measured
best by default: `reduce-overhead` plus bf16 weights in bf16, `max-autotune` in
fp32 (`PYTORCH_PLAIN=1` reverts). The eager fallback now stages through the
same fixed buffer the graph paths use, so the engines are compared on equal
terms.

What it is worth to PyTorch on an RTX 3060 Laptop at batch 8,192: bf16
6.59 -> 7.30 M samples/s (+11%), fp32 3.85 -> 4.19 M (+9%).

Three engines alternated, three rounds each, medians (RTX 3060 Laptop, WSL2,
batch 8,192, 28-1024-1024-1):

| Precision | OpenNN | PyTorch best | TensorFlow | OpenNN / PyTorch | OpenNN / TF |
|---|---:|---:|---:|---:|---:|
| bf16 | **9,442,101 samples/s** | 7,617,578 | 8,219,183 | **1.24x** | **1.15x** |
| fp32 (TF32) | **4,666,954 samples/s** | 4,116,292 | 4,017,197 | **1.13x** | **1.16x** |

The RTX 4080 figures below predate this correction, as did the RTX 5070 Ti
artifact `results/gpu-higgs-dense-inference-speed-20260819T101642Z.json`
(OpenNN 1.28x PyTorch bf16, 1.30x fp32): both compared against the
uncompiled-autocast path, so their PyTorch cells were ~10% low. The 5070 Ti
cells have since been re-measured under this protocol and are the Results table
below -- PyTorch's fp32 margin fell from 1.308x to 1.113x, which is the size of
the effect. The RTX 4080 cells have not been re-measured and should not be
quoted.

## Results

*Superseded by "The batch sweep" above, and kept for provenance. These cells were
measured at one batch size with PyTorch pinned to `reduce-overhead` in bf16 and
`max-autotune` in fp32 - which is 29% under PyTorch's best at this batch in bf16.
Do not cite them.*

Five alternated rounds per precision, medians, GPU clock pinned. OpenNN is
ahead in all five rounds of every cell. Artifact:
`results/gpu-higgs-dense-inference-speed-20260820T102740Z.json`.

| Precision | Framework | Median throughput | Median batch time | OpenNN speedup |
|---|---|---:|---:|---:|
| fp32 (TF32) | OpenNN | **18,364,173 samples/s** | **0.446 ms** | 1.000x |
| fp32 (TF32) | PyTorch | 16,502,523 samples/s | 0.496 ms | **1.113x** |
| fp32 (TF32) | TensorFlow | 17,824,397 samples/s | 0.460 ms | **1.030x** |
| bf16 | OpenNN | **35,531,341 samples/s** | **0.231 ms** | 1.000x |
| bf16 | PyTorch | 28,409,959 samples/s | 0.288 ms | **1.251x** |
| bf16 | TensorFlow | 33,493,091 samples/s | 0.245 ms | **1.061x** |

TensorFlow ran its compiled batch loop in bf16 and per-batch dispatch in fp32;
each cell reports its better path.

TensorFlow ran the compiled batch loop in bf16 (35.59M against 32.24M
per-batch) and per-batch dispatch in fp32 (18.61M against 17.65M compiled).
Both cells report its better path.

### Superseded: RTX 4080, pre-dispatch-fix

Kept for provenance only. These were measured with the TensorFlow driver that
paid per-batch eager dispatch, on a machine no longer available, and their
artifact was never committed, so they cannot be re-checked or re-run. Do not
cite them.

| Precision | Framework | Median throughput | Median batch time | OpenNN speedup |
|---|---|---:|---:|---:|
| fp32 (TF32) | OpenNN | 17,125,371 samples/s | 0.478 ms | 1.00x |
| fp32 (TF32) | PyTorch | 16,970,000 samples/s | 0.483 ms | 1.01x |
| fp32 (TF32) | TensorFlow | 17,220,000 samples/s | 0.476 ms | 0.99x |
| bf16 | OpenNN | 34,610,952 samples/s | 0.237 ms | 1.000x |
| bf16 | PyTorch | 31,904,566 samples/s | 0.257 ms | 1.085x |
| bf16 | TensorFlow | 32,421,696 samples/s | 0.253 ms | 1.068x |

## Discussion

Against TensorFlow this is a near-tie in both precisions -- 1.06x and 1.03x. That
is the result once TensorFlow gets the same dispatch amortization the two
graph-replaying engines already had; before the driver fix the same machine
reported 1.21x bf16, and the difference was Python, not TensorFlow.

Against PyTorch the ~1.30x margin holds in both precisions and has a mechanism
rather than a shrug. Timing a captured PyTorch graph with and without its two
ReLU kernels gives 0.5760 vs 0.4771 ms, so unfused activation accounts for
0.099 ms of the 0.131 ms fp32 gap. Note that ~4% of the bf16 margin is PyTorch's
autocast casting weights inside the replay (0.2907 vs 0.2798 ms with native bf16
weights), which is a methodology difference rather than OpenNN being faster.

fp32 has no headroom left for anyone. A standalone cuBLAS probe (`gemm_probe.cu`)
puts the 1024x1024 forward GEMM at 0.3666 ms of the 0.423 ms batch, cuBLASLt's
best-of-8 heuristic search finds nothing faster than its default, and OpenNN
lands within 6% of the isolated L1+L2 cost. TF32 measures exactly half of BF16
throughput on this silicon, which is why the bf16 margin exists and the fp32 one
cannot be manufactured.

OpenNN's own bf16 path is 1.94x its fp32 path for this model and batch -- the
tensor-core ratio plus halved activation traffic.

These are steady-state, device-resident figures: five-run medians, all
executions successful, every engine on its captured-graph or compiled path.

## Conclusions

*Current, from [the batch sweep](#the-batch-sweep-2026-08-21). The list that was
here before described the superseded single-batch measurement.*

- **fp32 (TF32): OpenNN leads both engines at every batch size from 256 to
  65,536**, by 1.12x-1.22x over PyTorch and 1.02x-1.16x over TensorFlow, ahead
  in every round.
- **bf16: OpenNN leads TensorFlow at every batch size** (1.06x-1.35x, ahead in
  every round) and leads PyTorch below 8,192 (1.26x, 1.46x, 1.04x). 8,192 and
  16,384 are ties; 65,536 goes to PyTorch by 6.4%.
- The engine each comparison is run against has to be re-checked *per batch
  size*, not once. PyTorch's best configuration flips between two that differ by
  2.9x at one end of the ladder and 1.44x at the other, and pinning one of them
  measured it 2.7x under itself at batch 256.
- OpenNN's inference graph is three CUDA graph nodes a batch against PyTorch's
  four. Fusion is not what the remaining bf16 gap is about; cuBLASLt's kernels
  against inductor's Triton templates is.
- fp32 is GEMM-bound at the hardware ceiling for every engine - 44 TFLOP/s of
  TF32 with fp32 accumulate on this consumer silicon - so what is left there is
  not the kernel but how the call is shaped: issuing it in chunks of rows above
  16,384 is worth 3.6-4.7% at the top of the ladder in both precisions.
- bf16 is not at the ceiling, and the 28->1024 layer is where the headroom is.

## Reproducing

The batch sweep, which is what the current result comes from:

```bash
sudo benchmarks/tools/gpu_clocks.sh lock 2700

python run_higgs_infer_sweep.py \
  --batches 256,1024,4096,8192,16384,65536 \
  --runs 5 --rounds 6 --soak 1 --precision both
```

It writes `benchmarks/results/gpu-higgs-infer-sweep-<label>-<run_id>.json`
holding every round's per-pass times in temporal order, the rotation used, and
the path each engine chose at each rung.

The same runner does the A/Bs, with two arms of one engine alternating against
each other under that protocol:

```bash
python run_higgs_infer_sweep.py --batches 256,1024,8192 --rounds 4 \
  --arm "opennn:on:" --arm "opennn:off:OPENNN_SINGLE_OUTPUT_ACTIVATION=0"
```

`run_higgs_infer.py` remains the single-batch publication runner:

```bash
python run_higgs_infer.py \
  --test "$OPENNN_BENCH_DATA/higgs/higgs_test.csv" \
  --batch 8192 --hidden 1024 --hidden-layers 2 \
  --activation relu --precision both --runs 5
```

## References

- [HIGGS dataset, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/280/higgs).
- [Searching for exotic particles in high-energy physics with deep learning](https://www.nature.com/articles/ncomms5308).
- [OpenNN source repository](https://github.com/Artelnics/opennn).
- [PyTorch](https://pytorch.org/).
- [TensorFlow](https://www.tensorflow.org/).
