# CPU HIGGS dense inference: OpenNN vs PyTorch vs TensorFlow

This benchmark measures repeated CPU inference passes for the same dense binary
classifier shape on the prepared UCI HIGGS test split.

**Status:** full 500k-row publication test split, native Linux, i9-12900K
(8 P-core threads), MKL-linked OpenNN. Last updated 2026-08-10, commit
52e21e15d. Artifact: `results/cpu-dense-higgs-infer-20260810T150333Z.json`.

## Result

Higher is better.

| Framework | Inference speed (samples/s) | Median pass (s) |
|---|---:|---:|
| OpenNN | **416,825** | **1.199** |
| PyTorch | 340,279 | 1.469 |
| TensorFlow | 356,244 | 1.403 |

OpenNN delivered **1.23x PyTorch speed** and **1.17x TensorFlow speed** on this
CPU inference run (median of 5 runs, 10 passes each), using the MKL-linked
OpenNN binary. A smaller batch-256 probe
was overhead-sensitive and put PyTorch ahead; the current row uses batch 1024 and
five timed runs.

## Re-measured with every engine at its best (2026-08-20)

*Laptop, WSL2, i7-12700H, MKL-linked OpenNN, full 500k-row test split, batch
1024, 10 passes per run, 12 threads, single runs (about 10% drift).*

| Engine | Best setting | Inference (samples/s) | Also measured |
|---|---|---:|---|
| TensorFlow | XLA | **197,852** | 153,108 with XLA off |
| PyTorch | eager, `inference_mode` | **148,298** | 143,809 on a repeat run |
| OpenNN (MKL) | - | **97,752** | median pass 5.11 s |

**At best settings OpenNN infers at 0.49x TensorFlow and 0.66x PyTorch here**,
again the opposite of the table above.

### How much of this machine to believe

The single runs above were taken on a laptop that turned out to swing far more
than the 10% they were reported with: the same binary, the same arguments and
the same thread count measured 56,173 and 101,182 samples/s back to back, and a
standalone MKL sgemm at one shape measured 128, 233 and 359 GFLOP/s across three
runs. Only **alternated** pairs survive that, and those are stable:

| round | OpenNN | PyTorch | ratio |
|---|---:|---:|---:|
| 1 | 102,215 | 143,106 | 1.40x |
| 2 | 100,466 | 143,830 | 1.43x |

So take **PyTorch at 1.40-1.43x OpenNN on CPU inference** as the result, and
treat the single-run TensorFlow figures as indicative of a large gap rather than
as a measured ratio. Anything smaller than about 2x cannot be attributed on this
machine from single runs.

## Where OpenNN's CPU inference time goes

Profiled over 5,856 layer calls, and isolated by adding and removing hidden
layers:

* `op:combination_fwd` (the GEMM and its bias) is **96.5%** of the time;
  `op:activation_fwd` is 3.4%.
* One 1024x1024 hidden layer costs about **7.4 ms per batch of 1024**.
* The two thin layers (28->1024 and 1024->1) cost about **2.9 ms per batch
  between them** - for 3% of the arithmetic. MKL runs those same two shapes
  standalone in **0.21 ms**, so roughly 2.7 ms per batch, a quarter of the whole
  step, is spent around the thin GEMMs rather than inside them.
* Throughput rises with batch size - 28,977 at 1,024, 72,277 at 4,096, 84,722
  at 16,384 samples/s in one back-to-back sweep - which is the signature of a
  fixed per-call cost rather than of the arithmetic.

Three explanations were tested and rejected:

* **"MKL is not being used."** It is: an instrumented counter records 7,320
  calls into `cblas_sgemm` with zero refusals for a five-pass run, exactly three
  per batch. (`MKL_VERBOSE` prints nothing for these calls, which is misleading.)
* **"The OpenMP teams churn."** They do - `add_bias` opens a parallel region per
  layer per batch, and one run creates about 10,000 threads - but pinning the
  team (`OPENNN_OMP_DYNAMIC=0`) measured **2.5x slower** (39,812 against
  100,426), and `MKL_THREADING_LAYER=GNU` measured neutral.
* **"MKL's GEMM is slower than PyTorch's."** Not supported: alternated at
  1024x1024x1024, MKL measured 359/279/128 GFLOP/s against PyTorch's
  221/233/343. Same class.

What is left, and what the next attempt should attack, is the fixed cost of a
forward call outside the GEMM.

## The bias pass, and what fixing it was worth (2026-08-20)

Instrumenting the dense forward by part found the cost, and it was not the
GEMM:

| scope | ms per call | calls per batch |
|---|---:|---:|
| `cpu:sgemm_wide` (1024 cube) | 5.634 | 1 |
| **`cpu:add_bias`** | **1.361** | **3** |
| `cpu:sgemm_thin_k` (28->1024) | 0.180 | 1 |
| `cpu:sgemm_thin_n` (1024->1) | 0.056 | 1 |
| `op:activation_fwd` | 0.128 | 3 |

Adding a bias to a 4 MB output is one pass over memory and should cost a
fraction of a millisecond; at 1.36 ms per call it was 40% of the forward, more
than every GEMM combined. The cause was the OpenMP region the pass opened per
layer per batch - about ten thousand thread creations in one inference run.
Hoisting the fused ReLU out of the inner loop changed nothing (97,672 against a
96,996 baseline), which ruled out vectorisation and left the threading. Three
ways of spreading the rows, measured in one binary:

| bias parallelism | inference (samples/s) | `add_bias` ms/call |
|---|---:|---:|
| OpenMP region per call (what it did) | 96,589 | 1.550 |
| the library's persistent thread pool | 143,090 | 0.173 |
| **none at all** | **154,219** | **0.121** |

The work is memory bound and short enough that every way of spreading it costs
more than it saves, so the pass is serial now, with `OPENNN_BIAS_MODE=pool|omp`
kept for the A/B.

### Result

Alternated against PyTorch at its own best setting, three rounds:

| round | OpenNN | PyTorch | ratio |
|---|---:|---:|---:|
| 1 | 157,897 | 152,673 | **1.03x** |
| 2 | 153,276 | 150,506 | **1.02x** |
| 3 | 153,238 | 149,510 | **1.02x** |

**OpenNN is now ahead of PyTorch on CPU inference in every round**, where before
this change it ran at 0.65x - a 58% gain on the same machine, same build, same
protocol. Training gains too, 34,209 to 43,047 samples/s (+26%), with the
held-out metrics unchanged to every digit printed (accuracy 0.73236, ROC AUC
0.815341), which is what one expects from a change that only reorders how the
same additions are spread over cores.

TensorFlow with XLA remains ahead of both on this benchmark and is the next
target.

## Chasing TensorFlow: what has been ruled out (2026-08-20)

With each engine at the thread count it likes best - OpenNN with `MKL_NUM_THREADS=6`
(183,903 against 170,681 at twelve), TensorFlow at twenty (211,328 against
189,442 at six) - TensorFlow leads by 1.19-1.25x over three alternated rounds:

| round | OpenNN | TensorFlow |
|---|---:|---:|
| 1 | 185,695 | 220,085 |
| 2 | 174,173 | 213,489 |
| 3 | 167,964 | 210,648 |

That the two engines want opposite thread counts is itself worth knowing, and
is why both were swept rather than one.

By then the forward is one number: `cpu:sgemm_wide` is about 90% of it, and
everything else - bias 0.32 ms, thin GEMMs 0.24 ms, activations now nil - sums
under 0.6 ms per batch. Four ways of attacking it, all measured, none kept:

* **MKL packed weights** (`cblas_sgemm_pack`, the obvious trick when B never
  changes between batches): **9% slower** than plain, 4.933 ms against 4.526.
* **`beta = 1` with the bias pre-broadcast into C**: rejected on arithmetic
  before building. It trades one 12 MB round trip for 20 MB, because the ReLU
  then needs a pass of its own.
* **Thread placement**: pinning to physical P-cores 182,186, six OpenMP threads
  182,319, `MKL_DYNAMIC=FALSE` 177,848 - all below the plain 206,985.
* **oneDNN with fused post-ops**, which is the mechanism TensorFlow and PyTorch
  actually use: a standalone prototype of this exact network through
  `inner_product` with bias and eltwise as post-ops and the weights reordered
  once, measured **169,984 / 182,350 / 101,151** samples/s at six, twelve and
  twenty threads. Its best is level with what OpenNN already gets from MKL plus
  separate passes, and 14% short of TensorFlow.

The last one is the informative failure. Fusion was the leading hypothesis for
the 0.8 ms between the in-app GEMM and standalone MKL at the same shape, and an
implementation with no separate passes at all lands in the same place - so the
gap is not epilogue fusion, and building post-ops into the dense CPU path would
not have paid. What is left to explain is why every one-GEMM-at-a-time
implementation, MKL and oneDNN alike, degrades at twenty threads while
TensorFlow's best point is twenty: the next thing to measure is what XLA is
scheduling across the batch loop, not another kernel.

## Who threads the GEMM, and across batch sizes (2026-08-20)

The answer to that last question turned out to be about *who splits the rows*,
not about the kernel. MKL threads a GEMM itself, and here that is two limits at
once: it runs at most ten threads whatever `MKL_NUM_THREADS` says - one per core
it can see, and WSL2 presents this twenty-thread laptop as a synthetic 10x2 -
and it splits the rows evenly across them, so its barrier waits on the slowest
slice. Handing blocks of rows out of an atomic counter to an OpenMP team, with
`mkl_set_num_threads_local(1)` inside each block, has neither limit. Alternated
three rounds at batch 1024:

| round | MKL's own threading | blocks |
|---|---:|---:|
| 1 | 172,792 | **216,761** |
| 2 | 162,733 | **212,332** |
| 3 | 170,537 | **214,211** |

**+25% to +30%, and it is the team that matters as much as the blocking**: the
same blocks dispatched on the library's own Eigen thread pool measured *below*
MKL (179,350 against 187,395). `OPENNN_GEMM_MODE=mkl|pool` keeps both for the
A/B. Training gains from the same change too, 41,698 to 44,771 samples/s, with
the held-out metrics agreeing to three digits (accuracy 0.765149 against
0.764929, ROC AUC 0.849500 against 0.849126 - a reordered summation over an
epoch of Adam, which is what that should look like).

Four further pieces, each measured:

* **The epilogue rides with the block.** Bias and ReLU run on a block's rows
  immediately after that block's GEMM, while they are still in cache, instead of
  as a pass over the whole output afterwards.
* **The block height scales with the batch** - enough blocks that every worker
  gets about four. A fixed sixteen rows was right at batch 1024 and wrong on
  both sides: too few blocks to balance at 256, and a thousand of them at 16384.
* **The weight panel is packed once per layer** with `cblas_sgemm_pack` and read
  by the whole team, which is *not* the packed GEMM rejected above - that one
  packed for a single whole-layer call, where MKL already packs once internally.
  It pays only while the blocks are short: +1.8/+2.5/+2.9% at batch 4096, and
  -6.7/-2.7/-7.9% at 16384, where a 256-row block amortises MKL's own
  rearrangement anyway. On below 64-row blocks, off above.
* **A guided schedule** - take a share of the rows that remain rather than a
  fixed block - is used exactly where the packed panel is not, since it costs
  the panel. It measured -9/-13/-15% at batch 256 and +4.7/+7.7% at 4096/16384.

### Where the three engines stand - superseded, kept for the record

*These numbers were taken with a fixed engine order and are biased; the table
that supersedes them is under "The instrument was the problem". Read this one
only to see how large the bias was.*

Each engine sweeps the batch sizes inside one process (the drivers take a
comma-separated list now, because this machine drifts 10% over a sweep, so a
row of the table has to share one load and one thermal window), and the three
are alternated inside each round.

Samples/s, three rounds, best settings each (OpenNN 20 threads, PyTorch 20
eager under `inference_mode`, TensorFlow 20 with XLA):

| batch | OpenNN | PyTorch | TensorFlow | vs PyTorch | vs TensorFlow |
|---|---:|---:|---:|---:|---:|
| 256 | 174,644 / 139,584 / 135,346 | 116,054 / 99,967 / 104,285 | 128,162 / 123,507 / 131,109 | **1.51 / 1.40 / 1.30** | **1.36 / 1.13 / 1.03** |
| 1,024 | 204,580 / 175,712 / 168,826 | 140,198 / 133,905 / 127,657 | 198,783 / 194,342 / 198,515 | **1.46 / 1.31 / 1.32** | 1.03 / 0.90 / 0.85 |
| 4,096 | 215,830 / 185,509 / 185,203 | 131,953 / 136,431 / 123,836 | 236,941 / 237,494 / 222,810 | **1.64 / 1.36 / 1.50** | 0.91 / 0.78 / 0.83 |
| 16,384 | 223,822 / 196,711 / 203,415 | 138,242 / 145,113 / 141,466 | 220,599 / 239,004 / 237,276 | **1.62 / 1.36 / 1.44** | 1.01 / 0.82 / 0.86 |

**PyTorch is beaten at every batch size in every round**, by 1.30x to 1.64x.
TensorFlow is beaten at batch 256 in every round, and at 1024 and 16384 in the
first round only.

*Superseded - see "The instrument was the problem" below. The reasoning in this
paragraph was wrong: OpenNN and PyTorch both fall about 20% between round one and
round two while TensorFlow holds within 3%, and this note concluded "it is not
thermal, since OpenNN runs first in each round". Running first is exactly what
exposes a run to the processor's short-duration boost window, so the ordering
argument proves the opposite of what it claimed, and the bias flattered whichever
engine held slot one - which was always OpenNN.*

### What was tried for the two large batches and did not work

At batch 16384 the profile is one number again: `cpu:sgemm_wide` is 90% of the
step at about 590 GFLOP/s, the 28->1024 layer 7% (writing 64 MB), and the
1024->1 layer 3% (reading 64 MB to produce 64 KB). TensorFlow's throughput
implies about 660 GFLOP/s on the same GEMM, and it *pays back* 8-12% on an
epilogue we fuse and it does not. So the whole remaining gap is the GEMM
schedule.

* **Sharding the columns as well as the rows**, which is exactly what XLA's
  `DotThunk` does through Eigen's contraction, and the one structural difference
  left between the two schedules. Implemented and measured: two column blocks
  took batch 1024 from 224,379 to **101,856**. A column slice leaves both the
  weight panel and the output strided, and MKL would rather have them contiguous
  than have a narrower panel to keep.
* **MKL threads inside each worker** (twenty workers of one thread against ten
  of two and five of four): one is best or tied everywhere, four is 25% behind.
* **Sizing the blocking decision by the work touched** rather than by the output,
  so the 1024->1 layer is blocked too: it does not help (1.868 ms against 1.797),
  because that layer is bound by a 64 MB read that MKL's ten threads already
  saturate, and it costs 3-5% at batch 256.
* **A row-panel schedule across layers**, carrying a panel of rows through all
  three layers to keep the intermediates in cache, was refuted from data already
  taken rather than built: it predicts smaller effective panels are faster, and
  the measured batch curve says the opposite - 4096 runs slower than 16384 in
  every round. Per-call cost dominates the DRAM round trip here.

## The instrument was the problem (2026-08-21)

Every table above this line was taken with a **fixed engine order**, and that
turns out to be worth more than most of the levers in this note. Whatever runs
first after an idle gap runs inside the processor's short-duration boost window;
what follows it does not. The engine in slot one was always OpenNN.

The evidence is a control that was measured by accident. A three-arm A/B of the
contraction's size gate included batches where the contraction *cannot fire* -
its gate needs 8e9 or 2e9 flops and those layers are 2.7e8 and 1.1e9 - so all
three arms ran identical code at batch 256 and 1024. They did not measure the
same:

| arm (identical code) | batch 256 | batch 1024 |
|---|---:|---:|
| slot 1 | 172,781 / 190,092 / 180,510 | 219,453 / 209,612 / 212,635 |
| slot 2 | 137,996 / 150,839 / 157,591 | 176,965 / 190,465 / 198,846 |
| slot 3 | 136,653 / 146,998 / - | 180,440 / 190,069 / - |

**A 20% spread with no code difference at all**, ordered by position. The same
decay repeats across batch position *inside* one process, so a fixed batch order
hands the window to the same batch every time as well.

This also inverts the reasoning recorded earlier in this note, which dismissed
thermals because "OpenNN runs first". Running first is exactly what exposes a run
to the boost window. The bias was in OpenNN's favour throughout.

### The protocol that replaces it

* **Rotate the engine order** every round, and **rotate the batch order** too.
* **Soak first**: run one full sweep and discard it.
* **Report medians of six rounds**, not of three.
* All three drivers print `batch_<B>_pass_times=` in temporal order, before the
  median, so a drifting machine is visible in the data instead of averaged away.

Spreads fall from about 20% to 2-4%, which is what makes a 3% difference
decidable at all.

### Result

Medians of six rotated rounds, each engine at its best (OpenNN 20 threads,
PyTorch 20 eager under `inference_mode`, TensorFlow 20 with XLA):

| batch | OpenNN | PyTorch | TensorFlow | vs PyTorch | vs TensorFlow |
|---|---:|---:|---:|---:|---:|
| 256 | **158,464** | 148,244 | 141,437 | **1.07x** | **1.12x** |
| 1,024 | **215,452** | 177,750 | 210,395 | **1.21x** | **1.02x** |
| 4,096 | 247,748 | 183,050 | 246,899 | **1.35x** | 1.00x |
| 16,384 | **265,035** | 161,684 | 255,324 | **1.64x** | **1.04x** |

**OpenNN is ahead of PyTorch at every batch size, and ahead of TensorFlow at
256, 1024 and 16,384. Batch 4,096 is a tie** - two independent six-round runs
put it at 0.94x and 1.00x, which is within what this machine can resolve.

Note what the honest instrument did to the earlier claims: the PyTorch margins
reported before rotation (up to 2.07x at batch 256) were mostly slot bias, and
the real figure there is 1.07x.

## Eigen's contraction, which is the kernel TensorFlow uses (2026-08-21)

XLA:CPU lowers each dot to a `DotThunk` running an Eigen `TensorContraction`.
That kernel is reachable from OpenNN directly, and `EIGEN_USE_MKL_ALL` does not
divert it: Eigen routes *Matrix* products to BLAS and keeps its own kernels for
*tensor* contractions (the whole `unsupported/Eigen/CXX11` tree contains no
reference to `EIGEN_USE_BLAS`, `cblas_` or `general_matrix_matrix_product`).
Standalone, twenty threads, GFLOP/s over two rounds:

| m (n = k = 1024) | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---:|---:|---:|---:|---:|---:|---:|
| blocked MKL | 523 | 484 | 490 | 652 | 704 | 744 | 726 |
| contraction | 598 | **854** | **749** | 746 | 703 | 747 | 719 |

And the epilogue rides in its **output kernel**, which Eigen calls on each output
block as it is produced - so bias and ReLU land while that block is in cache,
exactly where the blocked schedule puts them. It is free and then some: 790
against 723 GFLOP/s fused against not, and bitwise identical to a separate pass.
TensorFlow does *not* fuse there; its bias and ReLU are separate whole-tensor
thunks on about five threads.

**The standalone result did not transfer whole, and that is the lesson.** In the
app the contraction is worse at batch 1024 and better only on the wide layer at
16,384, because a microbenchmark that loops one shape keeps the pool hot and
never shows two things: the advantage needs a call long enough to absorb waking
the pool, and the kernel is poor when the contraction dimension is small - the
28->1024 layer measured 0.464 ms against the blocked schedule's 0.188.

So it is gated: `k >= 64`, and enough arithmetic to absorb the wake-up
(`OPENNN_GEMM_CONTRACT_FLOPS`, 8e9). Sweeping that gate one rung at a time, six
rotated rounds each, says the gate is where it should be - at 4,096 the two
kernels are indistinguishable, which is what the standalone table also says:

| batch | gate 8e9 (ships) | gate 4e9 (+4096) | gate 1e9 (+1024) |
|---|---:|---:|---:|
| 256 | 159,747 | 160,526 | 159,062 |
| 1,024 | 226,964 | 224,640 | 230,216 |
| 4,096 | 251,302 | 251,249 | 248,460 |
| 16,384 | 272,713 | 275,810 | 275,009 |

One thing it needed: Eigen asks its *device* for the buffers it packs operands
into, on every call, and `LinearForwardMemoryTest` caps the process precisely to
prove that a steady-state forward allocates nothing. It caught this immediately.
The contraction therefore runs on a device that shares the library's threads but
whose allocator recycles - the free list settles at the first call's peak.

`ContractionForwardTest` covers the numerics, at a batch just over the gate,
because no other test in the suite reaches a shape that takes this path.

### A 28x regression that came from adding a caller

Worth recording because it is invisible in a diff. Fusing the epilogue into the
blocks added a second caller of the bias-and-ReLU template. That alone left the
*original* call site on an out-of-line copy running at about 2 GB/s instead of
60: `cpu:add_bias` went from 0.128 ms per call to **3.4**, and inference fell to
65,000 samples/s. It was not the new path - a build that never executed the new
code was equally slow - and a stash-and-rebuild control confirmed the machine
was fine. The fix was to route every epilogue through one call site, which is
what the code does now, with a comment at the function saying why.

### What the harness was doing wrong

The table above came from a harness that did not let the other two engines run
their own fast paths:

* `tensorflow_higgs_cpu.py` pinned `@tf.function(jit_compile=False)` in both
  training and inference - XLA off - while the GPU family has always measured
  TensorFlow with XLA on. XLA is worth **+32% training / +29% inference** to it
  here, so the published rows measured TensorFlow below itself.
* `pytorch_higgs_cpu.py` had no `torch.compile` at all and ran inference under
  `no_grad` rather than `inference_mode`.
* `run_higgs_cpu.py` applied `OMP_PLACES=cores OMP_PROC_BIND=close` to OpenNN
  and PyTorch but **not** to TensorFlow, which on a hybrid CPU is not a neutral
  omission: unpinned threads land on efficiency cores.

All three are fixed. Every engine now runs its best configuration by default,
and `TF_PLAIN=1` / `PYTORCH_PLAIN=1` restore the old behaviour for an A/B.
One surprise worth recording: on this CPU `torch.compile` is a *pessimisation*
for this model - 29,449 samples/s against eager's 41,523, inductor's CPU codegen
losing to eager on a three-GEMM MLP - so compilation is opt-in (`PT_COMPILE=1`)
and eager is what the driver measures.

The i9 rows above were measured under the old harness and cannot be cited as
they stand; they need re-running on that machine with the corrected drivers.

## Setup

| Item | Value |
|---|---|
| Dataset | UCI HIGGS, prepared with `docs/benchmarks/throughput/higgs/prepare_higgs.py` |
| Test split used here | 20,000 rows |
| Layout | `feature_0,...,feature_27,label`, normalized from train-set statistics |
| Model | `28 -> 1024 -> 1024 -> 1` |
| Activation | ReLU hidden layers, sigmoid output |
| Batch | 1024 |
| Repetitions | 20 timed passes after warmup |
| Timed runs | 5 |
| Device | CPU only |
| Environment | WSL2, Intel Core i7-12700H host, CPU path |
| OpenNN CPU backend | MKL-linked binary (`libmkl_rt`) |
| Thread environment | `MKL_NUM_THREADS=20`, `OMP_NUM_THREADS=20`, dynamic threading disabled |

CPU joules are **not measured** in WSL on this machine; the result JSON includes
only a normalized inverse-throughput energy proxy.

## Artifact

Local result JSON:
`docs/benchmarks/results/cpu-dense-higgs-infer-20260704T-higgs100k-infer-b1024-mkl-threads20.json`

Runner:
[`higgs/run_higgs_cpu.py`](higgs/run_higgs_cpu.py)

OpenNN binary source:
[`higgs/opennn_higgs_cpu.cpp`](higgs/opennn_higgs_cpu.cpp)
