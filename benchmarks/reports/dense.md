# Dense: HIGGS classification

OpenNN against PyTorch 2.13 on the UCI HIGGS binary classifier, a
28 → 1,024 → 1,024 → 1 multilayer perceptron with 1,080,321 parameters,
trained with Adam and run for inference, on the GPU at batch 8,192 in bf16 and
on the CPU at batch 4,096 in fp32. Session `2026-09-06-publish`, commit
`e76425bd3`:

| cell | OpenNN | PyTorch | throughput | memory | energy |
|---|---|---|---|---|---|
| `cuda-dense-train` | 11,406,741 samples/s | 10,048,603 | **1.135×** | 1.244× | 1.154× |
| `cuda-dense-infer` | 39,387,890 samples/s | 38,689,107 | **1.018×** | 1.108× | 1.057× |
| `cpu-dense-train` | 70,120 samples/s | 54,520 | **1.286×** | 2.58× | 1.130× |
| `cpu-dense-infer` | 220,512 samples/s | 170,329 | **1.295×** | 2.03× | 1.086× |

All four cells win all three axes. `cuda-dense-infer` won all three a session
ago too, but against a slower draw of PyTorch's autotuner; the section on it
below is about what winning against the faster draw cost. One number in that
table should not be read as a point estimate: `cuda-dense-train`'s throughput
ratio is 1.13× to 1.26× depending on which draw of PyTorch it is measured
against. 1.135× is the published session's median and the conservative end.

The *Why* section argues each margin from a measured ceiling. On the GPU at
batch 8,192 the network is one large matrix product plus two small layers, and
both engines now run that product at essentially the same speed — OpenNN
through a cuDNN engine at 189.8 µs, PyTorch through an autotuned Triton kernel
at 189.6 µs — so the inference cell (1.018×) is decided by the two small
layers around it, and the training cell (1.135×) by how the step is issued:
one captured CUDA graph against Inductor's cudagraph-tree replay. On the CPU
the same MKL kernels run on both sides, and OpenNN's row-blocked layers —
which apply the bias and the activation to each block while it is still in
that core's cache, on the same thread pool as the GEMM — put the whole batch at
88.5% of the eight P-cores' fp32 peak at inference against PyTorch's 68.9%, and
84% against 67% in training.

## What is measured

**The network.** A binary classifier on the UCI HIGGS set: 28 features →
Dense(1,024) → ReLU → Dense(1,024) → ReLU → Dense(1) → sigmoid, 1,080,321
parameters, which both engines print and the runner compares before any
throughput is accepted. The loss is binary cross-entropy — OpenNN as a sigmoid
output layer under `CrossEntropy`, PyTorch as `BCEWithLogitsLoss` on the bare
linear output, the same function with the sigmoid folded into the loss — and
the optimiser is Adam at its default learning rate on both sides.

**The data.** `prepare.py dense` writes HIGGS feature-first (OpenNN's tabular
reader takes the last column as the target) and normalised with the training
split's statistics only; both engines read the identical CSV, so neither pays a
transformation the other does not. Training uses a 250,000-row split, so an
epoch at batch 8,192 is 30 whole batches (245,760 samples) and at batch 4,096
is 61 (249,856). Inference runs the 500,000-row test split, 61 batches of
8,192 (499,712 samples) per pass.

**The cells.** Warmup is excluded from every window (PROTOCOL §5), and the
CUDA cells run long enough for the energy sampler to resolve them: the rule is
at least fifty NVML power samples, one second. 100 epochs for
`cuda-dense-train` and 200 passes for `cuda-dense-infer` clear it with margin.

| cell | device | batch | precision | timed |
|---|---|---|---|---|
| `cuda-dense-train` | RTX 5070 Ti | 8,192 | bf16 autocast (TF32 for fp32 GEMMs) | 100 epochs |
| `cuda-dense-infer` | RTX 5070 Ti | 8,192 | bf16 | 200 passes |
| `cpu-dense-train` | P-cores, CPUs 0-15 | 4,096 | fp32 | 3 epochs |
| `cpu-dense-infer` | P-cores, CPUs 0-15 | 4,096 | fp32 | 5 passes |

**Each engine at its best.** On CUDA both engines keep the whole split
resident on the device. PyTorch takes each batch as a contiguous slice of the
resident tensors (`range(0, n - batch + 1, batch)`, no shuffling); OpenNN
reshuffles the training indices every epoch (the optimizer's default) and
gathers each batch by index on the device, which is strictly more work per
batch — it is the engine's normal training path, left as is. OpenNN captures the
entire Adam step — forward, loss, backward, update — into one CUDA graph and
replays it per batch. For inference OpenNN slices the resident test split as
device views and launches the three layers directly, with no graph, because
three launches per 0.2 ms batch are already inside what one host thread queues
ahead of the GPU. The A/B against the graphed path
(`OPENNN_DENSE_INFER_GATHER=1`, an index copy and a gather kernel per batch,
replayed as one graph) was measured this round: 36,657,201 samples/s against
the published 39,387,890 — 0.948× of PyTorch instead of 1.018× — over three
rounds (session `2026-09-06-dense-variants`,
`cuda-dense-infer-gather-20260906T192735Z`), so the graph is worth less than
nothing here and the resident views are the right path.
PyTorch's mode is chosen per cell and each was measured (the driver's
`compiled()` docstring has every mode): training runs
`torch.compile(mode="reduce-overhead")` — Inductor's fused Triton kernels plus
CUDA graphs, the closest analogue to what OpenNN does, and about 1.4× faster
than Inductor without graphs on a step that is launch-bound (~9.9 M against
~7.0 M samples/s; three-epoch runs with ~70 ms windows, and the pair spans two
builds, `T022718Z` at `38ad27e16` against `T045015Z` at `918805ce1`); inference
runs `max-autotune-no-cudagraphs`, whose autotuned Triton GEMM beats cuBLAS on
the 28-wide first layer and which is 1.33× faster than `reduce-overhead` here
(37,131,220 against 27,939,975, both `2026-09-02-variants` at `8e47e7662`),
where cudagraph-tree replay copies and bookkeeping cost more than three
launches. Inference also stores the weights
in bf16 once (`PT_INFER_CAST=weights`, the default) instead of re-casting them
under autocast on every call, the way OpenNN's inference deployment holds its
parameters; that is worth 3.4% on this cell (37,131,220 against 35,903,250)
and more on the larger networks. Those mode figures predate the autotuner
re-tune described below and are the basis for the *choice* of mode, not for
the published PyTorch number. On CPU every launch is `taskset`-pinned to CPUs
0-15 — the eight P-cores and their SMT siblings, with the twelve E-cores
excluded — at each engine's own default thread count, which the runner neither
equalises nor records (see the caveats), with MKL as the BLAS (`blas=mkl` is
printed by both and recorded per launch) and `GOMP_SPINCOUNT=300000` set for
both; PyTorch runs eager because Inductor's CPU code generation loses on a
small stack of GEMMs — the `compiled()` docstring's measurement, eager 93,156
against compiled 83,722 samples/s, comes from a configuration that reads far
above this cell on both engines and is quoted for its direction only.

**The gates.** Parameter counts must agree (they do: 1,080,321). This is the
one family with a quality gate: each training launch prints its test-set
accuracy, and per batch every launch must sit within 2% of the mean across
engines. The published `cpu-dense-train` launches read 0.727 (OpenNN, all
three) against 0.723635 (PyTorch, all three), and `cuda-dense-train` 0.715 /
0.716 / 0.717 against 0.706933 on all three PyTorch launches. Inference has no
gate — it computes no metric — beyond the identical parameter count and input
file.

## Results

Session `2026-09-06-publish`, commit `e76425bd3`, median of three rounds.

| cell | batch | precision | OpenNN samples/s | PyTorch samples/s | OpenNN / PyTorch | peak memory MiB (OpenNN / PyTorch) | energy Wh (OpenNN / PyTorch) |
|---|---|---|---|---|---|---|---|
| `cuda-dense-train` | 8,192 | bf16 | 11,406,741 | 10,048,603 | **1.135×** | 507.6 / 631.6 | 0.13174 / 0.15197 |
| `cuda-dense-infer` | 8,192 | bf16 | 39,387,890 | 38,689,107 | **1.018×** | 371.3 / 411.3 | 0.17313 / 0.18301 |
| `cpu-dense-train` | 4,096 | fp32 | 70,120 | 54,520 | **1.286×** | 305.2 / 787.3 | 0.08623 / 0.09742 |
| `cpu-dense-infer` | 4,096 | fp32 | 220,512 | 170,329 | **1.295×** | 280.4 / 568.7 | 0.09791 / 0.10630 |


`cuda-dense-train` — batch 8,192, bf16, epochs 100 per launch, 3 rounds. Artifact `cuda-dense-train-publish-20260906T133020Z.json`, commit `e76425bd3`, clean tree, quiet True (busy 0.8% before, 0.2% after, 0.3% max during, threshold 3%), clocks locked True, shape gate True, quality gate True.

| engine | median samples/s | min | max | peak device MiB | Wh (board) |
|---|---|---|---|---|---|
| OpenNN | 11,406,741 | 11,406,056 | 11,436,759 | 507.6 | 0.13174 |
| PyTorch | 10,048,603 | 9,521,984 | 10,144,606 | 631.6 | 0.15197 |
| **ratio** | **1.135×** | | | 1.24× less | 1.154× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 11,406,056 | 10,048,603 |
| 2 | pytorch → opennn | 11,406,741 | 10,144,606 |
| 3 | opennn → pytorch | 11,436,759 | 9,521,984 |

OpenNN's three launches span 0.27%; PyTorch's span 6.5%, and its median has
read five different values across publish runs, an 11% band. The cell reads
**1.13× to 1.26×** depending on the draw; the section below lists the
draws and says which of them are evidence-grade.


`cuda-dense-infer` — batch 8,192, bf16, passes 200 per launch, 3 rounds. Artifact `cuda-dense-infer-publish-20260906T132837Z.json`, commit `e76425bd3`, clean tree, quiet True (busy 0.2% before, 0.5% after, 0.5% max during, threshold 3%), clocks locked True, shape gate True, quality gate True (no metric).

| engine | median samples/s | min | max | peak device MiB | Wh (board) |
|---|---|---|---|---|---|
| OpenNN | 39,387,890 | 39,386,263 | 39,388,828 | 371.3 | 0.17313 |
| PyTorch | 38,689,107 | 38,679,908 | 38,697,811 | 411.3 | 0.18301 |
| **ratio** | **1.018×** | | | 1.11× less | 1.057× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 39,388,828 | 38,697,811 |
| 2 | pytorch → opennn | 39,387,890 | 38,689,107 |
| 3 | opennn → pytorch | 39,386,263 | 38,679,908 |

Both engines are steady here to better than 0.05%, which is why a 1.8% margin
is reportable at all.


`cpu-dense-train` — batch 4,096, fp32, epochs 3 per launch, 3 rounds. Artifact `cpu-dense-train-publish-20260906T134749Z.json`, commit `e76425bd3`, clean tree, quiet True (busy 0.0% before, 0.1% after, 0.1% max during, threshold 3%), clocks locked True, shape gate True, quality gate True.

| engine | median samples/s | min | max | peak RssAnon MiB | Wh (RAPL package-0) |
|---|---|---|---|---|---|
| OpenNN | 70,120 | 69,954 | 70,126 | 305.2 | 0.08623 |
| PyTorch | 54,520 | 53,035 | 54,953 | 787.3 | 0.09742 |
| **ratio** | **1.286×** | | | 2.58× less | 1.130× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 70,120 | 54,520 |
| 2 | pytorch → opennn | 70,126 | 53,035 |
| 3 | opennn → pytorch | 69,954 | 54,953 |


`cpu-dense-infer` — batch 4,096, fp32, passes 5 per launch, 3 rounds. Artifact `cpu-dense-infer-publish-20260906T134425Z.json`, commit `e76425bd3`, clean tree, quiet True (busy 0.1% before, 0.0% after, 0.1% max during, threshold 3%), clocks locked True, shape gate True, quality gate True (no metric).

| engine | median samples/s | min | max | peak RssAnon MiB | Wh (RAPL package-0) |
|---|---|---|---|---|---|
| OpenNN | 220,512 | 220,493 | 220,647 | 280.4 | 0.09791 |
| PyTorch | 170,329 | 169,852 | 170,765 | 568.7 | 0.10630 |
| **ratio** | **1.295×** | | | 2.03× less | 1.086× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 220,493 | 170,329 |
| 2 | pytorch → opennn | 220,647 | 169,852 |
| 3 | opennn → pytorch | 220,512 | 170,765 |

## Why

Both engines hand the matrix products to the same vendor libraries — cuBLASLt
or cuDNN on the GPU, MKL on the CPU — so the margins come from what happens
*around* the products: how many passes over the activations a layer costs, how
much host work each batch carries, and which thread pool the passes run on.

### Where the energy goes

Energy is power times time, and the two GPU cells now pay in different
proportions than they did a session ago.

`cuda-dense-infer` used to be the cell where OpenNN won on **power** — 186.4 W
against 248.6 W in `2026-09-03-publish`, for an energy margin of 1.339× on a
throughput margin of 1.004×. It is not that cell any more. The published
launches read 245.1 W against 254.5 W, and the energy margin decomposes almost
exactly: 1.018× on time times 1.039× on power gives the 1.057× measured. That
fall from 1.339× to 1.058× was chosen, not suffered, and the section below
gives the variant table that prices it.

`cuda-dense-train` wins on time at a board power the run cannot separate:
OpenNN's three launches read 219.5 / 220.1 / 219.8 W and PyTorch's 219.9 /
209.6 / 216.1, so PyTorch's own spread straddles OpenNN's value and neither
engine is established as the hotter one. The 1.149× energy margin is a time
margin, measured over windows that do not exactly coincide (see the caveats).
The energy ratio is also the *steadier* of the two
measurements here — 1.146 / 1.156 / 1.149 across the three rounds, against a
throughput ratio that swings 1.121 / 1.196 / 1.129 — because both engines
process exactly 24,576,000 samples per launch, so the undivided watt-hours are
already energy per fixed workload while the samples/s figure is not.

The two CPU cells win on **time while drawing more power**, and the honest
statement is that OpenNN is not the more frugal engine here. Over the same
window OpenNN pulls 31.1 W of package power against PyTorch's 26.1 W on
inference, and 29.1 W against 25.7 W on training — 19% and 13% more. It wins
1.086× and 1.130× anyway because it finishes 1.295× and 1.286× sooner. The
extra watts are the point, not a defect: the RAPL counter is measuring cores
held closer to their sgemm peak (88.5% against 68.9%), and a core at higher
occupancy costs more per second and less per sample.

### Where the memory goes

The two devices report different metrics and neither is a workload figure. On
CUDA `peak_mib` is `device_used_minus_idle`: NVML's device-used at peak, minus
a pre-launch idle reading of 243–255 MiB taken the same way for both engines.
It therefore contains each process's CUDA context, its loaded kernel images
and, on PyTorch's side, caching-allocator blocks that are reserved and free.
On CPU it is `process_peak_anonymous_rss`, the whole process including its
framework baseline.

The CUDA margins are small and mostly context. Both engines hold the same
resident split and the same 1,080,321 parameters — 4 MiB of fp32, 2 MiB of
bf16 — so the 124 MiB of the training cell (507.6 against 631.6) and the 40
of the inference cell (371.3 against 411.3) are allocator and context rather
than workload. The inference figure fell from 377.9 to 371.3 this round
because the driver now deploys the parameters through
`upload_parameters_bf16_inference()`, which releases the fp32 master the
forward pass never reads (see `transformer.md`, where the same change is
worth 245 MiB); on a 1 M-parameter network it is worth 6.

The CPU margins are not workload figures at all, and they are the largest
numbers in the table. Both drivers print a baseline: OpenNN 208.5 MiB against
PyTorch 761.7 on inference and 833.0 on training. Those gaps, 553 and 624 MiB,
are each larger than the whole peak difference being reported as the win (288
and 482 MiB). On top of that the two sides hold the data in different kinds of
memory: OpenNN's reader `mmap`s the CSV read-only (`io_utilities.cpp`), so the
158 MB test file lands in `RssFile` and outside the metric entirely, while the
PyTorch driver materialises it anonymously two to three times over — a pandas
frame, an `np.ascontiguousarray` copy of it, then
`torch.from_numpy(...).contiguous()`. That is the driver's construction, not
PyTorch's, and it is ours. Both effects push the same way, so read the 2.58×
and the 2.03× as process-footprint ratios bounded above by those two
asymmetries and not as what the two engines cost to run.

One part of the CPU figures did change this round, and it was OpenNN's to
fix: both cells read 34–35 MiB less than at `93cc90e07` (315.7 → 280.4 and
339.2 → 305.2) because a CPU-only process no longer creates a CUDA context.
The device backend used to create its streams and cuBLASLt/cuDNN handles in
its constructor, and the CPU GEMM path reaches that singleton for its thread
pool — so every CPU cell was holding a 226 MiB context on the GPU and the
driver's host-side state for it in its own resident set. `e76425bd3`
initialises the CUDA side on first CUDA use; a CPU process no longer maps
`/dev/nvidia*` at all. This is a library fix, not a benchmark one, and it
applies to every CPU-only user of a CUDA build.

### `cuda-dense-infer`, 1.018×: the cell that was losing

**Start with what changed, because the previous revision of this document was
measured against a PyTorch that no longer exists.** In `2026-09-03-publish`
this cell read 1.004× on throughput and 1.339× on energy. Then PyTorch got 4.0%
faster on it with nothing changed on its side that we made or recorded:
37,178,529 samples/s became 38,681,438. The likely cause is a cleared
`max-autotune` cache and a re-tune, but nothing in the store records a cache
state, an uptime or a boot, so the mechanism is inferred and only the 4.0% is
measured. Under PROTOCOL §3 each
engine is measured at its best, so the new, faster PyTorch is the honest
comparison — and against it the configuration we had published read **0.966×**,
which is the `OPENNN_CUDNN_MATMUL=0` row of the variant table below. Our
1.004× was measured against an unlucky draw of somebody else's autotuner: a
margin of a fraction of a percent on this cell was never a result, it was a
coin landing our way.

Nothing about it is traced. No profile of the re-tuned build exists to say
which of its kernels got faster, so the 4.0% is recorded, not attributed.

**The time ceiling.** Three constants carried from an earlier session and not
re-measured in this one: at the locked clock a bf16 8,192³ matmul sustains
94.0 TFLOPS on this card, and the 48 MB L2 serves 1.65 TB/s. The
8,192 × 1,024 × 1,024 hidden layer is 17.2 GFLOP, so it cannot take less than
183 µs. The first layer, 28 → 1,024, is 0.47 GFLOP — a fraction of a
microsecond of arithmetic — but it writes a 16.8 MB bf16 activation, 10.2 µs at
that bandwidth, and the last layer reads the same 16.8 MB back, another 10.2.
So a batch cannot take less than about 203 µs, 40.3 M samples/s, and every
number in this cell has to be read against that. OpenNN's published throughput
implies 207.8 µs a batch; PyTorch's implies 211.8 µs. Both are within 5% of a
floor set by data movement, which is the whole reason the margin is 1.9% and
not 20%.

**Per-kernel, one batch of 8,192.** The trace below is from `nsys` at commit
`6b7179dde`, the previous published build. It is kept because the two small
layers are unchanged and it is the only per-kernel decomposition that exists;
its hidden-layer rows describe **neither** engine now — OpenNN has moved to a
cuDNN engine and PyTorch has re-tuned.

| layer | OpenNN at `6b7179dde` | PyTorch at `6b7179dde` |
|---|---|---|
| 28 → 1,024, bias + ReLU | `small_k_linear_kernel<bf16,relu>` 12.8 µs | Triton autotuned GEMM + fused epilogue 16.5 |
| 1,024 → 1,024, bias + ReLU *(superseded)* | `nvjet` bf16 GEMM, 256×160 tile, 200.4 | Triton autotuned GEMM + fused epilogue 190.1 |
| 1,024 → 1 | `linear_forward_single_output_kernel`, sigmoid fused, 6.9 | Triton GEMV 12.9 |
| traced time per batch | 219.2 | 220.5 |

The last row is the traced batch time, not the sum of the kernel means above
it. OpenNN's output layer computes the sigmoid inside the timed pass and
PyTorch's does not — `dense.py` folds it into `BCEWithLogitsLoss` and applies
it only in `evaluate()`, outside this cell — so the comparison charges OpenNN
an elementwise pass PyTorch never runs. At that commit OpenNN won the two small
layers by 9.7 µs and lost the big one by 10.3 µs, and the cell came out at
1.004×. Both halves of that sentence have
since moved, and the consistency check holds: 219.2 µs traced against 219.5 µs
implied by the published 37,317,959 samples/s, and today 207.8 µs implied by
39,412,929 against 209.5 µs for the three current kernels measured
individually — 189.8 for the cuDNN engine plus the two small layers' 12.8 and
6.9. The arithmetic agrees to 1%; what it does *not* license is a
layer-by-layer attribution of today's 3.9 µs margin, because PyTorch's 8.6 µs
improvement is unattributed. Treat the decomposition as inference.

*The first layer is a write, not a GEMM.* With a contraction of 28, cuBLASLt
has no aligned kernel to pick — 28 bf16 are 56 bytes, so it falls to an
`align2` variant that loads the operands two bytes at a time and spends 22 µs
producing what a memset produces in 10. Inductor does better (16.5 µs) because
`max-autotune` benchmarks its own Triton templates against cuBLAS for every
shape and keeps the winner; that is why the published PyTorch mode is
`max-autotune-no-cudagraphs`. OpenNN runs a contraction of at most 32 through
its own kernel (`opennn/core/cuda/kernel_small_k_linear.cu`): `mma.sync` bf16
tensor-core fragments loaded straight from global memory, the 28 × 1,024
weights staged once through shared memory into fragments that stay in
registers for the block's lifetime, bias and ReLU in the epilogue, and the
bf16 output staged through shared memory so that every store writes four full
128-byte rows. It takes 12.8 µs — 80% of the memset floor's bandwidth. The
controlled comparison, `OPENNN_SMALL_K_LINEAR=0`, reads 37,750,841 samples/s
against the published 39,387,890 — 0.975× of PyTorch instead of 1.018× — over
three rounds each way (session `2026-09-06-dense-variants`,
`cuda-dense-infer-smallk0-20260906T192919Z`; energy 0.17415 against 0.17313
Wh, memory 374 against 371 MiB). The kernel is worth 4.3% of the cell, and it
is the 4.3% that turns the cell.

*The last layer is one kernel.* OpenNN's single-output path
(`linear_forward_single_output_kernel`, one warp per row, the sigmoid fused)
reads the activation once at 6.9 µs; Inductor lowers the 1,024 → 1 product to a
GEMV at 12.9 µs, with no sigmoid in the timed path to fuse.

*The middle layer is 90% of the batch, and it is where this session's work
went.*

#### Two kernels, and a hole where the third should be

The hidden layer is m 1,024, n 8,192, k 1,024, NN, bf16 in and out, fp32
accumulate, `CUBLASLT_EPILOGUE_RELU_BIAS`. Kernels doing exactly that
arithmetic at the same locked clock, timed back to back with board power
integrated from the driver's 20 ms sample ring:

| source | time | power | TFLOPS | energy per GEMM |
|---|---|---|---|---|
| PyTorch's autotuned Triton kernel | 189.6 µs | 227 W | 90.6 | 43.0 mJ |
| cuDNN engine 1 cfg 10 — what OpenNN now runs | 189.8 µs | 227.7 W | 90.5 | 43.2 mJ |
| cuBLASLt's fastest, the 64×64 | 192.4 µs | 265 W | 89.3 | 51.0 mJ |
| cuBLASLt's leanest, the nvjet 256×160 | 201.2 µs | 169 W | 85.4 | 34.0 mJ |
| best of 5 CUTLASS 3.8 instantiations | 211.0 µs | 210 W | 81.4 | 44.3 mJ |
| best of 6 hand-written `mma.sync` kernels | 232.5 µs | 161 W | 73.9 | 37.4 mJ |

The energy column is the product of the two measured columns, not a separate
measurement.

The 96 W spread across kernels doing identical work is not arithmetic — it is
data movement. A tile of `rows × columns` reads one row-strip of A and one
column-strip of B per output tile, so it moves `1/rows + 1/columns` bytes
through L2 and shared memory per multiply-add, and board power follows that
ratio almost exactly:

| tile | 1/rows + 1/columns | watts |
|---|---|---|
| 64×64 | 0.0312 | 265 |
| 64×128 (Triton's) | 0.0234 | 227 |
| 64×640 | 0.0172 | 189 |
| 80×512 | 0.0145 | 182 |
| 128×240 | 0.0120 | 179 |
| 128×320 | 0.0109 | 172 |
| 256×160 | 0.0102 | 169 |

Those points were measured in an earlier session's standalone kernel probe and
were not re-measured this session; the six nvjet rows are the table
`device_backend.cpp` carries, which sets the library's traffic budget at 0.0120
on the record that every tile at or below it measured within 10 W of the 169 W
floor while the 64×64 sits at 0.0312 and 265 W. What that supports is
monotonicity across cuBLASLt's nvjet tiles at this shape and nothing wider:
Triton's 64×128 is not one of them, and the relation does not extrapolate off
the family — it prices the cuDNN engine below at 265 W where it measures 227.7.

**Why the answer is not a lookup table.** The shortlist, the counts and the two
experiments below are the same standalone probe's, carried over with the
tile-power table. The probe (`cudnn_matmul_probe.cu`, which `cudnn_matmul.h`
names) is not in the tree and its output is not in the artifact store, so what
can still be checked against the library source is the 13,460 count, the 0.86 s
enumeration, the bound that nothing under 198 µs draws less than 265 W, and the
64×64 at 192.0 µs against the 256×160 at 201.4 — a quarter of a percent from
the 192.4 and 201.2 the kernel table above reports for the same two tiles.

cuBLASLt's heuristic returns a shortlist ranked by expected speed. For this
GEMM it returns eight candidates, and timing every one of them gives:

| rank | tile | time |
|---|---|---|
| 0 | 64×64 | 192.0 µs |
| 1 | 80×192 | 237.0 µs |
| 2 | 128×160 | 206.8 µs |
| 3 | 128×128 | 232.9 µs |
| 4 | 128×256 | 210.7 µs |
| 5 | 128×176 | 222.6 µs |
| 6 | 64×104 | 268.3 µs |
| 7 | 128×120 | 248.3 µs |

The autotuner kept the fastest, which was correct on its own terms and picked
the 265 W kernel. The efficient kernel is not on the list: **256×160 is never
offered by the heuristic.** So the search was widened. Enumerating every
configuration cuBLASLt will accept for this problem — 20 algorithm ids, each
advertised tile, stage and custom option, validated with
`cublasLtMatmulAlgoCheck` — gives 13,460 valid configurations, enumerated in
0.86 s. Timing all of them settles the shape of the trade:

| faster than | configurations |
|---|---|
| 198 µs | 1 |
| 199 µs | 6 |
| 200 µs | 23 |
| 205 µs | 79 |

There is exactly one configuration below 198 µs and it is the power-hungry
one: **the frontier has a hole precisely where the answer would be.** cuBLASLt
offers 192.4 µs at 265 W or 201.2 µs at 169 W, and neither wins both. Run as
whole cells, the 265 W configuration read 1.043× throughput at 0.919× energy
(`...publish-20260902T121416Z`, commit `bc6f4c2d0`) and the 169 W one reads
0.966× throughput at 1.315× energy, the `OPENNN_CUDNN_MATMUL=0` row below.
Three attempts to find a middle inside cuBLASLt all failed: the TN layout's
best is 193.1 µs at 271 W and its own 256×160 lands at 200.4 µs and 172 W;
splitting the batch between the two kernels measured slower than either alone
(best split 203.5 µs against 200.1 pure), because halving `n` costs the
wide-tile kernel more in tail waves than the mix saves; and 64×64 is fast only
at 12 stages — at 16, 18 and 25 it takes 410, 247 and 240 µs.

#### Eleven kernels that did not work

The previous revision of this document proposed writing the missing kernel: a
low-traffic tile at Triton's speed, taking both axes outright. That work was
done. Eleven kernels — five CUTLASS 3.8 tile-shape instantiations and six
hand-written `mma.sync` designs — were written for this shape and verified
correct against a cuBLASLt reference. None reached the 191.8 µs the selection
rule required: the best CUTLASS instantiation is 211.0 µs, the best
hand-written kernel 232.5 µs.

The diagnostic that closed the line of work: at Triton's *own* tile, CUTLASS
is 11% slower than Triton. The gap is mainloop scheduling, not geometry, so no
amount of tile-shape search was going to reach it.

#### The engine cuDNN had

cuBLASLt and cuDNN ship different matmul engine sets. Enumerating cuDNN's
configurations for this graph — 60 built, executed and verified against a
cuBLASLt reference — finds thirteen under both the time and the energy bound,
the best at 189.8 µs and 227.7 W. That is Triton's operating point, reached
through NVIDIA's own library rather than a generated kernel.

**cuDNN's own heuristic puts a 226 µs engine first.** Taking the first
suggestion would have lost the entire benefit. Enumerating rather than
trusting a heuristic is what found this, and it is the third time in this
codebase that enumerating beat trusting one: the cuBLASLt heuristic's
shortlist above, which never offers the 256×160, and the custom-option scan
that took the first valid option per tile, are the other two. The pattern is
written into the code rather than left to a reader — the candidate accessors
in `cudnn_matmul.h` say the candidates are deliberately **not** ranked,
because the caller must time them.

Two implementation notes:

- **Correctness is checked, not argued.** Every cuDNN candidate is verified at
  warmup against the cuBLASLt answer on the caller's own data — three
  4,096-element windows compared RMS-relative at 2e-2 — and a candidate that
  disagrees is dropped with its engine named. It costs three 8 KB copies and
  no device memory.
- **The implementation corrected its own brief.** OpenNN stores Dense weights
  `[inputs][neurons]`, not `[neurons][inputs]`: `dense_layer.cpp` indexes
  `i*outputs+j` and `linear_forward_lt_gpu` passes them as A with `OP_N` and
  `lda = m`. The copy-free cuDNN stride is therefore the transposed literal,
  which happens to be the cooler of the two winning engines — 227.7 W against
  235.4 W. An 8 W saving that was budgeted as a future layout change was
  already collected.

#### The rule, and the variant that isolates it

Selection is two stages, ordered by the quality of the evidence
(`device_backend.cpp`).

**Stage 1, inside cuBLASLt**, is anchored on the fastest cuBLASLt candidate
and takes a lower-traffic tile only when three conditions hold: traffic at or
below `OPENNN_LT_TRAFFIC_BUDGET` (0.0120), strictly lower modelled energy than
the fastest candidate, and at most `OPENNN_LT_TILE_TOLERANCE` percent slower
(default 10). The old rule was a flat 5% window, and the chosen 256×160 sat at
4.9% inside it — one tenth of a point — so any newly timed candidate at
191.8 µs or less would have pushed it out and reverted the cell to a 265 W
kernel silently, since no cell reports which tile it ran. Three conditions
instead of one window is the fix; a faster high-traffic kernel appearing can no
longer change the pick.

**Stage 2, between libraries**, is measured time alone: a cuDNN candidate wins
only when it beats stage 1's choice by more than
`OPENNN_MATMUL_CROSS_SOURCE_GAIN` percent, default 2. There is deliberately no
modelled energy term. `tile_traffic` reads a cuBLASLt tile id and
`tile_power_watts` is fitted over cuBLASLt nvjet tiles; asking either about a
cuDNN engine does not give a conservative estimate, it gives a fabricated one —
the model prices this engine at 265 W where it measures 227.7 — and the rule
would then use that fabrication to overrule a measurement. A model may not be
used outside the family it was calibrated on to overturn a real number.

The knobs are the experiment. Three rounds each, same commit (`93cc90e07`),
same machine, one variable in the environment, each ratio taken against the
PyTorch measured in the same run. The default row is the published cell; the
other three are the `2026-09-05-variants` session.

| configuration | throughput | energy | artifact |
|---|---|---|---|
| **default (published at `93cc90e07`)** | 39,412,929/s, **1.019×** | 0.17289 Wh, **1.058×** | `...publish-20260905T105536Z` |
| `OPENNN_CUDNN_MATMUL=0` | 37,387,210/s, **0.966×** | 0.13732 Wh, **1.315×** | `...20260905T112041Z` |
| `OPENNN_MATMUL_CROSS_SOURCE_GAIN=100` | 37,392,713/s, **0.966×** | 0.13676 Wh, **1.321×** | `...20260905T112201Z` |
| `OPENNN_LT_TILE_TOLERANCE=0` | 39,412,842/s, **1.019×** | 0.17190 Wh, **1.057×** | `...20260905T111857Z` |

Removing cuDNN from the candidate set returns the cell to exactly the 0.966× it
read before the work — the whole 5.4% of throughput is that one plan. Raising
the cross-source gain to 100% gives the identical result by a different route,
which confirms the mechanism rather than merely the outcome: it is stage 2
taking the cuDNN engine, not some side effect of linking cuDNN, and it is the
row that shows cuDNN winning here by more than the 2% margin. The third row is
not the same experiment. With `OPENNN_LT_TILE_TOLERANCE=0` the code takes the
branch `chosen = best_any` — fastest measured time across both libraries, the
cross-source margin never consulted — so it says only that the cuDNN engine is
the fastest candidate outright, which is why throughput and energy land on the
default's values. Its peak memory does not: 372.1 / 381.5 / 387.4 MiB across
its three launches, against 377.9 on all three of the default's, which is a
spread rather than a difference.

Stage 1 still governs every shape cuDNN declines, and `cudnn_matmul.cpp` lists
them: not bf16 in and out, a non-zero beta, an AUX or GELU epilogue, an m or k
that is not a multiple of eight, or a product below a minimum GFLOP.
`cuda-dense-train`'s hidden layer is none of those — it is the same
1,024 × 8,192 × 1,024 bf16 `RELU_BIAS` problem, built at warmup before capture
— and `device_backend.cpp`'s own comment records two of that cell's three GEMMs
qualifying for stage 2 as well. So the training cell is not a shape this rule
protects from the trade; see its section below.

#### What the trade cost, stated as a trade

The energy on this cell fell from **1.339× to 1.058×**. That is not an
improvement and this document will not present it as one. It is what winning
the throughput axis cost, and the variant table is the price list: 0.13732 Wh
at 0.966× throughput, or 0.17289 Wh at 1.019×. The cell now burns 25.9% more
energy than the configuration it replaced, to run 5.4% faster.

The rule that decided it was stated before the measurement, not after: clear
the constraint on every axis first, then maximise. A configuration that wins
two axes and loses the third does not clear the constraint, whatever its
geometric mean looks like. Under a different rule — maximise energy subject to
not losing throughput badly — `OPENNN_CUDNN_MATMUL=0` is the better
configuration, and the environment variable is there so that a reader who
holds that rule can have it in one line.

Two things about this trade are uncomfortable and both belong here.

**Stage 2 hands back a trade stage 1 made, without pricing it.** The
cross-source gain is measured against `times[best]` — the kernel stage 1
chose, which stage 1 was allowed to slow down by up to 10% to buy modelled
energy. On this GEMM that is the difference between a 6.1% margin (against the
lean 201.2 µs tile) and a 1.5% one (against cuBLASLt's fastest at 192.4 µs) —
and 1.5% is below the 2% noise floor. **So the anchor, not the margin, is why
cuDNN is taken here.** `OPENNN_MATMUL_CROSS_SOURCE_ANCHOR_FASTEST=1` moves the
anchor to the fastest cuBLASLt kernel, which is the stricter reading of the
claim; that A/B is not in the published artifact store.

**The rule cannot see a bad trade coming.** On a shape where a cuDNN engine is
4% faster and much hotter, stage 2 takes the throughput and loses the energy,
and nothing in the library notices; `cuda-dense-train` below looks like exactly
that shape. Closing that needs a per-candidate energy
measurement the tuner cannot afford: NVML's board power is a one-second
average whose sample ring needs about a second of window with the ends
discarded, so the probe spent 1.5 s per configuration, and sixty
configurations at 1.5 s is ninety seconds per shape inside a plan cache that
fills at warmup. Until that changes it is a visible default with three knobs
rather than a decision buried in a tie-break.

#### What is not to OpenNN's credit

- **The two engines do not share a cuDNN.** OpenNN links the system
  `libcudnn.so.9` at 9.25.1; PyTorch loads the 9.23.2 its wheel ships
  (PROTOCOL §1). This cell's throughput margin now rests entirely on an engine
  found by enumerating cuDNN's set, and nothing here establishes that the same
  engine exists in 9.23.2. The asymmetry cannot be corrected without building
  one engine's cuDNN from source, and its effect on this cell is unmeasured.
- **PyTorch would not look in cuDNN for a `Linear` anyway.** Inductor
  generates a Triton kernel and eager goes to cuBLAS; routing a matmul through
  cuDNN's engine set is not a thing the comparison engine does or is trying to
  do. OpenNN did not write a faster kernel — it went shopping in a library
  nobody searches for a matmul, and found NVIDIA's kernel there. Triton's
  190 µs and cuDNN's 190 µs are the same kernel-quality outcome reached two
  ways.
- **The hidden layer is a tie, so the cell's margin is the small layers.** At
  1.019× this is the narrowest cell in the matrix and the least interesting
  one to quote. What it demonstrates is a deficit removed, not a lead built.
- **Enumerating cuDNN costs peak memory.** OpenNN's peak went from 367.8 MiB
  at `6b7179dde` to 377.9 at `93cc90e07`, and `OPENNN_CUDNN_MATMUL=0` reads
  367.4. It is not the running plan's workspace:
  `OPENNN_MATMUL_CROSS_SOURCE_GAIN=100` does not select the cuDNN engine — its
  37,392,713 samples/s says so — and still reads 377.6, so the 10 MiB is spent
  building and timing the candidates at warmup whether or not one is taken.
  The memory margin narrowed from 1.12× to 1.090× to pay for the throughput.
  `cuda-dense-train`'s narrowed the same way over the same range and is not
  attributed: 491.8 MiB to 507.9 against a PyTorch that did not move (631.8 to
  631.9), so 1.285× to 1.244×.

#### What is left on the table

No kernel measured here is both fast and lean: the cheap 169 W point costs
201.2 µs and everything at Triton's 190 µs draws 227 W or more. cuDNN can also
fuse a second contraction into the same graph, which cuBLASLt structurally
cannot — a layer removed from the caller rather than a kernel from a plan, not
attempted. And stage 2 will keep taking trades it cannot price until
per-candidate energy is measurable at warmup cost.

**Why the margin is small at 8,192 and large below it.** The published batch
is the one that saturates the GEMM, so it is the batch at which the two
engines have the least room to differ; below it the batch is mostly issue
cost, where OpenNN's three launches from C++ compete against Inductor's
kernels plus their Python guards. The batch curve, each batch in its own
process (session `2026-09-06-dense-variants`,
`cuda-dense-infer-sweep-20260906T193102Z`, one round per rung, 200 passes,
quiet, at `bfd4e58fb`):

| batch | OpenNN samples/s | PyTorch samples/s | OpenNN / PyTorch | peak MiB ON / PT | Wh ON / PT |
|---|---|---|---|---|---|
| 512 | 26,556,660 | 4,561,814 | **5.82×** | 334 / 472 | 0.1928 / 0.4214 |
| 1,024 | 31,376,476 | 10,168,077 | **3.09×** | 332 / 470 | 0.1483 / 0.2599 |
| 2,048 | 33,970,788 | 19,435,814 | **1.75×** | 338 / 470 | 0.1370 / 0.1939 |
| 4,096 | 38,104,246 | 36,339,617 | **1.05×** | 356 / 518 | 0.1822 / 0.1822 |
| 8,192 (published) | 39,429,162 | 38,711,898 | **1.02×** | 368 / 412 | 0.1708 / 0.1810 |
| 16,384 | 37,596,397 | 38,220,735 | **0.98×** | 404 / 584 | 0.2019 / 0.2096 |

The shape is the one the mechanism predicts: below the published batch the
margin opens fast — 5.8× at 512, where PyTorch's compiled kernels plus their
guards cost more per batch than the batch's arithmetic — and at the published
batch the GEMM saturates and the two engines meet. One rung above it PyTorch
is ahead: at 16,384 OpenNN reads 37.6M against 38.2M, 0.984×, because its
three-layer pass falls off the L2-resident regime (the 16,384 × 1,024 bf16
hidden activation is 32 MiB, two of them no longer fit the 48 MB L2) while
Inductor's autotuned GEMM for that shape does not care. A reader deploying
this network at 16,384 or above should expect PyTorch to be marginally faster
on time and OpenNN to remain ahead on memory and energy; below 4,096 the
margin is not marginal at all.

CUDA graphs should be worth nothing to OpenNN on this cell: the published
launch runs the resident split as views with the graph off, and three launches
per 0.21 ms batch are well inside what one host thread queues ahead of the GPU.
The measurement agrees: the `OPENNN_DENSE_INFER_GATHER=1` A/B above reads
0.948× against 1.018×, the gather kernel and index copy costing more than the
launches they replace save. They are worth less than nothing
to PyTorch here: `reduce-overhead` costs it 25% against
`max-autotune-no-cudagraphs`, because cudagraph-tree replay copies each input
slice into its static placeholder and runs its bookkeeping in Python before
every replay.

### `cuda-dense-train`, 1.13× to 1.26×: a launch-bound step

The Adam step on this network is small — roughly 52 GFLOP of GEMM per batch of
8,192, about 0.56 ms of tensor-core time — and both engines spend the rest of
the batch waiting on the host. That is what the compile modes say: PyTorch
eager (every kernel launched from Python) makes ~4.8 M samples/s, Inductor
without graphs ~7.0 M, Inductor with CUDA graphs ~9.9 M (`20260902T045145Z`,
`T045015Z`, `T022718Z` — three-epoch runs with ~70 ms windows, quoted for the
mode choice and not as measurements of this cell); the same work runs about
twice as fast depending only on how it is *issued*. OpenNN captures the whole
step — gather of the shuffled batch, forward, loss, backward, Adam update —
into one graph. The A/B for that is bounded rather than measured: 10,874,476
against 10,616,248 samples/s (`OPENNN_NO_CUDA_GRAPH=1`, `20260902T045428Z`)
pairs `38ad27e16` with `918805ce1` over ~70 ms windows, inside which the
graphed run's own three launches span 3.2%. The graph is worth at most a few
percent here, because OpenNN's host path is already short.

**Quote this cell as a range.** PyTorch's median has read 10,115,756 /
9,865,309 / 9,021,160 / 10,091,130 across four runs — an 11% band — while
OpenNN's three published launches span 0.006%. The four are not one
configuration repeated. Only 10,091,130 is the published run at `93cc90e07`;
10,115,756 is `6b7179dde`; the other two are `1b8c65971` with a dirty tree, and
9,865,309 comes from a run the quiet gate rejected at 4.9% busy. Against
OpenNN's published median the four give 1.127× to 1.263×, so the cell reads
1.13× to 1.26× depending on the draw, with the upper end resting on a
dirty-tree run. The 1.267× quoted at the commit that introduced the cuDNN work
pairs the 9,021,160 draw with that same scratch run's OpenNN launch rather than
the published one. 1.135× is the published session's median and the
conservative end.

**What moved since `6b7179dde`**, where this cell read 10,768,761 samples/s at
0.12009 Wh: twelve commits later it reads 11,396,057 at 0.13201 Wh — 5.8%
faster for 10% more energy. The gain arrives in at least two steps and no clean
A/B isolates either. The scratch runs at `1b8c65971` are dirty-tree, so they
bracket rather than prove: they read 11,109,937 and 11,145,061 samples/s at
0.11419 and 0.11371 Wh — about three points of the throughput, with the energy
*below* `6b7179dde`'s. The same tree with the cuDNN patch applied reads 11,431,172 at
0.13226 Wh, and the published `93cc90e07` cell sits on that side of the step.
On that evidence `cuda-dense-train` made the same trade `cuda-dense-infer`
made: the last 2.6% of throughput cost 16% of energy. This document has no
cleaner A/B than that to price it with, and it does not claim the cell's energy
was won by finishing sooner alone.

Two mechanisms are present in that range, neither of them an accounting for the
5.8%. The backward dX bypassed the plan entirely — `multiply_gpu` →
`cublasGemmStridedBatchedEx` with `CUBLAS_GEMM_DEFAULT`: no cache, no timing,
no known-tile injection, no tie-break, for 25.7% of the dense training step —
and the rank-2 case now goes through `run_lt_matmul_cached`, where the
tie-break does the same m, n, k in 200.4 µs at 172 W against the default pick's
202.31 µs (`tensor_operations.cpp`), which is faster and cooler rather than
hotter. And a 4 kB memset sat inside the captured step, becoming a graph node
the driver schedules on a different internal stream: 3.68 µs × 3,000 =
11.03 ms, which the commit removing it recorded as 32% of that cell's GPU idle
— about 23% of the 48.1 ms the trace below reports.

The nsys trace below is from `6b7179dde` and describes the shape of the step —
launch counts, occupancy, the gap distribution — not its current timing.

| | OpenNN at `6b7179dde` | PyTorch at `6b7179dde` |
|---|---|---|
| published median / under `nsys` (samples/s) | 10,768,761 / 10,725,143 | 10,115,756 / 7,687,898 |
| timed window traced | 2.302 s | 3.171 s |
| kernel launches | 63,101 (27,411/s) | 104,987 (33,108/s) |
| GPU busy with kernels | 98.1% | 73.1% |
| GPU busy with any work (kernels, copies, memsets) | 98.3% | 73.1% |
| gaps between kernels, median / p90 | 0.2 / 2.1 µs | 0.2 / 20.4 µs |
| idle between kernels, total | 48.1 ms | 853.4 ms |

OpenNN's column does not close — 2.302 s at 98.1% busy leaves 43.7 ms, not the
48.1 stated, and its kernel shares implied more kernel time than the window
holds — and the trace is not in the store to recheck it, so the share column is
dropped below and the busy figures should be read as approximate. PyTorch's
column reconciles exactly. What the pair supports is 98% against 73%.

OpenNN, top kernels by summed time in the window:

| launches | mean µs | kernel |
|---|---|---|
| 3,000 | 204.9 | `cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bgrada_bf16_128x128...` |
| 3,000 | 204.3 | `nvjet_sm120_tst_mma_256x160x32_3_64x80x32_tmaAB_alignCD4_bx_biast_r...` |
| 3,000 | 202.3 | `nvjet_sm120_tst_mma_80x192x64_2_80x24x64_tmaAB_alignCD4_bz_TNNN` |
| 3,000 | 32.5 | `gather_rows_kernel<__nv_bfloat16>` |
| 3,000 | 29.7 | `cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_64...` |
| 3,001 | 25.2 | `adam_update_kernel(int, int, float *, float *, float *, const float...` |

PyTorch, top kernels by summed time in the window:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 27.3% | 3,000 | 210.7 | `cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x...` |
| 24.8% | 3,000 | 191.3 | `cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x6...` |
| 24.1% | 2,999 | 186.2 | `cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x6...` |
| 3.5% | 3,000 | 27.2 | `cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x...` |
| 2.9% | 6,000 | 11.2 | `triton_poi_fused_relu_3` |
| 2.4% | 2,999 | 18.5 | `triton_for_fused_1` |

The mechanism the trace shows is still the mechanism: OpenNN replays one
captured graph and keeps the card busy; PyTorch's cudagraph-tree replay leaves
it idle a quarter of the window. That is why the cell is won on time and not
on watts.

The tile-selection rule's contribution to this cell's *energy* is not quoted.
The only `OPENNN_LT_TILE_TOLERANCE` A/B in the store is at `1b8c65971`, inside
a cluster of runs that read 7.6-7.9 M samples/s whether or not they passed the
quiet gate — `cuda-dense-train-tol0-20260904T192841Z` is quiet at 0.8% busy and
still reads 7,943,400 — against 11.1 M for runs at the same commit forty
minutes later. Whatever depressed that cluster was never established, so none
of it is quotable, and no A/B for this variable exists at `93cc90e07` at all.

### `cpu-dense-infer`, 1.295×: a layer at 89% of the cores' peak

On the CPU the arithmetic is fixed and large: 8.83 GFLOP per batch of 4,096,
of which 8.59 are the 1,024 × 1,024 layer. The eight P-cores at their locked
2.1 GHz can retire 538 GFLOP/s of fp32 FMA (two 256-bit units per core, no
AVX-512 on this part) — another machine constant carried from an earlier
session and not re-measured here — so the batch cannot take less than 16.4 ms.
OpenNN takes 18.56 ms (88.5% of peak); PyTorch takes 23.83 ms (68.9%).

**The profile says the cell is three MKL calls.** Profiling OpenNN's published
inference configuration (`cpu-dense-infer-prof-20260906T063011Z.json`,
commit `93cc90e07`, 218,983 samples/s — 0.8% below the published median, so
the profile describes the published cell):

| primitive | ms/call | share of `cpu:linear_fwd` |
|---|---|---|
| `cpu:sgemm_wide` — the 1,024 × 1,024 layer | 17.877 | 95.7% |
| `cpu:sgemm_thin_k` — the 28-wide first layer | 0.649 | 3.5% |
| `cpu:sgemm_thin_n` — the 1,024 → 1 layer | 0.153 | 0.8% |

Those three are 99.9% of an 18.70 ms batch, and `op:activation_fwd` runs at
0.001 ms a call over 2,562 calls, so the ReLU is not a separate pass and no
framework work in this cell is worth measuring. Read the 89% carefully: the
scope is `PROFILE_SCOPE_HOST` around the whole of `blocked_linear_forward`, not
around the `cblas_sgemm` inside it, so what it times is OpenNN's blocked layer
with its epilogue in it — 8.590 GFLOP in 17.877 ms, 480 GFLOP/s, 89% of the 538
the eight cores can retire. It cannot separate MKL's share from the blocking's.

The margin over PyTorch on this cell is not OpenNN's arithmetic against
PyTorch's: both call MKL. What the profile establishes is that OpenNN spends
essentially nothing outside its three GEMM calls. It does not establish where
PyTorch's extra 5.27 ms a batch goes, and the epilogue cannot be all of it —
two out-of-place elementwise passes over a 16 MiB activation move well under a
millisecond at this machine's bandwidth. The rest most likely sits in how MKL
is threaded on each side, one threaded `sgemm` over the whole matrix against
eight single-threaded block `sgemm`s on the OpenMP pool, and nothing here
measured it.

It also closes a roadmap item by measurement. A CPU small-K kernel — the host
analogue of the CUDA `small_k_linear` above — was gated on the 28-wide first
layer exceeding 1 ms a call. It is 0.649 ms and 3.5% of the batch, so even a
perfect kernel there is worth under 3.5% and the item is closed rather than
deferred.

OpenNN's layer is `blocked_linear_forward` (`tensor_operations.cpp`): the rows
of the batch are split into blocks, each block is one single-threaded MKL
`sgemm` on one of the OpenMP workers, and the bias and ReLU are applied to the
block while it is still in that core's cache. PyTorch's eager `Linear` is
`addmm` — MKL's threaded `sgemm` over the whole matrix — followed by a separate
`torch.nn.ReLU()` pass, which the driver builds out of place (`inplace` left at
its default), so each hidden activation is read once and a fresh 16 MiB tensor
written. The PyTorch-side profile confirms the count (`torch.profiler`, CPU
activities, on the driver's own model and step, pinned to CPUs 0–15 with the
harness's `GOMP_SPINCOUNT`, 100 batches after 20 warm; per batch of 4,096):
`aten::addmm` 18.57 ms, 87.1% of self CPU time, three calls; `aten::copy_`
1.21 ms, 5.7%, three calls — `addmm` writing the broadcast bias into the
output before the GEMM, a full write pass per layer; `aten::clamp_min` 1.45
ms, 6.8%, two calls — the out-of-place ReLU, a read and a write pass per
hidden layer; everything else under 0.2%. So each hidden activation is
written by the bias copy, overwritten by the GEMM and read and rewritten by
the ReLU — three passes where OpenNN's blocked path makes one, in cache —
and those passes are 12.5% of PyTorch's batch. The profiler's per-batch
total, 21.3 ms, sits between the two engines' published batches (18.6 and
24.0 ms), as a profiled run does.

The variant shows what the pool choice is worth: with the same MKL kernels
running the row blocks on Eigen's thread pool (`OPENNN_GEMM_MODE=contract`),
OpenNN falls to 118,188 samples/s (`20260902T044422Z`), 0.54× of itself,
because the last layer's `sgemv` is an OpenMP region and libgomp's workers
spin for 300,000 iterations after it — on the same logical CPUs the Eigen pool
is trying to use for the next batch's GEMM. Running the GEMM on the OpenMP
pool makes the spinners and the workers the same threads. This was a loss
(0.70×) until it was found, and it is the reason the runner pins the OpenMP
wait policy for both engines (PROTOCOL §6).

### `cpu-dense-train`, 1.286×: the same GEMM, three times

Training is the forward GEMMs plus two more products of the same shape per
layer in the backward pass — 26.3 GFLOP per batch, 48.9 ms at peak — and the
Adam update over 1.08 M parameters, which is memory traffic and small. OpenNN
takes 58.4 ms per batch (84% of peak), PyTorch 73.4 ms (67%). The profile
(`cpu-dense-train-prof-20260906T063112Z.json`, commit `93cc90e07`, 69,576
samples/s, 0.8% below the published median) covers the forward sections only —
`fwd:Dense` and below, no backward, no bias gradient, no Adam — which is
18.5 ms of the 58.9 ms step, about a third of it. What it establishes is that
the forward call costs in training what it costs in inference:
`cpu:sgemm_wide` at 17.684 ms a call, 95.6% of `cpu:linear_fwd`, against
17.877 ms there. The backward and the update are unprofiled, and the shape of
the backward work below is read from `try_linear_backward` in source rather
than measured.

The same pool and the same blocking as inference: OpenNN's backward GEMMs are
single-threaded MKL calls on the OpenMP pool (`try_linear_backward`,
`tensor_operations.cpp`), the input delta split into row blocks of the batch
and the weight gradient tiled over *both* of its output axes — the batch is
that product's reduction axis, so it is the one axis that cannot be split —
and the bias gradient is a blocked column sum over the same delta with
per-block partials. The epilogue is where the similarity to inference stops:
there is no fused ReLU derivative on this path. `try_linear_backward` computes
the two GEMMs and nothing else, and the DReLU epilogue that fuses the ReLU
backward into the input-delta GEMM is CUDA-only — `linear_backward` refuses a
mask on a host tensor and reports `fused_input_relu` false — so OpenNN's ReLU
backward is its own pass over the activation, exactly as PyTorch's is.
PyTorch's backward GEMMs are threaded `sgemm` calls with the elementwise work
— the ReLU backward, the bias reduction, the Adam step (`torch.optim.Adam` in
the default *foreach* implementation we did not override) — as separate
passes. The PyTorch-side profile (same method as for inference, 100 steps
after 20 warm, per batch of 4,096, 61.9 ms of self CPU time): `aten::mm`
36.71 ms, 59.3%, five calls — the backward data and weight GEMMs — and
`aten::addmm` 18.50 ms, 29.9%, the three forward ones, so the GEMMs are 89.2%
of the step; `aten::threshold_backward` 1.91 ms, 3.1%, the ReLU backward as
two separate passes; `aten::clamp_min` 0.97 ms; `aten::sum` 0.56 ms for the
bias gradients; the Adam step 0.52 ms plus its *foreach* element-wise
kernels (`addcdiv_`, `lerp_`, `mul_`, `div`, `sqrt`, `add_`, `addcmul_`)
1.22 ms, together 2.8%; 29 `copy_` calls 0.62 ms. Nothing outside the GEMMs
is large on either side; the margin is the GEMMs' threading, below. The
contract variant costs OpenNN 20% here (56,308, `20260902T044716Z`) rather
than 46%, because the backward GEMMs on Eigen's pool are longer than the spin
they collide with.

## Asymmetries and caveats

- **PyTorch's inference number depends on the state of a cache the protocol
  does not control.** `max-autotune` writes its autotuning results to disk.
  Between `2026-09-03-publish` and `2026-09-05-publish` the same command
  produced a PyTorch 4.0% faster on `cuda-dense-infer` with nothing changed on
  its side by us; a cleared cache and a re-tune is the likely cause and the
  store records nothing that confirms it. Nothing in the runner, the gates or
  the artifact records which draw of the autotuner a `cuda-dense-infer` number
  was taken against. Our own previously published 1.004× on that cell was
  measured against the slower draw, and this document's 1.019× is measured
  against the faster one — but the next re-tune could move it again in either
  direction, and no re-run of ours would detect that it had.
- **The two engines do not share a cuDNN**, and `cuda-dense-infer`'s
  throughput margin now depends on one. OpenNN links the system cuDNN 9.25.1;
  PyTorch loads the 9.23.2 its wheel bundles (PROTOCOL §1). The engine OpenNN
  selects was enumerated from 9.25.1 and nothing here shows it exists, or is
  as fast, in 9.23.2. Unmeasured, and not correctable without a source build.
- **`cuda-dense-train` reads 1.13× to 1.26×.** The four PyTorch draws behind
  that band are at three different commits; one failed the quiet gate, two are
  dirty trees, and the upper end comes from a dirty-tree draw (the section
  above lists them). The direction survives the spread — OpenNN's slowest
  published launch, 11,395,417, is above the fastest PyTorch figure in any of
  the draws, 10,163,656 — but the *size* of the margin is not a stable
  measurement.
- **`cuda-dense-train`'s energy window is not its throughput window.** The
  engine-reported throughput implies 2.157 s for OpenNN and 2.435 s for
  PyTorch, against timed windows of 2.163 s and 2.527 s — a 0.3% discrepancy
  on one side and 3.8% on the other. The energy figure therefore covers a
  little more than the timed samples, and more of it for PyTorch. What the
  extra 92 ms contains was not established, so the watts above are quoted as
  per-launch ranges and not attributed to the timed samples.
- **Batch selection is not identical.** PyTorch walks the resident split in
  contiguous slices with no shuffling; OpenNN reshuffles the training indices
  every epoch and gathers each batch by index on the device. That is extra
  work on OpenNN's side of the training cells, left in because it is what
  OpenNN does when a user trains a network. The inference cells are symmetric:
  both engines run a fixed order.
- **The loss is written differently.** OpenNN evaluates the sigmoid as an
  output layer and binary cross-entropy on probabilities; PyTorch folds the
  sigmoid into `BCEWithLogitsLoss`. Same function, different rounding; the
  accuracy gate is the check that it does not matter.
- **The CPU memory ratios are process footprints, not workloads.** The
  baseline the drivers print is total RSS, not commensurable with the
  `RssAnon` the metric uses, so it is not subtracted and each CPU launch
  records a `workload_note` saying why — and those baselines, OpenNN 208.5 MiB
  against PyTorch 758.7 and 829.8, are already larger than the peak
  differences reported as the win. *Where the memory goes* above has the rest.
- **The CPU cells run under `GOMP_SPINCOUNT=300000`, set by the runner for
  both engines.** PyTorch's wheel bundles a libgomp that spins that long by
  default; the system libgomp OpenNN links (GCC 14) spins once on hybrid CPUs,
  so without the variable the two engines would be measured under different
  OpenMP wait policies — PROTOCOL §6 has the argument. No A/B for it exists on
  this family at this commit: the artifacts recorded without the variable are
  at earlier commits, before the GEMM-mode fix, and are not comparable. It
  mattered for the LSTM.
- **Different MKL builds.** OpenNN links MKL 2026.0.1; PyTorch's wheel bundles
  its own (PROTOCOL §1). Both print `blas=mkl`; the
  version is not something the runner can equalise, and the CPU cells are
  95%+ one MKL `sgemm` call, so this is the asymmetry that matters most on
  those two cells.
- **The CPU thread count is neither equalised nor recorded.** `run.py` sets
  `OMP_NUM_THREADS` and `torch.set_num_threads` only when `--threads` is
  passed, and it was not: every published CPU launch records
  `threads: "engine default"` and nothing records the count either engine
  chose. OpenNN's OpenMP pool takes the sixteen CPUs in the affinity mask;
  PyTorch does its own detection on a 28-CPU part while confined to those
  sixteen. The runner's own note says forcing a count inverted the result on
  this family (OpenNN 96,204 unset against 84,057 at eight threads, PyTorch
  93,417 against 89,378), which is why it is left free — and why an
  oversubscribed PyTorch would inflate both CPU margins by an unmeasured
  amount. `--threads 8` is the controlled variant.
- **Two PyTorch defaults were left where the constructor put them.** The
  driver builds `torch.nn.ReLU()` out of place rather than `inplace=True`, and
  `torch.optim.Adam(model.parameters())` without `fused=True` on either
  device, while OpenNN runs a fused Adam kernel inside its captured graph.
  Both are one word we did not write, both cost PyTorch time and — for the
  activation — anonymous peak memory, and neither effect is measured.
- **The timed inference is not the same arithmetic on both sides.** OpenNN's
  network ends in a Sigmoid `Dense` layer and computes it inside the timed
  pass; PyTorch's `Sequential` ends at the bare linear output and the driver
  applies the sigmoid only in `evaluate()`, outside the cell. Both inference
  cells therefore charge OpenNN one elementwise pass PyTorch never runs. It is
  small, and it is in OpenNN's disfavour.
- **Only training has an accuracy gate.** Inference computes no metric; its
  correctness rests on the identical parameter count, the identical input
  file, and the training gate passing for the same network definition.
- **An earlier version of this table timed OpenNN inference over zeros.**
  Until commit `4338506c8` the OpenNN inference drivers filled each batch
  (`Batch::fill()`) and never uploaded it: `fill()` only stages the rows on
  the host, and in the library the transfer is issued by the optimizer, which
  an inference driver does not run. The forward pass ran over all-zero inputs
  for every published inference cell before that commit. Nothing in the gates
  could see it — the parameter count, the sample count and the input file are
  the same whatever the batch holds, and the arithmetic of a forward pass does
  not depend on the values — and it was found by printing outputs (every one
  read `sigmoid(0) = 0.5`). For this network the zeros change no kernel and no
  memory traffic. The rows above are from the fixed drivers, which upload the
  batch (`upload_to_device_batch_async()`) after filling it and, where the
  split is resident, take the batch as a device view.

## Reproduce

```bash
export OPENNN_BENCH_SESSION=$(date +%F)-mine
python run.py --family dense --mode train --device cuda --batch 8192 --precision bf16 --epochs 100 --rounds 3
python run.py --family dense --mode infer --device cuda --batch 8192 --precision bf16 --repeats 200 --rounds 3
python run.py --family dense --mode train --device cpu  --batch 4096 --precision fp32 --epochs 3 --rounds 3
python run.py --family dense --mode infer --device cpu  --batch 4096 --precision fp32 --repeats 5 --rounds 3
```

The variants quoted above are the same commands with `--engines opennn` or
`--engines pytorch` and one variable in the environment: `OPENNN_CUDNN_MATMUL=0`,
`OPENNN_MATMUL_CROSS_SOURCE_GAIN=100`, `OPENNN_LT_TILE_TOLERANCE=0`,
`OPENNN_MATMUL_CROSS_SOURCE_ANCHOR_FASTEST=1`, `OPENNN_GEMM_MODE=contract`,
`PT_COMPILE_MODE=default|reduce-overhead|max-autotune|max-autotune-no-cudagraphs|eager`,
`PT_INFER_CAST=autocast`, `OPENNN_NO_CUDA_GRAPH=1`, `OPENNN_SMALL_K_LINEAR=0`
or `OPENNN_DENSE_INFER_GATHER=1`; `--threads 8` is the thread-count variant.
The CPU profile tables are the same commands with `--engines opennn --rounds 1`
and `OPENNN_PROFILE` set. `OPENNN_CUDNN_MATMUL_VERBOSE=1` prints which engine
the plan chose and what it beat, on any shape, which is what would settle
whether cuDNN serves the training GEMMs. The standalone kernel probe is not
part of `run.py` and its source is not in the tree. `prepare.py dense` builds
the HIGGS split the first time.
