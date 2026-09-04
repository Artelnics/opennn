# Dense: HIGGS classification

OpenNN against PyTorch 2.13 on the UCI HIGGS binary classifier, a
28 → 1,024 → 1,024 → 1 multilayer perceptron with 1,080,321 parameters,
trained with Adam and run for inference, on the GPU at batch 8,192 in bf16 and
on 16 CPU threads at batch 4,096 in fp32. Session `2026-09-03-publish`:

| cell | OpenNN | PyTorch | OpenNN / PyTorch |
|---|---|---|---|
| `cuda-dense-train` | 10,768,761 samples/s | 10,115,756 | **1.065×** |
| `cuda-dense-infer` | 37,317,959 samples/s | 37,178,529 | **1.004×** |
| `cpu-dense-train` | 70,025 samples/s | 55,177 | **1.27×** |
| `cpu-dense-infer` | 220,962 samples/s | 170,916 | **1.29×** |

OpenNN wins all four cells, and the *Why* section argues each margin from a
measured ceiling. On the GPU at batch 8,192 both engines hand the
1,024 × 1,024 layer to a tensor-core GEMM running at 96% of the card's
measured peak, so the inference cell (1.004×) is decided by the two
small layers around it — OpenNN's own tensor-core kernel for the 28-wide
first layer and its single-kernel output layer against Inductor's — and the
training cell (1.065×) by how the step is issued: one captured CUDA
graph against Inductor's cudagraph-tree replay. On the CPU the same MKL
kernels run on both sides, and OpenNN's row-blocked layers — which apply the
bias and the activation to each block while it is still in
that core's cache, on the same thread pool as the GEMM — reach
89% of the eight P-cores' fp32 peak at inference against
PyTorch's 68%, and 84% against 66% in
training.

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

**The cells.**

| cell | device | batch | precision | timed |
|---|---|---|---|---|
| `cuda-dense-train` | RTX 5070 Ti | 8,192 | bf16 autocast (TF32 for fp32 GEMMs) | 3 epochs after 2 untimed |
| `cuda-dense-infer` | RTX 5070 Ti | 8,192 | bf16 | 5 passes after 2 untimed |
| `cpu-dense-train` | 16 threads, P-cores | 4,096 | fp32 | 3 epochs after 2 untimed |
| `cpu-dense-infer` | 16 threads, P-cores | 4,096 | fp32 | 5 passes after 2 untimed |

**Each engine at its best.** On CUDA both engines keep the whole split
resident on the device. PyTorch takes each batch as a contiguous slice of the
resident tensors (`range(0, n - batch + 1, batch)`, no shuffling); OpenNN
reshuffles the training indices every epoch (the optimizer's default) and
gathers each batch by index on the device, which is strictly more work per
batch — it is the engine's normal training path, left as is. OpenNN captures the
entire Adam step — forward, loss, backward, update — into one CUDA graph and
replays it per batch. For inference OpenNN slices the resident test split as
device views and launches the three layers directly, with no graph: the
graphed path (`OPENNN_DENSE_INFER_GATHER=1`, a gather kernel per batch and
graph replay) measures the same within noise, because three launches per
0.2 ms batch are already inside what one host thread queues ahead of the GPU.
PyTorch's mode is chosen per cell, and each was measured (the driver's
`compiled()` docstring has every mode): training runs
`torch.compile(mode="reduce-overhead")` — Inductor's fused Triton kernels plus
CUDA graphs, the closest analogue to what OpenNN does, and 1.4× faster than
Inductor without graphs on a step that is launch-bound; inference runs
`max-autotune-no-cudagraphs`, whose autotuned Triton GEMM beats cuBLAS on the
28-wide first layer and which is 1.33× faster than `reduce-overhead` here,
where cudagraph-tree replay copies and bookkeeping cost more than three
launches. Inference also stores the weights in bf16 once (`PT_INFER_CAST=weights`,
the default) instead of re-casting them under autocast on every call, the
way OpenNN keeps a bf16 mirror of its parameters; that is worth 3% on this
cell and more on the larger networks. On CPU both engines run 16 threads
pinned to the P-cores with MKL as the BLAS (`blas=mkl` is printed by both and
recorded per launch); PyTorch runs eager because Inductor's CPU code
generation loses on a small stack of GEMMs (eager 93,156 against compiled
83,722 samples/s when measured).

**The gates.** Parameter counts must agree (they do: 1,080,321). This is the
one family with a quality gate: each training launch prints its test-set
accuracy, and per batch every launch must sit within 2% of the mean across
engines. The published `cpu-dense-train` launches read 0.723635 / 0.727 and
`cuda-dense-train` 0.706933 / 0.715 / 0.716; inference has no gate (it computes no
metric) beyond the identical parameter count and input file.

## Results

Session `2026-09-03-publish`, the last run of every cell, median of three rounds.

| cell | batch | precision | OpenNN samples/s | PyTorch samples/s | OpenNN / PyTorch | peak memory MiB (OpenNN / PyTorch) | energy Wh (OpenNN / PyTorch) |
|---|---|---|---|---|---|---|---|
| `cuda-dense-train` | 8,192 | bf16 | 10,768,761 | 10,115,756 | **1.065×** | 492 / 632 | 0.1201 / 0.1518 |
| `cuda-dense-infer` | 8,192 | bf16 | 37,317,959 | 37,178,529 | **1.004×** | 368 / 412 | 0.1389 / 0.1860 |
| `cpu-dense-train` | 4,096 | fp32 | 70,025 | 55,177 | **1.27×** | 339 / 787 | 0.0871 / 0.0969 |
| `cpu-dense-infer` | 4,096 | fp32 | 220,962 | 170,916 | **1.29×** | 314 / 575 | 0.0984 / 0.1066 |


`cuda-dense-train` — batch 8192, bf16, epochs 100 per launch, 3 rounds. Artifact `cuda-dense-train-publish-20260903T100431Z.json`, commit `6b7179dde`, quiet True (busy 0.2% before, 0.1% after), clocks locked True, shape gate True, quality gate True.

| engine | median samples/s | min | max | peak device MiB | Wh (board) |
|---|---|---|---|---|---|
| OpenNN | 10,768,761 | 10,765,337 | 10,770,038 | 492 | 0.12009 |
| PyTorch | 10,115,756 | 8,826,381 | 10,219,469 | 632 | 0.15183 |
| **ratio** | **1.065×** | | | 1.28× less | 1.26× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 10,770,038 | 8,826,381 |
| 2 | pytorch → opennn | 10,768,761 | 10,219,469 |
| 3 | opennn → pytorch | 10,765,337 | 10,115,756 |


`cuda-dense-infer` — batch 8192, bf16, passes 200 per launch, 3 rounds. Artifact `cuda-dense-infer-publish-20260903T100312Z.json`, commit `6b7179dde`, quiet True (busy 0.5% before, 0.4% after), clocks locked True, shape gate True, quality gate True.

| engine | median samples/s | min | max | peak device MiB | Wh (board) |
|---|---|---|---|---|---|
| OpenNN | 37,317,959 | 37,316,956 | 37,334,401 | 368 | 0.13888 |
| PyTorch | 37,178,529 | 37,146,520 | 37,188,163 | 412 | 0.18599 |
| **ratio** | **1.004×** | | | 1.12× less | 1.34× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 37,334,401 | 37,188,163 |
| 2 | pytorch → opennn | 37,317,959 | 37,178,529 |
| 3 | opennn → pytorch | 37,316,956 | 37,146,520 |


`cpu-dense-train` — batch 4096, fp32, epochs 3 per launch, 3 rounds. Artifact `cpu-dense-train-publish-20260903T104301Z.json`, commit `6b7179dde`, quiet True (busy 0.4% before, 0.1% after), clocks locked True, shape gate True, quality gate True.

| engine | median samples/s | min | max | peak RssAnon MiB | Wh (RAPL package-0) |
|---|---|---|---|---|---|
| OpenNN | 70,025 | 70,021 | 70,032 | 339 | 0.08708 |
| PyTorch | 55,177 | 54,538 | 55,911 | 787 | 0.09688 |
| **ratio** | **1.269×** | | | 2.32× less | 1.11× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 70,021 | 55,177 |
| 2 | pytorch → opennn | 70,025 | 54,538 |
| 3 | opennn → pytorch | 70,032 | 55,911 |


`cpu-dense-infer` — batch 4096, fp32, passes 5 per launch, 3 rounds. Artifact `cpu-dense-infer-publish-20260903T103938Z.json`, commit `6b7179dde`, quiet True (busy 0.1% before, 0.2% after), clocks locked True, shape gate True, quality gate True.

| engine | median samples/s | min | max | peak RssAnon MiB | Wh (RAPL package-0) |
|---|---|---|---|---|---|
| OpenNN | 220,962 | 220,694 | 221,010 | 314 | 0.09842 |
| PyTorch | 170,916 | 169,752 | 170,961 | 575 | 0.10658 |
| **ratio** | **1.293×** | | | 1.83× less | 1.08× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 220,694 | 170,961 |
| 2 | pytorch → opennn | 220,962 | 169,752 |
| 3 | opennn → pytorch | 221,010 | 170,916 |

## Why

The four cells have four different explanations, and only one of them is "a
GEMM is faster" — and that one is a layer too narrow to be a GEMM at all.
Everywhere else both engines hand the matrix products to the same vendor
libraries (cuBLASLt on the GPU, MKL on the CPU), and the margins come from
what happens *around* the products: how many passes over the activations a
layer costs, how much host work each batch carries, and which thread pool the
passes run on.

### Where the energy goes

Energy is power times time, and this family contains one cell of each kind.

`cuda-dense-infer` is the cell where OpenNN wins on **power**: 186.5 W against
248.5 W for the same 100.1 M samples, with both engines holding the card at
99% occupancy (99.6% against 98.7% of the window in kernels). When neither
engine is waiting for the host, the watts are decided by what the kernels
move rather than by how busy they are, which is the subject of the next
section.

`cuda-dense-train` wins on both at once, and its profile shows why. OpenNN
holds the GPU busy for 98.3% of its window and issues 63,101 kernels;
PyTorch manages 73.1% and issues 104,987 — it is partly launch-starved, and
its kernels also draw more: 189.0 W against 211.6 W. That is 1.065× on time
and 1.120× on power, which compounds to 1.26× on energy. Some
of the power difference is the tile-selection rule described below: with
`OPENNN_LT_TILE_TOLERANCE=0` this cell reads 0.13425 Wh, against
0.12009 Wh with the rule on — 11.7% of the cell's energy, for 1.7% of
its throughput.

The two CPU cells win on **time while drawing more power**, and the honest
statement is that OpenNN is not the more frugal engine here. Over the same
window OpenNN pulls 31.3 W of package power against PyTorch's 26.2 W on
inference, and 29.3 W against 25.7 W on training — about 19% and 14% more.
It wins 1.083× and 1.11× anyway because it
finishes 1.29× and 1.27× sooner. The extra watts are the point,
not a defect: the RAPL counter is measuring sixteen cores held closer to
their sgemm peak (89% against 68%), and a core at
higher occupancy costs more per second and less per sample.

### `cuda-dense-infer`, 1.004× throughput and 1.34× energy: one GEMM, two ways to pay for it

This is the narrowest throughput margin in the matrix and the widest energy
one, and both facts have the same cause. At batch 8,192 the network is one
large matrix product plus two small layers; both engines hand that product to
a tensor-core kernel running within a few percent of what the card can do, so
there is almost nothing left to win on time. What is left to win — and it is
a third of the cell's energy — is *which* of the card's equally fast kernels
runs it.

**The time ceiling.** Measured on this card with the clock locked: a bf16
8,192³ matmul sustains 94.0 TFLOPS. The 8,192 × 1,024 × 1,024 hidden layer is
17.2 GFLOP, so it cannot take less than 183 µs. The first layer, 28 → 1,024,
is 0.47 GFLOP — a fraction of a microsecond of arithmetic — but it writes a
16.8 MB bf16 activation, and a `memset` of that many bytes takes 10.2 µs (the
write is absorbed by the 48 MB L2 at 1.65 TB/s). The last layer, 1,024 → 1,
reads the activation back once: 6 µs at L2 bandwidth. So a batch cannot take
less than about 201 µs, 40.7 M samples/s, and every number in this cell has
to be read against that.

**Per-kernel, one batch of 8,192** (`nsys`, direct launches of the published
commands, kernel durations in µs):

| layer | OpenNN, published | PyTorch, `max-autotune-no-cudagraphs` |
|---|---|---|
| 28 → 1,024, bias + ReLU | `small_k_linear_kernel<bf16,relu>` 12.8 | Triton autotuned GEMM + fused epilogue 16.5 |
| 1,024 → 1,024, bias + ReLU | `nvjet` bf16 GEMM, 256×160 tile, 200.4 | Triton autotuned GEMM + fused epilogue 190.1 |
| 1,024 → 1, bias + sigmoid | `linear_forward_single_output_kernel` 6.9 | Triton GEMV 12.9 + sigmoid 0.5 |
| kernels per batch | 3, 219.2 | 4, 220.5 |

| | OpenNN | PyTorch |
|---|---|---|
| published / under `nsys` (samples/s) | 37,334,401 / 37370202 | 37,188,163 / 37202329 |
| timed window traced | 2.708 s | 2.716 s |
| kernel launches | 36,599 (13,515/s) | 48,799 (17,967/s) |
| GPU busy with kernels | 99.6% | 98.7% |
| GPU busy with any work (kernels, copies, memsets) | 99.6% | 98.7% |
| gaps between kernels, median / p90 | 0.2 / 0.3 µs | 0.3 / 0.3 µs |
| idle between kernels, total | 11.6 ms | 34.3 ms |

OpenNN, top kernels by summed time in the window:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 91.1% | 12,200 | 201.3 | `nvjet_sm120_tst_mma_256x160x32_3_64x80x32_tmaAB_alignCD4_bx_biast_r...` |
| 6.0% | 12,199 | 13.2 | `<unnamed>::small_k_linear_kernel<__nv_bfloat16, 1>(int, int, int, i...` |
| 2.9% | 12,200 | 6.5 | `linear_forward_single_output_kernel<__nv_bfloat16>(int, int, const ...` |

PyTorch, top kernels by summed time in the window:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 86.3% | 12,200 | 189.8 | `triton_tem_fused_addmm_relu_t_1` |
| 7.5% | 12,199 | 16.6 | `triton_tem_fused_addmm_relu_t_0` |
| 5.9% | 12,200 | 12.9 | `triton_tem_fused_addmm_relu_t_3` |
| 0.2% | 12,200 | 0.5 | `triton_poi_fused_addmm_relu_t_2` |

OpenNN wins the two small layers by 10.4 µs and loses the big one by 10.3 µs.
That is the whole cell: 1.004×.

*The first layer is a write, not a GEMM.* With a contraction of 28, cuBLASLt
has no aligned kernel to pick — 28 bf16 are 56 bytes, so it falls to an
`align2` variant that loads the operands two bytes at a time and spends
22 µs producing what a memset produces in 10. Inductor does better (16.5 µs)
because `max-autotune` benchmarks its own Triton templates against cuBLAS for
every shape and keeps the winner; that is why the published PyTorch mode is
`max-autotune-no-cudagraphs` and not `reduce-overhead` (37.1 M against
27.9 M samples/s). OpenNN runs a contraction of at most 32 through its own
kernel (`opennn/core/cuda/kernel_small_k_linear.cu`): `mma.sync` bf16
tensor-core fragments loaded straight from global memory, the 28 × 1,024
weights staged once through shared memory into fragments that stay in
registers for the block's lifetime, bias and ReLU in the epilogue, and the
bf16 output staged through shared memory so that every store writes four full
128-byte rows. It takes 12.8 µs — 80% of the memset floor's bandwidth — and
`OPENNN_SMALL_K_LINEAR=0` puts cuBLASLt back for the controlled comparison
(*[pending the final measurement round]* against 37,317,959 samples/s).

*The last layer is one kernel, not two.* OpenNN's single-output path
(`linear_forward_single_output_kernel`, one warp per row, the sigmoid fused)
reads the activation once at 6.9 µs; Inductor lowers the 1,024 → 1 product to
a GEMV and the sigmoid to a separate kernel, 13.4 µs together.

*The middle layer is 90% of the batch, and it is where the energy is.*

#### The kernel that is fast and the kernel that is cheap

Both engines are near the arithmetic ceiling on the hidden layer, so it is
natural to assume there is nothing to choose between their kernels. There is:
they draw different amounts of power for the same arithmetic. Measured
directly, back-to-back launches of that exact GEMM (m 1,024, n 8,192, k 1,024,
NN, bf16 in and out, fp32 accumulate, `CUBLASLT_EPILOGUE_RELU_BIAS`), with
board power integrated from the driver's own 20 ms sample ring:

| kernel | time | power | energy per GEMM |
|---|---|---|---|
| cuBLASLt 64×64, the fastest configuration that exists | 192.4 µs | 265 W | 51.0 mJ |
| PyTorch's Triton kernel, 64×128 | 190.1 µs | 227 W | 43.6 mJ |
| PyTorch eager, cuBLAS's own pick | 226.0 µs | 192 W | 43.8 mJ |
| cuBLASLt 256×160, the cheapest configuration that exists | 200.9 µs | 169 W | 33.9 mJ |

The spread is 96 W across kernels doing identical work at the same locked
clock. It is not arithmetic — it is data movement. A tile of `rows × columns`
reads one row-strip of A and one column-strip of B per output tile, so it
moves `1/rows + 1/columns` bytes through L2 and shared memory per
multiply-add, and board power follows that ratio almost exactly:

| tile | 1/rows + 1/columns | watts |
|---|---|---|
| 64×64 | 0.0312 | 265 |
| 64×128 (Triton's) | 0.0234 | 227 |
| 64×640 | 0.0172 | 189 |
| 80×512 | 0.0145 | 182 |
| 128×240 | 0.0120 | 179 |
| 128×320 | 0.0109 | 172 |
| 256×160 | 0.0102 | 169 |

Six of those seven points are cuBLASLt configurations measured here; the
seventh is PyTorch's Triton kernel, which the same relation predicts to
within 3 W without being fitted to it.

**Why OpenNN was on the wrong one, and why the fix is not a lookup table.**
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
the 265 W kernel. Worse, of those eight only the 64×64 keeps the cell ahead of
PyTorch at all — every other one puts the batch above 220.5 µs. The efficient
kernel is not on the list: **256×160 is never offered by the heuristic.**

So the search was widened. Enumerating every configuration cuBLASLt will
accept for this problem — 20 algorithm ids, each advertised tile, stage and
custom option, validated with `cublasLtMatmulAlgoCheck` — gives 13,460 valid
configurations, enumerated in 0.86 s at 1.8 µs a check. Timing all of them
settles the shape of the trade:

| faster than | configurations |
|---|---|
| 192 µs | 1 |
| 198 µs | 1 |
| 199 µs | 6 |
| 200 µs | 23 |
| 205 µs | 79 |

There is exactly one configuration below 198 µs and it is the power-hungry
one. The frontier has two useful points and nothing between them, and three
further attempts to find a middle all failed:

- **The TN layout,** normally the fastest orientation for tensor cores and a
  load-time change rather than a per-batch cost, is no better here: its best
  is 193.1 µs at 271 W, and its own 256×160 lands at 200.4 µs and 172 W.
- **Splitting the batch** between the two kernels — part of the columns on
  64×64, the rest on 256×160 — should in principle buy any point on the line
  between them. It buys none: every split measured slower than either kernel
  run alone (best split 203.5 µs against 200.1 µs pure), because halving `n`
  costs the wide-tile kernel more in tail waves than the mix saves.
- **A different stage count for the winning tile** does not exist: 64×64 is
  fast only at 12 stages; at 16, 18 and 25 stages it takes 410, 247 and
  240 µs.

#### What OpenNN does now

`autotune_lt_plan` still times candidates and still knows which is fastest,
but the plan no longer only sees the heuristic's shortlist. It adds the tiles
of `lt_known_tiles` as candidates of its own — lowest-traffic first, over
every stage and custom option each algorithm advertises, each validated by
`cublasLtMatmulAlgoCheck` before it is ever launched — and then, among the
candidates that came within `OPENNN_LT_TILE_TOLERANCE` percent of the fastest
(default 5), it takes the one that moves the least data.

The knob is the experiment. Three rounds each, same commit, same machine, the
only difference the selection rule:

| | throughput | energy | mean board power |
|---|---|---|---|
| `OPENNN_LT_TILE_TOLERANCE=0`, by time alone | 38,787,604/s, 1.044× | 0.20245 Wh, **0.919×** | 273 W |
| default, by data movement | 37,381,192/s, 1.006× | 0.13855 Wh, **1.338×** | 186 W |

Read that as a choice, not a discovery: 4% of the GEMM's time buys a third of
its energy. The protocol requires one published configuration per engine, and
the default is the one that wins every axis rather than the one that wins two
and loses the third. Before this rule, `cuda-dense-infer` energy was the only
number in the whole matrix where PyTorch beat OpenNN.

For the same 100.1 M samples the two engines now read:

| | window | energy | mean power | per sample |
|---|---|---|---|---|
| OpenNN | 2.683 s | 0.1390 Wh | 186.5 W | 5.0 nJ |
| PyTorch | 2.694 s | 0.1859 Wh | 248.4 W | 6.7 nJ |

The work is equal by construction — `--repeats 200` over the same HIGGS test
set, 1,080,321 parameters on both sides and the shape gate agreeing — so the
undivided watt-hours are already energy per fixed workload. The win is almost
entirely in power, which is why it costs so little time.

#### What is left on the table

The honest reading of the table above is that **NVIDIA's library kernels are
Pareto-dominated here by a generated one.** Triton's 64×128 kernel does
190.1 µs at 227 W; cuBLASLt's own 64×128 takes 214.4 µs. Same tile shape,
12% apart — so the 199–200 µs floor on the low-traffic cuBLASLt kernels is a
property of those implementations, not of the bandwidth they need. A
hand-written kernel at a low-traffic tile and Triton's speed — 128×256, which
the relation puts near 175 W, at 190 µs — would make this cell 1.055× on
throughput *and* about 1.32× on energy, taking both axes outright instead of
trading between them. That kernel does not exist in OpenNN yet; the small-K
layer above is the proof that the approach works, and it is the obvious next
piece of work on this cell.

**Why the margin is small at 8,192 and large below it.** The published batch
is the one that saturates the GEMM, so it is the batch at which the two
engines have the least room to differ. Running each batch size in its own
process (the runner does this; an in-process sweep under-reads the large
batches — see the caveats):

*[pending the final measurement round]*

At 1,024 the batch is 27 µs of GEMM and the rest is issue: OpenNN's three
launches from C++ against cudagraph-free Inductor's four launches and their
Python guards. The published cell is the fair one — it is the batch size the
protocol fixed before any of this was measured — but a reader deploying this
network at a smaller batch is looking at the left of that table, not the
right.

CUDA graphs are worth nothing to OpenNN on this cell — the published launch
runs the resident split as views with the graph off, and the graphed path
(`OPENNN_DENSE_INFER_GATHER=1`, a gather kernel per batch and graph replay)
reads the same within noise — because three launches per 0.22 ms batch are
well inside what one host thread queues ahead of the GPU. They are worth less
than nothing to PyTorch here: `reduce-overhead` costs it 25% against
`max-autotune-no-cudagraphs`, because cudagraph-tree replay copies each input
slice into its static placeholder and runs its bookkeeping in Python before
every replay.

### `cuda-dense-train`, 1.065× (1.065× over a long window): a launch-bound step

The Adam step on this network is small — roughly 52 GFLOP of GEMM per batch of
8,192, about 0.56 ms of tensor-core time — and both engines
spend the rest of the batch waiting on the host. That is what the compile
modes say: PyTorch eager (every kernel launched from Python) makes
4,770,607 samples/s, Inductor without graphs 7,031,465, Inductor with CUDA
graphs 9,903,911 (`20260902T045145Z`, `T045015Z`, `T022718Z`); the same work
runs 2.1× faster depending only on how it is *issued*. OpenNN captures the
whole step — gather of the shuffled batch, forward, loss, backward, Adam
update — into one graph; replaying it against launching it from the host is
10,874,476 against 10,616,248 samples/s (`OPENNN_NO_CUDA_GRAPH=1`,
`20260902T045428Z`, the previous session's build), a 2.4% effect, because
OpenNN's host path is already short.

| | OpenNN | PyTorch |
|---|---|---|
| published / under `nsys` (samples/s) | 10,770,038 / 10725143 | 8,826,381 / 7687898 |
| timed window traced | 2.302 s | 3.171 s |
| kernel launches | 63,101 (27,411/s) | 104,987 (33,108/s) |
| GPU busy with kernels | 98.1% | 73.1% |
| GPU busy with any work (kernels, copies, memsets) | 98.3% | 73.1% |
| gaps between kernels, median / p90 | 0.2 / 2.1 µs | 0.2 / 20.4 µs |
| idle between kernels, total | 48.1 ms | 853.4 ms |

OpenNN, top kernels by summed time in the window:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 26.0% | 3,000 | 204.9 | `cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bgrada_bf16_128x128...` |
| 26.0% | 3,000 | 204.3 | `nvjet_sm120_tst_mma_256x160x32_3_64x80x32_tmaAB_alignCD4_bx_biast_r...` |
| 25.7% | 3,000 | 202.3 | `nvjet_sm120_tst_mma_80x192x64_2_80x24x64_tmaAB_alignCD4_bz_TNNN` |
| 4.1% | 3,000 | 32.5 | `gather_rows_kernel<__nv_bfloat16>` |
| 3.8% | 3,000 | 29.7 | `cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_64...` |
| 3.2% | 3,001 | 25.2 | `adam_update_kernel(int, int, float *, float *, float *, const float...` |

PyTorch, top kernels by summed time in the window:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 27.3% | 3,000 | 210.7 | `cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x...` |
| 24.8% | 3,000 | 191.3 | `cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x6...` |
| 24.1% | 2,999 | 186.2 | `cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x6...` |
| 3.5% | 3,000 | 27.2 | `cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x...` |
| 2.9% | 6,000 | 11.2 | `triton_poi_fused_relu_3` |
| 2.4% | 2,999 | 18.5 | `triton_for_fused_1` |

The published window is short: three epochs of 30 batches, 0.07 s of timed
work for the winner, and the round-to-round spread on the PyTorch side is
visible in the launches (8,826,381 / 10,219,469 / 10,115,756, against OpenNN's
10,770,038 / 10,768,761 / 10,765,337). Run the same cell for 30 epochs and the spread
closes: OpenNN 10,896,472 against PyTorch 10,226,830 (`--epochs 30`,
`20260902T045634Z`, both engines in the same run), a 1.065× margin. That is
the number to quote if one number is wanted; the 1.065× in the table is
the protocol's cell, and both windows put OpenNN's slowest launch above
PyTorch's fastest.

### `cpu-dense-infer`, 1.29×: a GEMM at 89% of the cores' peak

On the CPU the arithmetic is fixed and large: 8.83 GFLOP per batch of 4,096,
of which 8.59 are the 1,024 × 1,024 layer. The eight P-cores at their locked
2.1 GHz can retire 538 GFLOP/s of fp32 FMA (two 256-bit units
per core, no AVX-512 on this part; MKL's own 4,096³ `sgemm` reaches
*[pending the final measurement round]* on 16 threads), so the batch cannot take less than
16.4 ms. OpenNN takes 18.5 ms (89% of peak);
PyTorch takes 24.0 ms (68%).

OpenNN's layer is `blocked_linear_forward` (`tensor_operations.cpp`): the rows
of the batch are split into blocks, each block is one single-threaded MKL
`sgemm` on one of the 16 OpenMP workers, and the bias and ReLU are applied to
the block while it is still in that core's cache. PyTorch's eager `Linear` is
`addmm` — MKL's threaded `sgemm` over the whole matrix — followed by `relu_`
as a separate pass, and under `torch.no_grad` each pass is a full read and
write of the 16 MiB activation from memory. *[pending the final measurement round]*

The variant shows what the pool choice is worth: with the same MKL kernels
running the row blocks on Eigen's thread pool (`OPENNN_GEMM_MODE=contract`),
OpenNN falls to 118,188 samples/s (`20260902T044422Z`), 0.54× of itself,
because the last layer's `sgemv` is an OpenMP region and libgomp's workers
spin for 300,000 iterations after it — on the same logical CPUs the Eigen
pool is trying to use for the next batch's GEMM. Running the GEMM on the
OpenMP pool makes the spinners and the workers the same threads. This was a
loss (0.70×) until it was found, and it is the reason the runner pins the
OpenMP wait policy for both engines (PROTOCOL §7).

### `cpu-dense-train`, 1.27×: the same GEMM, three times

Training is the forward GEMMs plus two more per layer in the backward pass —
26.5 GFLOP per batch, 49.3 ms at peak — and the Adam update over
1.08 M parameters, which is memory traffic and small. OpenNN takes
58.5 ms per batch (84% of peak), PyTorch 74.2 ms
(66%). The same pool and the same blocking as inference:
OpenNN's backward GEMMs are single-threaded MKL calls on the OpenMP pool
(`try_linear_backward`, `tensor_operations.cpp`), the input delta split into
row blocks of the batch and the weight gradient tiled over *both* of its
output axes — the batch is that product's reduction axis, so it is the one
axis that cannot be split — and the bias gradient is a blocked column sum
over the same delta with per-block partials. The epilogue is where the
similarity to inference stops: there is no fused ReLU derivative on this
path. `try_linear_backward` computes the two GEMMs and nothing else, and the
DReLU epilogue that fuses the ReLU backward into the input-delta GEMM is
CUDA-only — `linear_backward` refuses a mask on a host tensor and reports
`fused_input_relu` false — so OpenNN's ReLU backward is its own pass over the
activation, exactly as PyTorch's is. PyTorch's backward GEMMs are threaded
`sgemm` calls with the elementwise work — the ReLU backward, the bias
reduction, the Adam step (`torch.optim.Adam` in its default *foreach*
implementation) — as separate passes. *[pending the final measurement round]* The contract variant costs
OpenNN 20% here (56,308, `20260902T044716Z`) rather than 46%, because the
backward GEMMs on Eigen's pool are longer than the spin they collide with.

## Asymmetries and caveats

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
- **`cuda-dense-train` has a short window.** Three epochs at 8,192 are 737,280
  samples, which the winner finishes in about 0.07 s; the energy sampler
  cannot resolve that (its column reads `--` for two of OpenNN's three
  launches) and the round-to-round spread is the widest in the matrix — the
  PyTorch launches read 8,826,381 / 10,219,469 / 10,115,756. The margin survives the
  spread: OpenNN's slowest launch is above PyTorch's fastest. The
  `--epochs 30` variant in the *Why* section is the same cell with a window
  ten times longer.
- **The CPU cells run under `GOMP_SPINCOUNT=300000`, set by the runner for
  both engines.** PyTorch's wheel bundles a libgomp that spins that long by
  default; the system libgomp OpenNN links (GCC 14) spins once on hybrid
  CPUs, so without the variable the two engines would be measured under
  different OpenMP wait policies — PROTOCOL §7 has the argument. On this
  family the variable is worth almost nothing to OpenNN once the GEMM runs
  on the OpenMP pool (inference 221k against 217k samples/s with and without the spin, training 70.0k against 69.2k); it mattered for the LSTM.
- **Different MKL builds.** OpenNN links MKL 2026.0.1; PyTorch's wheel
  bundles its own MKL (`BLAS_INFO=mkl`, oneDNN v3.12.0). Both print `blas=mkl`; the
  version is not something the runner can equalise.
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
  memory traffic, and the cell moved only by what the small-K kernel added.
  The rows above are from the fixed drivers, which upload the batch
  (`upload_to_device_batch_async()`) after filling it and, where the split is
  resident, take the batch as a device view; the previous session's rows read
  36,672,161 samples/s for OpenNN, before the small-K kernel, against a
  PyTorch under `reduce-overhead`.

## Reproduce

```bash
export OPENNN_BENCH_SESSION=$(date +%F)-mine
python run.py --family dense --mode train --device cuda --batch 8192 --precision bf16 --epochs 3 --rounds 3
python run.py --family dense --mode infer --device cuda --batch 8192 --precision bf16 --repeats 5 --rounds 3
python run.py --family dense --mode train --device cpu  --batch 4096 --precision fp32 --epochs 3 --rounds 3
python run.py --family dense --mode infer --device cpu  --batch 4096 --precision fp32 --repeats 5 --rounds 3
```

The variants quoted above are the same commands with `--engines opennn` or
`--engines pytorch` and one variable in the environment: `OPENNN_GEMM_MODE=contract`,
`PT_COMPILE_MODE=default|reduce-overhead|max-autotune|max-autotune-no-cudagraphs|eager`,
`PT_INFER_CAST=autocast`, `OPENNN_NO_CUDA_GRAPH=1`, `OPENNN_SMALL_K_LINEAR=0`,
`OPENNN_DENSE_INFER_GATHER=1`, or `--epochs 30` for the long window. `prepare.py dense` builds the HIGGS split the first time.
