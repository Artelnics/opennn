# LSTM: forecasting Beijing PM2.5

OpenNN against PyTorch 2.13 on a single-layer LSTM of 128 units with a linear
output (73,857 parameters) forecasting the next hour of PM2.5 from 24 hours
of 15 features, 43,800 windows, batch 256 — bf16 on the RTX 5070 Ti, fp32 on
the sixteen P-cores, where both engines run oneDNN's LSTM primitive. Session
`2026-09-05-publish`, commit `93cc90e07`:

| cell | OpenNN | PyTorch | OpenNN / PyTorch |
|---|---|---|---|
| `cuda-lstm-train` | 290,819 windows/s | 104,815 | **2.775×** |
| `cuda-lstm-infer` | 918,696 windows/s | 524,191 | **1.753×** |
| `cpu-lstm-train` | 23,147 windows/s | 13,016 | **1.778×** |
| `cpu-lstm-infer` | 80,394 windows/s | 69,269 | **1.161×** |

`cuda-lstm-train` is the widest margin in the twelve-cell matrix, and one of
the least repeatable cells in it; the caveats give its band, and the
published row is the less favourable of the two publish runs at this commit.

**On the GPU, PyTorch has the better algorithm and still loses.** Both
engines call cuDNN's RNN API and do not get the same kernels. PyTorch's
`nn.LSTM` reaches cuDNN's *persistent* LSTM kernel —
`RNN_blockPersist_fp_LSTM_HMMA`, one launch of 38.12 µs that runs the whole
24-step recurrence with the weights resident in registers, 65.2% of its
inference profile. OpenNN takes cuDNN's `STANDARD` path, which unrolls the
recurrence into 24 `elemWiseRNNcell` launches and 23 recurrent GEMMs per
batch. What selects the path is the precision, and the gate that was
observed is on OpenNN's side: `persist_algo_active` in `cudnn_rnn.cpp:199`
requires, among other conditions, `!bf16 && H <= 128` and, for an LSTM,
`batch < 512`, so a cell declared bf16 never asks cuDNN for the persistent
algorithm at all — OpenNN's own descriptor dump records `algo=STANDARD,
dtype=BFLOAT16`. PyTorch, asked for bf16 autocast, runs the recurrence in
fp16 and gets the persistent kernel. Whether cuDNN would offer one at bf16
was not tested, and the two engines do not link the same cuDNN (9.25.1
against the wheel's 9.23.2; see the caveats).

The persistent kernel does the LSTM layer in 46.4 µs of GPU time per batch
against OpenNN's 87.0 µs, and PyTorch's whole batch costs 58.5 µs of GPU
time against OpenNN's 90.6 µs. OpenNN wins the cell 1.753× anyway, because
at this size neither engine is limited by the GPU: the batch is 279 µs of
wall time for OpenNN and 488 µs for PyTorch around 58–91 µs of kernels, so
what decides it is how fast each host path issues. The margin is an
issue-rate margin, not an arithmetic one, and the better recurrent kernel is
on the other side.

On the CPU both engines run the same oneDNN LSTM primitive, and there the
cell is almost nothing but that primitive: 3.185 ms of a 3.290 ms
instrumented batch, 98.4% of OpenNN's LSTM layer. Inference (1.161×) is the
0.51 ms by which PyTorch's batch is longer, and where that time goes is not
measured; training (1.778×) is 11.06 ms against 19.67 ms, and the 8.61 ms
difference is unattributed too. Both are in the *Why* section as arithmetic,
not as attribution.

## What is measured

**Network.** One LSTM layer of 128 units over a window of 24 hourly rows of 15
features, followed by a linear output of one unit: 4 × (15×128 + 128×128 + 128)
+ (128 + 1) = **73,857 trainable parameters** on both sides. PyTorch's
`nn.LSTM(15, 128, batch_first=True)` owns two bias vectors whose sum is what
the gates see; the driver zeroes and freezes `bias_hh` so the trainable
parameterisation is identical to OpenNN's single bias, and OpenNN's
`set_parameters_pytorch()` draws the initial weights from PyTorch's
distribution rather than merely from the same seed. Training minimises the
mean squared error with Adam at default hyper-parameters, no clipping. On
CUDA, OpenNN pads the cuDNN input width to a multiple of 8 whenever it is
not on the persistent path (`cudnn_input_features`, 15 → 16), which is why
its input-projection GEMM is an `align8` kernel and PyTorch's an `align1`
one, and which costs OpenNN a sixteenth of that GEMM; the artifacts' own
descriptor dump reads `F=16`. The logical feature count the shape gate
compares is 15 on both sides.

Two networks are timed, by design. Inference times exactly `nn.LSTM +
nn.Linear` on both sides (OpenNN: `LongShortTermMemory` + `Dense(Identity)`).
Training uses OpenNN's full `ForecastingLstmNetwork`, which is Scaling → LSTM
→ Dense → Unscaling → Clamping(none): OpenNN standardises the inputs inside
the network and un-standardises the output, while the PyTorch driver
standardises the CSV once, outside the timed window. The three element-wise
layers are part of what OpenNN's training step does per batch and PyTorch's
does not; they favour PyTorch and are left in because that is the network an
OpenNN user trains.

**Data.** UCI Beijing PM2.5, 43,824 hourly rows from 2010 to 2014, prepared
once by `prepare.py lstm`: the wind direction is one-hot encoded (NE, NW, SE,
cv), the gaps in the PM2.5 series are linearly interpolated, and the target
is the last column — 15 columns in all, with the target's own history used as
an input, as forecasting requires. A window of 24 rows predicts the next
reading, which yields **43,800 windows**; both drivers use the complete file
(the chronological 60/20/20 split OpenNN would normally install is
overridden). Whole batches only: at batch 256 an epoch or a pass covers 171
batches = **43,776 samples**.

The two engines do not hold that data the same way, and the difference is
visible in the trace. The PyTorch driver materialises the whole window
tensor on the device once (`torch.from_numpy(windows).to(device)`, 43,800 ×
24 × 15 fp32 = 60 MiB) and slices it there; its training trace contains six
host-to-device copies in total. OpenNN sets `GPUPersistantData` on the
dataset but still assembles and uploads each batch: its training trace shows
one 184,320-byte host-to-device copy (256 × 24 × 15 bf16) and one of 1,024
bytes per batch, 17.8 µs of copy time per 880 µs batch. That
asymmetry costs OpenNN throughput and saves it memory; both effects are
counted in the results below. For inference both engines fill the first batch
of 256 windows once and replay it 171 times per pass — one host-to-device
copy in OpenNN's whole inference trace — so the inference cells time the
resident forward pass alone.

**Cells.**

| cell | device | batch | precision | timed window |
|---|---|---|---|---|
| `cuda-lstm-train` | RTX 5070 Ti | 256 | bf16 | 20 epochs after 2 untimed |
| `cuda-lstm-infer` | RTX 5070 Ti | 256 | bf16 | 50 passes after 1 call + 1 untimed pass |
| `cpu-lstm-train` | CPUs 0–15 (P-cores) | 256 | fp32 | 3 epochs after 2 untimed |
| `cpu-lstm-infer` | CPUs 0–15 (P-cores) | 256 | fp32 | 5 passes after 1 call + 1 untimed pass |

**Each engine at its best.** PyTorch is **eager on CUDA in this family**, and
that is the unusual case. Compiling was measured and lost: session
`2026-09-02-variants`, commit `918805ce1`, `PT_COMPILE_MODE=reduce-overhead`
reads 89,518 samples/s training (`cuda-lstm-train-variant-compile-ro-…`)
and 462,572 inference (`cuda-lstm-infer-variant-compile-ro-…`), against
eager PyTorch measured the same day across four publish runs at 95,821 to
104,590 training and 522,694 to 535,766 inference; the driver's docstring
records the same result from its own sweep (train 87,628 compiled against
108,614 eager, inference 457,209 against 528,734). Dynamo breaks the graph at
`zero_grad` and Inductor cannot fuse anything into the opaque cuDNN RNN call,
so compilation buys a recompile and fuses nothing that matters. Compiled
training also peaks at 582 MiB, against 507–531 MiB for that day's eager
publish runs. That measurement is four days and one reboot older than the
published rows and was not repeated at `93cc90e07` (see the caveats). For
dense, cnn and transformer compiling wins, so the suite takes PyTorch's
better mode per family.

OpenNN asks for a CUDA graph of the Adam step and of the inference pass and
is refused: in both CUDA cells its driver prints that `cudnnRNNForward`
returned cuDNN status 4000 and invalidated the capture, the artifacts record
`cuda_graph=failed`, and OpenNN runs the cell eagerly as well — launch for
launch against PyTorch. OpenNN's own message is explicit that it does not
know *which* forbidden operation cuDNN performed (an allocation, a
synchronisation or an event query), only that its own launches in the
captured region did not throw; no artifact in this session narrows it
further, so this document does not claim a mechanism.

On CPU both engines run oneDNN's LSTM primitive — OpenNN links its own build
and PyTorch's wheel bundles a different one; neither version is recorded in
this session's artifacts — with MKL as the BLAS. Both processes are
`taskset`-pinned to CPUs 0–15, the P-cores, and each chooses its own thread
count: the runner sets none (`pinning.threads = "engine default"` in every
CPU artifact), OpenNN reads the affinity mask and takes 16
(`Backend::set_threads_number`, `device_backend.cpp:1250`), and what
PyTorch's own default resolves to is recorded nowhere (see the caveats).
Both drivers enable flush-to-zero (`flush_denormals=on` is recorded), and
the runner sets `GOMP_SPINCOUNT=300000` for both — PROTOCOL §6 explains why
the GCC 14 libgomp default would otherwise penalise the engine that links
the system runtime.

**Gates.** Samples (43,800 windows), inputs (15), past (24), hidden (128) and
parameters (73,857) must agree between the engines. There is **no loss or
accuracy gate** in this family: the artifacts record `quality_gate.agrees =
true`, but every accuracy in them is `NaN` because neither driver prints one.
The drivers' `quality` mode exists for that and is not part of the published
matrix.

## Results

Session `2026-09-05-publish`, commit `93cc90e07`. Throughput and energy are
the median of the three rounds; peak memory is the highest of the three
launches, not a median, and on CUDA it is a difference against an idle
baseline that itself drifts (see *Where the memory goes*).


`cuda-lstm-train` — batch 256, bf16, epochs 20 per launch, 3 rounds. Artifact `cuda-lstm-train-publish-20260906T062439Z.json`, commit `93cc90e07`, clean tree, quiet True (busy 1.0% before, 0.5% after, 1.9% max during), clocks locked True, shape gate True.

| engine | median samples/s | min | max | peak device MiB | Wh (board) |
|---|---|---|---|---|---|
| OpenNN | 290,819 | 290,028 | 298,968 | 326.5 | 0.05610 |
| PyTorch | 104,815 | 97,006 | 106,586 | 531.6 | 0.12116 |
| **ratio** | **2.775×** | | | 1.63× less | 2.16× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 298,968 | 97,006 |
| 2 | pytorch → opennn | 290,819 | 106,586 |
| 3 | opennn → pytorch | 290,028 | 104,815 |


`cuda-lstm-infer` — batch 256, bf16, passes 50 per launch, 3 rounds. Artifact `cuda-lstm-infer-publish-20260905T105936Z.json`, commit `93cc90e07`, clean tree, quiet True (busy 0.46% before, 0.36% after), clocks locked True, shape gate True.

| engine | median samples/s | min | max | peak device MiB | Wh (board) |
|---|---|---|---|---|---|
| OpenNN | 918,696 | 916,771 | 920,489 | 293.9 | 0.04592 |
| PyTorch | 524,191 | 505,792 | 536,438 | 441.9 | 0.06325 |
| **ratio** | **1.753×** | | | 1.50× less | 1.38× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 918,696 | 524,191 |
| 2 | pytorch → opennn | 920,489 | 536,438 |
| 3 | opennn → pytorch | 916,771 | 505,792 |


`cpu-lstm-train` — batch 256, fp32, epochs 3 per launch, 3 rounds. Artifact `cpu-lstm-train-publish-20260905T112532Z.json`, commit `93cc90e07`, clean tree, quiet True (busy 0.0% before, 0.06% after), clocks locked True, shape gate True.

| engine | median samples/s | min | max | peak RssAnon MiB | Wh (RAPL package-0) |
|---|---|---|---|---|---|
| OpenNN | 23,147 | 22,855 | 23,218 | 242.3 | 0.04290 |
| PyTorch | 13,016 | 12,887 | 13,465 | 591.5 | 0.06093 |
| **ratio** | **1.778×** | | | 2.44× less | 1.42× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 22,855 | 12,887 |
| 2 | pytorch → opennn | 23,218 | 13,465 |
| 3 | opennn → pytorch | 23,147 | 13,016 |


`cpu-lstm-infer` — batch 256, fp32, passes 5 per launch, 3 rounds. Artifact `cpu-lstm-infer-publish-20260905T112347Z.json`, commit `93cc90e07`, clean tree, quiet True (busy 0.5% before, 0.0% after), clocks locked True, shape gate True.

| engine | median samples/s | min | max | peak RssAnon MiB | Wh (RAPL package-0) |
|---|---|---|---|---|---|
| OpenNN | 80,394 | 79,606 | 80,418 | 189.2 | 0.021654 |
| PyTorch | 69,269 | 67,145 | 69,569 | 462.7 | 0.023409 |
| **ratio** | **1.161×** | | | 2.45× less | 1.08× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 79,606 | 69,269 |
| 2 | pytorch → opennn | 80,418 | 69,569 |
| 3 | opennn → pytorch | 80,394 | 67,145 |

## Why

The GPU profiles below come from `nsys profile --trace=cuda
--cuda-graph-trace=node` runs of the same binaries at the same commit, with
only the repeat count cut so the capture stays manageable (10 passes instead
of 50, 3 epochs instead of 20). **No trace export was kept under
`results/`.** The session's evidence bundle carries only part of what those
traces showed — for `cuda-lstm-infer`, OpenNN's `elemWiseRNNcell` (34.2%,
45,192 launches) and its nvjet matmul (52.1%), and PyTorch's
`RNN_blockPersist_fp_LSTM_HMMA` (65.2%, 1,882 launches of 38.12 µs). Every
other kernel row, launch count, gap and traced-window figure in this section
is transcribed from runs that were not archived and **cannot be checked at
source**; re-running the profiler is the only way to confirm them. Nothing
outside this section depends on them: the batch times, throughputs, memory
and energy all come from the artifacts named above.

**The profiler slows both host paths**, and by different amounts: OpenNN
reads 544,175 windows/s under `nsys` against 918,696 published, PyTorch
397,264 against 524,191 — 1.69× against 1.32×. Kernel *durations* are
device-side and unaffected, so the per-batch GPU times below are usable
directly. The traced busy fractions and gaps are not published-run
quantities: busy fractions are rescaled to the published batch where they
are stated and the division is shown, and the gaps are left as traced and
used only as a ratio and a shape. The asymmetry runs OpenNN's way — the
profiler costs it more — so an untraced gap ratio would if anything be
wider.

### Where the energy goes

All four cells win energy the same way, and it is not by drawing less power.
On the GPU cells OpenNN pulls *more* watts than PyTorch — over the published
launches, 69.3 W against 54.5 W on inference and 65.7 W against 51.4 W on
training, 27% and 28% more — and still comes out 1.38× and 2.16× ahead on
energy, because it finishes 1.753× and 2.775× sooner. The arithmetic closes:
1.753 ÷ 1.272 = 1.38 against the measured 1.377, 2.775 ÷ 1.278 = 2.17
against the measured 2.160.

The extra watts are the card being busy rather than idle. Dividing the
traced GPU time per batch by the published batch time, OpenNN keeps the card
in kernels 32.5% of the inference window (90.6 µs of a 279 µs batch) against
PyTorch's 12.0% (58.5 µs of 488 µs), and 32.2% of the training window
(283.2 µs of GPU work in a 880 µs batch) against 10.3% (251.8 µs of 2,442
µs). A card that is busy about three times as often draws more power per
second and much less power per sample. It also means a large part of both
board readings in these two cells is the card powered on and waiting, which
is why the energy ratios are *smaller* than the throughput ratios. A reader
who wants the energy of the arithmetic rather than the energy of a mostly
idle GPU should read the CPU cells, where the device is saturated.

The CPU cells have the same shape without the idle-card distortion. RAPL
package-0 energy over the timed window gives OpenNN 28.5 W against PyTorch's
26.7 W on inference and 27.2 W against 21.9 W on training — the published
launches; across all three rounds, 28.50–28.69 against 25.84–26.69 and
27.03–27.26 against 21.60–22.17, so the direction holds in every
pairing — and OpenNN wins 1.08× and 1.42× by finishing 1.161× and 1.778×
sooner. Why the package power differs is *not* established here. Both
engines run under the same `GOMP_SPINCOUNT`, which the runner sets precisely
to remove the spin-versus-sleep difference between the two libgomps, and no
PyTorch-side CPU profile exists in this session to say what the extra watts
buy.

### Where the memory goes

The two devices report different quantities. On CUDA the figure is
`device_used_minus_idle`: device memory in use during the run minus an idle
baseline read just before it — 251 MiB in every `cuda-lstm-infer` launch,
254–272 MiB across the `cuda-lstm-train` launches. Each engine's CUDA
context is inside its number, and the 18 MiB spread in that baseline is
inside the training figures. On CPU the figure is
`process_peak_anonymous_rss`: the whole process, framework baseline
included, which is a caveat below rather than a decomposition.

On the GPU, two components of the gap are identified and the rest is not.
PyTorch's driver holds the whole 43,800 × 24 × 15 fp32 window tensor on the
device in both cells — 60 MiB — which is 60 of the 148 MiB gap in
`cuda-lstm-infer` (294 against 442) and 60 of the 205 MiB gap in
`cuda-lstm-train` (326 against 532). Inside OpenNN's own 326 MiB, cuDNN's
scratch is 24 MiB: the descriptor dump in
`cuda-lstm-train-publish-20260906T062439Z.json` reads `workspace=17254656 B,
reserve=7864576 B`, on a network of 73,857 parameters — under 300 KB of
weights even at fp32. The remaining 88 MiB and 145 MiB are **not
attributed** by anything in this session — allocator behaviour, activations
and framework overhead in unknown proportion. Launch to launch, OpenNN
reads 293.9 MiB in all three inference launches and 315.5–326.5 in
training; PyTorch 437.9–441.9 and 515.3–531.6. The tables publish the
highest of each.

On CPU nothing decomposes the 2.44× and 2.45×; the caveat below says why the
instrument cannot.

### `cuda-lstm-infer`, 1.753×: two different cuDNN algorithms

A batch of 256 windows × 24 steps × 128 units is small: 279 µs per batch for
OpenNN, 488 µs for PyTorch, both eager (no graph replays on this family, see
the caveats). The traced batch is where the difference is:

| | OpenNN | PyTorch |
|---|---|---|
| published / under `nsys` (samples/s) | 918,696 / 544,175 | 524,191 / 397,264 |
| timed window traced | 0.813 s | 1.102 s |
| kernel launches in that window | 87,213 (107,273/s) | 27,367 (24,834/s) |
| GPU busy with kernels, traced | 19.1% | 9.1% |
| GPU busy with kernels, published batch | 32.5% | 12.0% |
| gaps between kernels, median / p90 | 5.0 / 10.1 µs | 15.6 / 65.6 µs |
| idle between kernels, traced total | 658.1 ms | 1001.6 ms |

The two kernel tables below cover the whole traced run, warmup included, not
only the timed window in the table above.

OpenNN, top kernels over the traced run:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 52.1% | 43,309 | 2.05 | `nvjet_sm120_tst_mma_64x32x64_8_16x32x64_tmaAB_alignCD4_bz_TNNN` |
| 34.2% | 45,192 | 1.29 | `elemWiseRNNcell<__nv_bfloat16, __nv_bfloat16, float, 2, 1>` |
| 9.7% | 1,883 | 8.79 | `cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>` |
| 1.9% | 1,883 | 1.71 | `transpose_padded_batch_time_kernel<__nv_bfloat16, 0>` |
| 1.1% | 1,883 | 1.02 | `linear_forward_single_output_kernel<__nv_bfloat16>` |
| 1.0% | 1,883 | 0.86 | `time_slice_kernel<__nv_bfloat16, 1>` |

PyTorch, top kernels over the traced run:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 65.2% | 1,882 | 38.12 | `RNN_blockPersist_fp_LSTM_HMMA<__half, __half, float, 1, 128>` |
| 14.1% | 1,882 | 8.27 | `cutlass::Kernel2<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_tn_align1>` |
| 8.5% | 13,174 | 0.71 | `at::native::vectorized_elementwise_kernel<4, float16_copy_kernel_cuda…>` |
| 3.1% | 1,882 | 1.84 | `at::native::unrolled_elementwise_kernel<direct_copy_kernel_cuda…>` |
| 2.7% | 1,882 | 1.59 | `at::native::elementwise_kernel<128, 4, …direct_copy…>` |
| 2.3% | 1,882 | 1.36 | `internal::gemvx::kernel<…__nv_bfloat16…>` |
| 1.7% | 3,765 | 0.49 | `at::native::vectorized_elementwise_kernel<4, FillFunctor<float>>` |

Per batch, GPU time by kernel class (kernels grouped by name; 1,883 traced
batches for OpenNN, 1,882 for PyTorch):

| kernel class | OpenNN launches / batch | OpenNN µs / batch | PyTorch launches / batch | PyTorch µs / batch |
|---|---|---|---|---|
| recurrent | 24.0 | 31.0 | 1.0 | 38.1 |
| gemm | 24.0 | 56.0 | 2.0 | 9.6 |
| elementwise | 0.0 | 0.0 | 13.0 | 10.8 |
| other (OpenNN's own) | 3.0 | 3.6 | 0.0 | 0.0 |
| **total** | **51.0** | **90.6** | **16.0** | **58.5** |

**PyTorch's recurrence is the better one.** Its single
`RNN_blockPersist_fp_LSTM_HMMA` launch runs all 24 steps with the weights
held in registers; adding its 8.27 µs input-projection GEMM, its LSTM layer
costs 46.4 µs of GPU time per batch (38.12 + 8.27). OpenNN's `STANDARD` path
costs 87.0 µs for the same layer — 24 `elemWiseRNNcell` launches of 1.29 µs,
one per time step, 23 recurrent GEMMs of 2.05 µs, and its own 8.79 µs input
projection: 30.96 + 47.15 + 8.79 = 86.9. (Twenty-three, not twenty-four:
`cudnn_rnn_forward_` passes `nullptr` for the initial hidden and cell
states, which cuDNN treats as zeros, and the first step's recurrent GEMM
does not appear.) OpenNN is not missing the persistent algorithm by
oversight — it implements it and gates it on the precision, as the summary
above describes — but in this cell, at this precision, it does not get it.

PyTorch does the entire batch in 58.5 µs of GPU time against OpenNN's 90.6
and loses the cell by 1.753× regardless, because at this size what decides
it is how fast each host path can issue: PyTorch's batch is 488 µs around
those 58.5 µs of kernels where OpenNN's is 279 µs around 90.6.

OpenNN's 51 launches per batch are issued from C++; PyTorch's 16 come out of
the Python interpreter and the dispatcher, and what separates the two traced
gap distributions is the tail, not the middle: a 15.6 µs median and a
65.6 µs 90th percentile against 5.0 and 10.1. The medians alone decide nothing —
51 gaps of 5.0 µs and 16 of 15.6 µs are the same total — and neither figure
describes the published run. Thirteen of PyTorch's sixteen launches are
element-wise, and seven of those are `float16_copy_kernel_cuda` — 13,174
launches over the traced run, 8.5% of its GPU time. They follow
from the driver running `nn.LSTM` under `torch.autocast` over an fp32 model
(see the caveats). OpenNN makes no per-batch cast and no per-batch copy: the
packed cuDNN weight space is built once (`rnn_copy_regions_kernel` and
`cast_kernel` appear exactly once each in the whole inference trace, against
1,883 batches) and the states are null, so nothing is zeroed per call.

### `cuda-lstm-train`, 2.775×: eager against eager, and the host is the bottleneck

Training is the same forward, cuDNN's two backward calls (data, then
weights), the output layer's backward, the loss and the Adam update. Per
batch: OpenNN 880 µs, PyTorch 2,442 µs, with 283 µs and 252 µs of GPU work
in them respectively. This is a cell decided by launch overhead on both
sides:

| | OpenNN | PyTorch |
|---|---|---|
| published / under `nsys` (samples/s) | 290,819 / 173,440 | 104,815 / 78,689 |
| timed window traced | 0.761 s | 1.677 s |
| kernel launches in that window | 58,963 (77,481/s) | 29,752 (17,741/s) |
| GPU busy with kernels + copies, traced | 19.1% | 7.7% |
| GPU busy with kernels + copies, published batch | 32.2% | 10.3% |
| gaps between kernels, median / p90 | 9.3 / 12.6 µs | 25.1 / 115.3 µs |

The two kernel tables below cover the whole traced run — about 870 batches
for OpenNN and 855 for PyTorch, warmup included — not the 513 batches of the
timed window above.

OpenNN, top kernels over the traced run:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 22.7% | 19,849 | 2.60 | `nvjet_sm120_tst_mma_16x64x128_4_16x16x128_tmaAB_alignCD4_bz_NNNN` |
| 17.7% | 19,849 | 2.03 | `nvjet_sm120_tst_mma_64x32x64_8_16x32x64_tmaAB_alignCD4_bz_TNNN` |
| 13.5% | 20,880 | 1.47 | `elemWiseRNNcell<__nv_bfloat16, __nv_bfloat16, float, 2, 1>` |
| 12.8% | 20,880 | 1.40 | `LSTM_elementWise_bp1<__nv_bfloat16, __nv_bfloat16, float>` |
| 9.3% | 870 | 24.21 | `GENERIC_elementWise_bp2<__nv_bfloat16, __nv_bfloat16, float, 4, 1>` |
| 4.4% | 863 | 11.58 | `nvjet_sm120_tst_mma_64x64x64_6_32x16x64_tmaAB_alignCD4_splitK_NTNN` |
| 3.6% | 879 | 9.39 | `magma_sgemmEx_kernel<float, __nv_bfloat16, float, …>` |
| 3.4% | 863 | 8.84 | `cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>` |

PyTorch, top kernels over the traced run:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 18.2% | 855 | 45.92 | `RNN_blockPersist_bp_LSTM_HMMA<__half, __half, float, 128>` |
| 16.4% | 855 | 41.19 | `RNN_blockPersist_fp_LSTM_HMMA<__half, __half, float, 1, 128>` |
| 8.7% | 855 | 21.93 | `GENERIC_elementWise_bp2<__half, __half, float, 4, 2>` |
| 4.5% | 855 | 11.36 | `at::native::multi_tensor_apply_kernel<…>` |
| 4.3% | 855 | 10.93 | `cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_f16_64x64_32x10_nt_align8>` |
| 3.7% | 855 | 9.33 | `at::native::multi_tensor_apply_kernel<…>` |
| 3.7% | 855 | 9.32 | `at::native::multi_tensor_apply_kernel<…>` |
| 3.7% | 855 | 9.30 | `cutlass::Kernel2<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_32x1_nt_align1>` |

Per batch, GPU time by kernel class, over that same whole trace:

| kernel class | OpenNN launches / batch | OpenNN µs / batch | PyTorch launches / batch | PyTorch µs / batch |
|---|---|---|---|---|
| recurrent | 49.3 | 93.7 | 4.0 | 110.5 |
| gemm | 54.6 | 152.5 | 10.0 | 44.1 |
| optimizer | 2.0 | 2.7 | 7.0 | 57.8 |
| elementwise | 0.0 | 0.0 | 35.0 | 36.8 |
| reduction | 0.0 | 0.0 | 2.0 | 2.5 |
| other (OpenNN's own) | 9.1 | 13.9 | 0.0 | 0.0 |
| copy / memset | 5.1 | 20.3 | 0.0 | 0.0 |
| **total** | **120.0** | **283.2** | **58.0** | **251.8** |

The two engines spend almost the same GPU time per batch — 283 µs against
252 µs, PyTorch again the cheaper — and OpenNN gets 2.775× the throughput
out of it. The rest of each published batch is host time: 597 µs of it for
OpenNN, 2,190 µs for PyTorch. OpenNN issues its 120 launches from C++;
PyTorch issues 58 from Python, and each of its gaps is the interpreter and
the dispatcher: `zero_grad`, the forward under autocast, `backward()`
through autograd (the cuDNN backward is one node, but every cast, the
`Linear`, the loss and the bias sum are separate ones), then `Adam.step()`.
Traced, the difference is again a tail — 25.1 µs median and 115.3 µs at
the 90th percentile against 9.3 and 12.6 — but those microseconds are
inflated by the profiler and do not multiply out to the host budget above.

Two structural differences are worth naming because they are not launch
overhead.

**The optimiser.** OpenNN's Adam update runs over one contiguous gradient
buffer — the LSTM's weight regions and the dense layer's two are views into
it — as `adam_prepare_kernel` plus `adam_update_kernel`, two launches
totalling 2.7 µs per batch, 1.0% of its GPU time. PyTorch's `Adam` at its
defaults selects the *foreach* implementation: seven
`multi_tensor_apply_kernel` launches per batch, 57.8 µs, **23.0% of
PyTorch's GPU time — more than its LSTM backward kernel** (18.2%).
Updating 73,857 parameters cannot cost 57.8 µs of arithmetic; this is
per-tensor latency, and it is a PyTorch default the driver did not override
(see the caveats).

**The weight-space traffic.** cuDNN wants its own packed weight layout, so
OpenNN moves between that and its gradient arena twice per batch:
`rnn_copy_regions_kernel`, 2.0 launches of 3.09 µs, 6.2 µs and 2.2% of its
283.2 µs of GPU time. Add the per-batch upload of the window batch (17.8 µs
across a 184,320-byte and a 1,024-byte copy) and the memsets cuDNN needs,
and OpenNN pays 20.3 µs per batch in copies and memsets where PyTorch pays
none. Both are costs, and both are already inside the 2.775×.

`torch.compile` cannot close the gap — it was measured slower, twice, in the
artifacts and in the driver's own sweep. Neither engine replays a CUDA graph
here, but not for the same reason: OpenNN asks for one and cuDNN refuses it
(status 4000, `cuda_graph=failed` in both CUDA artifacts), while PyTorch's
CUDA-graph path, `compile:reduce-overhead`, was available, was measured at
89,518 samples/s against an eager band of 95,821–104,590, and lost. So the
cell measures two eager launch paths over comparable amounts of GPU work.

### `cpu-lstm-infer`, 1.161×: one primitive, two runtimes

The oneDNN LSTM primitive is the whole forward pass on OpenNN's side, and
the profile says how completely. Running the published binary with
`OPENNN_PROFILE=1` (artifacts `cpu-lstm-infer-cpuprofile-20260906T062628Z`
and `cpu-lstm-infer-lprof-20260906T063256Z`, same commit; the table below is
the second):

| section | ms / call | share |
|---|---|---|
| `fwd:LongShortTermMemory` | 3.237 | — |
| ` rnn:onednn_forward` | **3.185** | **98.4% of the LSTM layer** |
| ` rnn:onednn_transpose_input` | 0.020 | 0.6% |
| ` rnn:onednn_arguments` | 0.005 | 0.2% |
| ` rnn:onednn_final_state`, `_pack_bias`, `_plan` | 0.005 | 0.2% |
| `fwd:Dense` | 0.043 | — |
| ` cpu:sgemm_thin_n` | 0.037 | 86% of the dense layer |

Three profiled runs at this commit agree to 0.010 ms/call — 3.185, 3.190 and
3.195, the third from `scratch/cpu-lstm-infer-prof-20260906T063154Z`, which
fails the quiet gate. Read inside one run: the instrumented batch is 3.290 ms
(77,812 windows/s under the profiler against 80,394 published, so the
instrument costs about 3%) and the oneDNN primitive is 3.185 ms of it —
**96.8% of the batch**, 98.4% of the LSTM layer. Everything OpenNN adds
around the primitive, the 128 → 1 output layer included, is about 0.10 ms.
That residual is the size of the instrument's own cost, so it bounds any
runtime effect on OpenNN's side at roughly 3% of the batch, not at zero.

PyTorch's batch is 3.696 ms — 0.512 ms longer. Where that half-millisecond
goes is *not* measured: no PyTorch-side CPU profile exists for this cell, so
this document cannot attribute it, and it cannot rule out that PyTorch's
oneDNN build executes the primitive faster or slower than OpenNN's, or with
a different thread count (see the caveats). What can be said is that it is
not in OpenNN.

Two runtime effects were found while this cell was being tuned; both are now
neutralised for both engines by the runner, so neither is inside the
published 1.161×. GCC 14's libgomp — the system runtime OpenNN links —
sets its spin count to 1 on hybrid CPUs, so its workers futex-sleep between
parallel regions where PyTorch's bundled libgomp spins: same primitive, same
descriptor, 3.45 ms against 3.14 ms per batch, which the runner removes by
setting `GOMP_SPINCOUNT=300000` for both (PROTOCOL §6). And libgomp re-sizes
its pool to every region's team, so `omp_set_dynamic(1)` (team = CPUs minus
the fifteen-minute load average) and MKL's own heuristic for the 128 → 1
output layer (10 threads for a small GEMM) were each shrinking the team
between oneDNN's 16-thread regions, at six `pthread_create` calls per batch
and 10% of the forward pass; both are fixed in `Backend::set_threads_number`
(`device_backend.cpp:1239`, dynamic off and `mkl_set_dynamic(0)`).

Both drivers print `flush_denormals=on`.

### `cpu-lstm-train`, 1.778×: measured, not attributed

Training runs oneDNN's LSTM forward (training mode, with its workspace) and
backward primitives on both sides. OpenNN's batch is 11.06 ms against
PyTorch's 19.67 ms, a difference of 8.61 ms.

*[pending the final measurement round]* — **that 8.61 ms is unattributed.**
The profiling round did cover this cell, twice at this commit
(`results/scratch/cpu-lstm-train-cpuprofile-20260906T062648Z` and
`-prof-20260906T063214Z`, 20,932 and 20,570 windows/s against 23,147
published), and both returned a section table holding only
`device:deallocate` (15 calls) and `fp:dtor` (2 calls): OpenNN's training
path carries no profile sections, so the instrument has nothing to report
there. No PyTorch-side CPU profile exists in this session either. Until
sections are added, this document cannot say which of the two backward
primitives, the loss, the optimiser or the extra layers accounts for the
8.61 ms, and the inference profile above does not transfer: it covers a
forward pass with no workspace, no backward and no optimiser.

What can be said without a profile: OpenNN's extra layers — the scaling
layer in front of the LSTM and the unscaling and clamping layers behind the
output, which the inference network does not have and the PyTorch network
does not carry at all — are three element-wise passes over a 256 × 24 × 15
batch and a 256 × 1 output, and they run in OpenNN's 11.06 ms batch, not
PyTorch's 19.67 ms one. Whatever their cost, it counts against OpenNN.

## Asymmetries and caveats

- **PyTorch reaches the better recurrent kernel on the GPU and OpenNN does
  not.** PyTorch's `nn.LSTM` gets cuDNN's persistent LSTM kernel; OpenNN's
  own persistent path is gated on `!bf16` and this cell is bf16, so it calls
  `cudnnRNNForward` with `algo=STANDARD` and gets the unrolled path — 87.0
  µs of GPU time for the layer against PyTorch's 46.4, and 90.6 against 58.5
  across the whole batch (the *Why* section has the kernels). **OpenNN wins
  these cells on issue rate, not on kernels**, and any reader who wants a
  claim about kernel quality should read that sentence the other way round.
  What OpenNN would do at fp16, where its own gate would let the persistent
  algorithm through, has not been measured.
- **The two GPU cells do not run the recurrence in the same precision.** The
  cell is declared `bf16` and OpenNN's kernels are `bf16` throughout
  (`elemWiseRNNcell<__nv_bfloat16, __nv_bfloat16, float, …>`). PyTorch's
  driver asks for `torch.autocast(dtype=torch.bfloat16)`, and what actually
  runs is `fp16`: the `__half` persistent kernel, `float16_copy_kernel_cuda`
  casts, `FillFunctor<c10::Half>`. Why bf16 autocast yields an fp16
  recurrence inside PyTorch is not established by these artifacts, and this
  document does not guess at it. Sixteen-bit floats of either kind cost the
  same per operation on this card, so the precision is not itself a speed
  advantage — but it is what selects the algorithm, and here it selected the
  better one for PyTorch. The two engines are also **not bit-comparable** in
  these cells, and the family has no accuracy gate that would notice.
- **The two engines do not link the same cuDNN.** OpenNN's descriptor dump
  in both CUDA artifacts ends `cuDNN 92501` (9.25.1); every artifact records
  `torch_built_cudnn = 92302` (9.23.2) for the PyTorch wheel. That
  difference is not equalised, and nothing here separates a version
  difference in which engines cuDNN offers from the precision effect above.
- **PyTorch's Adam is at its default, and the default is expensive here.**
  The driver builds `torch.optim.Adam(model.parameters())` and never asks
  for `fused=True`, so PyTorch takes the *foreach* path: seven
  `multi_tensor_apply_kernel` launches per batch, 57.8 µs, 23.0% of its GPU
  time in `cuda-lstm-train` — more than the LSTM backward kernel. PyTorch
  ships a fused Adam that would plausibly cut most of that. It was **not
  measured**, so the size of the effect is unknown, but a reader should
  treat some part of `cuda-lstm-train`'s 2.775× as a PyTorch default this
  suite did not override rather than as an OpenNN result. The per-batch
  casts are the same kind of thing: `families/lstm.py:151` wraps the step in
  `torch.autocast` over an fp32 model, so PyTorch re-casts its weights every
  batch — seven `float16_copy_kernel_cuda` launches per batch on inference,
  8.5% of its GPU time. A model held in bf16 weights, which is effectively
  what OpenNN runs, was not measured, so part of the launch-count gap is
  that driver choice rather than a PyTorch limit.
- **OpenNN uploads each training batch; PyTorch keeps the dataset
  resident.** PyTorch's driver moves the whole 43,800 × 24 × 15 fp32 window
  tensor to the device once — 60 MiB — and slices it there. OpenNN's
  training trace shows one 184,320-byte and one 1,024-byte host-to-device
  copy per batch — 17.8 µs of copy time on the timeline of a 880 µs batch,
  so at most about 2% of the cell. This cuts both ways and both ways are in
  the published numbers: roughly 60 MiB of the 205 MiB memory gap in
  `cuda-lstm-train` is that resident tensor rather than anything OpenNN
  does better.
- **The same 60 MiB inflates OpenNN's inference memory win.** The PyTorch
  inference driver also uploads the complete window tensor and then reads
  only its first 256 rows, because the inference cells replay one resident
  batch. About 60 MiB of the 148 MiB gap in `cuda-lstm-infer` (294 against
  442 MiB) is therefore a tensor our driver told PyTorch to allocate and
  never used. The comparison would be closer if the driver sliced before
  uploading.
- **`cuda-lstm-train` is one of the least repeatable cells in the matrix,
  and the ratio depends on which run is published.** Two full publish runs
  exist at this commit, in the same session, on a quiet machine with clocks
  locked: `cuda-lstm-train-publish-20260905T110036Z` reads OpenNN 289,372
  against PyTorch 95,954, which is **3.016×**;
  `cuda-lstm-train-publish-20260906T062439Z` reads 290,819 against 104,815,
  which is **2.775×**. The published row is the later one — the less
  favourable of the two. Nothing in those two artifacts distinguishes them:
  the quiet gate, the clock lock and the shape gate pass in both, and their
  OpenNN launches span 289,321–298,968, about 3%. Four single-launch runs at
  the same commit are also kept: 294,210
  (`cuda-lstm-train-adam1-20260906T062904Z`, quiet true) and, under
  `results/scratch/`, 279,991, 279,878 and 274,530
  (`-adam1-20260906T062929Z`, `-adam0-20260906T062942Z` and
  `-adam0-20260906T062917Z`), all three of which **fail the quiet gate** —
  4.9–5.6% busy before, against a 3% threshold. Counting them, OpenNN's
  side spans 274,530–298,968; the low half is the half that fails the gate.
  (The `adam0`/`adam1` labels differ, but the command lines are
  byte-identical and no artifact records an environment, so what the labels
  distinguish is not in the record.) PyTorch's side moves further,
  95,459–106,586 across the two publish runs, 11.7% on the low end. **Read
  this cell as 2.77–3.02×, not as a point value.**
- **The window is not short any more, so a short window does not explain
  that spread.** The cell times 20 epochs after 2 untimed: 2.96–3.07 s for
  OpenNN and 8.29–9.13 s for PyTorch per launch. The earlier publication of
  this table ran three epochs and blamed the spread on the window; that
  explanation no longer holds and no replacement has been established.
- **OpenNN's training network is larger than its inference network.** The
  training cell builds `ForecastingLstmNetwork`, which wraps the LSTM and its
  output layer in a scaling layer in front and an unscaling and a (disabled)
  clamping layer behind — the layers OpenNN's users get from the model
  builder, and each a pass over the batch that PyTorch's `nn.LSTM` + `Linear`
  does not make. The inference cell builds the bare LSTM and dense layers.
  The parameter count is the same either way (73,857) because the scaling
  layers have none.
- **PyTorch's LSTM has a second bias the model does not use.** cuDNN's and
  PyTorch's LSTM carry two bias vectors per gate; OpenNN's carries one. The
  driver zeroes `bias_hh` and freezes it, prints the parameter count without
  it so the gate agrees, and cuDNN still adds the zeros — a negligible
  amount of work on PyTorch's side that is there so both engines drive the
  same cuDNN RNN descriptor.
- **Neither engine replays a CUDA graph on this family.** PyTorch runs eager
  because compiling measured slower (89,518 samples/s training and 462,572
  inference in the `2026-09-02-variants` artifacts, against eager bands of
  95,821–104,590 and 522,694–535,766 in that day's publish runs, and the
  same result in the driver's own sweep); OpenNN asks for a graph and is
  refused — both CUDA artifacts record `cuda_graph=failed` and the engine
  continues eager. The driver's message
  reports that `cudnnRNNForward` returned cuDNN status 4000 and invalidated
  the capture, and states plainly that it does not know which forbidden
  operation caused it. The margin is therefore between two eager launch
  streams, and what separates them is in the *Why* section. The compiled
  measurement is at a different commit and session and predates the reboot
  that cleared PyTorch's inductor autotune cache — the event that moved
  `cuda-dense-infer` by 4% this session. It was not repeated at
  `93cc90e07`, so read the compiled rows as the best measurement available
  for this family rather than as PyTorch's best mode under the published
  conditions; the 582 MiB compiled peak likewise belongs beside that day's
  eager peaks of 507–531 MiB, not beside the 531.6 published here.
- **Different oneDNN builds on CPU, and neither the versions nor the thread
  counts are equalised.** OpenNN links its own oneDNN with the OpenMP
  runtime and PyTorch's wheel bundles a different build; the versions were
  not recorded in this session's artifacts. Both CPU cells run the same
  oneDNN LSTM primitive with the same descriptor (checked with
  `DNNL_VERBOSE` when the CPU inference cell was being tuned) under the same
  `GOMP_SPINCOUNT`, which the runner sets for both (PROTOCOL §6). But the
  runner sets **no thread count**: every CPU artifact records
  `pinning.threads = "engine default"`, OpenNN derives 16 from the `taskset`
  mask and PyTorch computes its own default from the machine, not from the
  affinity mask, and no artifact records what that resolves to. So the two
  engines may not run the same number of threads, and the descriptors are
  not claimed to match on `nthr`. With no measurement of PyTorch's per-call
  primitive time here either, a version difference and a thread-count
  difference are both inside the 1.161× and the 1.778× and this document
  cannot separate them out.
- **The CPU memory ratios are process peaks with each framework's baseline
  inside them.** Every CPU artifact carries the instrument's own warning:
  "no baseline_anonymous_rss_mib: peak_mib is the whole process, framework
  baseline included. The baseline the drivers do print is a total-RSS
  reading, which is not commensurable with process_peak_anonymous_rss".
  Those printed baselines are 208.3–208.6 MiB for the OpenNN binary against
  671.8–672.0 MiB for the Python process with `torch` imported. No
  baseline-subtracted figure was recorded, so an unknown and probably large
  share of the 2.44× and 2.45× is the interpreter and its imported libraries
  rather than anything OpenNN does with the model. `footprint.md` measures
  that floor on its own.
- **Flush-to-zero on both CPU sides.** The PyTorch driver calls
  `torch.set_flush_denormal(True)` on CPU, the usual practice for recurrent
  inference; OpenNN's driver sets the same MXCSR bits before any OpenMP team
  exists so the workers inherit them. Both print `flush_denormals=on`, so a
  denormal slowdown cannot land on one engine only.
- **No accuracy gate.** Neither driver prints a loss. The runner compares
  the window count (`samples`), `inputs`, `past`, `hidden` and the parameter
  count, and the cells are held to that shape gate. The artifacts' quality
  gate records `agrees: true` over a list of `NaN`s; it is vacuous here and
  should not be read as agreement on outputs.
- **An earlier version of this table timed OpenNN inference over zeros.**
  Until commit `4338506c8` the OpenNN inference drivers filled each batch
  (`Batch::fill()`) and never uploaded it: `fill()` only stages the rows on
  the host, and in the library the transfer is issued by the optimizer, which
  an inference driver does not run. The forward pass ran over all-zero inputs
  for every published inference cell before that commit. Nothing in the gates
  could see it — the parameter count, the sample count and the input file are
  the same whatever the batch holds, and the arithmetic of a forward pass does
  not depend on the values — and it was found by printing outputs (every one
  read `sigmoid(0) = 0.5`). The cuDNN RNN over a zero window runs the same
  kernels for the same time, and the cell did not move beyond its spread:
  the affected row read 923,205 windows/s
  (`cuda-lstm-infer-publish-20260902T022430Z`) against the 918,696 published
  here, and the publish runs of this cell have read between 908,788 and
  923,205 across the whole series. The rows above are from the fixed
  drivers, which upload the batch (`upload_to_device_batch_async()`) after
  filling it.

## Reproduce

```bash
export OPENNN_BENCH_SESSION=$(date +%F)-mine
python run.py --family lstm --mode train --device cuda --batch 256 --precision bf16 --epochs 20 --rounds 3
python run.py --family lstm --mode infer --device cuda --batch 256 --precision bf16 --repeats 50 --rounds 3
python run.py --family lstm --mode train --device cpu  --batch 256 --precision fp32 --epochs 3 --rounds 3
python run.py --family lstm --mode infer --device cpu  --batch 256 --precision fp32 --repeats 5 --rounds 3
```

`prepare.py lstm` downloads the Beijing PM2.5 set and writes the 15-column
CSV both engines read. `PT_COMPILE_MODE=reduce-overhead` is the PyTorch knob
(the compiled rows above are its result); `OPENNN_PROFILE=1` prints the
section table quoted for `cpu-lstm-infer`; the OpenNN CPU build needs
`-DOpenNN_ENABLE_ONEDNN=ON` — without it the recurrent layer runs a fallback
and the CPU cells are a different comparison.
