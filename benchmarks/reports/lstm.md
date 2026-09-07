# LSTM: forecasting Beijing PM2.5

OpenNN against PyTorch 2.13 on a single-layer LSTM of 128 units with a linear
output (73,857 parameters) forecasting the next hour of PM2.5 from 24 hours
of 15 features, 43,800 windows, batch 256 — bf16 on the RTX 5070 Ti, fp32 on
the sixteen P-cores, where both engines run oneDNN's LSTM primitive. Session
`2026-09-06-publish`, commit `e76425bd3`:

| cell | OpenNN | PyTorch | OpenNN / PyTorch |
|---|---|---|---|
| `cuda-lstm-train` | 823,255 windows/s | 95,842 | **8.590×** |
| `cuda-lstm-infer` | 2,716,548 windows/s | 513,813 | **5.287×** |
| `cpu-lstm-train` | 23,251 windows/s | 13,103 | **1.774×** |
| `cpu-lstm-infer` | 80,285 windows/s | 69,304 | **1.158×** |

`cuda-lstm-train` is the widest margin in the twelve-cell matrix, 8.590×, and
`cuda-lstm-infer` the second, 5.287×. Both are new this round: at commit
`93cc90e07` the same cells read 2.775× and 1.753×, and the difference is one
line in the library, described below.

**On the GPU, PyTorch has the better algorithm and still loses.** Both
engines call cuDNN's RNN API and do not get the same kernels. PyTorch's
`nn.LSTM` reaches cuDNN's *persistent* LSTM kernel —
`RNN_blockPersist_fp_LSTM_HMMA`, one launch of 38.1 µs that runs the whole
24-step recurrence with the weights resident in registers, 65.1% of its
inference profile. OpenNN takes cuDNN's `STANDARD` path, which unrolls the
recurrence into 24 `elemWiseRNNcell` launches and 23 recurrent GEMMs per
batch. What selects the path is the precision: `persist_algo_active` in
`cudnn_rnn.cpp` requires, among other conditions, `!bf16 && H <= 128` and,
for an LSTM, `batch < 512`, so a cell declared bf16 never asks cuDNN for the
persistent algorithm — OpenNN's own descriptor dump records `algo=STANDARD,
dtype=BFLOAT16`. PyTorch, asked for bf16 autocast, runs the recurrence in
fp16 and gets the persistent kernel. Asking cuDNN for the persistent
algorithm at bf16 was tried in this round: it is granted only with the
`CUDNN_RNN_DOUBLE_BIAS` layout and refused with `SINGLE_INP_BIAS`, and taken
that way it was worth 0.6% on the cell, so the gate stays where it is.

The persistent kernel does PyTorch's whole inference batch in 62.4 µs of GPU
time against OpenNN's 86.6 µs, and its training batch in 252 µs against
OpenNN's 280. OpenNN wins the cells 5.3× and 8.6× anyway, because of what
surrounds the kernels. Until this round neither engine replayed a CUDA graph
on this family: PyTorch's `reduce-overhead` mode measured slower than eager
and was not taken, and OpenNN asked for a graph and was refused by cuDNN
with status 4000. The refusal turned out to be OpenNN's own: the backend
created its lane-0 compute stream with `cudaStreamDefault`, the blocking
kind, while every other stream in the process was non-blocking, and cuDNN's
RNN implementation touches that stream in a way that is illegal inside a
capture (`cudaErrorStreamCaptureImplicit`). Commit `8cb810339` makes the
stream non-blocking; the capture succeeds, both CUDA artifacts now record
`cuda_graph=captured`, and the 51 kernels of an inference batch and the 120
of a training batch are issued by the graph executor 0.2 µs apart instead of
by the host 5–9 µs apart. A batch that was 279 µs of wall time around 91 µs
of kernels is now 94 µs around 87, and the GPU, not the host, sets the
throughput. PyTorch's batch is still 498 µs of wall time around 62 µs of
kernels. The margin is therefore an issue-path margin fully realised on one
side; the better recurrent kernel remains on the other, and a PyTorch path
that captured its batch would not lose this cell by 5×.

On the CPU both engines run the same oneDNN LSTM primitive, and there the
cell is almost nothing but that primitive: 3.185 ms of a 3.290 ms
instrumented batch, 98.4% of OpenNN's LSTM layer. Inference (1.158×) is the
0.5 ms by which PyTorch's batch is longer: 0.2 ms more inside the oneDNN
call itself and 0.3 ms of copies and concatenations around it; training
(1.774×) is 11.0 ms against 19.5 ms, and the 8.5 ms is mostly inside
PyTorch's two oneDNN calls, with 3 ms of autograd's own time around the
backward node. Both are in the *Why* section with the profiles.

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
bytes per batch, 16.0 µs of copies and memsets per 311 µs batch on the
captured path. That asymmetry costs OpenNN throughput and saves it memory;
both effects are counted in the results below. For inference both engines fill the first batch
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

OpenNN captures a CUDA graph of the inference pass and of the whole training
step (forward, cuDNN's two backward calls, the output layer, the loss and
Adam), and replays it once per batch. At `93cc90e07` it asked for the same
graphs and was refused: `cudnnRNNForward` returned cuDNN status 4000 inside
the capture, the artifacts recorded `cuda_graph=failed`, and the cell ran
eagerly. The cause was found by reading the backend's constructor rather
than cuDNN: its lane-0 compute stream was the only blocking stream in the
process, and a blocking stream is implicitly synchronised with the legacy
default stream, which cuDNN's RNN path exercises and which capture forbids
(`cudaErrorStreamCaptureImplicit`, surfacing as cuDNN's generic 4000).
`8cb810339` creates that stream with `cudaStreamNonBlocking`, as every other
stream already was; nothing else changed on this family between the two
tables.

On CPU both engines run oneDNN's LSTM primitive — OpenNN's linked oneDNN
3.11 and the 3.12 PyTorch's wheel bundles — with MKL as the BLAS. Both processes are
`taskset`-pinned to CPUs 0–15, the P-cores, and each chooses its own thread
count: the runner sets none (`pinning.threads = "engine default"` in every
CPU artifact), OpenNN reads the affinity mask and takes 16
(`Backend::set_threads_number`, `device_backend.cpp:1250`), and what
PyTorch's own default resolves to is recorded nowhere (see the caveats).
Both drivers enable flush-to-zero (`flush_denormals=on` is recorded), and
the runner sets `GOMP_SPINCOUNT=300000` for both — PROTOCOL §7 explains why
the GCC 14 libgomp default would otherwise penalise the engine that links
the system runtime.

**Gates.** Samples (43,800 windows), inputs (15), past (24), hidden (128) and
parameters (73,857) must agree between the engines. There is **no loss or
accuracy gate** in this family: the artifacts record `quality_gate.agrees =
true`, but every accuracy in them is `NaN` because neither driver prints one.
The drivers' `quality` mode exists for that and is not part of the published
matrix.

## Results

Session `2026-09-06-publish`, commit `e76425bd3`. Throughput and energy are
the median of the three rounds; peak memory is the highest of the three
launches, not a median, and on CUDA it is a difference against an idle
baseline that itself drifts (see *Where the memory goes*).


`cuda-lstm-train` — batch 256, bf16, epochs 20 per launch, 3 rounds. Artifact `cuda-lstm-train-publish-20260906T133548Z.json`, commit `e76425bd3`, clean tree, quiet True (busy 0.4% before, 0.4% after, 0.5% max during, threshold 3%), clocks locked True, shape gate True.

| engine | median samples/s | min | max | peak device MiB | Wh (board) |
|---|---|---|---|---|---|
| OpenNN | 823,255 | 811,863 | 828,105 | 315.6 | 0.02920 |
| PyTorch | 95,842 | 95,839 | 96,827 | 511.6 | 0.13200 |
| **ratio** | **8.590×** | | | 1.62× less | 4.521× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 828,105 | 95,839 |
| 2 | pytorch → opennn | 823,255 | 95,842 |
| 3 | opennn → pytorch | 811,863 | 96,827 |


`cuda-lstm-infer` — batch 256, bf16, passes 500 per launch, 3 rounds. Artifact `cuda-lstm-infer-publish-20260906T133237Z.json`, commit `e76425bd3`, clean tree, quiet True (busy 0.2% before, 0.4% after, 1.6% max during, threshold 3%), clocks locked True, shape gate True.

| engine | median samples/s | min | max | peak device MiB | Wh (board) |
|---|---|---|---|---|---|
| OpenNN | 2,716,548 | 2,716,530 | 2,716,629 | 293.6 | 0.25753 |
| PyTorch | 513,813 | 508,239 | 524,147 | 441.6 | 0.64779 |
| **ratio** | **5.287×** | | | 1.50× less | 2.515× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 2,716,548 | 508,239 |
| 2 | pytorch → opennn | 2,716,629 | 524,147 |
| 3 | opennn → pytorch | 2,716,530 | 513,813 |


`cpu-lstm-train` — batch 256, fp32, epochs 3 per launch, 3 rounds. Artifact `cpu-lstm-train-publish-20260906T134139Z.json`, commit `e76425bd3`, clean tree, quiet True (busy 0.2% before, 0.1% after, 0.1% max during, threshold 3%), clocks locked True, shape gate True.

| engine | median samples/s | min | max | peak RssAnon MiB | Wh (RAPL package-0) |
|---|---|---|---|---|---|
| OpenNN | 23,251 | 22,938 | 23,273 | 209.5 | 0.04305 |
| PyTorch | 13,103 | 13,102 | 13,348 | 591.4 | 0.06077 |
| **ratio** | **1.774×** | | | 2.82× less | 1.412× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 22,938 | 13,348 |
| 2 | pytorch → opennn | 23,273 | 13,102 |
| 3 | opennn → pytorch | 23,251 | 13,103 |


`cpu-lstm-infer` — batch 256, fp32, passes 5 per launch, 3 rounds. Artifact `cpu-lstm-infer-publish-20260906T133954Z.json`, commit `e76425bd3`, clean tree, quiet True (busy 0.2% before, 0.2% after, 0.1% max during, threshold 3%), clocks locked True, shape gate True.

| engine | median samples/s | min | max | peak RssAnon MiB | Wh (RAPL package-0) |
|---|---|---|---|---|---|
| OpenNN | 80,285 | 80,233 | 80,640 | 156.5 | 0.02156 |
| PyTorch | 69,304 | 64,643 | 69,820 | 458.8 | 0.02332 |
| **ratio** | **1.158×** | | | 2.93× less | 1.082× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 80,285 | 64,643 |
| 2 | pytorch → opennn | 80,640 | 69,820 |
| 3 | opennn → pytorch | 80,233 | 69,304 |

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
launches, 114.9 W against 54.8 W on inference and 98.8 W against 52.4 W on
training, 2.10× and 1.89× more — and still comes out 2.52× and 4.52× ahead
on energy, because it finishes 5.287× and 8.590× sooner. The arithmetic
closes: 5.287 ÷ 2.097 = 2.52 against the measured 2.515, 8.590 ÷ 1.885 =
4.56 against the measured 4.521.

The extra watts are the card being busy rather than idle. Dividing the
traced GPU time per batch by the published batch time, OpenNN keeps the card
in kernels about 92% of the inference window (86.6 µs of a 94.2 µs batch)
against PyTorch's 12.5% (62.4 µs of 498 µs), and about 90% of the training
window (280 µs of GPU work in a 311 µs batch) against 9.4% (252 µs of
2,671 µs). A card that is busy seven to nine times as often draws twice the
power per second and a fraction of the power per sample. It also means most
of PyTorch's board reading in these two cells is the card powered on and
waiting, which is what the 54.8 W and 52.4 W means are: an idle-ish GPU
attached to a busy host. At `93cc90e07`, when OpenNN was also eager, its
busy share was 32%, its board power 69 W and its energy margins 1.38× and
2.16×; the capture is what moved all three.

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
`cuda-lstm-infer` (294 against 442) and 60 of the 196 MiB gap in
`cuda-lstm-train` (316 against 512). Inside OpenNN's own 316 MiB, cuDNN's
scratch is 24 MiB (`workspace=17254656 B, reserve=7864576 B` in the
descriptor dump), on a network of 73,857 parameters — under 300 KB of
weights even at fp32. The library's own attribution (`OPENNN_MEMORY_DEBUG=1`)
accounts for under 30 MiB of buffers in either cell; the rest of both
engines' figures is the CUDA context and the loaded kernel images, which is
why the two inference numbers — 294 and 442 — sit so close to what an empty
process holding cuDNN and cuBLAS costs. Launch to launch, OpenNN reads
293.6 MiB in all three inference launches and 315.6 in all three training
launches; PyTorch 441.6 and 511.6. The tables publish the highest of each.

On CPU the OpenNN figures fell by 33 MiB this round in both cells (189 →
156 and 242 → 210) for a reason that has nothing to do with the LSTM: the
library's device backend created its CUDA streams and cuBLASLt/cuDNN handles
in its constructor, and the CPU GEMM path reaches that singleton for its
thread pool, so a CPU-only process was holding a 226 MiB CUDA context on the
GPU and the driver's host-side state for it in its own resident set.
`e76425bd3` creates the CUDA resources on first CUDA use; a CPU process no
longer maps `/dev/nvidia*` at all. Nothing else in the 2.93× and 2.82×
decomposes; the caveat below says why the instrument cannot.

### `cuda-lstm-infer`, 5.287×: one graph launch per batch against sixteen kernels from Python

A batch of 256 windows × 24 steps × 128 units is small. Per batch at the
published rates: OpenNN 94.2 µs of wall time, PyTorch 498 µs. The traced
runs, `nsys --cuda-graph-trace=node` at `e76425bd3` with 10 passes instead
of 500, show where that goes:

| | OpenNN | PyTorch |
|---|---|---|
| published / under `nsys` (samples/s) | 2,716,548 / 2,686,631 | 513,813 / 393,846 |
| whole trace | 0.356 s | 1.630 s |
| kernel launches in the trace | 96,034 (269,728/s) | 30,137 (18,485/s) |
| GPU busy with kernels, traced | 45.8% | 7.2% |
| GPU busy with kernels, published batch | 92% | 12.5% |
| gaps between kernels, median / p90 | 0.2 / 0.3 µs | 15.7 / 66.6 µs |
| how the kernels are issued | one `cudaGraphLaunch` per batch | 16 launches from the interpreter |

The 0.2 µs gaps are graph-node latency; at `93cc90e07`, on the eager path,
the same kernels were 5.0 µs apart at the median and 10.1 at the 90th
percentile, and the traced GPU busy share was 19.1%. (Under the profiler the
graph replay itself is slowed — 2.69M against 2.72M published — while
PyTorch's eager path is slowed far more, 394k against 514k, because every
launch is traced; the published batch figures in the table are computed
from the published rates and the traced per-batch kernel time.)

OpenNN, top kernels over the traced run:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 49.3% | 43,309 | 1.85 | `nvjet_sm120_tst_mma_64x32x64_8_16x32x64_tmaAB_alignCD4_bz_TNNN` |
| 36.8% | 45,192 | 1.33 | `elemWiseRNNcell<__nv_bfloat16, __nv_bfloat16, float, 2, 1>` |
| 10.0% | 1,883 | 8.63 | `cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>` |
| 1.9% | 1,883 | 1.66 | `transpose_padded_batch_time_kernel<__nv_bfloat16, 0>` |
| 1.2% | 1,883 | 1.01 | `linear_forward_single_output_kernel<__nv_bfloat16>` |
| 0.9% | 1,883 | 0.76 | `time_slice_kernel<__nv_bfloat16, 1>` |

PyTorch, top kernels over the traced run:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 65.1% | 1,882 | 38.14 | `RNN_blockPersist_fp_LSTM_HMMA<__half, __half, float, 1, 128>` |
| 14.2% | 1,882 | 8.28 | `cutlass::Kernel2<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_tn_align1>` |
| 8.5% | 13,174 | 0.71 | `at::native::vectorized_elementwise_kernel<4, float16_copy_kernel_cuda…>` |
| 3.1% | 1,882 | 1.84 | `at::native::unrolled_elementwise_kernel<direct_copy_kernel_cuda…>` |
| 2.7% | 1,882 | 1.59 | `at::native::elementwise_kernel<128, 4, …direct_copy…>` |
| 2.3% | 1,882 | 1.34 | `internal::gemvx::kernel<…__nv_bfloat16…>` |
| 1.7% | 3,765 | 0.49 | `at::native::vectorized_elementwise_kernel<4, FillFunctor<float>>` |

Per batch, GPU time by kernel class (kernels grouped by name; 1,883 traced
batches for OpenNN, 1,882 for PyTorch):

| kernel class | OpenNN launches / batch | OpenNN µs / batch | PyTorch launches / batch | PyTorch µs / batch |
|---|---|---|---|---|
| recurrent | 24.0 | 31.8 | 1.0 | 38.1 |
| gemm | 24.0 | 51.3 | 1.0 | 8.3 |
| elementwise | 0.0 | 0.0 | 13.0 | 10.8 |
| other (OpenNN's own) | 3.0 | 3.4 | 1.0 | 1.3 |
| copy / memset | 0.0 | 0.0 | 0.0 | 3.9 |
| **total** | **51.0** | **86.6** | **16.0** | **62.4** |

**PyTorch's recurrence is still the better one.** Its single
`RNN_blockPersist_fp_LSTM_HMMA` launch runs all 24 steps with the weights
held in registers; adding its 8.28 µs input-projection GEMM, its LSTM layer
costs 46.4 µs of GPU time per batch. OpenNN's `STANDARD` path costs 83.1 µs
for the same layer — 24 `elemWiseRNNcell` launches of 1.33 µs, one per time
step, 23 recurrent GEMMs of 1.85 µs, and its own 8.63 µs input projection.
(Twenty-three, not twenty-four: `cudnn_rnn_forward_` passes `nullptr` for
the initial hidden and cell states, which cuDNN treats as zeros, and the
first step's recurrent GEMM does not appear.) PyTorch does the entire batch
in 62.4 µs of GPU time against OpenNN's 86.6.

What decides the cell is that OpenNN's 86.6 µs of kernels are the batch:
the graph executor issues the 51 nodes back to back, the GPU is busy about
92% of the published batch time, and the remaining 8 µs is the graph launch
and the pass-end synchronisation. PyTorch's 62.4 µs sit inside 498 µs of
interpreter and dispatcher time — sixteen launches, thirteen of them
element-wise and seven of those `float16_copy_kernel_cuda`, the casts that
follow from running `nn.LSTM` under `torch.autocast` over an fp32 model (see
the caveats). Its `compile:reduce-overhead` mode, which would capture the
batch, measured slower than eager in the `2026-09-02-variants` artifacts
because Dynamo breaks the graph at the cuDNN RNN call; so the published
PyTorch is eager, at its measured best, and the 5.287× is the ratio of a
GPU-bound batch to a host-bound one.

### `cuda-lstm-train`, 8.590×: the whole step is one graph

Training is the same forward, cuDNN's two backward calls (data, then
weights), the output layer's backward, the loss and the Adam update. Per
batch at the published rates: OpenNN 311 µs of wall time, PyTorch 2,671 µs,
with 280 µs and 252 µs of GPU work in them respectively.

| | OpenNN | PyTorch |
|---|---|---|
| published / under `nsys` (samples/s) | 823,255 / 794,683 | 95,842 / 78,689 |
| kernel launches per batch | 119.8 | 58.0 |
| GPU time per batch, traced | 279.9 µs | 251.8 µs |
| GPU busy with kernels, published batch | 90% | 9.4% |
| gaps between kernels, median / p90 | 0.2 / 0.2 µs | 25.1 / 115.3 µs |

PyTorch's row is the `93cc90e07` trace: its path did not change between the
two commits (the same eager step, the same kernels, 104,815 → 95,842
samples/s across the two publish runs, inside its own band), and it was not
re-traced. OpenNN's is a fresh trace at `e76425bd3`; its kernel mix is the
one the previous document listed, at the same per-batch cost — 283 µs then,
280 now — with the gaps between kernels gone from 9.3 / 12.6 µs to 0.2 / 0.2.

OpenNN, top kernels over the traced run:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 20.7% | 19,849 | 2.40 | `nvjet_sm120_tst_mma_16x64x128_4_16x16x128_tmaAB_alignCD4_bz_NNNN` |
| 15.9% | 19,849 | 1.84 | `nvjet_sm120_tst_mma_64x32x64_8_16x32x64_tmaAB_alignCD4_bz_TNNN` |
| 13.7% | 20,880 | 1.51 | `elemWiseRNNcell<__nv_bfloat16, __nv_bfloat16, float, 2, 1>` |
| 12.5% | 20,880 | 1.37 | `LSTM_elementWise_bp1<__nv_bfloat16, __nv_bfloat16, float>` |
| 9.5% | 870 | 25.17 | `GENERIC_elementWise_bp2<__nv_bfloat16, __nv_bfloat16, float, 4, 1>` |
| 5.1% | 863 | 13.53 | `nvjet_sm120_tst_mma_64x64x64_6_32x16x64_tmaAB_alignCD4_splitK_NTNN` |
| 3.7% | 863 | 9.73 | `nvjet_sm120_tst_mma_16x64x64_9_16x16x64_tmaAB_alignCD4_splitK_NTNN` |
| 3.6% | 886 | 9.28 | `magma_sgemmEx_kernel<float, __nv_bfloat16, float, …>` |
| 3.3% | 863 | 8.65 | `cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>` |
| 2.8% | 2,048 | 3.09 | `cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, …>` |

Per batch, GPU time by kernel class, OpenNN over the whole trace (870
batches, warmup included) against PyTorch's `93cc90e07` trace:

| kernel class | OpenNN launches / batch | OpenNN µs / batch | PyTorch launches / batch | PyTorch µs / batch |
|---|---|---|---|---|
| recurrent | 51.0 | 100.5 | 4.0 | 110.5 |
| gemm | 52.1 | 146.3 | 10.0 | 44.1 |
| optimizer | 2.0 | 2.6 | 7.0 | 57.8 |
| elementwise | 0.0 | 0.0 | 35.0 | 36.8 |
| reduction | 2.4 | 7.3 | 2.0 | 2.5 |
| other (OpenNN's own) | 7.1 | 7.2 | 0.0 | 0.0 |
| copy / memset | 5.3 | 16.0 | 0.0 | 0.0 |
| **total** | **119.8** | **279.9** | **58.0** | **251.8** |

The two engines spend almost the same GPU time per batch — 280 µs against
252, PyTorch again the cheaper — and OpenNN gets 8.590× the throughput out
of it. The whole of the difference is what surrounds the kernels. OpenNN's
step is one captured graph: the 120 nodes replay 0.2 µs apart, the host
does nothing per batch but launch the graph and fill the next window batch,
and the GPU is busy about 90% of the 311 µs. PyTorch issues 58 launches
from Python, and each of its gaps is the interpreter and the dispatcher:
`zero_grad`, the forward under autocast, `backward()` through autograd (the
cuDNN backward is one node, but every cast, the `Linear`, the loss and the
bias sum are separate ones), then `Adam.step()` — 2,420 µs of host time
around 252 µs of kernels.

Two structural differences are worth naming because they are not launch
overhead.

**The optimiser.** OpenNN's Adam update runs over one contiguous gradient
buffer — the LSTM's weight regions and the dense layer's two are views into
it — as `adam_prepare_kernel` plus `adam_update_kernel`, two launches
totalling 2.6 µs per batch, 0.9% of its GPU time. PyTorch's `Adam` at its
defaults selects the *foreach* implementation: seven
`multi_tensor_apply_kernel` launches per batch, 57.8 µs, **23.0% of
PyTorch's GPU time — more than its LSTM backward kernel** (18.2%).
Updating 73,857 parameters cannot cost 57.8 µs of arithmetic; this is
per-tensor latency, and it is a PyTorch default the driver did not override
(see the caveats).

**The weight-space traffic.** cuDNN wants its own packed weight layout, so
OpenNN moves between that and its gradient arena twice per batch:
`rnn_copy_regions_kernel`, two launches per batch. Add the per-batch upload
of the window batch and the memsets cuDNN needs, and OpenNN pays 16.0 µs per
batch in copies and memsets where PyTorch pays none. Both are costs, and
both are already inside the 8.590×.

`torch.compile` cannot close the gap — it was measured slower, twice, in the
artifacts and in the driver's own sweep — and it is the only route by which
PyTorch's step could be captured. So the cell measures a captured step
against an eager one over comparable amounts of GPU work; the 8.6× is the
cost of issuing 58 launches from Python, and it would shrink to the ratio
of the two kernel budgets, about 0.9×, on a PyTorch path that captured.

### `cpu-lstm-infer`, 1.158×: one primitive, two runtimes

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

PyTorch's batch is 3.696 ms — 0.512 ms longer, and this round it was
profiled (`torch.profiler`, CPU activities, the driver's own model replaying
its resident window, 200 batches after 20 warm; 3.683 ms of self CPU time
per batch, within 0.4% of the published batch): `aten::mkldnn_rnn_layer`
3.329 ms, 90.4%, against OpenNN's 3.124 ms in the same primitive; two
`aten::copy_` per batch, 0.186 ms; `aten::cat`, `aten::fill_`,
`aten::lstm` and the output `aten::addmm` together 0.10 ms. So 0.2 ms of the
half-millisecond is inside the primitive as PyTorch's oneDNN build reaches
it (a different build from OpenNN's, see the caveats, so not a statement
about either engine's call), and 0.3 ms is the copies and concatenations
`nn.LSTM` wraps around it, which OpenNN does not make.

Two runtime effects were found while this cell was being tuned; both are now
neutralised for both engines by the runner, so neither is inside the
published 1.158×. GCC 14's libgomp — the system runtime OpenNN links —
sets its spin count to 1 on hybrid CPUs, so its workers futex-sleep between
parallel regions where PyTorch's bundled libgomp spins: same primitive, same
descriptor, 3.45 ms against 3.14 ms per batch, which the runner removes by
setting `GOMP_SPINCOUNT=300000` for both (PROTOCOL §7). And libgomp re-sizes
its pool to every region's team, so `omp_set_dynamic(1)` (team = CPUs minus
the fifteen-minute load average) and MKL's own heuristic for the 128 → 1
output layer (10 threads for a small GEMM) were each shrinking the team
between oneDNN's 16-thread regions, at six `pthread_create` calls per batch
and 10% of the forward pass; both are fixed in `Backend::set_threads_number`
(`device_backend.cpp:1239`, dynamic off and `mkl_set_dynamic(0)`).

Both drivers print `flush_denormals=on`.

### `cpu-lstm-train`, 1.774×: measured, not attributed

Training runs oneDNN's LSTM forward (training mode, with its workspace) and
backward primitives on both sides. OpenNN's batch is 11.01 ms against
PyTorch's 19.54 ms, a difference of 8.5 ms.

Both sides were profiled this round, on the P-cores with the harness's
`GOMP_SPINCOUNT`. OpenNN under `OPENNN_PROFILE=1` (three epochs, the last
one's table, 171 batches of 256; the profiled epoch ran at 22,891 windows/s
against 23,251 published, so the instrument costs about 1.5%):

| section | ms / batch | share of the 11.5 ms batch |
|---|---|---|
| `step:bwd_total` | 7.497 | 65.2% |
| — `rnn:onednn_backward` (the primitive) | 6.222 | 54.4% |
| — the rest of `bwd:LongShortTermMemory` (unpacking the packed gradients, transposes) | 1.195 | 10.4% |
| `step:fwd_total` | 3.711 | 32.3% |
| — `rnn:onednn_forward` | 3.496 | 30.6% |
| — packing and reordering the weights, transposing the input | 0.118 | 1.0% |
| `step:fill` (the window batch, gathered on the host) | 0.238 | 2.1% |
| `step:optim_total` (Adam over the 73,857 parameters) | 0.039 | 0.3% |
| the dense output layer, forward and backward | 0.051 | 0.4% |
| scaling, unscaling, clamping, loss | 0.005 | 0.0% |

PyTorch under `torch.profiler` (CPU activities, the driver's own model and
step, 200 steps after 20 warm; 20.86 ms of self CPU time per batch against
the 19.54 ms published batch, the difference being the profiler):

| op | ms / batch | share |
|---|---|---|
| `aten::mkldnn_rnn_layer_backward` | 10.083 | 48.3% |
| `aten::mkldnn_rnn_layer` (forward) | 5.112 | 24.5% |
| `autograd::engine::evaluate_function: MkldnnRnnLayerBackward0` (self time, the engine around that node) | 3.007 | 14.4% |
| `Optimizer.step#Adam.step` plus its *foreach* element-wise kernels | 0.81 | 3.9% |
| `aten::fill_`, `aten::copy_`, `aten::_to_copy`, `aten::empty` | 0.94 | 4.5% |
| `aten::mm` (the output layer) and `aten::cat` | 0.27 | 1.3% |

So the 8.5 ms is, in order: about 3.9 ms inside the backward primitive
(10.08 against 6.22 — the same oneDNN LSTM backward, called through
PyTorch's `mkldnn_rnn_layer_backward` operator with whatever that operator
does around the primitive counted in its self time); about 1.6 ms inside the
forward primitive (5.11 against 3.50); 3.0 ms that PyTorch's autograd engine
spends on the backward node itself, outside the primitive, which has no
counterpart in OpenNN's 1.2 ms of gradient unpacking; and about 1.5 ms of
fills, copies and the *foreach* Adam against OpenNN's 0.3 ms of fill and
update. The two engines link different oneDNN builds (the caveats), so the
first two figures are "the primitive as each engine reaches it", not a
statement that OpenNN calls it better; the last two are framework, and are
OpenNN's to claim.

OpenNN's extra layers — the scaling layer in front of the LSTM and the
unscaling and clamping layers behind the output, which the PyTorch network
does not carry — are 0.005 ms of the batch, measured, and count against it.

## Asymmetries and caveats

- **PyTorch reaches the better recurrent kernel on the GPU and OpenNN does
  not.** PyTorch's `nn.LSTM` gets cuDNN's persistent LSTM kernel; OpenNN's
  own persistent path is gated on `!bf16` and this cell is bf16, so it calls
  `cudnnRNNForward` with `algo=STANDARD` and gets the unrolled path — 83.1
  µs of GPU time for the layer against PyTorch's 46.4, and 86.6 against 62.4
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
  treat some part of `cuda-lstm-train`'s 8.590× as a PyTorch default this
  suite did not override rather than as an OpenNN result — though the
  section above puts a bound on it: PyTorch's whole kernel budget per batch
  is 252 µs of a 2,671 µs batch, so the 57.8 µs of Adam is 2% of the cell. The per-batch
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
  copy per batch — 16.0 µs of copies and memsets on the timeline of a 311 µs
  batch, about 5% of the cell now that the batch is GPU-bound. This cuts
  both ways and both ways are in the published numbers: roughly 60 MiB of
  the 196 MiB memory gap in `cuda-lstm-train` is that resident tensor rather
  than anything OpenNN does better.
- **The same 60 MiB inflates OpenNN's inference memory win.** The PyTorch
  inference driver also uploads the complete window tensor and then reads
  only its first 256 rows, because the inference cells replay one resident
  batch. About 60 MiB of the 148 MiB gap in `cuda-lstm-infer` (294 against
  442 MiB) is therefore a tensor our driver told PyTorch to allocate and
  never used. The comparison would be closer if the driver sliced before
  uploading.
- **`cuda-lstm-train` was the least repeatable cell in the matrix; captured,
  it is ordinary.** At `93cc90e07`, eager, two publish runs at the same
  commit read 3.016× and 2.775× and OpenNN's launches spanned 3%, because a
  host-bound batch moves with everything else the host is doing. At
  `e76425bd3` the published run's OpenNN launches read 828,105, 823,255 and
  811,863 — a 2.0% band — and PyTorch's 95,839, 95,842 and 96,827, 1.0%; the
  cell reads between 8.4× and 8.6× on any pairing of them. The inference
  cell is tighter still: 2,716,530–2,716,629 on OpenNN's side (0.004%)
  against 508,239–524,147 on PyTorch's, 5.2×–5.3×. The eager-era artifacts
  (`cuda-lstm-train-publish-20260905T110036Z`, `…20260906T062439Z`, and the
  `adam0`/`adam1` single launches, three of which fail the quiet gate) are
  kept in the store and describe a path the library no longer takes.
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
- **OpenNN replays a CUDA graph on this family; PyTorch does not.** PyTorch
  runs eager: `reduce-overhead` compilation measured 89,518 samples/s
  training and 462,572 inference in the `2026-09-02-variants` artifacts,
  against eager bands of 95,821–104,590 and 522,694–535,766 (the driver's
  own sweep gives the same result), so eager is its best mode and the one
  published. OpenNN's capture failed at `93cc90e07` with cuDNN status 4000
  and succeeds from `8cb810339`, where the backend's lane-0 stream became
  non-blocking like every other stream; both CUDA artifacts at `e76425bd3`
  record `cuda_graph=captured`. The margin is therefore between a captured
  step and an eager one, and the section above says what it would be on a
  captured PyTorch: PyTorch's kernels per batch are the cheaper ones.
- **Different oneDNN builds on CPU, and neither the versions nor the thread
  counts are equalised.** OpenNN links its own oneDNN with the OpenMP
  runtime and PyTorch's wheel bundles a different build; the versions were
  not recorded in this session's artifacts. Both CPU cells run the same
  oneDNN LSTM primitive with the same descriptor (checked with
  `DNNL_VERBOSE` when the CPU inference cell was being tuned) under the same
  `GOMP_SPINCOUNT`, which the runner sets for both (PROTOCOL §7). But the
  runner sets **no thread count**: every CPU artifact records
  `pinning.threads = "engine default"`, OpenNN derives 16 from the `taskset`
  mask and PyTorch computes its own default from the machine, not from the
  affinity mask, and no artifact records what that resolves to. So the two
  engines may not run the same number of threads, and the descriptors are
  not claimed to match on `nthr`. With no measurement of PyTorch's per-call
  primitive time here either, a version difference and a thread-count
  difference are both inside the 1.158× and the 1.774× and this document
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
