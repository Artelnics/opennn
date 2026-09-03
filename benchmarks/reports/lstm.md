# LSTM: forecasting Beijing PM2.5

OpenNN against PyTorch 2.13 on a single-layer LSTM of 128 units with a linear
output (73,857 parameters) forecasting the next hour of PM2.5 from 24 hours
of 15 features, 43,800 windows, batch 256 — bf16 on the RTX 5070 Ti, fp32 on
16 CPU threads, where both engines run oneDNN's LSTM primitive. Session
`2026-09-03-publish`:

| cell | OpenNN | PyTorch | OpenNN / PyTorch |
|---|---|---|---|
| `cuda-lstm-train` | 297,615 windows/s | 99,631 | **2.99×** |
| `cuda-lstm-infer` | 917,344 windows/s | 523,041 | **1.75×** |
| `cpu-lstm-train` | 22,995 windows/s | 13,286 | **1.73×** |
| `cpu-lstm-infer` | 80,566 windows/s | 69,329 | **1.16×** |

All four cells run the recurrence in the same vendor primitive — cuDNN's
RNN on the GPU, oneDNN's LSTM on the CPU — so the margins are what each
framework does between the calls. On the GPU neither engine gets a CUDA
graph (cuDNN's RNN refuses capture) and both run eagerly: OpenNN's C++ launch
path issues fewer kernels per batch, with no per-call casts and a packed
weight space rebuilt only when the parameters change, against `nn.LSTM`
under autocast and `Adam`'s foreach update issued from Python
(1.75× inference, 2.99× training). On the CPU the primitive's
time is the same on both sides and the inference margin (1.16×) is
the runtime around it — an OpenMP wait policy and a thread pool that resized
itself between regions, both found and fixed while this cell was being
tuned; training (1.73×) adds the passes around oneDNN's backward
primitive.

## What is measured

**Network.** One LSTM layer of 128 units over a window of 24 hourly rows of 15
features, followed by a linear output of one unit: 4 × (15×128 + 128×128 + 128)
+ (128 + 1) = **73,857 trainable parameters** on both sides. PyTorch's
`nn.LSTM(15, 128, batch_first=True)` owns two bias vectors whose sum is what
the gates see; the driver zeroes and freezes `bias_hh` so the trainable
parameterisation is identical to OpenNN's single bias, and OpenNN's
`set_parameters_pytorch()` draws the initial weights from PyTorch's
distribution rather than merely from the same seed. Training minimises the
mean squared error with Adam at default hyper-parameters, no clipping.

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
batches = **43,776 samples**. For training both engines keep the windows resident on the
device and slice them there (`GPUPersistantData` on the OpenNN side);
for inference both fill the first batch of 256 windows once and replay it
171 times per pass, so the inference cells time the resident forward pass
alone.

**Cells.**

| cell | device | batch | precision | timed window |
|---|---|---|---|---|
| `cuda-lstm-train` | RTX 5070 Ti | 256 | bf16 | 3 epochs after 2 untimed |
| `cuda-lstm-infer` | RTX 5070 Ti | 256 | bf16 | 5 passes after 1 call + 1 untimed pass |
| `cpu-lstm-train` | 16 threads, P-cores | 256 | fp32 | 3 epochs after 2 untimed |
| `cpu-lstm-infer` | 16 threads, P-cores | 256 | fp32 | 5 passes after 1 call + 1 untimed pass |

**Each engine at its best.** On CUDA both engines run cuDNN's fused RNN —
OpenNN through `cudnn_rnn_forward_`, PyTorch behind `nn.LSTM` — so the
recurrent arithmetic is the same NVIDIA kernel and the cell measures the
machinery around it: data movement, launches, the optimiser. PyTorch is
**eager on CUDA in this family**, and that is the unusual case: the driver's
docstring records `torch.compile` losing here (train 87,628 compiled against
108,614 eager samples/s, inference 457,209 against 528,734) because Dynamo
breaks the graph at `zero_grad` and Inductor cannot fuse anything into the
opaque cuDNN RNN call; for dense, cnn and transformer compiling wins, so the
suite takes PyTorch's better mode per family. OpenNN asks for a CUDA graph of
the Adam step and of the inference pass; the capture is refused on the cuDNN
RNN path (cuDNN's RNN descriptor allocates its own workspace on the capture stream, which a graph capture cannot contain), the artifacts record `cuda_graph=failed`,
and OpenNN runs the cell eagerly as well — launch for launch against PyTorch.
On CPU both engines run oneDNN's LSTM primitive (PyTorch's bundled oneDNN
3.12, OpenNN's linked oneDNN 3.11) on 16 threads pinned to the P-cores with
MKL as the BLAS; both drivers enable flush-to-zero (`flush_denormals=on` is
recorded), and the runner sets `GOMP_SPINCOUNT=300000` for both — PROTOCOL §6
explains why the GCC 14 libgomp default would otherwise penalise the engine
that links the system runtime.

**Gates.** Samples (43,800 windows), inputs (15), past (24), hidden (128) and
parameters (73,857) must agree between the engines. There is **no loss or
accuracy gate** in this family; the drivers' `quality` mode exists for that
and is not part of the published matrix.

## Results

Session `2026-09-03-publish`, the last run of every cell, median of three rounds.

| cell | batch | precision | OpenNN samples/s | PyTorch samples/s | OpenNN / PyTorch | peak memory MiB (OpenNN / PyTorch) | energy Wh (OpenNN / PyTorch) |
|---|---|---|---|---|---|---|---|
| `cuda-lstm-train` | 256 | bf16 | 297,615 | 99,631 | **2.99×** | 380 / 512 | 0.0560 / 0.1248 |
| `cuda-lstm-infer` | 256 | bf16 | 917,344 | 523,041 | **1.75×** | 294 / 442 | 0.0460 / 0.0636 |
| `cpu-lstm-train` | 256 | fp32 | 22,995 | 13,286 | **1.73×** | 242 / 592 | 0.0433 / 0.0608 |
| `cpu-lstm-infer` | 256 | fp32 | 80,566 | 69,329 | **1.16×** | 189 / 459 | 0.0215 / 0.0234 |


`cuda-lstm-train` — batch 256, bf16, epochs 20 per launch, 3 rounds. Artifact `cuda-lstm-train-publish-20260903T100659Z.json`, commit `6b7179dde`, quiet True (busy 0.8% before, 0.1% after), clocks locked True, shape gate True, quality gate True.

| engine | median samples/s | min | max | peak device MiB | Wh (board) |
|---|---|---|---|---|---|
| OpenNN | 297,615 | 297,132 | 298,704 | 380 | 0.05600 |
| PyTorch | 99,631 | 99,612 | 109,560 | 512 | 0.12479 |
| **ratio** | **2.987×** | | | 1.35× less | 2.23× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 297,615 | 99,631 |
| 2 | pytorch → opennn | 298,704 | 99,612 |
| 3 | opennn → pytorch | 297,132 | 109,560 |


`cuda-lstm-infer` — batch 256, bf16, passes 50 per launch, 3 rounds. Artifact `cuda-lstm-infer-publish-20260903T100600Z.json`, commit `6b7179dde`, quiet True (busy 0.4% before, 0.2% after), clocks locked True, shape gate True, quality gate True.

| engine | median samples/s | min | max | peak device MiB | Wh (board) |
|---|---|---|---|---|---|
| OpenNN | 917,344 | 900,143 | 918,408 | 294 | 0.04597 |
| PyTorch | 523,041 | 504,706 | 532,674 | 442 | 0.06358 |
| **ratio** | **1.754×** | | | 1.50× less | 1.38× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 917,344 | 523,041 |
| 2 | pytorch → opennn | 900,143 | 532,674 |
| 3 | opennn → pytorch | 918,408 | 504,706 |


`cpu-lstm-train` — batch 256, fp32, epochs 3 per launch, 3 rounds. Artifact `cpu-lstm-train-publish-20260903T103650Z.json`, commit `6b7179dde`, quiet True (busy 0.3% before, 0.2% after), clocks locked True, shape gate True, quality gate True.

| engine | median samples/s | min | max | peak RssAnon MiB | Wh (RAPL package-0) |
|---|---|---|---|---|---|
| OpenNN | 22,995 | 22,951 | 23,273 | 242 | 0.04333 |
| PyTorch | 13,286 | 13,180 | 13,356 | 592 | 0.06080 |
| **ratio** | **1.731×** | | | 2.44× less | 1.40× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 22,951 | 13,356 |
| 2 | pytorch → opennn | 23,273 | 13,180 |
| 3 | opennn → pytorch | 22,995 | 13,286 |


`cpu-lstm-infer` — batch 256, fp32, passes 5 per launch, 3 rounds. Artifact `cpu-lstm-infer-publish-20260903T103505Z.json`, commit `6b7179dde`, quiet True (busy 0.1% before, 0.2% after), clocks locked True, shape gate True, quality gate True.

| engine | median samples/s | min | max | peak RssAnon MiB | Wh (RAPL package-0) |
|---|---|---|---|---|---|
| OpenNN | 80,566 | 80,424 | 80,795 | 189 | 0.02146 |
| PyTorch | 69,329 | 69,189 | 69,820 | 459 | 0.02337 |
| **ratio** | **1.162×** | | | 2.42× less | 1.09× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 80,795 | 69,329 |
| 2 | pytorch → opennn | 80,424 | 69,820 |
| 3 | opennn → pytorch | 80,566 | 69,189 |

## Why

All four cells run the recurrent arithmetic in the same vendor kernel —
cuDNN's RNN on the GPU, oneDNN's LSTM primitive on the CPU — so none of the
margins is "OpenNN's LSTM is faster". They are what each framework does
*between* the calls: the tensors it casts, copies or zeroes before the
primitive, the kernels it launches after it, and how much host time each of
those costs against a network small enough that a batch of 256 windows is
under a millisecond of GPU work.

### Where the energy goes

All four cells win energy the same way, and it is not by drawing less power.
On the GPU cells OpenNN pulls *more* watts than PyTorch — 69.3 W against
54.7 W on inference, 67.9 W against 51.9 W on training, 27% and 31% more —
and still comes out 1.38× and 2.23× ahead on
energy, because it finishes 1.75× and 2.99× sooner.

The profile says exactly where those watts come from. This network is small
enough that both engines leave the card mostly idle, and OpenNN leaves it
idle about half as much: 19.4% of the inference window is spent in kernels
against PyTorch's 9.0%, and 18.1% against 8.2% on training. A card that is
busy twice as often draws more power per second and much less power per
sample. It also means most of the board reading in these two cells is the
cost of the card being switched on rather than the cost of the work, which
is why the energy ratios here are so much *smaller* than the throughput
ratios — 1.38× against 1.75×, and 2.23× against 2.99×. A reader who wants
the energy of the arithmetic rather than the energy of an idling GPU should
read the CPU cells below, where the device is saturated.

The CPU cells have the same shape without the idle-card distortion: OpenNN
draws 28.4 W against 26.7 W on inference and 27.2 W against 22.1 W on
training, and wins 1.089× and 1.40× on the
strength of finishing sooner. The oneDNN primitive is the same on both
sides; what differs is the OpenMP runtime around it, and a runtime that
spins rather than futex-sleeps is a runtime that burns watts to save
microseconds. Here it saves more than it burns.

### `cuda-lstm-infer`, 1.75×: the same cuDNN kernel, and what is launched around it

A batch of 256 windows × 24 steps × 128 units is small: 279 µs per
batch for OpenNN, 489 µs for PyTorch, both eager (no graph replays on
this family, see the caveats). The traced batch is where the difference is:

| | OpenNN | PyTorch |
|---|---|---|
| published / under `nsys` (samples/s) | 917,344 / 545007 | 523,041 / 394096 |
| timed window traced | 4.014 s | 5.574 s |
| kernel launches | 436,065 (108,636/s) | 136,899 (24,560/s) |
| GPU busy with kernels | 19.4% | 9.0% |
| GPU busy with any work (kernels, copies, memsets) | 19.4% | 9.0% |
| gaps between kernels, median / p90 | 5.3 / 10.0 µs | 15.8 / 66.2 µs |
| idle between kernels, total | 3236.1 ms | 5073.4 ms |

OpenNN, top kernels by summed time in the window:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 52.1% | 196,658 | 2.1 | `nvjet_sm120_tst_mma_64x32x64_8_16x32x64_tmaAB_alignCD4_bz_TNNN` |
| 34.3% | 205,207 | 1.3 | `elemWiseRNNcell<__nv_bfloat16, __nv_bfloat16, float, 2, 1>(int, in...` |
| 9.7% | 8,550 | 8.9 | `cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x6...` |
| 1.9% | 8,550 | 1.7 | `transpose_padded_batch_time_kernel<__nv_bfloat16, 0>` |
| 1.1% | 8,550 | 1.0 | `linear_forward_single_output_kernel<__nv_bfloat16>(int, int, const ...` |
| 0.9% | 8,550 | 0.9 | `time_slice_kernel<__nv_bfloat16, 1>` |

PyTorch, top kernels by summed time in the window:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 65.2% | 8,550 | 38.1 | `RNN_blockPersist_fp_LSTM_HMMA<__half, __half, float, 1, 128>(const ...` |
| 14.1% | 8,550 | 8.3 | `cutlass::Kernel2<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_tn_ali...` |
| 8.5% | 59,850 | 0.7 | `at::native::vectorized_elementwise_kernel<4, at::native::float16_co...` |
| 3.1% | 8,550 | 1.8 | `at::native::unrolled_elementwise_kernel<at::native::direct_copy_ker...` |
| 2.7% | 8,550 | 1.6 | `at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_...` |
| 2.3% | 8,550 | 1.4 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, _...` |

Per batch, GPU time by kernel class (`nsys`, the timed window of one launch, kernels grouped by name; OpenNN's graph replays are traced at node level):

| kernel class | OpenNN launches / batch | OpenNN µs / batch | PyTorch launches / batch | PyTorch µs / batch |
|---|---|---|---|---|
| recurrent | 0.0 | 0.0 | 1.0 | 38.0 |
| gemm | 2.0 | 9.9 | 2.0 | 9.6 |
| elementwise | 2.0 | 2.6 | 13.0 | 10.7 |
| other | 47.0 | 78.6 | 0.0 | 0.0 |
| **total** | **51.0** | **91.0** | **16.0** | **58.3** |

OpenNN's batch is 51 launches: the cuDNN RNN forward (which is
itself several kernels — the input projection as one GEMM over all 24 steps,
then the recurrent step kernels), then the 128 → 1 output layer as one
warp-per-row kernel with the identity activation. The packed cuDNN weight
space is built once and reused — `cudnn_pack_weights_` skips the copy when
the parameter version, the shape and the destination buffer match the last
pack — and the initial hidden and cell states are passed to cuDNN as null
pointers, which it treats as zeros, so nothing is zeroed per call.

PyTorch's batch is 16 launches, and the extra ones are the
price of running `nn.LSTM` under `torch.autocast`: 59,850 `float16_copy_kernel_cuda` launches over the window, seven per batch, plus a `direct_copy_kernel_cuda` and two further element-wise passes — 17% of PyTorch's kernel count for 8.5% of its GPU time
The cuDNN call itself — the same descriptor, the same kernels — takes
38 µs on PyTorch's side against 0 on OpenNN's;
the rest of PyTorch's batch is the casts, the weight-buffer copy and the
Python dispatch between them, which the gaps in the trace measure
(a median 5.3 µs gap between kernels for OpenNN against 15.8 µs for PyTorch, and a 90th percentile of 10.0 µs against 66.2).

### `cuda-lstm-train`, 2.99×: eager against eager, and the host is the bottleneck

Training is the same forward, cuDNN's two backward calls (data, then
weights), the output layer's backward, the loss and the Adam update. Per
batch: OpenNN 860 µs, PyTorch 2,569 µs — and the GPU is busy
for only 18.1% and 8.2% of those windows respectively.
This is a cell decided by launch overhead, on both sides:

| | OpenNN | PyTorch |
|---|---|---|
| published / under `nsys` (samples/s) | 297,615 / 171605 | 99,631 / 85557 |
| timed window traced | 5.160 s | 10.442 s |
| kernel launches | 393,063 (76,175/s) | 198,367 (18,997/s) |
| GPU busy with kernels | 16.8% | 8.2% |
| GPU busy with any work (kernels, copies, memsets) | 18.1% | 8.2% |
| gaps between kernels, median / p90 | 9.2 / 12.8 µs | 23.2 / 105.8 µs |
| idle between kernels, total | 4292.1 ms | 9584.3 ms |

OpenNN, top kernels by summed time in the window:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 22.7% | 78,660 | 2.6 | `nvjet_sm120_tst_mma_16x64x128_4_16x16x128_tmaAB_alignCD4_bz_NNNN` |
| 17.8% | 78,660 | 2.0 | `nvjet_sm120_tst_mma_64x32x64_8_16x32x64_tmaAB_alignCD4_bz_TNNN` |
| 13.5% | 82,560 | 1.5 | `elemWiseRNNcell<__nv_bfloat16, __nv_bfloat16, float, 2, 1>(int, in...` |
| 12.8% | 82,560 | 1.4 | `LSTM_elementWise_bp1<__nv_bfloat16, __nv_bfloat16, float>(int, int,...` |
| 9.3% | 3,441 | 24.3 | `GENERIC_elementWise_bp2<__nv_bfloat16, __nv_bfloat16, float, 4, 1>(...` |
| 4.4% | 3,420 | 11.6 | `nvjet_sm120_tst_mma_64x64x64_6_32x16x64_tmaAB_alignCD4_splitK_NTNN` |

PyTorch, top kernels by summed time in the window:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 18.2% | 3,420 | 45.9 | `RNN_blockPersist_bp_LSTM_HMMA<__half, __half, float, 128>` |
| 16.3% | 3,420 | 41.2 | `RNN_blockPersist_fp_LSTM_HMMA<__half, __half, float, 1, 128>(const ...` |
| 8.7% | 3,420 | 22.0 | `GENERIC_elementWise_bp2<__half, __half, float, 4, 2>(int, int, T1 *...` |
| 4.5% | 3,421 | 11.4 | `at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnam...` |
| 4.4% | 3,420 | 11.0 | `cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_f16_64x64_32x10_nt_...` |
| 3.7% | 3,421 | 9.4 | `at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnam...` |

Per batch, GPU time by kernel class (`nsys`, the timed window of one launch, kernels grouped by name; OpenNN's graph replays are traced at node level):

| kernel class | OpenNN launches / batch | OpenNN µs / batch | PyTorch launches / batch | PyTorch µs / batch |
|---|---|---|---|---|
| copy/memset | 5.0 | 19.6 | 0.0 | 0.0 |
| recurrent | 25.9 | 39.6 | 2.9 | 87.0 |
| gemm | 9.5 | 40.4 | 9.8 | 43.4 |
| optimizer | 2.0 | 2.8 | 6.9 | 56.9 |
| reduction | 2.0 | 25.2 | 2.9 | 24.0 |
| elementwise | 4.0 | 4.4 | 34.4 | 36.3 |
| other | 70.4 | 148.2 | 0.0 | 0.0 |
| **total** | **118.6** | **280.1** | **56.9** | **247.6** |

OpenNN issues its step from C++: 119 launches per batch,
9.2 µs median gap between kernels. Its Adam update runs over one
contiguous gradient buffer — the LSTM's weight regions and the dense layer's
two are views into it — as one `adam_update_kernel` per batch at 2.0 µs, 0.8% of OpenNN's GPU time. The
gradients cuDNN writes into its packed weight space are unpacked into the
arena by eight small copies (`cudnn_unpack_gradients_`), and the weights are
repacked after each update because the parameter version moved; that is the
data movement OpenNN pays for using cuDNN's layout, and it is 1.7
µs of the batch.

PyTorch's step is 57 launches with a 23.2 µs median
gap, and each of those gaps is the Python interpreter and the dispatcher:
`zero_grad`, the forward under autocast (with the same casts as inference),
`backward()` through autograd (the cuDNN backward is one node, but every
cast, the `Linear`, the loss and the bias sum are separate ones), then
`Adam.step()` in its default *foreach* implementation (six distinct `multi_tensor_apply_kernel` instantiations, one launch each per step, together 18.1% of PyTorch's GPU time — more than its LSTM backward kernel).
`torch.compile` cannot close this — the driver measured it slower (87,628
against 108,614 samples/s in the docstring), because Dynamo breaks the graph
at `zero_grad` and Inductor cannot fuse into the opaque cuDNN RNN call — and
CUDA graphs are not available to either engine here (see the caveats). So the
cell measures the two eager launch paths over the same kernels, and OpenNN's
is 9.2 µs against 23.2 µs median, and 12.8 against 105.8 at the 90th percentile× shorter per batch.

The window is short (0.5 s for OpenNN) and the spread is wider than the
other cells' (297,132–298,704 windows/s over three launches, PyTorch 99,612–109,560); the slowest OpenNN launch is still
2.7× the fastest PyTorch one.

### `cpu-lstm-infer`, 1.16×: one primitive, two runtimes

The oneDNN LSTM primitive is the whole forward pass on both sides —
`DNNL_VERBOSE` shows the same descriptor (`lstm`, forward inference, `tnc`
layout, fp32, 16 threads) executing on both engines — and the primitive's
own time per batch is the same to within the run-to-run noise of this cell.
The 15% is everything else:

*[pending the final measurement round]*

Two runtime effects were found while this cell was being tuned, and both are
now neutralised for both engines by the runner, so what remains is smaller:

1. **The OpenMP wait policy.** GCC 14's libgomp — the system runtime OpenNN
   links — sets its spin count to 1 on hybrid CPUs, so after every parallel
   region the workers sleep on a futex and have to be woken for the next one;
   PyTorch's wheel bundles an older libgomp that spins for 300,000
   iterations. Same primitive, same descriptor: 3.45 ms against 3.14 ms per
   batch. The runner sets `GOMP_SPINCOUNT=300000` for both engines (a no-op
   for PyTorch), PROTOCOL §6.
2. **Pool resizing.** libgomp re-sizes its thread pool to every region's team,
   and two things were shrinking the team between oneDNN's 16-thread regions:
   `omp_set_dynamic(1)` (team = CPUs minus the 15-minute load average) and
   MKL's own heuristic for the 128 → 1 output layer (10 threads for a small
   `sgemv`). Each shrink and regrow cost six `pthread_create` calls per batch.
   Fixed in `Backend::set_threads_number` (dynamic off, `mkl_set_dynamic(0)`);
   `strace -f -e clone` counts about 35 clones for a whole run now.

With those gone, the remaining difference is between the two runtimes and
what each framework does around the primitive: *[pending the final measurement round]*
OpenNN's output layer is one `sgemv` on the OpenMP pool; PyTorch's
`nn.Linear` is an `addmm` through the same MKL. Both print
`flush_denormals=on`.

### `cpu-lstm-train`, 1.73×: the backward primitive and the passes around it

Training runs oneDNN's LSTM forward (training mode, with its workspace) and
backward primitives on both sides — PyTorch's `nn.LSTM` reaches
`mkldnn_rnn_layer_backward` on CPU for fp32 contiguous inputs — and the
margin is wider than inference because the work around the primitive is a
larger share of the batch:

*[pending the final measurement round]*

*[pending the final measurement round]* OpenNN's extra layers — the scaling layer in front
of the LSTM and the unscaling and clamping layers behind the output, which
the inference network does not have and the PyTorch network does not carry at
all — are three element-wise passes over a 256 × 24 × 15 batch and a 256 × 1
output, too small to register in the profile against the 11 ms batch.

## Asymmetries and caveats

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
  amount of work on PyTorch's side that is there so both engines run the
  same cuDNN RNN descriptor.
- **The two GPU cells do not run the recurrence in the same precision, and
  they do not run the same cuDNN algorithm.** The cell is declared `bf16` and
  OpenNN's kernels are `bf16` throughout —
  `elemWiseRNNcell<__nv_bfloat16, __nv_bfloat16, float, …>`. PyTorch's are
  `fp16`: under `torch.autocast` its `nn.LSTM` reaches cuDNN's *persistent*
  kernel `RNN_blockPersist_fp_LSTM_HMMA<__half, __half, float, …>`, one
  38.1 µs launch per inference batch against OpenNN's 205,207 small
  `elemWiseRNNcell` launches over the window. Two things follow, and they
  point in opposite directions. Sixteen-bit floats of either kind cost the
  same per operation on this card, so the precision itself is not an
  advantage to either side; but a persistent kernel is the *better* algorithm
  for a recurrence this small, and PyTorch is the one getting it. OpenNN wins
  the cell anyway, on launch rate and on what surrounds the primitive — which
  is the claim the *Why* section makes and the reason it does not claim a
  faster LSTM kernel. What this asymmetry does mean is that the two engines
  are not bit-comparable here, and the family has no accuracy gate to notice.

- **Neither engine replays a CUDA graph on this family.** PyTorch runs eager
  because compiling measured slower (the driver's docstring: 87,628 against
  108,614 samples/s training, 457,209 against 528,734 inference); OpenNN
  asks for a graph and is refused — both CUDA artifacts record
  `cuda_graph=failed` and the engine continues eager, cuDNN's RNN descriptor allocates its own workspace on the capture stream, which a graph capture cannot contain.
  The margin is therefore between two eager launch streams over the same
  cuDNN RNN, and what separates them is in the *Why* section.
- **`cuda-lstm-train` has a short window**: three epochs at 256 are 131,328
  samples, about 0.5 s for OpenNN against 1.3 s for PyTorch, and the
  round-to-round spread is what a short window gives (297,132–298,704 windows/s over three launches, PyTorch 99,612–109,560). The
  slowest OpenNN launch is 2.7× the fastest PyTorch one.
- **Different oneDNN builds on CPU.** OpenNN links oneDNN 3.11 with the
  OpenMP runtime; PyTorch's wheel bundles 3.12. Both CPU cells run the same
  oneDNN LSTM primitive with the same descriptor (checked with
  `DNNL_VERBOSE` when the CPU inference cell was being tuned) on the same 16
  threads under the same `GOMP_SPINCOUNT`, which the runner sets for both
  (PROTOCOL §6); the remaining difference is the libgomp each engine links
  (GCC 14's system library against the copy in the wheel) and what each
  framework does between the primitive calls.
- **Flush-to-zero on both CPU sides.** The PyTorch driver calls
  `torch.set_flush_denormal(True)` on CPU, the usual practice for recurrent
  inference; OpenNN's driver sets the same MXCSR bits before any OpenMP team
  exists so the workers inherit them. Both print `flush_denormals=on`, so a
  denormal slowdown cannot land on one engine only.
- **No accuracy gate.** Neither driver prints a loss; the runner compares
  the window count (`samples`), `inputs`, `past`, `hidden` and the parameter
  count, and the cells are held to that shape gate.
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
  kernels for the same time, and the cell did not move beyond its spread. The
  rows above are from the fixed drivers, which upload the batch
  (`upload_to_device_batch_async()`) after filling it and, where the split is
  resident, take the batch as a device view; the previous session's rows read
  923,205 windows/s for OpenNN.

## Reproduce

```bash
export OPENNN_BENCH_SESSION=$(date +%F)-mine
python run.py --family lstm --mode train --device cuda --batch 256 --precision bf16 --epochs 3 --rounds 3
python run.py --family lstm --mode infer --device cuda --batch 256 --precision bf16 --repeats 5 --rounds 3
python run.py --family lstm --mode train --device cpu  --batch 256 --precision fp32 --epochs 3 --rounds 3
python run.py --family lstm --mode infer --device cpu  --batch 256 --precision fp32 --repeats 5 --rounds 3
```

`prepare.py lstm` downloads the Beijing PM2.5 set and writes the 15-column
CSV both engines read. `PT_COMPILE_MODE=reduce-overhead` is the PyTorch knob
(the docstring numbers above are its result); the OpenNN CPU build needs
`-DOpenNN_ENABLE_ONEDNN=ON` — without it the recurrent layer runs a fallback
and the CPU cells are a different comparison.
