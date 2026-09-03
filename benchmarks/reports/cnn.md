# CNN: ResNet-50 on the ImageNet subset

OpenNN against PyTorch 2.13 on ResNet-50 v1.5 (25,557,032 parameters) over
the 50,000-image ILSVRC2012 validation set laid out as 1,000 class folders,
224×224, batch 128, bf16 with TF32 GEMMs, on the RTX 5070 Ti. Session
`2026-09-03-publish`:

| cell | OpenNN | PyTorch | OpenNN / PyTorch |
|---|---|---|---|
| `cuda-cnn-train` | 1,650 samples/s | 1,406 | **1.17×** |
| `cuda-cnn-infer` | 6,942 samples/s | 5,833 | **1.19×** |

Both engines run all 53 convolutions in cuDNN and the convolution time per
batch is close; the margins are the kernels between the convolutions. At
inference OpenNN folds every batch normalisation into the weights of the
convolution in front of it and runs conv → bias → residual → ReLU as one
cuDNN graph, the whole pass replayed as one CUDA graph per batch
(1.19×); in training it fuses the batch-norm forward and backward
with the residual and the ReLU in cuDNN graphs and captures the whole Adam
step (1.17×). PyTorch's Inductor keeps the convolutions as external
cuDNN calls and lowers the normalisations, adds and activations to Triton
passes over activations of up to 25 MB each, issued one by one.

## What is measured

**Network.** ResNet-50 v1.5 — the variant with the stride in the 3×3
convolution — as bottleneck stages [3, 4, 6, 3] over widths [64, 128, 256, 512]
with the full 1000-class head (2048 → 1000). PyTorch uses the library
definition, `torchvision.models.resnet50(weights=None)`, so the comparison is
against the citable network rather than against a transcription of it; OpenNN
builds the same graph from `ResNet(input_shape, {3,4,6,3}, {64,128,256,512},
output_shape, bottleneck=true)`. Both report **25,557,032 parameters**, which the
runner checks. Training uses cross-entropy and Adam with default
hyper-parameters on both sides, no regularisation, no gradient clipping.

**Data.** The pinned ImageNet subset: the ILSVRC2012 validation set laid out as
class folders, 1000 classes × 50 JPEGs = 50,000 images, resized to 224×224×3 and
scaled to [0, 1]. Every cell processes whole batches only, so at batch 128 an
epoch or an inference pass covers 390 batches = **49,920 samples**; the
throughput figure divides that by the median epoch or pass time.

The two engines feed the network differently, and the training cell measures
that difference on purpose. PyTorch decodes the JPEGs on every epoch:
`ImageFolder` → `Resize(224)` → `CenterCrop(224)` → `ToTensor`, in a
`DataLoader` with 8 worker processes (`PT_WORKERS`), pinned memory, persistent
workers and `drop_last`, converted to `channels_last` on the device. OpenNN's
`ImageDataset` decodes each JPEG once (libjpeg, bilinear resize straight to
224×224, no crop) into a pre-decoded cache — `.cache/images.bin`, 7.5 GB of
uint8 HWC pixels in the same sorted-folder/sorted-file order `ImageFolder`
uses, with a signature trailer — and every epoch reads its batches from that
file with `pread`, casts and scales them in an OpenMP team, and stages the
next batch on a prefetch thread (`OPENNN_BATCH_WORKERS`, default 2) while the
current one trains. The cache is built on the first run for a given image
size and is excluded from every timed window; the runner's warmup epochs also
leave it warm in the page cache (31 GB of RAM against 7.5 GB of pixels). This
is the engine's normal training path, not a benchmark-only shortcut, but it is
an asymmetry that favours OpenNN and section "Why" quantifies it with the
`PT_INPUT=cache` variant, which feeds PyTorch from the same file.

**Cells.**

| cell | device | batch | precision | timed window |
|---|---|---|---|---|
| `cuda-cnn-train` | RTX 5070 Ti | 128 | bf16 | 2 epochs after 2 untimed (OpenNN) / 1 untimed (PyTorch) |
| `cuda-cnn-infer` | RTX 5070 Ti | 128 | bf16 | 5 passes after 1 untimed |

The warmups differ by one epoch because the OpenNN driver's first epoch also
captures the CUDA graph and autotunes the convolution plans, while the
PyTorch driver's first epoch also runs `torch.compile`; both warmups are
outside the window, and the median of the timed epochs is insensitive to the
count. Inference fills one batch once and replays it 390 times per pass, so the
inference cell times the resident forward pass alone — the decode cost is
already accounted for by the training cell, and paying it again here would
measure the input pipeline twice.

**Each engine at its best.** bf16 autocast (`torch.autocast`, and OpenNN's
`Configuration::set(CUDA, BF16)`), TF32 allowed, `cudnn.benchmark = True` on
the PyTorch side against cuDNN plan autotuning on the OpenNN side
(`OPENNN_CONV_AUTOTUNE`, on by default, with a 16 MiB workspace cap for
inference that the driver's comment explains: the winning ResNet-50 plans fit
under it and larger candidates only added a cold-start memory peak). PyTorch's
`torch.compile` mode is chosen per cell and each was measured (the driver's
`compiled()` docstring has every mode): the training step runs
`max-autotune-no-cudagraphs`, Inductor with its GEMM and pointwise templates
benchmarked per shape and no CUDA graphs (1,398 samples/s against 1,364 in
the default mode and 1,366 under `reduce-overhead`); the inference forward
runs the default mode (5,604, against 5,597 for `max-autotune-no-cudagraphs`
at 940 MiB more resident memory and 5,574 for `reduce-overhead`), with the
weights stored in bf16 once (`PT_INFER_CAST=weights`, the default: 5,604
against 5,498 under autocast, which re-casts every convolution's weights on
every call). `PT_COMPILE_MODE=default|reduce-overhead|max-autotune-no-cudagraphs|eager`
and `PT_INFER_CAST=autocast` remain available. OpenNN
captures the whole Adam step (forward, loss, backward, update) into one CUDA
graph per batch shape and replays it, and the inference forward pass is a
captured graph as well; `OPENNN_NO_CUDA_GRAPH=1` turns that off for the
controlled comparison. Both engines use tensor cores through NHWC
convolutions: PyTorch by converting the model and inputs to `channels_last`,
OpenNN because its image tensors are HWC natively.

**Gates.** The shape gate checks parameters (25,557,032) and samples per
window (49,920) on both sides. There is **no accuracy gate** on this family:
two epochs of ResNet-50 from scratch on 50 images per class do not produce a
meaningful test accuracy, so the runner only requires both engines to run the
same network over the same number of samples. The `quality` mode of both
drivers exists for longer runs and is not part of the published matrix.

## Results

Session `2026-09-03-publish`, the last run of every cell, median of three rounds.

| cell | batch | precision | OpenNN samples/s | PyTorch samples/s | OpenNN / PyTorch | peak memory MiB (OpenNN / PyTorch) | energy Wh (OpenNN / PyTorch) |
|---|---|---|---|---|---|---|---|
| `cuda-cnn-train` | 64 | bf16 | 1,650 | 1,406 | **1.17×** | 3,608 / 4,208 | 3.8627 / 4.2546 |
| `cuda-cnn-infer` | 64 | bf16 | 6,942 | 5,833 | **1.19×** | 718 / 868 | 2.4425 / 2.9421 |


`cuda-cnn-train` — batch 64, bf16, epochs 2 per launch, 3 rounds. Artifact `cuda-cnn-train-publish-20260903T120534Z.json`, commit `6b7179dde`, quiet True (busy 0.2% before, 0.1% after), clocks locked True, shape gate True, quality gate True.

| engine | median samples/s | min | max | peak device MiB | Wh (board) |
|---|---|---|---|---|---|
| OpenNN | 1,650 | 1,650 | 1,650 | 3608 | 3.86270 |
| PyTorch | 1,406 | 1,405 | 1,406 | 4208 | 4.25465 |
| **ratio** | **1.174×** | | | 1.17× less | 1.10× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 1,650 | 1,406 |
| 2 | pytorch → opennn | 1,650 | 1,405 |
| 3 | opennn → pytorch | 1,650 | 1,406 |


`cuda-cnn-infer` — batch 64, bf16, passes 5 per launch, 3 rounds. Artifact `cuda-cnn-infer-publish-20260903T100822Z.json`, commit `6b7179dde`, quiet True (busy 0.3% before, 0.1% after), clocks locked True, shape gate True, quality gate True.

| engine | median samples/s | min | max | peak device MiB | Wh (board) |
|---|---|---|---|---|---|
| OpenNN | 6,942 | 6,941 | 6,943 | 718 | 2.44248 |
| PyTorch | 5,833 | 5,833 | 5,833 | 868 | 2.94215 |
| **ratio** | **1.190×** | | | 1.21× less | 1.20× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 6,941 | 5,833 |
| 2 | pytorch → opennn | 6,942 | 5,833 |
| 3 | opennn → pytorch | 6,943 | 5,833 |

## Why

ResNet-50 is 53 convolutions, and both engines run every one of them in
cuDNN. The convolution kernels are the same family on both sides (each
engine's autotuner picks from the same `sm100`/`sm120` implicit-GEMM
kernels, and the profile below lists which), and they are about 79% of OpenNN's GPU time in the inference cell, spread over eleven distinct cuDNN kernels rather than one
of the batch. The margins in this family are what happens between the
convolutions: batch normalisation, the residual adds and the ReLUs — 53
convolutions carry 53 normalisations, 49 ReLUs and 16 residual adds — and
each one of those, run as its own kernel, is a full read and write of an
activation tensor that at batch 128 in bf16 is between 3 and 25 MB.

### Where the energy goes

This is the family where the two engines draw the *same* power, so the energy
margin is almost exactly the throughput margin. On inference OpenNN reads
244.3 W against PyTorch's 247.2 W — within the sampling noise — and both hold
the card at 99.7% and 99.6% occupancy; the 1.20× energy win is
the 1.19× throughput win and nothing else. On training OpenNN draws
229.6 W against 215.4 W, 6.6% more, and wins 1.10× on energy
against 1.17× on time.

Two consequences worth stating. First, a fully saturated convolution workload
is the case where OpenNN has the least energy advantage available to it:
there is no idle time to reclaim and no launch overhead to remove, so the
only lever is the arithmetic itself, which both engines hand to the same
cuDNN kernels. Second, the GEMM tile-selection rule that carries the dense
and transformer energy margins does nothing here, and this was measured
rather than assumed — the same cell at `OPENNN_LT_TILE_TOLERANCE=0` and at
the default reads 2.427 Wh and 2.430 Wh on inference, 3.848 and 3.843 on
training. ResNet-50's only matrix product is its 2,048 → 1,000 classifier
head, three orders of magnitude smaller than the convolutions around it, so
which kernel runs it cannot move the cell.

### `cuda-cnn-infer`, 1.19×: a convolution is one kernel

At inference OpenNN folds each batch normalisation into the weights of the
convolution in front of it (`Convolutional::forward_propagate_folded`: the
per-channel scale rescales the kernel, the shift becomes a bias, once per
parameter version) and builds the convolution as one cuDNN graph — `conv →
bias → [+ residual] → ReLU` (`convolution_operator.cpp`, `build_forward`) —
so a bottleneck block is three kernels and the whole forward pass is
62 kernels, captured once and replayed as one CUDA graph per
batch. The residual add is in the graph on purpose; the comment in the source
records that as a separate pass it was 26% of GPU time.

PyTorch's forward is `torch.compile` in its default mode with the weights in
bf16. Inductor keeps the convolution as an external cuDNN call
(`aten.convolution`) and lowers what follows it — the eval-mode batch norm
(a per-channel scale and shift, since the running statistics are frozen),
the residual add and the ReLU — to a Triton pointwise kernel, one per
convolution: 105 kernels per batch, 3,269
µs of them pointwise passes over activations the convolution had just
written and the next convolution is about to read again.

| | OpenNN | PyTorch |
|---|---|---|
| published / under `nsys` (samples/s) | 6,941 / 6681 | 5,833 / 5479 |
| timed window traced | 37.336 s | 45.629 s |
| kernel launches | 230,394 (6,171/s) | 410,033 (8,986/s) |
| GPU busy with kernels | 99.7% | 99.6% |
| GPU busy with any work (kernels, copies, memsets) | 99.7% | 99.6% |
| gaps between kernels, median / p90 | 0.2 / 0.5 µs | 0.3 / 0.9 µs |
| idle between kernels, total | 109.2 ms | 165.6 ms |

OpenNN, top kernels by summed time in the window:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 15.4% | 50,765 | 112.9 | `cudnn_generated_fort_native_sm80_convFwd_pointwise_pointwise_pointw...` |
| 14.0% | 27,335 | 190.0 | `sm80_xmma_fprop_implicit_gemm_indexed_bf16bf16_bf16f32_f32_nhwckrsc...` |
| 13.2% | 3,905 | 1253.8 | `convolve_common_engine_float_NHWC<__nv_bfloat16, __nv_bfloat16, 128...` |
| 9.6% | 11,715 | 303.9 | `cudnn_generated_fort_native_sm80_convFwd_pointwise_pointwise_pointw...` |
| 8.8% | 23,430 | 140.1 | `cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x...` |
| 8.1% | 15,620 | 192.6 | `sm80_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_ti...` |

PyTorch, top kernels by summed time in the window:

| share | launches | mean µs | kernel |
|---|---|---|---|
| 23.9% | 82,005 | 132.6 | `cutlass__5x_cudnn::Kernel<cutlass_tensorop_bf16_s16816fprop_optimiz...` |
| 11.1% | 31,241 | 161.6 | `cutlass__5x_cudnn::Kernel<cutlass_tensorop_bf16_s16816fprop_optimiz...` |
| 9.7% | 3,905 | 1131.3 | `convolve_common_engine_float_NHWC<__nv_bfloat16, __nv_bfloat16, 128...` |
| 7.1% | 7,810 | 412.4 | `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4` |
| 5.4% | 11,715 | 209.8 | `cutlass__5x_cudnn::Kernel<cutlass_tensorop_bf16_s16816fprop_optimiz...` |
| 5.0% | 11,715 | 193.1 | `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8` |

Per batch, GPU time by kernel class (`nsys`, the timed window of one launch, kernels grouped by name; OpenNN's graph replays are traced at node level):

| kernel class | OpenNN launches / batch | OpenNN µs / batch | PyTorch launches / batch | PyTorch µs / batch |
|---|---|---|---|---|
| copy/memset | 3.0 | 1.2 | 0.0 | 0.0 |
| convolution | 39.1 | 7,786.1 | 44.0 | 7,445.2 |
| attention | 1.0 | 2.2 | 0.0 | 0.0 |
| gemm | 13.0 | 1,285.2 | 11.0 | 924.0 |
| normalisation | 0.0 | 0.0 | 50.0 | 3,269.4 |
| pooling | 2.0 | 183.3 | 0.0 | 0.0 |
| elementwise | 4.0 | 294.6 | 0.0 | 0.0 |
| **total** | **62.1** | **9,552.6** | **105.0** | **11,638.6** |

The convolution time is close on both sides (7,786 against
7,445 µs per batch): both engines are dispatched to the same `sm80_xmma_fprop_implicit_gemm` and `cudnn_generated_fort_native_sm80_convFwd` families, so the convolutions themselves are not what separates them What separates the
engines is the 3,269 µs of pointwise kernels on PyTorch's
side against 295 on OpenNN's, plus the difference in
launch overhead — one graph launch against 105 eager-issued
kernels (Inductor's default mode does not use CUDA graphs; its
`reduce-overhead` mode does and measured slower here, 5,574 against 5,604,
because the graph's input copy and Python-side bookkeeping cost more than the
launches they save at this batch).

### `cuda-cnn-train`, 1.17×: the same fusions in the backward pass, and the input pipeline is not the story

The training step is the forward with batch statistics, the backward through
every block, and Adam over 161 parameter tensors. OpenNN runs the batch
normalisation as cuDNN graphs too: forward `batchnorm → [+ residual] → ReLU`
with the running statistics updated in the same kernel
(`batch_norm_operator.cpp`, `build_bn_forward`), backward `ReLU' → batchnorm
backward` with the residual's delta forked out of the same graph
(`build_bn_backward`), and the data gradient of a convolution with the
residual delta added in the `dgrad` graph. The whole step — the gather of the
shuffled batch, forward, loss, backward, Adam as one kernel over the
contiguous gradient buffer — is one captured CUDA graph, *[pending the final measurement round]* kernels per batch.

PyTorch's step is `max-autotune-no-cudagraphs`: Inductor's Triton kernels for
the batch-norm statistics (a Welford reduction per layer), the normalisation,
the ReLU and the residual, benchmarked per shape; the convolutions and their
two backward passes as cuDNN calls; `torch.optim.Adam` in its *foreach*
implementation (*[pending the final measurement round]* multi-tensor kernels per step).
*[pending the final measurement round]* kernels per batch, issued from Python.

*[pending the final measurement round]*

*[pending the final measurement round]*

*[pending the final measurement round]*

**The input pipeline is not what the cell measures.** OpenNN reads its
batches from the pre-decoded cache and PyTorch decodes JPEGs in eight worker
processes (see *What is measured*), and the question the caveats raise is
whether PyTorch's GPU ever waits for its loader. The `PT_INPUT=cache` variant
answers it: fed from the same `uint8` cache OpenNN reads, through a
`DataLoader` over a memory-mapped file, PyTorch trains at 1,372 against 1,364 samples/s
— a 0.6% difference, inside the round-to-round spread. Eight workers decoding
50,000 JPEGs at 224×224 keep up with 1,400 samples/s on this CPU, so the
loader is hidden behind the GPU on both sides and the cell measures the
training step.

## Asymmetries and caveats

- **The training input pipelines are not the same work.** OpenNN reads the
  images from a pre-decoded `uint8` cache (`.cache/images.bin`, 7.5 GB: every
  image resized once to 224×224 with a bilinear resize, no crop) and converts
  each batch to `float` in an OpenMP team while the previous batch trains.
  PyTorch decodes the JPEGs every epoch in eight `DataLoader` workers with
  torchvision's `Resize(224)` + `CenterCrop(224)`, then pins and copies. Both
  are what a user of each framework gets by default, but the cache is cheaper
  per image, and if PyTorch's GPU ever waits on its loader the cell is
  measuring the loader. Whether it does is the first question the *Why*
  section answers, with a profile of both launches and a variant that feeds
  PyTorch from the same cache (`PT_INPUT=cache`, 1,372 against 1,364 samples/s).
  The inference cells do not have this asymmetry: both engines fill one batch
  and replay it, so what they time is the resident forward pass.
- **Different cuDNN builds.** OpenNN links cuDNN 9.25.1; PyTorch's wheel
  bundles 9.23.2. Both pick their convolution kernels by trial
  (`cudnn.benchmark` against OpenNN's autotune, which also caps the inference
  workspace at 16 MiB), so the kernels can differ; which kernels each engine
  ran is in the profile.
- **The resize is not the same resize.** OpenNN's cache is a direct bilinear
  resize to 224×224; torchvision resizes the short side to 224 and
  centre-crops. Same tensor shape, different pixels; the parameter count and
  sample count are what the gate compares, and two epochs from a random
  initialisation are not long enough for an accuracy that would separate the
  two.
- **No accuracy gate.** The runner's quality gate only reads `test_accuracy`,
  which only the dense training drivers report, and neither CNN driver prints
  a loss in its timed modes; this family is held to the shape gate —
  25,557,032 parameters, 49,920 samples per epoch at batch 128. Both drivers
  have a `quality` mode that trains longer and evaluates the test split, for
  a reader who wants to check that the two networks learn alike; it is not
  part of the published cells.
- **OpenNN warms up for two epochs, PyTorch for one.** Both are excluded from
  the window; OpenNN's second untimed epoch is where the conv autotune and the
  graph capture settle. Nothing in the timed epochs depends on which warmup
  count was used.
- **An earlier version of this table timed OpenNN inference over zeros.**
  Until commit `4338506c8` the OpenNN inference drivers filled each batch
  (`Batch::fill()`) and never uploaded it: `fill()` only stages the rows on
  the host, and in the library the transfer is issued by the optimizer, which
  an inference driver does not run. The forward pass ran over all-zero inputs
  for every published inference cell before that commit. Nothing in the gates
  could see it — the parameter count, the sample count and the input file are
  the same whatever the batch holds, and the arithmetic of a forward pass does
  not depend on the values — and it was found by printing outputs (every one
  read `sigmoid(0) = 0.5`). A ResNet-50 forward over a zero image runs exactly
  the kernels it runs over a photograph — the cuDNN plans, the batch-norm
  arithmetic and the graph replay are all value-independent — so the number
  did not move beyond the round-to-round spread. The rows above are from the
  fixed drivers, which upload the batch (`upload_to_device_batch_async()`)
  after filling it and, where the split is resident, take the batch as a
  device view; the previous session's rows read 7,049 samples/s for OpenNN.

## Reproduce

```bash
export OPENNN_BENCH_SESSION=$(date +%F)-mine
python run.py --family cnn --mode train --device cuda --batch 128 --precision bf16 --epochs 2 --rounds 3
python run.py --family cnn --mode infer --device cuda --batch 128 --precision bf16 --repeats 5 --rounds 3
```

`prepare.py cnn` lays out the ImageNet validation subset as class folders; the
first OpenNN launch builds the cache (libjpeg, one decode per image, on all cores)
and every launch after it checks the cache's signature. `PT_INPUT=cache`
feeds PyTorch from the same file, `PT_COMPILE_MODE` and `PT_INFER_CAST=autocast`
are its other knobs; `OPENNN_NO_CUDA_GRAPH=1` runs OpenNN without graph
replay; `OPENNN_CONV_WORKSPACE_MB` and `OPENNN_CONV_AUTOTUNE`
are the autotune knobs, and every value of them tried on this machine is
within 1% of the default.
