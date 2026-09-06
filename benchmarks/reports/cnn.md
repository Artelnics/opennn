# CNN: ResNet-50 on the ImageNet subset

OpenNN against PyTorch 2.13 on ResNet-50 v1.5 (25,557,032 parameters) over
the 50,000-image ILSVRC2012 validation set laid out as 1,000 class folders,
224×224, bf16 with TF32 GEMMs, on the RTX 5070 Ti. Session
`2026-09-06-energy-publish`, commit `520f0f2f8`:

| cell | batch | OpenNN | PyTorch | OpenNN / PyTorch |
|---|---|---|---|---|
| `cuda-cnn-train` | 64 | 1,667 samples/s | 1,401 | **1.190×** |
| `cuda-cnn-infer` | 128 | 7,060 samples/s | 5,634 | **1.253×** |

Both cells also win on memory (1.18× and 1.50×) and on energy (1.132× and
1.307×).

The margin is not in the convolutions. Per batch OpenNN spends *more* GPU
time on them than PyTorch does — 23,246 µs against 23,039 µs in training,
17,625 against 15,356 in inference — because the two engines are not given
the same kernels: of the convolution kernel names in the training profile,
48 are OpenNN's and 51 are PyTorch's and only 12 appear on both sides. The
margin is what happens *between* the convolutions. At inference OpenNN folds
every batch normalisation into the weights of the convolution in front of it
and runs conv → bias → residual → ReLU as one cuDNN graph, so the forward
pass is 58 kernels replayed from one CUDA graph against PyTorch's 104
eager-issued launches, 49 of which are Triton passes over activations. In
training the normalisation cannot be folded away — it is 31.6% of the step,
in kernels OpenNN wrote itself — and the win comes from doing it in 302
kernels rather than the 559 Inductor pointwise and reduction kernels PyTorch
issues, and from replaying the whole step as one captured graph. Part of the
training margin is not to OpenNN's credit: it uploads its batch in bf16 on a
transfer stream and hides it, PyTorch uploads the same batch in fp32 on the
compute stream and does not, and that alone is 2.04 ms of a 7.16 ms
per-batch difference.

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
scaled to [0, 1]. Every cell processes whole batches only, so the training
cell at batch 64 covers 781 batches = **49,984 samples** per epoch and the
inference cell at batch 128 covers 390 batches = **49,920 samples** per pass;
the throughput figure divides that by the median epoch or pass time.

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
an asymmetry that favours OpenNN, in two separate ways that section "Why"
separates: the decode, which the `PT_INPUT=cache` variant bounds at 0.6%, and
the upload, where OpenNN sends half the bytes on a stream of its own and
PyTorch does not.

**Cells.**

| cell | device | batch | precision | timed window |
|---|---|---|---|---|
| `cuda-cnn-train` | RTX 5070 Ti | 64 | bf16 | 2 epochs after 2 untimed (OpenNN) / 1 untimed (PyTorch) |
| `cuda-cnn-infer` | RTX 5070 Ti | 128 | bf16 | 5 passes after 1 untimed |

The two cells run at different batch sizes because they were tuned
separately and neither was re-run at the other's; nothing below compares a
training number to an inference number, and every ratio is between two
engines at the same batch.

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
under it and larger candidates only added a cold-start memory peak — and,
since `520f0f2f8`, a second stage that meters every candidate within 10% of
the fastest against the board's power samples and takes the one that costs
the least energy per run, `OPENNN_CONV_ENERGY_AUTOTUNE`; *Where the energy
goes* prices it). PyTorch's
`torch.compile` mode is chosen per cell and each was measured (the driver's
`compiled()` docstring has every mode). Those comparisons were all run at
batch 128, before the training cell moved to batch 64, so they establish the
ranking rather than the published number: the training step runs
`max-autotune-no-cudagraphs`, Inductor with its GEMM and pointwise templates
benchmarked per shape and no CUDA graphs (1,398 samples/s, artifact
`cuda-cnn-train-variant-compile-mano-20260902T065653Z.json`, against 1,366
under `reduce-overhead`, `-variant-compile-ro-20260902T051314Z`); the
inference forward runs the default mode (5,604,
`cuda-cnn-infer-variant-cast-default-20260902T061724Z`, against 5,597 for
`max-autotune-no-cudagraphs` at 941 MiB more resident memory — 2,207 against
1,266 — 5,574 for `reduce-overhead` and 3,625 in eager), with the weights
stored in bf16 once (`PT_INFER_CAST=weights`, the default). The two remaining
medians are not variant runs: the default training mode reads 1,364 and
autocast inference 5,498, both from session `2026-09-02-publish` at commit
`38ad27e16` (`cuda-cnn-train-publish-20260902T023438Z.json` and
`cuda-cnn-infer-publish-20260902T022835Z.json`), an earlier commit than any
of the variants, so they rank the modes more loosely than the rest.
`PT_COMPILE_MODE=default|reduce-overhead|max-autotune-no-cudagraphs|eager`
and `PT_INFER_CAST=autocast` remain available. OpenNN
captures the whole Adam step (forward, loss, backward, update) into one CUDA
graph per batch shape and replays it, and the inference forward pass is a
captured graph as well; `OPENNN_NO_CUDA_GRAPH=1` turns that off for the
controlled comparison. Both engines use tensor cores through NHWC
convolutions: PyTorch by converting the model and inputs to `channels_last`,
OpenNN because its image tensors are HWC natively.

**Gates.** The shape gate checks parameters (25,557,032) on both sides and
samples per window (49,984 for the training cell, 49,920 for inference).
There is **no accuracy gate** on this family: two epochs of ResNet-50 from
scratch on 50 images per class do not produce a meaningful test accuracy, so
the runner only requires both engines to run the same network over the same
number of samples. The quality gate is reported as agreeing because neither
driver emits an accuracy for it to disagree about — every value in it is
`NaN`. The `quality` mode of both drivers exists for longer runs and is not
part of the published matrix.

## Results

Session `2026-09-06-energy-publish`, commit `520f0f2f8`, median of three rounds.

| cell | batch | precision | OpenNN samples/s | PyTorch samples/s | OpenNN / PyTorch | peak memory MiB (OpenNN / PyTorch) | energy Wh (OpenNN / PyTorch) |
|---|---|---|---|---|---|---|---|
| `cuda-cnn-train` | 64 | bf16 | 1,667 | 1,401 | **1.190×** | 3,573 / 4,228 | 3.7574 / 4.2545 |
| `cuda-cnn-infer` | 128 | bf16 | 7,060 | 5,634 | **1.253×** | 846 / 1,267 | 2.3976 / 3.1327 |


`cuda-cnn-train` — batch 64, bf16, epochs 2 per launch, 3 rounds. Artifact `cuda-cnn-train-publish-20260906T172254Z.json`, commit `520f0f2f8`, clean tree, quiet True (busy 0.1% before, 0.2% after, 2.7% max during, threshold 3%), clocks locked True, shape gate True, quality gate vacuously True.

| engine | median samples/s | min | max | peak device MiB | Wh (board) | mean W |
|---|---|---|---|---|---|---|
| OpenNN | 1,667 | 1,664 | 1,667 | 3,572.9 | 3.75744 | 225.5 |
| PyTorch | 1,401 | 1,399 | 1,402 | 4,227.9 | 4.25447 | 214.9 |
| **ratio** | **1.190×** | | | 1.18× less | 1.132× less | 1.05× more |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 1,664 | 1,401 |
| 2 | pytorch → opennn | 1,667 | 1,402 |
| 3 | opennn → pytorch | 1,667 | 1,399 |


`cuda-cnn-infer` — batch 128, bf16, passes 5 per launch, 3 rounds. Artifact `cuda-cnn-infer-publish-20260906T171646Z.json`, commit `520f0f2f8`, clean tree, quiet True (busy 0.3% before, 0.1% after, 0.5% max during, threshold 3%), clocks locked True, shape gate True, quality gate vacuously True.

| engine | median samples/s | min | max | peak device MiB | Wh (board) | mean W |
|---|---|---|---|---|---|---|
| OpenNN | 7,060 | 7,060 | 7,060 | 845.8 | 2.39758 | 243.5 |
| PyTorch | 5,634 | 5,633 | 5,642 | 1,267.2 | 3.13269 | 254.4 |
| **ratio** | **1.253×** | | | 1.50× less | 1.307× less | 1.04× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 7,060 | 5,642 |
| 2 | pytorch → opennn | 7,060 | 5,633 |
| 3 | opennn → pytorch | 7,060 | 5,634 |

OpenNN's throughput is identical across all three rounds in both cells; the
run-to-run spread is on PyTorch's side, 1.2% in training and 0.2% in
inference. The identity is integer rounding, not a frozen measurement: the
three OpenNN training launches ran 121.744, 121.669 and 121.670 s and the
three inference launches 44.139, 44.167 and 44.116 s, spreads of 78 ms and
51 ms that samples/s reported as an integer cannot resolve.

## Why

Every profile number below is from four `nsys` traces taken in this session,
each restricted to the driver's own timed window so that warmup,
`torch.compile` and the graph capture are excluded. **All four are at commit
`cad7ca98e`, the commit before the batch-normalisation change described
below, and they were exported in-session rather than kept beside the run
artifacts under `results/`, so a reader cannot open them.** For the inference
cell that costs nothing — `93cc90e07` touches only the batch-normalisation
kernels, which the folded inference forward does not run — and under `nsys`
it read 7,074 and 5,634 samples/s against the published 7,077 and 5,635. For
the training cell it matters, and the section says so where it does.

ResNet-50 is 53 convolutions carrying 53 normalisations, 49 ReLUs and 16
residual adds. Each of those non-convolution operations, run as its own
kernel, is a full read and write of an activation tensor: at batch 64 the
largest is 51.4 M elements, 103 MB in bf16, and the 49 normalisations OpenNN
fuses itself pass 615.0 M elements per step between them. That traffic, not
the arithmetic, is what the two engines organise differently.

### Where the energy goes

This is the family where the two engines draw close to the *same* power, so
the energy margin tracks the throughput margin — with one deliberate
exception, new this round. On inference OpenNN reads 243.5 W against
PyTorch's 254.4 W, 4.3% less, and both hold the card busy with kernels
99.9% of the window; the cell wins 1.253× on time and 1.307× on energy, the
4% between them being the power. On training OpenNN draws 225.5 W against
214.9 W, 4.9% more, and wins 1.132× on energy against 1.190× on time.
Occupancy is part of that and not all of it: OpenNN's training window is
99.0% occupied by kernels against PyTorch's 94.3%, so at most about half of
the power gap is the card being kept busy and the rest is that the two
engines run different kernels. PyTorch's 5.7 points of non-kernel window
are also not idle in the power sense — 85% of that time is a 38.5 MB
host-to-device copy, which draws board power.

The exception is the convolution engine choice. cuDNN's engines for one
shape differ in power the way cuBLASLt's tiles do — tile shape, split-K,
occupancy — and the frontend's autotune ranks them by time alone. Since
`520f0f2f8` the library adds a second stage: every built plan within 10%
of the fastest runs back to back for 300 ms while the driver's power ring
samples the board (NVML's 20 ms samples, read through a runtime-loaded
`libnvidia-ml`; the cumulative energy counter reads zero on this GeForce),
and the plan that costs the least energy per run becomes the candidate,
provided it beats the fastest by more than 2% so that meter noise cannot
flip the choice between runs. It is the GEMM tile rule with a measured
power in place of a modelled one. On this network at batch 128 it swaps
about half of the forward plans (`OPENNN_CONV_ENERGY_VERBOSE=1` prints each
decision): typical swaps are 5-23% less energy for 0.5-5% more time per
convolution — 65.2 mJ at 382.5 µs over 84.6 mJ at 365.6 µs on one stem-sized
plan, 25.7 mJ at 156.2 µs over 33.1 mJ at 147.5 µs on a 1×1 — and the
fastest is kept where nothing cheaper is within tolerance.

What it is worth on the cells, publish row against publish row
(`520f0f2f8` against `e76425bd3`, both warm-cache, three rounds): inference
2.482 → 2.398 Wh, 3.4% less energy, for 7,075 → 7,060 samples/s, 0.2% less
throughput; training 3.845 → 3.757 Wh, 2.3% less, for 1,682 → 1,667, 0.9%
less. The one-round A/B at the commit itself, stage on against
`OPENNN_CONV_ENERGY_AUTOTUNE=0` in the same session, reads the same trade
with more throughput cost — 2.376 against 2.437 Wh at 6,992 against 7,086
on inference, 3.746 against 3.858 Wh at 1,667 against 1,681 on training —
so the price is between 0.2% and 1.3% of throughput for 2.3-3.4% of energy.
It is on by default because it is the rule this suite already applies to
GEMMs and the throughput axis stays won by 19% and 25%; a reader who wants
the last percent of throughput back sets the knob to 0.

Two consequences worth stating. First, a fully saturated convolution workload
is the case where OpenNN has the least energy advantage available to it:
there is no idle time to reclaim and, once the step is a captured graph, no
launch overhead to remove either — which is why the engine choice above is
the lever that was left, and why it is worth what it is worth. ResNet-50's only true matrix product is its
2,048 → 1,000 classifier head — one 11.3 µs `nvjet` kernel out of 18,117 µs
of GPU time per inference batch — so which GEMM kernel runs it cannot move
the cell. Second, and consistent with that, the GEMM tile-selection rule that
carries the dense-training and transformer energy margins buys no energy
here: at `OPENNN_LT_TILE_TOLERANCE=0` against tolerance 5 the cell read
2.42733 and 2.42981 Wh on inference (6,938 and 6,941 samples/s), 3.84804 and
3.84277 Wh on training (1,647 both ways). It does move memory, and against
the rule — 718.1 MiB against 1,200.1 on inference, 3,608.1 against 4,136.1 on
training, so the rule costs 482 and 528 MiB of the axis this cell also
claims. Those four runs are session `2026-09-02-gemmenergy`, commit
`bc6f4c2d0`, batch 64, one round each, two of the four failing the quiet
gate, and all four carrying an uncommitted `device_backend.cpp` because the
knob did not exist in the committed tree there; no artifact records an
environment block, so the mapping from run label to tolerance rests on the
label. They bound the rule's effect on this network rather than measuring it
on the published rows, where the default is tolerance 10.

### Where the memory goes

This round both cells have the library's own attribution behind them
(`OPENNN_MEMORY_DEBUG=1`, device figures; PyTorch's side was not traced, so
its half of each gap is measured and not decomposed).

*Training, 3,573 against 4,228 MiB.* OpenNN's forward arena is 2,667 MiB and
that is its planner's lower bound: the live set at the peak — the boundary
between forward and backward — is 261 saved tensors, 2,640 MiB, all in bf16,
the 72 backward entries co-planned into the same arena by lifetime. The
largest are nine of 98 MiB (the 56×56×256 block outputs of stage 0 and the
stem's 112×112×64), eleven of 49 MiB and twenty-eight of 24.5 MiB — the
same set of activations PyTorch's autograd saves for the same network, at the
same dtype. Around the arena: the fp32 master 97.5 MiB, its bf16 mirror 49,
the gradient 97.5, Adam's state 146 (bf16 first moment, fp32 second), cuDNN
and cuBLASLt scratch 64, batch buffers 37; the remainder of the 3,573 is the
CUDA context and kernel images. Nothing here shrinks without changing the
algorithm — recompute, which costs throughput on a 1.190× cell, or a bf16
second moment, which changes what Adam computes — so the 1.18× is where the
standard algorithm in bf16 lands against PyTorch's caching allocator, and
this document does not claim more for it.

The row is a warm-cache figure, and it has to be said because the cold one
is 500 MiB higher. cuDNN plans are autotuned once per shape and kept in a
plan cache on disk; a run whose cache is cold autotunes at warm-up, and the
frontend's autotune allocates the largest candidate's workspace — the
training path caps candidates at the largest activation slot, not at 16 MiB
— transiently while it times them. The harness reads the peak over the whole
process, so a cold run reads it: 4,066-4,086 MiB in the one-round A/B of
2026-09-06, whose cache was cold because the commit before had changed every
cache key, against 3,558-3,573 in every publish run, all of which were warm.
The publish protocol now warms the cache with one untimed run before the
measured one, which is the state a model runs in after its first session and
the state every earlier publish row was in; PyTorch's `cudnn.benchmark`
autotune is per process and runs inside every one of its launches, so its
rows are cold-cache by construction and would not move.

*Inference, 846 against 1,267 MiB, where the table before last read 944.* The
96 MiB that left is the fp32 master copy of the 25.6 M parameters, which
the driver had been keeping on the device beside the bf16 mirror the forward
pass reads. The library's release path, `upload_parameters_bf16_inference()`,
was only reached through the model-loading functions; the driver now calls
it before its warm-up pass (`families/cnn.cpp`), the same deployment step
the PyTorch driver takes with `model.to(torch.bfloat16)`. Throughput and
energy did not move with the release (7,077 → 7,075 samples/s, 2.474 → 2.482
Wh at `e76425bd3`, both inside the launch spread) and the quality gate is
unchanged; the energy that moved since is the engine choice, above. What remains is the
49 MiB bf16 mirror, the inference arena, cuDNN's convolution plans and the
CUDA context.

The mechanism the profiles point at for the rest of the inference gap is the
same fusion the throughput section describes: OpenNN's 58-kernel graph
writes 53 convolution outputs, PyTorch's 104-kernel forward writes a
convolution output *and* a pointwise output per convolution, the same
tensors its 7,259.8 µs pointwise row passes over. That is consistent with
the direction and the order of magnitude of the 420 MiB; it is not a
measurement of them.

Two things do have to be said about the column. It is device-used minus idle,
so it charges each engine for its CUDA context and for whatever its allocator
is holding rather than using (see the caveats). And PyTorch's half of it is
mode-dependent while the mode was chosen on throughput: at batch 128 the
inference cell reads 1,266 MiB in the default mode and 2,207 under
`max-autotune-no-cudagraphs`, a 941 MiB swing on a knob the reader did not
set.

### `cuda-cnn-infer`, 1.253×: a convolution is one kernel

At inference OpenNN folds each batch normalisation into the weights of the
convolution in front of it (`Convolutional::forward_propagate_folded`: the
per-channel scale rescales the kernel, the shift becomes a bias, once per
parameter version) and builds the convolution as one cuDNN graph — `conv →
bias → [+ residual] → ReLU` (`convolution_operator.cpp`, `build_forward`) —
so a bottleneck block is three kernels and the whole forward pass is 58
kernels: 53 convolutions, two poolings, the classifier GEMM, the input
scale and the softmax. All 58 are replayed from one captured CUDA graph, one
`cudaGraphLaunch` per batch. The residual add is in the graph on purpose; the
comment in the source records that as a separate pass it was 26% of GPU time.

PyTorch's forward is `torch.compile` in its default mode with the weights in
bf16. Inductor keeps the convolution as an external cuDNN call
(`aten.convolution`) and lowers what follows it — the eval-mode batch norm
(a per-channel scale and shift, since the running statistics are frozen),
the residual add and the ReLU — to a Triton pointwise kernel, one per
convolution: 104 kernels per batch, 104 launches, 7,260 µs of them pointwise
passes over activations the convolution had just written and the next
convolution is about to read again.

| | OpenNN | PyTorch |
|---|---|---|
| published / under `nsys` (samples/s) | 7,077 / 7,074 | 5,635 / 5,634 |
| timed window traced | 21.218 s | 26.581 s |
| kernels per batch | 58 | 104 |
| API launches per batch | 1 `cudaGraphLaunch` | 104 kernel launches |
| GPU busy with kernels | 99.9% | 99.9% |
| gaps between kernels, median / p90 | 0.19 / 0.42 µs | 0.26 / 0.26 µs |
| idle between kernels, total | 20.8 ms | 36.2 ms |

Per batch, GPU time by kernel class (`nsys`, the timed window; OpenNN's graph
replays are traced at node level; Inductor's generated kernels are classed by
their prefix, not by the operations their names list):

| kernel class | OpenNN launches / batch | OpenNN µs / batch | PyTorch launches / batch | PyTorch µs / batch |
|---|---|---|---|---|
| convolution | 53 | 17,625.4 | 53 | 15,356.2 |
| pointwise (Inductor) | 0 | 0.0 | 49 | 7,259.8 |
| pooling | 2 | 345.7 | 0 | 0.0 |
| input scale | 1 | 131.9 | 0 | 0.0 |
| gemm (classifier head) | 1 | 11.3 | 1 | 15.2 |
| reduction / softmax | 1 | 2.4 | 1 | 56.3 |
| **total** | **58** | **18,116.7** | **104** | **22,687.5** |

The convolution row counts 53 of ResNet-50's convolutions on each side.
Several of the thirty-six 1×1 convolutions are handed by cuDNN to a GEMM
engine on both sides — a 1×1 convolution over NHWC *is* a matrix product —
and they are counted here as the convolutions they are rather than by the
name of the kernel that ran them; the classifier head is not one of those and
gets its own row on both sides. PyTorch's pooling does not have a row because
Inductor fused it into the adjacent pointwise kernels. Its input scaling has
no row because it does not exist: `ToTensor` scales on the host, so PyTorch's
graph has nothing to scale, while OpenNN re-scales the resident batch inside
the graph on every replay — 131.9 µs, 0.73% of its step, and it counts
against it.

**OpenNN's convolutions are the slower half of this cell.** 17,625 µs per
batch against 15,356, 14.8% more, and the difference is the epilogue rather
than the arithmetic. Its graph asks cuDNN for conv → bias → [+ residual] →
ReLU and gets engines whose names carry that epilogue —
`cudnn_generated_fort_native_sm80_convFwd_pointwise_pointwise_pointwise`,
`cutlass_80_tensorop_..._gemm_relu_...`, `nvjet_..._biast_relu_...` — for 26
of its 53 convolutions and 38% of its convolution time; PyTorch asks for a
bare convolution and gets `cutlass__5x_cudnn` `fprop_optimized`
instantiations for 37 of its 53 and 58% of its own, which a fused graph
reaches for none. Only three kernel names appear on both sides, one of them
the 7×7 stem. That 2,269 µs is bought back three times over by what the
fusion removes: everything that is not a convolution or the head costs
PyTorch 7,316.1 µs per batch and OpenNN 480.0, plus the difference in launch
overhead — one graph launch against 104 eager-issued kernels. The graph is
the smaller half of that: `OPENNN_NO_CUDA_GRAPH=1` reads 7,003 samples/s
against 7,049 for the graph at the same batch and the adjacent commit
(`cuda-cnn-infer-variant-nograph-20260902T051136Z.json` at `918805ce1`
against `cuda-cnn-infer-publish-20260902T022835Z.json` at `38ad27e16`, batch
128), so replay is worth about 0.7% to OpenNN — both of those runs predate
the zero-input fix at `4338506c8`, which affects them equally. Inductor's
default mode does not use CUDA graphs; its `reduce-overhead` mode does and
measured slower here (5,574 against 5,604 at batch 128), because the graph's
input copy and Python-side bookkeeping cost more than the launches they
save.

### `cuda-cnn-train`, 1.190×: the normalisation is the step

The training step is the forward with batch statistics, the backward through
every block, and Adam over 161 parameter tensors. The batch normalisation is
fused with the residual add and the ReLU here too, but — unlike inference —
mostly not by cuDNN. `own_forward_kernel` (`batch_norm_operator.cpp`) sends a
normalisation to OpenNN's own kernels whenever a ReLU follows it and a mask
slot exists, which is **49 of the 53**: forward `batchnorm → [+ residual] →
ReLU` with the running statistics updated in the same pass and the ReLU's
mask written out, backward `ReLU' → batchnorm backward` reading that mask
back and forking the residual's delta. Each direction is three launches — a
reduce over the rows, a finalize over the per-block partials, and one apply
pass over the tensor. Only the 4 normalisations with no ReLU after them —
the downsample projections, whose output feeds an add rather than an
activation — take the cuDNN graph (`build_bn_forward` / `build_bn_backward`;
in the profile they are `nhwc_batch_norm_fwd` and `nhwc_batch_norm_bwd` at 4
launches per batch each, against 49 for every one of OpenNN's own). The data
gradient of a convolution still carries the residual delta inside its `dgrad`
graph. The whole step — the gather of the shuffled batch, forward, loss,
backward, Adam as one kernel over the contiguous gradient buffer — is one
captured CUDA graph: **561 kernels per batch behind a single
`cudaGraphLaunch`**, with two `cudaMemcpyAsync` calls outside it for the
batch upload. The capture is worth less than that framing suggests:
`OPENNN_NO_CUDA_GRAPH=1` reads 1,591 samples/s against 1,612 for the same
code with the graph, at batch 128 and the adjacent commit
(`cuda-cnn-train-variant-nograph-20260902T050835Z.json` at `918805ce1`
against `cuda-cnn-train-publish-20260902T023438Z.json` at `38ad27e16`), so
replay buys OpenNN about 1.3% of its own step.

PyTorch's step is `max-autotune-no-cudagraphs`: Inductor's Triton kernels for
the batch-norm statistics (a Welford reduction per layer), the normalisation,
the ReLU and the residual, benchmarked per shape; the convolutions and their
two backward passes as cuDNN calls where Inductor's autotuner preferred them
and as its own Triton convolution templates where it did not; and
`torch.optim.Adam` — which is inside the compiled region, so Inductor lowers
it too, into ten `triton_for_fused_*` kernels rather than the multi-tensor
`foreach` kernels an eager step would issue. **764 kernels per batch, every
one of them launched individually from Python** (654 `cuLaunchKernel`, 84
`cudaLaunchKernel`, 26 `cudaLaunchKernelExC`). Exactly one of the 764 is an
eager ATen kernel — `at::native::vectorized_elementwise_kernel` running a
`FillFunctor<float>`, 0.7 µs per batch — so the step compiled essentially
whole.

Both columns below are traced at `cad7ca98e`. PyTorch's binary is the
published one and read 1,403 samples/s under `nsys` against 1,399 published;
OpenNN's is one commit behind the published 1,682 and read 1,665, so that
pair measures the commit gap rather than the cost of tracing.

| | OpenNN | PyTorch |
|---|---|---|
| under `nsys` (samples/s) | 1,665 | 1,403 |
| timed window traced | 30.014 s | 35.608 s |
| kernels per batch | 561 | 764 |
| API launches per batch | 1 `cudaGraphLaunch` | 764 kernel launches |
| GPU busy with kernels | 99.0% | 94.3% |
| gaps between kernels, median / p90 | 0.19 / 0.19 µs | 0.26 / 0.26 µs |
| idle between kernels, total | 285.5 ms | 1,883.1 ms |

| kernel class | OpenNN launches / batch | OpenNN µs / batch | PyTorch launches / batch | PyTorch µs / batch |
|---|---|---|---|---|
| convolution and GEMM, all three directions | 192 | 23,245.6 | 191 | 23,038.6 |
| normalisation | 302 | 13,273.6 | — | — |
| pointwise (Inductor) | — | — | 273 | 13,165.5 |
| reduction (Inductor) | — | — | 286 | 5,594.0 |
| optimizer | 2 | 828.1 | 10 | 954.6 |
| pooling | 4 | 486.3 | 0 | 0.0 |
| padding (cuDNN) | 2 | 132.9 | 2 | 166.4 |
| pointwise (weight-gradient casts) | 53 | 89.6 | 1 | 91.1 |
| loss and softmax | 6 | 7.7 | 1 | 0.7 |
| **total** | **561** | **38,063.8** | **764** | **43,010.9** |

The first row counts, on both sides, every kernel that runs a convolution in
any direction — including the GEMM kernels the 1×1 convolutions and the
classifier head are lowered to, the Triton convolution templates Inductor
kept for itself, and the split-k reductions those kernels spawn. Binning them
any other way splits the same work differently on the two sides: PyTorch's
1×1 weight gradients arrive as `cutlass_80_tensorop_s16816gemm_bf16_*_nt`
kernels, which are convolution work under a GEMM name exactly as OpenNN's
`gemm_relu` kernels are.

PyTorch's normalisation work has no row of its own because Inductor's
kernels are not separable: `triton_poi_fused__native_batch_norm_legit_-
functional_add_convolution_relu_22` is a normalisation, an add and a ReLU in
one pass. The zeros in PyTorch's pooling row mean the same thing: its two
pooling passes are inside those fused kernels, not absent. So the comparison
has to be made on the aggregate. Everything that is neither convolution nor
optimizer costs OpenNN 13,990.1 µs in 367 launches and PyTorch 19,017.7 µs in
563 — a 5,027.6 µs per-batch advantage. Against it the convolution row gives
207.0 µs back and the optimizer row adds 126.5. Net kernel time, 4,947.1 µs
per batch, which is the whole of the 43,010.9 − 38,063.8 difference. The
remaining 2,216 µs of the 7,163 µs per-batch wall difference is PyTorch's
extra non-kernel time, and most of it is not launch overhead — see the input
pipeline below.

PyTorch's largest single kernel is
`cutlass_tensorop_bf16_s16816dgrad_optimized_bf16_128x128_32x3_nhwc_-
unity_stride_align8`, 16 launches and 1,755.8 µs per batch, 4.08% of its
step. OpenNN's is `batchnorm_backward_apply_kernel` at 12.47%. The two
profiles are shaped differently: PyTorch's step spreads across many
autotuned kernels, OpenNN's concentrates in four normalisation kernels.

**Batch normalisation is 31.6% of OpenNN's step** and the largest thing in
the profile that is not a convolution. Everything below is measured at
`cad7ca98e`, *before* the change described after it: this is the picture that
motivated the work, not the current state of the kernels. Seven kernels
carry the 49 layers OpenNN normalises itself:

| share of step | launches / batch | mean µs | µs / batch | kernel |
|---|---|---|---|---|
| 12.47% | 49 | 96.89 | 4,747.4 | `batchnorm_backward_apply_kernel` |
| 8.20% | 49 | 63.71 | 3,121.9 | `batchnorm_backward_reduce_kernel` |
| 6.15% | 16 | 146.23 | 2,339.7 | `batchnorm_forward_apply_kernel` (with residual) |
| 2.32% | 49 | 18.01 | 882.5 | `batchnorm_forward_reduce_kernel` |
| 2.04% | 33 | 23.51 | 775.8 | `batchnorm_forward_apply_kernel` (plain) |
| 0.28% | 49 | 2.14 | 104.9 | `batchnorm_forward_finalize_kernel` |
| 0.17% | 49 | 1.28 | 62.9 | `batchnorm_backward_finalize_kernel` |

The 16 forward applies that also add a residual are the `bn3` of each
bottleneck block; the 33 plain ones are `bn1`, `bn2` and the stem. Those
seven rows are 12,035.1 µs, 31.62% of the step. The four normalisations that
go to cuDNN instead cost a further 3.25% (`nhwc_batch_norm_bwd` 823.9
µs/batch, `nhwc_batch_norm_fwd` 414.6), which is why the table's
normalisation row reads 302 launches and 13,273.6 µs: it holds both.

These kernels are bandwidth-bound and they were not equally good at it. The
forward reduce ran at 1394 GB/s and the forward apply at 1040, both above the
card's 896 GB/s pin bandwidth because L2 serves part of the traffic; the
backward apply ran at 943 and the backward reduce at 812, below it. The
backward reduce was the outlier and it is the one the change aimed at.

**The obvious fusion was measured and rejected before it was written.**
Fusing the backward reduce into the backward apply, so that the tensor is
read once instead of twice, only pays where the second pass can be served
from cache. The apply re-reads `x`, `dy` and the ReLU mask — 2 + 2 + 0.125 =
**4.125 B/element** — so a layer is recovered only if that working set fits
in this card's 48 MiB L2 (50,331,648 B, as the driver reports it). It does
for the layers at 128×28×28 and below, and does not for anything at
64×112×112, 256×56×56, 128×56×56, 64×56×56, 512×28×28, 256×28×28 or
1024×14×14: **114.0 M of the 615.0 M normalised elements per step, 18.5%**.
Best case the fusion removes
0.470 GB of the 11.483 GB the four kernels move per step, about 1.3% of the
step at the bandwidth the backward apply already achieves — in exchange for a
grid-wide arrival barrier inside a captured graph and the loss of the
launcher's fixed summation order, which is what makes the gradients
reproducible run to run. The 4.125 B/element re-read is structural, not
waste: the apply needs `dgamma` and `dbeta`, which are reductions over every
element of the tensor, so nothing can be applied until everything has been
read. cuDNN's own `nhwc_batch_norm`, which runs the four layers OpenNN does
not fuse, moves the same 16 B/element with a persistent kernel; its advantage
is L2 behaviour, not fewer passes.

**What was wrong instead was that every element paid for branches it never
took.** The backward kernels chose the ReLU source with a runtime pointer
test, but the wrapper nulls `y` whenever a mask exists, so in this network
every launch takes the mask arm while still keeping `vy[VEC]` — eight
registers at VEC=8 — live for a load never issued; the apply likewise kept
`pre[VEC]` live on the 33 of 49 layers that have no residual fork. Both are
now template parameters (`RELU_SRC`, `HAS_DPRE`) resolved at compile time,
the reduce's `b[]`/`ig[]` initialisation moved under `if constexpr`, and the
backward reduce given `__launch_bounds__` so that ptxas targets the four
blocks per SM the grid was already sized for rather than spending registers
down to three.

Whether it lands is not shown. The traced kernel compiles to 77 registers per
thread, which rounds to 80 and yields floor(65536/(256·80)) = 3 resident
blocks; dropping the `y` vector returns about 8, landing near 69, still above
the 64 that four blocks require, so the pragma is asking ptxas for roughly
five registers more than the source change frees. No `nvcc -Xptxas -v`
register count is recorded for the published build, and the only profile
there is predates the change, so no post-change bandwidth for this kernel
exists either: the occupancy step was aimed at, not shown to be crossed, and
whether the gap to the forward reduce's 1394 GB/s narrowed is unmeasured.
The upside is bounded anyway: only 65.3% of this
kernel's time runs on grids that offer four or more blocks per SM, the other
34.7% being the narrow-channel layers, which are short of blocks rather than
of occupancy and which no register cap can help.

Every change is bit-identical by construction: no arithmetic moved, every
removed arm was unreachable given the pointer values the wrapper guarantees,
and the reduce's geometry — and so its summation order — depends only on
rows, channels and VEC, none of which changed. It was not checked by
comparing gradients before and after, and this path has no accuracy gate that
would have caught it if the argument were wrong — a gradient that changes in
the last bits is a silent correctness change. Two further optimisations were
rejected for failing the same test: folding the apply's five per-channel
loads into a precomputed affine form reassociates `x_hat·dgamma·inv_rows`,
and reading `y` instead of `x` in the backward moves the same byte count
anyway.

**Nothing here isolates the change, and the range should be read that way.**
Before it the cell read 1,647 and 1,650 samples/s (session
`2026-09-03-publish`, commit `6b7179dde`, artifacts `...20260903T101423Z`
and `...20260903T120534Z`); a run carrying the change as its only tree
difference read 1,667 (`cuda-cnn-train-bnfix-20260905T084015Z.json`, session
`2026-09-05-bn`, commit `cad7ca98e`, dirty in `kernel_normalization.cu`
alone); the published run at `93cc90e07` reads 1,682. That is 1.0% to 2.1%
on the cell, and every pair that produces it is weak. No clean baseline at
`cad7ca98e` was ever run, so both ends reach back to `6b7179dde` and both
therefore also span `e844bfe16` and `cad7ca98e`, which touched CUDA kernel
selection. The bnfix run failed the quiet gate (4.3% busy before the window
against a 3% threshold) and ran two rounds rather than three, which is the
likeliest reason the same code reads 1,667 there and 1,682 at `93cc90e07`.
One of the two baselines, the 1,647, failed the quiet gate as well (6.6%
peak during). 2.1% is the widest figure the artifacts support and it is an
upper bound on a range that a proper A/B has not been run for.

**Adam is not the story here.** OpenNN runs it as one kernel over the
contiguous gradient buffer, 828.1 µs per batch, 2.18% of the step; PyTorch's
compiled Adam is ten kernels and 954.6 µs, 2.22%. The graph capture already
buys what a multi-tensor apply buys, and there is nothing left in the
optimizer worth taking on this network.

**The input pipeline is not what the cell measures — but the upload is.**
The JPEG decode is not the problem: fed from the same `uint8` cache OpenNN
reads, through a `DataLoader` over a memory-mapped file, PyTorch trained at
1,372 against 1,364 samples/s decoding JPEGs, a 0.6% difference inside the
round-to-round spread. That pair is looser than a controlled A/B: 1,372 is
`cuda-cnn-train-variant-cache-20260902T050056Z.json` at `918805ce1` and 1,364
is the publish run at `38ad27e16`, different commits, both PyTorch in the
default compile mode at batch 128 rather than the batch 64 and
`max-autotune-no-cudagraphs` the cell now publishes.

What the variant cannot see is the upload, and the profile shows it plainly.
Per batch OpenNN issues one host-to-device copy of 19,267,584 B — the batch
in bf16 — plus 256,000 B of targets, on a dedicated transfer stream behind a
prefetch thread: 922.5 µs of copy against 366 µs of total inter-kernel idle,
so it is overlapped. PyTorch issues one copy of 38,535,168 B — the same
tensor in fp32, because `ToTensor` produces fp32 and `cnn.py` never casts
before `.to()` — and `images.to("cuda", non_blocking=True, ...)` lands it on
the compute stream, where it serialises with the kernels. Bucketing every
inter-kernel gap in PyTorch's window: 578,257 gaps below 1 µs carry 186 µs
per batch, which is what 764 Python-issued launches actually cost, while 782
gaps above 1 ms — one per batch — carry 2,050 µs. The copy is 2,038.7 µs of
that one stall. PyTorch's non-kernel time is 2,582 µs per batch against
OpenNN's 366, and the fp32 upload accounts for 2,039 of that 2,216 µs
difference; the launches account for the small remainder. `PT_INPUT=cache`
is blind to this because `CachedImages.__getitem__` ends in
`.float().div_(255.0)` and sends the identical 38.5 MB fp32 tensor.

## Asymmetries and caveats

- **The training input pipelines are not the same work.** OpenNN reads its
  images from a pre-decoded `uint8` cache and PyTorch decodes the JPEGs every
  epoch in eight `DataLoader` workers (see *What is measured*). The cache is
  cheaper per image, and it is an asymmetry that favours OpenNN. The decode
  itself is not what costs PyTorch — the `PT_INPUT=cache` variant in *Why*
  bounds it at 0.6% — but the upload is, and that is the next bullet. The
  inference cells do not have the asymmetry at all: both engines fill one
  batch and replay it.
- **PyTorch uploads twice the bytes, on the wrong stream, and we did not stop
  it.** Per training batch OpenNN copies 19,267,584 B (the batch in bf16) on
  a dedicated transfer stream behind a prefetch thread, overlapped with the
  kernels; PyTorch copies 38,535,168 B (the same tensor in fp32, because
  `ToTensor` produces fp32 and the driver never casts before `.to()`) on the
  compute stream, where it serialises. That copy is 2,038.7 µs of the 7,163
  µs per-batch wall difference, 28% of this cell's margin, and neither cause
  is to OpenNN's algorithmic credit: half is bytes we let PyTorch send and
  half is where we let it send them. At the copy's measured 18.9 GB/s a bf16
  upload would cost about 1.02 ms, which splits it roughly evenly.
  `PT_INPUT=cache` cannot detect either, since it also produces fp32.
- **The memory column is device-used minus idle.** `PROTOCOL.md` §5 measures
  peak GPU memory as whole-device used memory minus the idle reading, so both
  numbers include the CUDA context and every block PyTorch's caching
  allocator is holding but not using. That is deliberate — the memory is
  genuinely unavailable to anything else — but it charges an allocator that
  caches for caching, and it is not `torch.cuda.max_memory_allocated()`. A
  reader comparing the 1.19× and 1.34× against tensor accounting will get
  different numbers.
- **Different cuDNN builds.** OpenNN links the system cuDNN 9.25.1; PyTorch's
  wheel bundles 9.23.2. Both pick their convolution kernels by trial
  (`cudnn.benchmark` against OpenNN's autotune, which also caps the inference
  workspace at 16 MiB), so the kernels can and do differ: in the training
  window only 12 convolution kernel names appear on both sides, out of 48 on
  OpenNN's and 51 on PyTorch's. Some of the difference is the fused epilogue
  OpenNN asks for and some of it is the library version, and this profile
  cannot separate the two. On this network the difference runs *against*
  OpenNN — its convolutions cost 0.9% more per training batch and 14.8% more
  per inference batch — so the margins reported here are not resting on a
  luckier draw of convolution kernels.
- **PyTorch's autotuner beat OpenNN on some convolutions outright.** Under
  `max-autotune-no-cudagraphs` Inductor benchmarked its own Triton
  convolution templates against the cuDNN call and kept the Triton kernel for
  41 of its 191 convolution launches per training batch, 4,231 µs of its
  23,039. OpenNN has no equivalent: it chooses among cuDNN's engines only.
- **OpenNN pays for a cuDNN weight-gradient plan it cannot get.** For all 53
  convolutions, across 23 distinct shapes, cuDNN 9.25.1 reports no engine on
  this architecture for a weight gradient with an FP32 output tensor, so the
  driver falls back to a bf16 store followed by a widening cast — the 53
  `cast_kernel` launches and 89.6 µs per batch in the table above, and a
  weight gradient accumulated in bf16 rather than fp32. It is 0.24% of the
  step and it is a real numerical difference from PyTorch's path, not just a
  performance one.
- **The resize is not the same resize.** OpenNN's cache is a direct bilinear
  resize to 224×224; torchvision resizes the short side to 224 and
  centre-crops. Same tensor shape, different pixels; the parameter count and
  sample count are what the gate compares, and two epochs from a random
  initialisation are not long enough for an accuracy that would separate the
  two.
- **No accuracy gate.** The runner's quality gate only reads `test_accuracy`,
  which only the dense training drivers report, and neither CNN driver prints
  a loss in its timed modes; this family is held to the shape gate —
  25,557,032 parameters, 49,984 samples per training epoch at batch 64 and
  49,920 per inference pass at batch 128. The gate is recorded as agreeing
  because every accuracy in it is `NaN`, which is a weaker statement than it
  looks (see *What is measured* for the scope).
- **The two cells run at different batch sizes.** Training publishes at batch
  64 and inference at 128. Every ratio compares two engines at the same
  batch, but the memory and energy columns of the two rows are not
  comparable with each other, and the mode-selection and `PT_INPUT` variants
  quoted above were measured at batch 128 for both cells.
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
  device view; the last session to publish the unfixed inference cell at this
  batch read 7,049 samples/s for OpenNN against the 7,077 here.

## Reproduce

```bash
export OPENNN_BENCH_SESSION=$(date +%F)-mine
python run.py --family cnn --mode train --device cuda --batch 64  --precision bf16 --epochs 2 --rounds 3
python run.py --family cnn --mode infer --device cuda --batch 128 --precision bf16 --repeats 5 --rounds 3
```

`prepare.py cnn` lays out the ImageNet validation subset as class folders; the
first OpenNN launch builds the cache (libjpeg, one decode per image, on all
cores) and every launch after it checks the cache's signature. `PT_INPUT=cache`
feeds PyTorch from the same file, `PT_COMPILE_MODE` and `PT_INFER_CAST=autocast`
are its other knobs; `OPENNN_NO_CUDA_GRAPH=1` runs OpenNN without graph replay;
`OPENNN_CONV_WORKSPACE_MB` and `OPENNN_CONV_AUTOTUNE` control the convolution
plan search, and `OPENNN_LT_TILE_TOLERANCE` the GEMM tile rule — the tile runs
quoted above were at batch 64 on `bc6f4c2d0`, not at the published
configuration.

The retained sweeps say less about those than a reader would want. Capping
the inference workspace at 16, 32, 64 and 256 MiB moved the cell 7,024 /
7,042 / 7,029 / 7,038 samples/s, a 0.26% band (`cuda-cnn-infer-sweep-ws-*`,
batch 128, commit `e8bf14d5d`); no retained run has `conv_autotune` off, so
that half is unmeasured; and two sibling sweeps recorded the same
`conv_autotune=on`, `conv_workspace_mib=16` and read 7,042 and 7,182, 2.0%
apart, with nothing in either artifact saying what differed. None of them ran
at the published batch or commit.
