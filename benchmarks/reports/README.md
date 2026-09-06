# OpenNN against PyTorch: the results, and why

One document per family, each answering the same three questions: what was
measured, what the numbers are, and *why* the margin is what it is — with the
evidence for the why, so that a reader who distrusts a number can see what
would have to be wrong for it to be wrong.

| document | cells |
|---|---|
| [`dense.md`](dense.md) | HIGGS classifier: CUDA and CPU, training and inference |
| [`cnn.md`](cnn.md) | ResNet-50 v1.5 on the ImageNet subset: CUDA training and inference |
| [`transformer.md`](transformer.md) | the *Attention Is All You Need* base model on WMT14: CUDA training and inference |
| [`lstm.md`](lstm.md) | LSTM forecasting on Beijing PM2.5: CUDA and CPU, training and inference |
| [`footprint.md`](footprint.md) | what each framework costs before it does any work |

The contract the numbers were taken under is [`../PROTOCOL.md`](../PROTOCOL.md);
this page summarises only what a reader needs to interpret the table.
TensorFlow is not in the matrix: no build ships `sm_120` kernels, so it runs
here on driver-JIT'd PTX at a cost (~25% on a 4096³ matmul) with no native
build to difference against, and being unattributable is what disqualifies
it (PROTOCOL §2).

## The results

Twelve cells. Every cell wins every axis: throughput, peak memory, energy.
Session `2026-09-05-publish`, commit `93cc90e07`, clean tree, GPU clock locked
at 2,692 MHz, turbo off, every gate passing in every row.

Throughput is OpenNN / PyTorch; memory and energy are PyTorch / OpenNN, so
that above 1 always means OpenNN is ahead.

| cell | batch | precision | OpenNN /s | PyTorch /s | thr | peak MiB ON/PT | mem | Wh ON/PT | energy |
|---|---|---|---|---|---|---|---|---|---|
| `cpu-dense-infer` | 4,096 | fp32 | 220,647 | 171,867 | **1.284×** | 316 / 575 | 1.82× | 0.0978 / 0.1056 | 1.079× |
| `cpu-dense-train` | 4,096 | fp32 | 70,140 | 55,788 | **1.257×** | 339 / 790 | 2.33× | 0.0864 / 0.0960 | 1.111× |
| `cpu-lstm-infer` | 256 | fp32 | 80,394 | 69,269 | **1.161×** | 189 / 463 | 2.45× | 0.0217 / 0.0234 | 1.081× |
| `cpu-lstm-train` | 256 | fp32 | 23,147 | 13,016 | **1.778×** | 242 / 592 | 2.44× | 0.0429 / 0.0609 | 1.420× |
| `cuda-cnn-infer` | 128 | bf16 | 7,077 | 5,635 | **1.256×** | 944 / 1,268 | 1.34× | 2.4739 / 3.1247 | 1.263× |
| `cuda-cnn-train` | 64 | bf16 | 1,682 | 1,399 | **1.202×** | 3,557 / 4,235 | 1.19× | 3.8509 / 4.2685 | 1.108× |
| `cuda-dense-infer` | 8,192 | bf16 | 39,412,929 | 38,681,438 | **1.019×** | 378 / 412 | 1.09× | 0.1729 / 0.1829 | 1.058× |
| `cuda-dense-train` | 8,192 | bf16 | 11,396,057 | 10,091,130 | **1.129×** | 508 / 632 | 1.24× | 0.1320 / 0.1517 | 1.149× |
| `cuda-lstm-infer` | 256 | bf16 | 918,696 | 524,191 | **1.753×** | 294 / 442 | 1.50× | 0.0459 / 0.0633 | 1.377× |
| `cuda-lstm-train` | 256 | bf16 | 290,819 | 104,815 | **2.775×** | 326 / 532 | 1.63× | 0.0561 / 0.1212 | 2.160× |
| `cuda-transformer-infer` | 32 | bf16 | 5,335 | 4,695 | **1.136×** | 863 / 1,161 | 1.35× | 11.5286 / 14.9475 | 1.297× |
| `cuda-transformer-train` | 32 | bf16 | 1,329 | 1,145 | **1.161×** | 2,217 / 3,349 | 1.51× | 16.3427 / 22.6253 | 1.384× |
| **geomean** | | | | | **1.351×** | | **1.60×** | | **1.265×** |

Throughput and energy are the median of the three launches an engine makes in
a cell; peak memory is the highest of the three, so the memory column is the
worst case rather than the typical one. Memory on CUDA cells is device use
above idle; on CPU cells it is the process's peak anonymous RSS. Energy on
CUDA cells is whole-board NVML power integrated over the timed window; on CPU
cells it is the RAPL `package-0` counter. The energy geomean therefore pools
two different instruments measuring two different things, and is a summary of
twelve ratios rather than of any physical quantity.

| footprint question | OpenNN | PyTorch | PyTorch / OpenNN |
|---|---|---|---|
| memory | 0.123 s, 118 MiB | 3.182 s, 449 MiB | **25.9×** the time, 3.8× the memory |
| startup | 0.568 s, 325 MiB | 1.885 s, 375 MiB | **3.3×** the time, 1.2× the memory |
| export | 0.448 s, 156 MiB | 1.905 s, 376 MiB | **4.3×** the time, 2.4× the memory |

## What a sceptical reader should know before believing the table

**PyTorch is measured at its best, and on one cell it got better since we last
published.** Between the previous round and this one the machine was rebooted,
which cleared PyTorch's `max-autotune` cache; it re-tuned `cuda-dense-infer`
and found a kernel about 4% faster than the one it had been running. Its
median on that cell went from 37,178,529 samples/s in session
`2026-09-03-publish` to 38,681,438 here, on the same machine, the same driver
610.43.02 and the same PyTorch 2.13.0+cu130, with the dense drivers unchanged
between the two commits (`git diff 6b7179dde 93cc90e07 -- benchmarks/families/`
is empty). The protocol says each engine at its best, so the faster PyTorch is
the honest comparison — which means the **1.004×** we published for that cell
last round was measured against an unlucky draw of PyTorch's autotuner, and
should not be cited. Against the kernel PyTorch now finds, the matmul path we
published last round, reproduced here with `OPENNN_CUDNN_MATMUL=0`, reads
0.966×.

**`cuda-dense-infer` is the cell that changed, and it is the narrowest.** It
reads 1.019× — a 1.9% margin, against launches far tighter than that: OpenNN's
three read 39,412,929, 39,413,370 and 39,412,612, PyTorch's 38,690,821,
38,680,896 and 38,681,438, spreads of 0.002% and 0.026%. It reads 1.019× only
because of one intervention, isolated here by a single variable:

| configuration | throughput | energy |
|---|---|---|
| published cell | 39,412,929/s, 1.019× | 0.17289 Wh, 1.058× |
| `OPENNN_CUDNN_MATMUL=0` | 37,387,210/s, 0.966× | 0.13732 Wh, 1.315× |
| `OPENNN_MATMUL_CROSS_SOURCE_GAIN=100` | 37,392,713/s, 0.966× | 0.13676 Wh, 1.321× |
| `OPENNN_LT_TILE_TOLERANCE=0` | 39,412,842/s, 1.019× | 0.17190 Wh, 1.057× |

Row one is the published run itself; the three variant rows are session
`2026-09-05-variants` at the same commit, each carrying its own three PyTorch
launches, which is why their ratios are internally valid.

Disabling the cuDNN plan returns the cell to within 0.2% of the throughput it
had before the work — 37,387,210/s against 37,317,959/s on 3 September — that
is, to 0.966× against the kernel PyTorch now finds. Raising the cross-source
gain to 100%, a different knob that suppresses the same decision, gives the
identical result. That is the mechanism, not an inference from a profile.
Tolerance 0 reads within 0.0002% of the published cell, but that row proves
less than it looks: with the tolerance at zero the library picks by measured
time across both sources and the cross-source gain test is bypassed
altogether, so it shows the two paths reaching the same kernel rather than the
tile-traffic rule being irrelevant. That rule still governs every shape where
cuDNN declines, which is where `cuda-dense-train`'s energy margin comes from.

**Against cuBLASLt's fastest kernel, the cuDNN engine is inside the noise
floor.** It is 1.5% faster — 189.8 us against 192.4 — and the library's own
cross-source margin is 2%, which its source calls a noise floor rather than a
trade rate. What makes it selectable is the anchor: the margin is measured
against the 201.2 us tile stage 1 chose in order to buy modelled energy, which
turns 1.5% into 6.1%. `device_backend.cpp`'s stage-2 comment states it
outright — "the anchor, not the margin, is why cuDNN is taken here", and
"where stage 1 traded time for energy, stage 2 can hand the trade back without
pricing it". So this reversal is OpenNN's own energy-for-time policy being
unwound at a second stage, not a kernel cuBLASLt cannot reach: cuBLASLt
exposes one within 1.4% of it. The control that separates the two readings is
`OPENNN_MATMUL_CROSS_SOURCE_ANCHOR_FASTEST=1`, which keeps cuDNN in the
candidate set and changes only the anchor. No artifact in the store sets it.
It was not measured this round, and it is the open control on this cell.

**The energy on that cell was traded down deliberately, and is not an
improvement.** Winning the throughput axis required the faster kernel, and the
faster kernel is the hotter one: 1.019× throughput comes with 1.058× energy,
where the slower kernel gave 0.966× throughput at 1.315× energy. The rule
applied was to clear the constraint on every axis first and maximise
afterwards, so a cell that wins energy by 1.3× while losing throughput was not
acceptable and a cell that wins both narrowly was. The variant table above is
there so the price is visible: roughly 0.26× of energy ratio bought 0.053× of
throughput ratio. Anyone who cares more about joules than about samples per
second should set `OPENNN_CUDNN_MATMUL=0` and read row two.

**Two cells are noisier than their own third digit.** `cuda-dense-train` is
noisy on PyTorch's side: its three launches inside the published run read
10,163,656, 9,526,841 and 10,091,130, a 6.7% band, so against OpenNN's
11,396,057 the cell reads between 1.12× and 1.20× depending on the draw and
the 1.129× in the table sits near the low end. `cuda-lstm-train` varies 3.1%
on OpenNN's own side inside the published run — 298,968, 290,819, 290,028 —
and 1.7% across the three gate-passing medians at this commit (289,372,
290,819, 294,210); PyTorch's three launches in the same run spanned 97,006 to
106,586, so that cell's 2.775× is a point on a band of 2.7× to 3.1×. Neither
cell's ordering is in doubt; their third digit is.

## Where a win is not to OpenNN's credit

**The two engines do not share a cuDNN.** OpenNN links the system
`libcudnn.so.9` at 9.25.1; PyTorch loads the 9.23.2 its wheel bundles
(PROTOCOL §1). This matters more this round than last, because the kernel that
turned `cuda-dense-infer` positive *is a cuDNN engine* — one found by timing
cuDNN's engine configurations rather than trusting its heuristic, but selected
out of a library PyTorch does not have. Whether PyTorch's 9.23.2
exposes the same engine was not measured. The CNN and transformer families are
cuDNN-bound for the same reason, so no margin of a few percent in those cells
should be attributed to the framework without checking this first.

**On the GEMM itself, PyTorch is not behind.** Timed alone at the cell's shape
(m=1024, n=8192, k=1024, bf16, bias and ReLU fused), PyTorch's Triton kernel
takes 189.6 us and the cuDNN engine OpenNN now runs takes 189.8 us at 227.7 W.
(`cudnn_matmul.h` records the same engine at 189.6 us and 235 W; that is the
other operand layout the probe measured, and `cudnn_matmul.cpp` records why
the layout OpenNN already has gets the 189.8 us / 227.7 W one instead.) The
two matmuls are at parity within measurement. What is left is a subtraction,
not a profile: 8,192 samples take 207.9 us per batch on OpenNN against 211.8
on PyTorch, of which about 190 us is the same fused GEMM on both. OpenNN does
not win `cuda-dense-infer` with a faster matmul — the 1.9% is what remains
after a matmul of equal speed, and no in-situ profile of this cell exists this
round to attribute it further.

**On `cuda-lstm-infer`, PyTorch has the better algorithm, and both engines are
on cuDNN.** PyTorch reaches cuDNN's persistent kernel,
`RNN_blockPersist_fp_LSTM_HMMA`, 65.2% of its GPU time in 1,882 launches of
38.12 us. OpenNN asks for the same persistent algorithm —
`CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H` is its default (`cudnn_rnn.cpp`) — and
falls back to `CUDNN_RNN_ALGO_STANDARD` when cuDNN answers `NOT_SUPPORTED` for
its descriptor. What it runs is that standard path: `elemWiseRNNcell` at 34.2%
in 45,192 launches, plus the cuBLAS `nvjet` GEMMs the standard algorithm calls
at 52.1%. That is consistent with the cuDNN version asymmetry running against
OpenNN here, though the two descriptors were not compared. OpenNN still wins
the cell 1.753×; no measurement taken this round says why, and the launch
counts say it is not kernel quality.

**The CPU memory column is a whole-process peak.** The artifacts say so
explicitly: on CPU cells `peak_mib` is the process's peak anonymous RSS with
the framework baseline included, so much of the 1.8×–2.4× CPU memory margin is
a Python interpreter with `import torch` measured against a C++ binary, not an
allocator difference — a real cost to a user, but not an engineering result.
How much cannot be quantified from these runs. The baselines the drivers print
(758.7 MiB for PyTorch, 208.5 for OpenNN on `cpu-dense-infer`) are total-RSS
readings, which the artifact's own `workload_note` says are "not commensurable
with process_peak_anonymous_rss", and no anonymous-RSS baseline was captured.
The commensurable figures beside the column are the file-backed peaks: 455–527
MiB for PyTorch against 241.5–241.6 for OpenNN across the published launches.

**The four CPU cells are mostly not OpenNN's code.** This round's profiles put
95.7% of the dense inference layer and 95.4% of the dense training layer inside
one MKL call, `cpu:sgemm_wide`, and 98.4% of `cpu-lstm-infer` inside one oneDNN
call, `rnn:onednn_forward`. The two engines link different MKL builds
(PROTOCOL §1) and the oneDNN builds are not recorded per run at all; PyTorch's
CPU side was not profiled this round. A cell that is 98% one vendor primitive
cannot have its 1.161× attributed to the 1.6% of framework around it, and the
same holds for 1.284×, 1.257× and 1.778×. That is the argument this document
already accepts for cuDNN, applied to a third of the table.

**A harness setting moves one engine and not the other.** The runner exports
`GOMP_SPINCOUNT=300000` for both engines, because GCC 14's libgomp stops
spinning at barriers on a hybrid CPU while PyTorch's bundled copy still spins.
Measured, it is a no-op for PyTorch (69.0k to 69.3k samples/s) and takes OpenNN
from 62.6k to 72.3k, and PROTOCOL §6 records that this alone once turned a won
cell into a lost one. The defence is that it is libgomp's own documented
default, applied identically to both engines and recorded per launch as
`pinning.omp_wait` — but the CPU LSTM cells would read differently without it.

**`cuda-cnn-train` compares two different input pipelines.** PyTorch decodes
the JPEGs every epoch through worker processes; OpenNN reads its own
pre-decoded image cache. `PT_INPUT=cache` feeds PyTorch the same cache, but
the only run of it is at batch 128 under `compile:default` at commit
`918805ce1`, where the published cell is batch 64 under
`max-autotune-no-cudagraphs`, so the decode's share of the 1.202× is
unmeasured at the published configuration.

**OpenNN gets one more warmup than PyTorch on the CNN.** Training warms two
epochs for OpenNN against one for PyTorch, and inference gives OpenNN a warm
resident call *and* a full untimed pass against PyTorch's one pass. The
published artifact suggests it buys nothing measurable: PyTorch's epoch pairs
are 35.87/36.11, 35.63/35.65 and 35.64/35.72, with no first-epoch penalty to
warm away. It is an asymmetry all the same, and it favours OpenNN.

## The state of this round

All twelve cells are evidence-grade. `cuda-transformer-train`, which the
previous round could not publish because every attempt tripped the runner's
foreign-activity gate, was re-measured cleanly: `busy_during_max` 0.0156
against the 0.03 threshold. `cuda-lstm-train` was run twice under this session
id at this commit: the 5 September run passed every gate and read 3.016×
(289,372 against 95,954), and the 6 September re-run is published because it is
the later one, which puts the lower of the two valid readings in the table.
Every other row is that cell's publish run on 5 September.

Kernel-level evidence covers less than every cell. Profiles exist on OpenNN's
side for `cuda-cnn-train` (batch normalisation 31.5% of the step, convolutions
about 13%), `cuda-transformer-train`, `cuda-transformer-infer`, and the three
CPU cells a single vendor primitive dominates — `cpu:sgemm_wide` at 95.7% of
the dense inference layer and 95.4% of the dense training layer,
`rnn:onednn_forward` at 98.4% of `cpu-lstm-infer`. On both engines there is
`cuda-lstm-infer`; on PyTorch's side there is `cuda-cnn-train`, whose largest
single kernel is 19.9%. There is no profile of `cuda-dense-infer`,
`cuda-dense-train`, `cuda-cnn-infer` or `cuda-lstm-train` on either side this
round, and the GEMM timings quoted above are a standalone microbenchmark rather
than a trace. `cpu-lstm-train` was profiled twice on 6 September; both runs
were filed to scratch on the pre-launch quiet reading (0.065 and 0.064 against
the 0.03 threshold) and neither covered the training path, so that cell's
margin is unattributed. The traces themselves are not in `results/`, which
holds one JSON per cell and no kernel data, so a profile share is checkable
only against the document that quotes it. Where a family document still carries
*[pending the final measurement round]*, the measurement named there does not
exist and the document says what is missing rather than guessing.

Two interventions were measured and rejected, and belong here beside the one
that was adopted. Fusing the CNN's batch-normalisation backward would have
recovered a layer only where the re-read fits in the 48 MB L2, true for 114.0M
of 615.0M elements — about 1.3% of the step, against a grid-wide barrier inside
a captured graph; removing dead branches and registers instead was
bit-identical and moved the cell from 1.171× to 1.197× on the commit's own
two-round measurement (`cnn.md`). Adam was examined in the same pass and left
alone at 778.9 GB/s, 86.9% of pin; the launch-count hypothesis died because the
step is graph-captured, so its 74 inter-kernel gaps are 160-192 ns of graph-node
latency rather than the 3-8 us of a real launch (`transformer.md`).

The controlled variants measured at this commit are the three
`cuda-dense-infer` rows above. Any other variant a family document quotes was
taken at an earlier commit and is dated there.

## What "same work" means here

A cell compares one network definition, driven by two engines with the same
positional arguments and the same `key=value` output. Before any throughput is
reported, the runner checks:

- **shape** — sample count, sequence length, vocabulary, parameter count, as
  printed by each engine; a mismatch is not a comparison. The published
  transformer cells report 74,878,496 parameters on both engines, which is not
  the count PROTOCOL §4 currently states; the gate passes because it compares
  the engines against each other, and the contract is the side that is stale;
- **quality** — where a driver reports a test accuracy, which today is dense
  training only, both engines must agree within 2%; the other families are
  held to the shape gate, and each document says what that leaves unchecked;
- **whole batches only** — both engines drop the tail of the epoch;
- **warmup excluded** — training runs untimed epochs first (allocation, graph
  capture, autotuning, `torch.compile`: two for dense and LSTM, one for the
  transformer, two for OpenNN's CNN against one for PyTorch's); inference runs
  two untimed passes on each engine for dense, and for the CNN, transformer and
  LSTM one full untimed pass, which OpenNN and the LSTM driver precede with a
  single warm call. The family documents carry the exact counts;
- **what the inference cells time** — on the CNN and both LSTM inference cells
  each engine fills one batch, uploads it before the clock and replays it, so
  those cells measure the resident forward pass and exclude input handling on
  both sides. Dense inference walks the real test set.

Throughput is samples per second inside the engine's own timed window, which
begins after warmup and ends after a device synchronisation, so the clock stops
when the work is done and not when it was queued.

## Each engine at its best

An engine measured below its own ceiling makes the other look good for the
wrong reason, so both are configured the way their users would configure them:

| | GPU | CPU |
|---|---|---|
| OpenNN | captured CUDA graph (training, and CNN and transformer inference; the dense inference path runs three launches on device views, and the LSTM runs eager because the cuDNN RNN path refuses capture), device-resident split, MKL for CPU-side work | 16 threads on the P-cores, MKL, oneDNN for recurrent layers |
| PyTorch 2.13 | `torch.compile` in the mode measured best per cell (`reduce-overhead` for dense training, `max-autotune-no-cudagraphs` for dense inference, the transformer and CNN training, default for CNN inference), bf16 weights for inference, `channels_last`, TF32; eager for the LSTM, where compiling measured slower | eager (compiling measured slower), MKL, oneDNN for recurrent layers |

Every one of those choices was measured before it was adopted, and the driver
docstrings carry the numbers for every mode tried: CPU dense training reads
93,156 samples/s eager against 83,722 compiled, which is why PyTorch runs eager
on CPU (PROTOCOL §3). `PT_COMPILE_MODE`, `PT_INFER_CAST`, `PT_INPUT` and the
`OPENNN_*` variables override them, so the choices stay measurable rather than
baked in. The mode is recorded per launch for the training cells and for
`cuda-dense-infer`; the CNN, transformer and LSTM inference drivers print no
mode, so for those cells the table states the driver's own default rather than
something an artifact records.

## The machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti (`sm_120`), driver 610.43.02, 300 W limit, SM clock locked at 2,692 MHz (boost ceiling 3,090), persistence mode on |
| CPU | Intel Core i7-14700F: 8 P-cores (16 threads) + 12 E-cores; CPU cells pinned to the P-cores, CUDA cells given the whole CPU; governor `performance`, turbo disabled |
| RAM | 32 GB |
| OS | Ubuntu 24.04, Linux 7.0.0-30, glibc 2.39, native (not WSL) |
| CUDA / cuDNN | 13.3 / 9.25.1 (OpenNN); PyTorch's wheel bundles cuDNN 9.23.2 |
| PyTorch | 2.13.0+cu130, Python 3.12.3 |
| MKL | 2026.0.1 (OpenNN); PyTorch's wheel bundles its own |
| OpenNN | commit `93cc90e07`, branch `dev`, clean tree, Release, LTO, GCC 13.3 |

The oneDNN builds are deliberately absent: neither engine reports one and no
artifact records the field, which is the limit on how far the `cpu-lstm-infer`
attribution above can go. The rest of the table is in PROTOCOL §1 or in every
artifact's `frameworks`, `cpu` and `machine` blocks, except the last three
rows: RAM is the machine's, and the build type, LTO and compiler are
`build-bench`'s CMake cache, neither of which any artifact records.

The GPU clock is locked below its boost ceiling because a floating clock drifts
by more per session than the margins being measured (PROTOCOL §7). Turbo is
disabled on the CPU for the same reason. Both are recorded in every artifact.

The two shortest cells were lengthened this round so that their energy is
integrated over a usable number of samples rather than a handful:
`cuda-dense-train` now runs 100 timed epochs, about 22 s per launch and 108
power samples inside the window, and `cuda-lstm-train` 20 epochs, about 4.2 s
and 150 samples. Their throughput spread is still the widest in the table, for
the reasons given above.

## How to read the "why"

Each family document attributes its margin with two kinds of evidence:

- **A profile of the published launch command.** GPU cells were traced with
  Nsight Systems and the trace reduced to the GPU-busy fraction inside the
  engine's timed window, the kernel launch rate, the gaps between kernels, and
  the kernels that took the time. CPU cells were sampled with `perf record` and
  split by shared object and symbol inside the same window. The profiler
  invocation is not recorded in the artifacts and the traces are not in
  `results/`. A profiled launch is a separate run from the published one; both
  throughputs are quoted so the profiler's overhead is visible.
- **A controlled variant.** Where a single mechanism is claimed to explain a
  margin, the document shows the cell measured with that mechanism removed or
  swapped (`OPENNN_CUDNN_MATMUL`, `PT_COMPILE_MODE`, `PT_INPUT`,
  `GOMP_SPINCOUNT`, …), so the attribution is a measurement rather than an
  inference from a profile — except where the document says the control was not
  run, as with `OPENNN_MATMUL_CROSS_SOURCE_ANCHOR_FASTEST` above.

One method note, because it produced this round's only reversal: enumerate, do
not trust the heuristic. Twice on this one shape the heuristic was the thing in
the way. cuBLASLt's offers eight candidates for the dense inference GEMM and
never the 256x160 tile that costs 34% less energy for 4% more time;
cuDNN's mode A ranks a 226 us engine first where its best runs at 189.6
(`device_backend.cpp`, `cudnn_matmul.cpp`). Reaching that engine took
enumerating all 13,460 valid cuBLASLt configurations (exactly one is below
198 us, and it draws 265 W), instantiating five CUTLASS 3.8 tile shapes, and
hand-writing six `mma.sync` kernels — eleven kernels written and verified
correct against a cuBLASLt reference, none fast enough. The cuDNN side was
sampled rather than swept: sixty engine configurations were built, executed and
verified against a cuBLASLt reference, and thirteen beat both the time and the
energy bound (`cudnn_matmul.h`).

Where the margin owes something to an asymmetry that is not the framework —
a different cuDNN or oneDNN build, a pre-decoded image cache against a JPEG
decoder — the document says so and either says how much or says the control was
not run. Several on this page are the second kind.

## Reproducing

```bash
sudo ./gpu_clocks.sh lock 2700                   # PROTOCOL §7; the artifact records the clock
export OPENNN_BENCH_SESSION=$(date +%F)-mine
python run.py --family dense --mode train --device cuda --batch 8192 --precision bf16 --epochs 100 --rounds 3
python run.py --family dense --mode infer --device cuda --batch 8192 --precision bf16 --repeats 200 --rounds 3
python run.py --family cnn   --mode train --device cuda --batch 64   --precision bf16 --epochs 2 --rounds 3
python run.py --family cnn   --mode infer --device cuda --batch 128  --precision bf16 --repeats 5 --rounds 3
python run.py --family transformer --mode train --device cuda --batch 32 --precision bf16 --epochs 2 --rounds 3
python run.py --family transformer --mode infer --device cuda --batch 32 --precision bf16 --repeats 5 --rounds 3
python run.py --family lstm  --mode train --device cuda --batch 256  --precision bf16 --epochs 20 --rounds 3
python run.py --family lstm  --mode infer --device cuda --batch 256  --precision bf16 --repeats 50 --rounds 3
python run.py --family dense --mode train --device cpu  --batch 4096 --precision fp32 --epochs 3 --rounds 3
python run.py --family dense --mode infer --device cpu  --batch 4096 --precision fp32 --repeats 5 --rounds 3
python run.py --family lstm  --mode train --device cpu  --batch 256  --precision fp32 --epochs 3 --rounds 3
python run.py --family lstm  --mode infer --device cpu  --batch 256  --precision fp32 --repeats 5 --rounds 3
python run.py --family footprint
```

Run them from a shell whose CPU affinity is the whole machine (`taskset -pc $$`
should list every CPU): the runner pins CPU cells itself, but CUDA launches
inherit the shell's mask, and a shell confined to the E-cores halves every CUDA
number on both engines without failing any gate.

The artifacts behind this page are on the reference machine under
`benchmarks/results/`, one JSON per cell, named by `run_id`; the documents
quote every launch, not only the median, so the spread is visible without them.
