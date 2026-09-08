# Footprint: what a framework costs before it works

Three questions answered by processes that do almost nothing: what a process
weighs once the framework is loaded and an empty model exists, how long it
takes to make a first prediction, and what a trained model exports to. The
family sits outside the twelve-cell table and its geomeans — there is no
throughput here and no round structure, and each question is a single
unrepeated launch per engine, against the twelve cells' median of rounds.
Session `2026-09-06-publish`, commit `e76425bd3`, both engines pinned to the
P-cores; the three rows are in Results below.

None of the three is about a kernel. OpenNN is one executable that links
`libopennn.a` statically and twelve shared libraries dynamically, five of them
CUDA (`readelf -d build-bench/bin/footprint_opennn`), and its empty model
touches few of their pages. A PyTorch process is the CPython interpreter plus
`import torch`, which maps and runs the static initialisers of libtorch:
946 MiB of shared objects in `torch/lib` alone, `libtorch_cpu.so` 418 MiB and
`libtorch_cuda.so` 448 MiB, with fifteen `nvidia-*` packages of CUDA libraries
beside them. Both engines here are CUDA builds, so the gap is not a
build-configuration artefact — but most of what PyTorch carries is generality
OpenNN does not offer: device code for six SM architectures (`sm_75` through
`sm_120`, recorded as `torch_arch_list` in the artifact), cuFFT, cuSOLVER,
cuSPARSE and NCCL beside cuBLAS and cuDNN, and a dispatcher registering every
operator it ships. A reader who needs that breadth is buying the size on
purpose.

## What is measured

The footprint family asks what each framework costs merely by existing. Speed,
peak memory and energy are readings of a run in progress; these three
questions are answered by processes that do almost nothing, one process per
question, because a cost paid at startup is already paid by anything that
shares the process with it. All six launches are pinned to the P-cores like
the CPU cells. No launch runs a GPU benchmark, but OpenNN's `startup` executes
its forward pass on the card and its `export` creates a CUDA context — both
below — and `run.py` classes all six as CPU launches, so neither the card's
memory nor its power is sampled: whatever the GPU costs in those two launches
is outside every number in this document.

| question | OpenNN process | PyTorch process | recorded |
|---|---|---|---|
| `memory` | link the library, construct an empty `Network`, `TabularDataset` and `TrainingStrategy` | `import torch`, construct an empty `nn.Sequential` and an `Adam` over one tensor | peak anonymous resident set (`RssAnon` from `/proc/<pid>/status`, polled by the runner and reported as `peak_mib`); the drivers' own `/proc/self/statm` print is total RSS and appears separately as `baseline_ram_mb` |
| `startup` | construct `ApproximationNetwork({10},{64},{1})` and predict on one row of ones | construct `Linear(10,64) → Tanh → Linear(64,1)` and predict on one row of ones | seconds from process entry to the prediction, plus the whole-process wall time the runner measures around the launch |
| `export` | train `ApproximationNetwork({3},{64},{1})` for 50 epochs on a 512-row synthetic sum and write it as `.c` and `.py` through `ModelExpression` | `torch.jit.script` a `Linear(3,64) → Tanh → Linear(64,1)` and save it | bytes of the exported files, and a driver-declared `standalone_source` flag on the PyTorch side; neither export is executed by the run |

**What the numbers mean.** `memory` deliberately counts the import: for
OpenNN the equivalent cost is paid by the dynamic loader before `main()`, so
neither figure is "the library's own size" — both are what a process weighs
once the framework is available and an empty model exists, which is the
comparable quantity. The MiB column is peak anonymous resident set; the
caveats say what that excludes and which way the choice cuts. `startup` has
two readings on purpose, an in-process timer and the runner's `wall_seconds`;
the tables report the wall time, for the reason the caveats give. `export` is
the one question with a qualitative answer: OpenNN writes source that computes
the model with no framework runtime, and PyTorch has no equivalent —
TorchScript and ONNX both produce artifacts that need libtorch or onnxruntime
to execute — so the PyTorch driver reports the TorchScript size with
`standalone_source=0` rather than pretending the two files are the same kind
of thing.

**Energy.** The suite's third axis is missing from this family in this
session. All six launches record `energy_measurable: false` and
`energy_domain: null`: `energy_uj` is root-only after CVE-2020-8694, and it
was unreadable for every CPU launch between 11:02 and 11:09 UTC. The same
session read `package-0` normally for `cpu-dense-infer` at 11:28
(0.097716 Wh against PyTorch's 0.105261), so what changed was the counter's
readability, not the workload, and the family was not relaunched afterwards.
Earlier sessions did record it — in `2026-09-01-publish`, commit `b4abb3cc6`,
`memory` read 0.000268 Wh against 0.007880, `startup` 0.001266 against
0.004704 and `export` 0.001233 against 0.004669 — but those are that
session's numbers, not this table's missing column. Re-running
`run.py --family footprint` with the counter readable would fill it.

**Gates.** None beyond the process exiting with `RESULT=OK`; there is no
throughput here and nothing to agree on across engines.

## Results

Session `2026-09-06-publish`, commit `e76425bd3`, torch 2.13.0+cu130. The
family ran twice in that session; the table is the second run, and the first
is quoted below it.

| footprint question | OpenNN (wall, peak anon RSS) | PyTorch (wall, peak anon RSS) | PyTorch / OpenNN |
|---|---|---|---|
| memory | 0.123 s, 118.3 MiB | 3.201 s, 449.4 MiB | **26.0×** the time, 3.8× the memory |
| startup | 0.569 s, 320.5 MiB | 1.886 s, 374.8 MiB | **3.3×** the time, 1.17× the memory |
| export | 0.184 s, 124.3 MiB | 1.909 s, 375.6 MiB | **10.4×** the time, 3.0× the memory |

The session's other launch reads 0.123 s / 118.4 MiB, 0.569 s / 321.0 MiB and
0.184 s / 124.3 MiB on OpenNN's side, and 3.200 s, 1.905 s, 1.908 s on
PyTorch's — everything within a per cent of the table. The `export` row is
the one that moved since the previous table, where it read 0.448 s and
156.3 MiB: the reason is a library fix described under *Why*, and it is the
same fix that took 33–35 MiB off every CPU cell of the main matrix.

## Why

All three questions are about what a process has to load, initialise and
carry before the first useful instruction.

### `memory`: a linked library against an imported package

The OpenNN process links `libopennn.a` statically and maps twelve shared
libraries — MKL, oneDNN, libgomp, and the CUDA runtime, cuBLAS, cuBLASLt,
NVRTC and cuDNN. A mapped library costs resident memory only for the pages
that are touched, and constructing an empty `Network`, an empty
`TabularDataset` and a `TrainingStrategy` touches very few. Three readings
describe the result and they do not reconcile: the runner's polling records a
118.2 MiB anonymous peak and an 87.9 MiB file-backed peak, each an
independent high-water mark taken over the whole process, while the driver
prints 209.4 MiB of total resident at one instant near the end — its field is
named `baseline_ram_mb` but is computed in MiB. The two peaks need not
co-occur, and the point reading is a different quantity; the runner says so in
every launch's `workload_note`. The tables report the anonymous peak, reached
in 0.123 s.

None of that is a CUDA context. `Configuration::set` only records the
requested device and bumps a generation counter; `has_cuda_device()` is a
cached `cudaGetDeviceCount`; and the `Backend` that owns the streams, the
cuBLASLt handle and the cuDNN op descriptor is a function-local static in
`device_backend.cpp`, reached the first time something runs, which empty
objects never do. Until this round, reaching it was enough: its constructor
created the streams and both handles whenever `cudaGetDeviceCount` found a
device, `Device` setting notwithstanding, so a CPU-only process that trained
anything held a CUDA context. The `export` cell measured exactly that cost.
It runs `Device::CPU`, trains for 50 epochs and therefore certainly builds
the `Backend`; at `93cc90e07` it peaked at 156.3 MiB against this cell's
118.2 and took 0.448 s, and the previous version of this document bounded
"the CUDA state and everything 50 epochs of Adam allocates, together" at
38.1 MiB. `e76425bd3` creates the CUDA side on first CUDA use instead
(`Backend::ensure_cuda`), and the same cell now peaks at 124.3 MiB in
0.184 s: the context was 32 MiB of the 38 and 0.26 s of the 0.45, and 50
epochs of Adam on a network this size are the remaining 6 MiB. A process
that never touches the GPU no longer maps `/dev/nvidia*` at all, which the
runner can see in `/proc/<pid>/maps`.

The PyTorch process is the CPython interpreter plus `import torch`, and the
import is where the size goes: it loads `libtorch_cpu.so`, `libtorch_cuda.so`,
`libc10.so` and the cuDNN, cuBLAS and NCCL objects the `nvidia-*` wheels
install, running their static initialisers and the operator-registration code
that populates PyTorch's dispatcher with every kernel it ships. The process
ends at a 449.4 MiB anonymous peak and a 565.4 MiB file-backed peak — the
file-backed figure is the top of a 395.6–565.4 MiB band across the eight
publish launches, because it tracks page-cache state — with 815.8 MiB of total
resident by the driver's own instantaneous print, in 3.182 s. The ratio is a
property of the delivery, not of the empty model: a C++ program linking
libtorch would skip the interpreter and the Python imports but still pay the
shared-object initialisation, so it would land between the two columns rather
than beside OpenNN's.

What these artifacts do not show is how the 3.182 s divides. The `startup`
process imports the same `torch`, builds a real two-layer stack and predicts
with it in 1.513 s measured from module entry — which in `footprint.py` is
before the `import torch`, so that reading contains the import — against
1.885 s of wall, the 0.372 s difference being CPython's own start. Subtract
both and 3.182 − 0.372 − 1.513 ≈ 1.3 s of the `memory` cell is spent
elsewhere. The two processes differ in both directions: `memory` builds an
empty `nn.Sequential` and an `Adam` over one live tensor, `startup` calls
`manual_seed`, builds two `Linear` layers and runs a forward pass under
`no_grad`. Which of those spends the 1.3 s, or the 74.6 MiB by which
`memory`'s anonymous peak exceeds `startup`'s, is not recorded; an `Adam` over
one `torch.zeros(1, requires_grad=True)` allocates two one-element state
tensors, so the optimizer object itself cannot be the answer, whatever it
pulls in behind it. Until someone times `import torch` on its own (`-X
importtime` would do it), the 25.9× on this row should be read as the cost of
this particular pair of processes, not as `import torch` against
`libopennn.a`.

### `startup`: the first prediction

Both processes predict on one row of ones from a 10 → 64 → 1 stack, and the
arithmetic is a few thousand flops that appears in neither number. The stacks
are not identical: `ApproximationNetwork({10},{64},{1})` also builds scaling,
unscaling and clamping layers, five in all against PyTorch's three modules
(`models.cpp:85-108`). The extra arithmetic is negligible; the difference
matters for the export sizes below.

For OpenNN the in-process time to the prediction is 0.343 s (0.373 s in the
session's other launch). The `memory` cell is the same binary, the same
`Device::Auto`, the same loader work, and stops just short of running
anything: 0.123 s at 118.3 MiB. The difference between the two cells —
0.446 s of wall and 202 MiB — is what first use costs, and most of it is not
context creation. The `export` cell measured the context itself this round,
by losing it: 32 MiB and 0.26 s (see below), so about 170 MiB of the 202 is
first *GPU use*:
`Device::Auto` resolves to CUDA whenever a device is present
(`configuration.cpp`, `resolve_effective`), `ApproximationNetwork`'s
constructor compiles the network (`models.cpp:41-45`), and `calculate_outputs`
then takes the `is_gpu()` branch (`network.cpp:1117`), which loads CUDA
modules, builds the per-lane cuBLAS and cuDNN handles and allocates on the
device. How the 202 MiB and the 0.446 s divide between the context and the
rest of first use was not measured: the run that would separate them is the
same binary with the GPU hidden (`CUDA_VISIBLE_DEVICES=`), and no artifact
under `results/` contains it, so the split is left unmeasured rather than
estimated. The wall time the runner measures around the whole process adds the
loader's work — mapping the CUDA libraries is most of it — and reads 0.568 s.

For PyTorch the wall time is 1.885 s and the in-process reading 1.513 s, the
difference being the interpreter's own start. Nothing in the comparison is
unfair to PyTorch — the process is doing what any PyTorch script does before
its first line of model code — but the row means something narrower than it
looks. The 3.3× is "import torch" against "initialise CUDA and run on it",
and the PyTorch process never touches `torch.cuda`, so it never pays for a
context at all. The memory axis is that same trade read backwards: 1.16× is
not two comparable processes but OpenNN paying 202 MiB to reach the GPU and
still landing below a PyTorch process that pays none. Put either engine on the
other's footing — a CPU-only OpenNN, a PyTorch process that initialises CUDA —
and the row moves, in opposite directions.

### `export`: what the trained model becomes

This is the one question with a qualitative answer, and the timing is the
least interesting part of it. OpenNN's `ModelExpression` writes the network as
source: a C file of 15,104 bytes and a Python file of 14,990, with the trained
weights inlined as literals. The dependency-free property is read off the
emitted source rather than tested — `model.c` includes `<math.h>`, and
`<stdio.h>` unless `OPENNN_EXPORT_NO_MAIN` is defined, so it needs a C
standard library and no framework; no launch compiles or runs it. The process
also trains the model it exports (50 epochs of Adam at batch 32 on 512
synthetic rows), which is inside its 0.184 s. PyTorch's process scripts an
untrained model with `torch.jit.script` and saves it: 7,747 bytes of
TorchScript, a zip archive holding a serialised graph and the weights, which
runs only where libtorch is installed. The driver labels it
`standalone_source=0` — a declaration in the driver, not a test result —
because the two files are not the same kind of artifact: one is a deliverable,
the other a checkpoint for the framework that produced it. They also describe
different graphs, five OpenNN layers against three PyTorch modules. ONNX
export would give a third kind, a graph for onnxruntime, and would not change
the finding.

The memory ratio on this row, 3.0×, is two floors plus what each process then
does: OpenNN's 124.3 MiB is the 118.3 MiB `memory` floor plus 6 MiB for
every allocation 50 epochs of training makes, now that no CUDA context sits
between the two; PyTorch's 375.6 MiB is its import floor without the `Adam`
construction that carries its own `memory` cell to 449.4 MiB. The wall times
(0.184 s against 1.909 s) are once more the import against a small amount of
real work, with the training on OpenNN's side and none on PyTorch's. The
previous table's 0.448 s on this row was 0.26 s of CUDA context creation
plus the work; one launch in that session read 1.300 s, a reminder of how
little a single unrepeated wall time is worth, and the two launches of this
session agree to the millisecond.

## Asymmetries and caveats

- **`memory` does not include a CUDA context on OpenNN's side.**
  `Device::Auto` invites the suspicion that the OpenNN process initialises
  the CUDA runtime and carries a context the PyTorch process — which imports
  `torch` but never touches `torch.cuda` — does not pay. The `memory` section
  gives the code path: `Backend` is a function-local static reached on
  first use, three empty objects never reach it, and since `e76425bd3` even
  reaching it creates no CUDA state until a CUDA device is used — the
  `export` cell, which trains on the CPU for 50 epochs, now peaks 6 MiB
  above the `memory` cell where it peaked 38 MiB above it at `93cc90e07`.
- **The two `memory` processes do not construct equivalent objects.** OpenNN's
  `Network` has no layers, so `compile()` returns immediately
  (`network.cpp:546-548`): no device is resolved and no parameter
  storage is allocated, and its `TrainingStrategy` attaches a loss and an
  optimizer object to that empty network. PyTorch's process builds an empty
  `nn.Sequential` *and* a real `Adam` over a live
  `torch.zeros(1, requires_grad=True)`. By this document's own decomposition
  that difference could be roughly 1.3 s of the 3.182 s, so 25.9× is an upper
  bound on the cost of delivery, not a measurement of it.
- **The peak is sampled, not read.** `RssAnon` has no kernel-maintained
  high-water mark, so the runner polls `/proc/<pid>/status` every 20 ms and
  keeps the maximum. OpenNN's `memory` process lives 0.123 s — about six
  samples — against PyTorch's 3.182 s and about 159, and OpenNN's process is
  the shorter on every row (0.569 s against 1.886, 0.184 against 1.909). A
  sampled peak can be missed but never overstated, so the bias runs in
  OpenNN's favour by an unknown amount. `VmHWM` would settle it and is not
  recorded; the driver's own unsampled total-RSS print, 209.4 MiB, is a
  different quantity and neither confirms nor contradicts the sampled figure.
- **The MiB column is anonymous resident only.** File-backed pages are
  excluded because counting mapped pages penalises the engine that maps rather
  than copies (`common.py` gives the measurement behind that choice), and here
  the choice is the generous one to PyTorch: file-backed resident is where its
  delivery costs most, 565.4 MiB against OpenNN's 87.9 MiB on the `memory`
  cell, none of it in the 3.8×. It is also the least reproducible reading in
  the document — across the eight publish launches it runs 87.9–97.8 MiB on
  OpenNN and 395.6–565.4 MiB on PyTorch, tracking page-cache state rather than
  the framework — so the figures quoted above are one launch's, not a property
  of either engine.
- **The two `startup` timers do not start at the same place.** PyTorch's
  in-process timer is set at module entry, before `import torch`, so its
  1.513 s contains libtorch's loading and initialisation; what it misses is
  CPython's own start, 0.372 s of the 1.885 s wall. OpenNN's starts at
  `main()` and misses the dynamic loader entirely — `libopennn.a` is inside
  the binary, but MKL, oneDNN and the five CUDA libraries are `DT_NEEDED` and
  resolved before it. The asymmetry is on OpenNN's side, which is why the
  runner's `wall_seconds` around each launch is the like-for-like number and
  the tables report it.
- **The export sizes are different kinds of file**, as that section says, so
  they are reported for completeness rather than as a comparison. A reader who
  needs one should compare OpenNN's export against an ONNX file plus the
  runtime that executes it.
- **The `export` processes do different amounts of work.** OpenNN's trains the
  model it exports (50 epochs of Adam at batch 32 on 512 synthetic rows)
  before writing it; PyTorch's scripts a freshly constructed model and saves
  it, because there is nothing to gain from training a model whose export is
  the point. The asymmetry favours PyTorch on the wall clock and the OpenNN
  column still reads lower.
- **PyTorch's process is Python.** Every PyTorch number here includes the
  interpreter and the import of a 946 MiB `torch/lib`; an application that
  embeds libtorch from C++ would read lower on `memory` and `startup` than
  these cells, and higher than OpenNN by a smaller margin. The cells measure
  the framework as it is delivered to its users, which for PyTorch is the
  Python package.
- **The energy axis is unmeasured here.** No readable RAPL counter for these
  six launches, while the same session's CPU cells read `package-0` later the
  same morning; see *What is measured* for the artifact fields and for an
  earlier session's figures.
- **Nothing is repeated within a session.** Each question is one process,
  launched once, with no round structure; the spread has to come from
  relaunching the family. Across the eight publish-labelled launches from
  `2026-09-01-publish` to `2026-09-05-publish`, `memory` reads 0.123–0.144 s
  and 116.6–118.9 MiB on OpenNN and 3.182–3.241 s and 449.4–449.5 MiB on
  PyTorch; `startup` 0.568–0.610 s and 319.6–325.6 MiB against 1.885–1.946 s
  and 374.8–374.9 MiB; `export` 0.428–1.300 s and 156.3–156.4 MiB against
  1.888–1.964 s and 375.5–375.6 MiB (the two `2026-09-06-publish` launches,
  after the context fix, read 0.184 s and 124.3 MiB on that row). File-backed resident is far wider and has
  its own caveat above. Every reading is inside a few per cent of its median
  except one: OpenNN's `export` wall, 0.428–0.486 s in seven launches and
  1.300 s in the eighth. Those eight launches span five different commits, so
  the band is an upper bound on the run-to-run noise, not a measurement of it
  — and a single launch of a question that is never repeated can land where
  that eighth one did.

## Reproduce

```bash
export OPENNN_BENCH_SESSION=$(date +%F)-mine
python run.py --family footprint       # commit e76425bd3 for the table above
# The family ran twice in 2026-09-06-publish. The table is the second launch,
#   results/cuda-footprint-publish-20260906T133830Z.json
# and the readings quoted beside it are the first,
#   results/cuda-footprint-publish-20260906T133706Z.json
# Not yet run, and the way to split CUDA init from first GPU use:
CUDA_VISIBLE_DEVICES= build-bench/bin/footprint_opennn startup
```
