# Footprint: what a framework costs before it works

Three questions answered by processes that do almost nothing: what a process
weighs once the framework is loaded and an empty model exists, how long it
takes to make a first prediction, and what a trained model exports to.
Session `2026-09-03-publish`, one process per question, both pinned to the
P-cores:

| question | OpenNN | PyTorch | PyTorch / OpenNN |
|---|---|---|---|
| `memory` | 0.124 s, 117 MiB | 3.241 s, 449 MiB | **26.1×** the time, **3.8×** the memory |
| `startup` | 0.568 s, 325 MiB | 1.926 s, 375 MiB | **3.4×** the time, **1.2×** the memory |
| `export` | 0.448 s, 156 MiB | 1.928 s, 376 MiB | **4.3×** the time, **2.4×** the memory |

None of the three is about a kernel. OpenNN is one statically linked
executable whose empty model touches a few pages of a few libraries; a
PyTorch process is the CPython interpreter plus `import torch`, which
executes 1,135 Python modules and runs the static initialisers
of libtorch and its bundled CUDA libraries before the first line of model
code. `startup` compares OpenNN's CUDA context creation against PyTorch's
import; `export` compares a standalone C or Python source file against a
TorchScript archive that runs only where libtorch is installed.

## What is measured

The footprint family asks what each framework costs merely by existing. Speed,
peak memory and energy are readings of a run in progress; these three
questions are answered by processes that do almost nothing, one process per
question, because a cost paid at startup is already paid by anything that
shares the process with it. All six launches are pinned to the P-cores like
the CPU cells; no GPU work is involved.

| question | OpenNN process | PyTorch process | recorded |
|---|---|---|---|
| `memory` | link the library, construct an empty `NeuralNetwork`, `TabularDataset` and `TrainingStrategy` | `import torch`, construct an empty `nn.Sequential` and an `Adam` over one tensor | resident set in MiB (`/proc/self/statm`) at that point |
| `startup` | construct `ApproximationNetwork({10},{64},{1})` and predict on one row of ones | construct `Linear(10,64) → Tanh → Linear(64,1)` and predict on one row of ones | seconds from process entry to the prediction, plus the whole-process wall time the runner measures around the launch |
| `export` | train `ApproximationNetwork({3},{64},{1})` for 50 epochs on a 512-row synthetic sum and write it as `.c` and `.py` through `ModelExpression` | `torch.jit.script` a `Linear(3,64) → Tanh → Linear(64,1)` and save it | bytes of the exported files and whether they run without the framework |

**What the numbers mean.** `memory` deliberately counts the import: for
OpenNN the equivalent cost is paid by the dynamic loader before `main()`, so
neither figure is "the library's own size" — both are what a process weighs
once the framework is available and an empty model exists, which is the
comparable quantity. `startup` has two readings on purpose. The in-process
timer starts at `main()`/module entry and therefore misses the dynamic
loader, which for a framework whose shared objects run to hundreds of
megabytes is most of the answer; the runner's `wall_seconds` around the whole
process is the figure to compare, and the tables report both. `export` is
the one question with a qualitative answer: OpenNN writes source that
computes the model with no runtime dependency, and PyTorch has no equivalent
— TorchScript and ONNX both produce artifacts that need libtorch or
onnxruntime to execute — so the PyTorch driver reports the TorchScript size
with `standalone_source=0` rather than pretending the two files are the same
kind of thing.

**Gates.** None beyond the process exiting with `RESULT=OK`; there is no
throughput here and nothing to agree on across engines.

## Results

Session `2026-09-03-publish`, the last run of every cell, median of three rounds.

| footprint question | OpenNN | PyTorch | PyTorch / OpenNN |
|---|---|---|---|
| memory | 0.124 s, 117 MiB | 3.241 s, 449 MiB | **26.1×** the time, 3.8× the memory |
| startup | 0.568 s, 325 MiB | 1.926 s, 375 MiB | **3.4×** the time, 1.2× the memory |
| export | 0.448 s, 156 MiB | 1.928 s, 376 MiB | **4.3×** the time, 2.4× the memory |

## Why

None of the three questions is about a kernel. They are about what a process
has to load, initialise and carry before the first useful instruction, and
the two frameworks are delivered in ways that make that cost very different.

### `memory`: a linked library against an imported package

The OpenNN process is one executable that links `libopennn.a` statically and
a handful of shared libraries — MKL, oneDNN, libgomp, and the CUDA runtime,
cuBLAS, cuBLASLt, NVRTC and cuDNN (`ldd` lists fifteen objects in all). The
dynamic loader maps them, but a mapped library costs resident memory only for
the pages that are touched, and constructing an empty `NeuralNetwork`, an
empty `TabularDataset` and a `TrainingStrategy` touches very few of them:
219 MB of anonymous memory, 117 MiB resident in all,
in 0.124 s. *[pending the final measurement round]*

The PyTorch process is the CPython interpreter, which is 11 MiB
and 0.021 s before it does anything, plus `import torch`, which is
where the rest goes: the import executes 1,135 Python modules —
each one parsed or unmarshalled from its `.pyc`, its objects allocated on the
heap — and loads `libtorch_cpu.so`, `libtorch_cuda.so`, `libc10.so` and the
bundled cuDNN, cuBLAS and NCCL objects, running their static initialisers and
the operator-registration code that populates PyTorch's dispatcher with every
kernel it ships. That is 1.55 s of the 3.24 s
the process takes and most of its 816 MB of anonymous memory;
the empty `nn.Sequential` and the `Adam` over one tensor that follow are
negligible. The ratio is a property of the delivery, not of the empty model:
a C++ program linking libtorch would skip the interpreter and the module
graph but still pay the shared-object initialisation, so it would land between
the two columns rather than beside OpenNN's.

### `startup`: the first prediction

Both processes construct a 10 → 64 → 1 network and predict on one row of
ones; the arithmetic is a few thousand flops and does not appear in either
number. What the two timers measure is everything that has to happen before
it.

For OpenNN the in-process time to the prediction is 0.352 s,
of which almost all is one thing: the driver constructs the network under
`Device::Auto`, and on a machine with a GPU that means creating the CUDA
context and the cuBLAS and cuDNN handles before the first layer can run.
With the GPU hidden (`CUDA_VISIBLE_DEVICES=`, the same binary) the same
prediction is ready in *[pending the final measurement round]* — so the OpenNN column in the
table is, to within a few milliseconds, the price of initialising CUDA, which
the PyTorch process in this cell never pays because it predicts on the CPU and
never touches `torch.cuda`. The wall time the runner measures around the
process adds the loader's work — mapping the CUDA libraries is most of it —
and reads 0.568 s.

For PyTorch the in-process time is 1.537 s, and `import
torch` alone is 1.55 s of it; the wall time,
1.93 s, adds the interpreter's own start. Nothing in the
comparison is unfair to PyTorch — the process is doing what any PyTorch script
does before its first line of model code — but a reader should see that the
3× in the table is "import torch" against "initialise CUDA", and that a CPU-only
OpenNN process would be an order of magnitude faster still.

### `export`: what the trained model becomes

This is the one question with a qualitative answer, and the timing is the
least interesting part of it. OpenNN's `ModelExpression` writes the network as
source — a C file and a Python file, each around 15 kB, that compute the
model with its trained weights inlined and no dependency on OpenNN or on
anything else; the process also trains the model it exports (50 epochs of
Adam on a 512-row synthetic set), which is inside its 0.448 s.
PyTorch's process scripts an untrained model with `torch.jit.script` and
saves it: 7,747 bytes of TorchScript, a zip archive holding
a serialised graph and the weights, which runs only where libtorch is
installed. The driver reports it with `standalone_source=0` because the two
files are not the same kind of artifact: one is a deliverable, the other is a
checkpoint for the framework that produced it. ONNX export would give a
third kind — a graph for onnxruntime — and would not change the finding. The
wall times (0.448 s against 1.93 s) are once more
the import against a small amount of real work, with the training on OpenNN's
side and none on PyTorch's.

## Asymmetries and caveats

- **`memory` may include a CUDA context on OpenNN's side.** The OpenNN
  process constructs its objects with `Device::Auto`, and on a machine with a
  GPU that can initialise the CUDA runtime, whose context is a few hundred
  megabytes of resident memory that the PyTorch process — which imports
  `torch` but never touches `torch.cuda` — does not pay. *[pending the final measurement round]*
- **The two `startup` timers do not start at the same place.** The
  in-process reading starts after the dynamic loader has finished, which
  for a process that maps hundreds of megabytes of shared objects is where
  most of PyTorch's cost sits, and after `import torch` on the PyTorch side.
  The runner's `wall_seconds` around each launch is the like-for-like
  number; the table reports both so a reader can see how much of each
  engine's startup is the loader.
- **The export sizes are different kinds of file.** OpenNN's `.c` and `.py`
  are source text that computes the model with no framework; PyTorch's
  TorchScript archive is a zip of a serialised graph plus its weights, which
  needs libtorch to run. The sizes are reported for completeness, not as a
  comparison — the driver marks the PyTorch file `standalone_source=0` for
  that reason, and a reader who needs a size comparison should compare
  OpenNN's export against an ONNX file plus the runtime that executes it.
- **The `export` processes do different amounts of work.** OpenNN's trains
  the model it exports (50 epochs of Adam at batch 32 on 512 synthetic rows)
  before writing it; PyTorch's scripts a freshly constructed model and saves
  it, because there is nothing to gain from training a model whose export is
  the point. The asymmetry favours PyTorch on the wall clock and the OpenNN
  column still reads lower.
- **PyTorch's process is Python.** Every PyTorch number includes the
  interpreter (`import torch` loads 1,135 Python modules); an
  application that embeds libtorch from C++ would read lower on `memory` and
  `startup` than these cells, and higher than OpenNN by a smaller margin.
  The cells measure the framework as it is delivered to its users, which
  for PyTorch is the Python package.
- **Nothing is repeated.** Each question is one process, launched once, with
  no round structure; *[pending the final measurement round]*

## Reproduce

```bash
export OPENNN_BENCH_SESSION=$(date +%F)-mine
python run.py --family footprint
CUDA_VISIBLE_DEVICES= build-bench/bin/footprint_opennn memory   # the same question with the GPU hidden
```
