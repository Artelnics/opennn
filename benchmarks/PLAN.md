# The benchmark suite, rebuilt simple

Written 2026-08-25. This replaces `REORGANIZATION_PLAN.md`, which planned a
reorganisation of the existing suite. This plans a replacement of it.

Two decisions drive everything below.

**One execution, many observations.** Speed, peak memory and energy are not
three benchmarks. They are three instruments reading one run. Today
`run_higgs_dense_energy.py` launches the same `opennn_speed` binary the speed
benchmark launches and samples board power across it — the identical work, run
twice, in two thermal states, filed in two folders as two results that cannot
be cross-referenced. Attach the instruments to the run instead and the second
execution disappears, along with the folder, the runner and the report.

**One sweep, not four protocols.** Speed, capacity, peak-batch and quality
differ in which batch sizes they visit, how many epochs they run, and what they
do when a run fails. Those are arguments. They were four directories.

## What we are replacing

| | now | after |
|---|---:|---:|
| files (excluding `results/`) | **201** | **~16** |
| directories | **36** | **2** |
| C++ programs | 25 | 5 |
| Python scripts | 86 | 3 + 5 |
| Markdown | 56 | 2 |

## Target layout

```text
benchmarks/
├── README.md                  what is measured, how to run it, what it means
├── PROTOCOL.md                the contract; changing it forces a re-run
├── run.py                     the only entry point
├── common.py                  provenance · gpu sampling · metrics · binaries
├── prepare.py                 every dataset, one subcommand per family
├── families/
│   ├── dense.cpp        dense.py
│   ├── cnn.cpp          cnn.py
│   ├── transformer.cpp  transformer.py
│   ├── recurrent.cpp    recurrent.py
│   └── footprint.cpp    footprint.py
├── imagenet_subset.manifest
└── results/                   artifacts, and scratch/ for anything unclean
```

Sixteen files. Every family is exactly two files — the definition in each
engine — and everything that is not a model definition lives in one of three
Python files at the top.

## The primitive

One thing runs:

    run(family, engine, mode, batch, precision, device, epochs) -> observations

`mode` is `train` or `infer`. That is the whole vocabulary. Every run emits,
from that single execution:

| observation | how |
|---|---|
| throughput | timed epochs or passes, warmup excluded, median reported with every round kept |
| peak memory | `nvidia-smi` device-used sampled at 50 ms, minus the idle reading taken before the run |
| energy | board power integrated over the same window, between the marks the run already prints |
| quality | loss, accuracy and AUC of the network the run just trained |

Nothing re-runs to collect a second number.

## The sweeps

What used to be four protocol directories are arguments to the one runner:

| old benchmark | now |
|---|---|
| throughput | `run.py --family dense --mode train --batch 8192` |
| footprint/memory (peak) | falls out of the run above |
| energy | falls out of the run above |
| quality / accuracy / convergence | the same run, with enough epochs |
| capacity (max batch) | `--sweep 1024:OOM` — ascending until a launch fails |
| peak-batch | the same sweep, keeping throughput at every rung |

A sweep re-launches per batch because it must: a CUDA out-of-memory fault
leaves the context unusable, so the next attempt in the same process would
measure the wreck of the last one. Exit code 0 fits, 1 does not.

## What does not fold in

Being honest about the limits of "one execution":

* **Baseline footprint** — RAM after constructing empty objects, time to first
  prediction, exported-code size, lines of code. These ask what a framework
  costs *before* you run anything, so they cannot ride along on a run. They are
  one file, `footprint.py`, with a small C++ counterpart.
* **Energy needs a quiet machine.** Sampling power during every run is free, so
  we always do it, but the number is only quotable when nothing else used the
  GPU. Each artifact records the idle baseline and a `quiet` flag; a run that
  cannot assert quiet keeps its speed number and forfeits its energy one.
* **Failed capacity attempts** yield no speed, memory or energy — only the
  fact that they failed.

## Starting from scratch

Nothing old is cited. Concretely:

* **All 107 result artifacts go.** They cannot be salvaged: 39 record a dirty
  tree, 38 measured PyTorch eager or do not say how they measured it, and they
  span three different GPUs. They stay in git history; they leave the suite.
* **All 56 reports go**, because every one of them quotes those artifacts.
  `README.md` and `PROTOCOL.md` are written fresh and start empty of numbers.
* **`PRESENTATION_CLAIMS.md` goes.** Claims come back one at a time, each
  naming the artifact it rests on.
* **`REORGANIZATION_PLAN.md` and `DUPLICATION_LEDGER.md` go** once the code
  they describe is deleted. What the ledger found that still matters is
  recorded below, so deleting it loses nothing anyone needs.

### What the merge found, kept because it explains the numbers

Three ways the six dense definitions disagreed, all of which moved results:

1. The capacity site seeded with `0`; the other five with `42`. Capacity had
   never measured the same initialised network as speed and quality.
2. Four sites wrapped the model in `ClassificationNetwork`, which prepends a
   Scaling layer. PyTorch has no scaling stage, so those four timed a layer the
   other engine did not have.
3. PyTorch was launched eager while OpenNN ran a captured CUDA graph on a
   device-resident split. Correcting only that moved dense training from
   **1.29× to 1.06×**, and dense inference to **0.995×** — a tie.

The third is why the artifacts are being discarded rather than re-filed.

## Order of work

1. **`common.py`** — fold the four modules into one file.
2. **`run.py`** — the primitive, the instruments, the sweep, the artifact.
3. **`prepare.py`** — fold every `prepare_*.py` into one, subcommand per family.
4. **`families/dense.*`** — already written and verified against the programs
   it replaces; move and rename.
5. **Delete the dense generation** — six C++ programs, their runners, their
   reports, their folders.
6. **`families/cnn.*`** — ImageNet subset is prepared and waiting: 1000
   classes, 50 images each, 224×224. Includes the TensorFlow ResNet-50 that
   never existed, for when TensorFlow can run here.
7. **Repeat for transformer, recurrent, footprint**, deleting each old
   generation as its replacement passes.
8. **`PROTOCOL.md`**, then one re-baseline session on a clean tree with locked
   clocks, one session id, every number from one sitting.

Steps 1–5 leave the suite working with the dense family alone; nothing is
half-migrated at any point.

## A library bug this found, and fixed

Building the sweep surfaced a segfault in CUDA training, and it was worth the
detour: **it had nothing to do with the benchmarks.** The program being
replaced, `opennn_speed`, crashed identically on the same input. The old suite
never saw it because nothing ever swept batch sizes.

**The cause.** `Optimizer::run_graph_epoch` runs a *grouped* path that needs
`pipelines[0].slots[0..M-1]`. Two gates decided that, and they disagreed:

| | threshold | value |
|---|---|---|
| allocation, `set_up_batches` | `training_batches >= TrainingSession::group_size` | **8** |
| execution, `run_graph_epoch` | `batches_number >= M` | **2** |

An epoch of 2 to 7 batches therefore entered the grouped path and dereferenced
a slot that was never allocated -- a null `unique_ptr`, so a segfault rather
than an error. It reproduced exactly where that predicts, on HIGGS 250k:

| batch | whole batches | slots allocated (>=8) | grouped path (>=2) | result |
|---:|---:|---|---|---|
| 8,192 | 30 | yes | yes | ok |
| 28,672 | 8 | yes | yes | ok |
| 32,768 | **7** | **no** | **yes** | **SIGSEGV** |
| 65,536 | **3** | **no** | **yes** | **SIGSEGV** |
| 131,072 | 1 | no | no | ok |

Precision-independent, and `OPENNN_NO_CUDA_GRAPH=1` avoided it, which is what
placed the fault in the graph path rather than in the arithmetic.

**The fix** makes the consumer check its own precondition -- `can_group_batches`
now also requires the slot to exist -- so an unprovisioned epoch falls back to
the non-grouped path that already handles every other such case. Widening the
allocation instead would have spent memory to satisfy a gate the code can
simply test.

Verified: every batch from 8,192 to 131,072 now trains with
`cuda_graph=captured`, the sweep reaches the dataset's own limit, and the suite
is 1,090 of 1,100 passing. The two failures --
`ConvolutionalLayerTest.ProjectionResidualReuseGradientMatchesNumerical` and
`C2PSA.CpuAndGpuGradientsMatch` -- were confirmed pre-existing by rebuilding
without the fix and watching them fail identically.

**Kept regardless of the fix:** a crash is not a capacity limit. `--batch N:OOM`
reports `frontier_valid: false` when a launch dies on a signal instead of
reporting `fits=0`, because publishing a bug as a "max batch" is wrong in a way
no reader could detect.

Two smaller things found in passing, both the shadowing hazard AGENTS.md
documents -- unqualified names resolving to Eigen inside `namespace opennn`:
`tests/neural_network/network_topology_golden_test.cpp` used `array` where it
meant `std::array`, which broke the whole test build, and the same trap caught
`Dense` while writing `families/dense.cpp`.

## The contract, for `PROTOCOL.md`

Fixed now, and changing any of it forces a full re-run:

1. **Machine** — RTX 5070 Ti (`sm_120`, 16,303 MiB, 300 W), driver 610.43.02,
   native Linux, CUDA 13.3, cuDNN 9.25.
2. **Engines at their best** — OpenNN with captured CUDA graphs and a
   device-resident split; PyTorch with `torch.compile`. Written as commands.
3. **TensorFlow is not in the matrix.** No build ships `sm_120` kernels — not
   2.21.0, not the nightly — so it runs on driver-JIT'd PTX and any deficit
   partly measures NVIDIA's release schedule. Every cell records
   `engines: [opennn, pytorch]` until an NGC container changes that.
4. **Memory means device-used from `nvidia-smi`,** minus idle.
   `torch.cuda.max_memory_allocated()` never appears in a comparison: it
   excludes the CUDA context and cached blocks, and flatters PyTorch by
   construction.
5. **A dirty tree writes to `results/scratch/`.** Enforced in code, not asked
   for in prose.
6. **Comparisons are valid only within one session id.** Absolute throughput is
   provenance; ratios within a session are the durable output.
7. **Clocks locked** for anything published — the card drifts 8% across a day,
   so margins under ~2% are not resolvable otherwise.
