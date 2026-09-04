# Why the suite has this shape

History, not a plan. The work described here is done; what remains is the
reasoning, which the code cannot state about itself. The contract is
[`PROTOCOL.md`](PROTOCOL.md) and how to run it is [`README.md`](README.md).

This replaced a suite of **201 files across 36 directories** — 25 C++ programs,
86 Python scripts, 56 Markdown files — with roughly a dozen. Two decisions did
almost all of that.

## One execution, many observations

Speed, peak memory and energy are not three benchmarks. They are three
instruments reading one run.

The suite this replaced had `run_higgs_dense_energy.py` launch the same
`opennn_speed` binary the speed benchmark launched, and sample board power
across it: the identical work, run twice, in two thermal states, filed in two
folders as two results that could not be cross-referenced. Attaching the
instruments to the run instead made the second execution disappear, and with it
the folder, the runner and the report.

That is why `run.py` is one entry point and why an artifact carries `summary`
alongside `launches` — every reading in a cell comes from the same process, so
they can be compared to each other rather than merely filed together.

## One sweep, not four protocols

Speed, capacity, peak-batch and quality differ in which batch sizes they visit,
how many epochs they run, and what they do when a run fails. Those are
arguments. They had been four directories.

`--batch` is the only sweep axis, in three forms: one rung, several rungs, or
double-until-OOM. A capacity sweep re-launches per rung because it must — a CUDA
out-of-memory fault leaves the context unusable, so a second attempt in the same
process would measure the wreck of the first.

## What does not fold in

Being honest about the limits of "one execution":

- **Baseline footprint** — RAM after constructing empty objects, time to first
  prediction, exported-code size. These ask what a framework costs *before* it
  runs anything, so they cannot ride along on a run. They are the `footprint`
  family, whose modes are one process each.
- **Energy needs a quiet machine.** Sampling power costs nothing, so it always
  happens, but the figure is quotable only when nothing else used the device.
  A run that cannot assert quiet keeps its speed number and forfeits its
  energy one.
- **Failed capacity attempts** yield no speed, memory or energy — only the fact
  that they failed. A crash is not a capacity limit either, which is why
  `frontier_valid` exists.

## What this document used to be

A plan, written 2026-08-25, with a target layout and an order of work. Both
were achieved and both then drifted: the layout named `recurrent.cpp` where the
files are `lstm.cpp`, and the file count stopped matching. It also carried a
draft of the contract, later superseded by `PROTOCOL.md`, and a 53-line account
of a CUDA-graph slot bug that the suite's first batch sweep surfaced — a
library fault, not a benchmark one, and recorded where library findings belong,
in [`docs/status/engineering-audit.md`](../docs/status/engineering-audit.md).

Trimmed to this on 2026-09-01. A completed plan kept as a live document only
teaches readers to distrust the directory.
