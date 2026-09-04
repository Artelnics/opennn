# Result artifacts

One JSON per cell, written by [`run.py`](../run.py). Nothing here is committed:
`.gitignore` keeps `results/*` out of the tree and lets through only this file.
The artifacts live on disk, and the claim they support lives in the commit or
note that cites their `run_id`.

## Two directories, and the difference is enforced

| | |
|---|---|
| `results/` | the evidence store: every gate passed |
| `results/scratch/` | everything else |

`common.result_destination()` decides, not the caller. A run lands in
`scratch/` when the tree is dirty, when the machine was busy, or -- on CUDA --
when the GPU clock was not locked. That is a function, not a convention,
because the suite this replaced stated the rule in prose and checked it
nowhere: 39 of its 107 artifacts were dirty-tree results filed as reproducible
ones.

`scratch/` is disposable by definition. Delete it whenever; nothing reads it.

## Naming

```text
<benchmark_id>[-<label>]-<run_id>.json
cuda-lstm-train-final-20260901T123049Z.json
```

`benchmark_id` is `<device>-<family>-<mode>`. `--label` is free text.

**A label is not an identity.** Labels repeat across sittings -- `final` names a
2026-08-26 run and a 2026-09-01 one -- so anything that collects artifacts must
filter on `session_id`, which PROTOCOL item 7 already makes the unit of
comparison. Selecting by label alone once produced a 24% spread on a cell whose
real spread was 3%, by averaging two different machines' numbers together.

## What an artifact contains

`schema_version: 1`, plus:

| key | |
|---|---|
| `session_id`, `run_id`, `label` | which sitting, which run |
| `git` | commit and `dirty` |
| `machine`, `cpu`, `frameworks` | GPU state, core layout, governor, versions |
| `clocks_locked` | GPU clock pinned for this run |
| `machine_quiet` | non-idle fraction before and after, and the threshold |
| `shape_gate` | sample count, sequence, vocabulary, parameters, per engine |
| `quality_gate` | accuracies per batch, and whether they agree |
| `summary` | per engine: median/min/max samples/s, peak and workload MiB, energy |
| `launches` | every launch kept, so a drifting session is visible |
| `datasets` | path, size and hash of what was read |

Read the gates before the numbers. A `summary` whose `shape_gate.agrees` is
false is not a comparison between two engines; it is two engines doing
different work.

## Reading energy

`summary[engine].energy_wh` is board power on GPU and the RAPL package counter
on CPU. They are **different quantities and must not share a column** -- each
launch records `energy_metric` and, on CPU, `energy_domain`. A run whose timed
window held too few samples reports `energy_measurable: false` and says why,
rather than reporting `0.0000 Wh`, which would be a claim instead of an
absence.

## Immutability

Never edit an artifact. A re-run writes a new file; that is what `run_id` is
for. The point of keeping every launch rather than a summary is that a reader
can find the drift themselves instead of trusting that someone else looked.

Older artifacts from the suite this replaced used an incompatible shape
(`protocol`, `results`, `git_commit`) under the same `schema_version: 1`. They
were archived out of this directory on 2026-09-01; if any resurface, the
current shape is the one with `launches`.
