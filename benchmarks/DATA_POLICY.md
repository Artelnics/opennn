# Where the data lives

Datasets never enter the repository. The tree holds benchmark code, the two
documents that define the measurement, and one manifest; everything a run reads
comes from outside it.

## The data root

`$OPENNN_BENCH_DATA`, defaulting to `~/opennn-benchmark-data`:

```bash
export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
```

Both [`prepare.py`](prepare.py) and [`run.py`](run.py) read that one variable,
so a machine with its data elsewhere sets it once and nothing else changes. No
benchmark resolves a dataset by a path relative to itself, and none carries a
machine-specific absolute path.

## One preparer, one subcommand per family

```bash
python prepare.py dense cnn transformer lstm     # or: all
```

| family | subcommand | lands in | read as |
|---|---|---|---|
| dense | `dense` | `higgs/` | `higgs_train_250k.csv`, `higgs_test.csv` |
| cnn | `cnn` | `imagenet_subset/` | `train/` — 1000 classes |
| transformer | `transformer` | `wmt14/` | `wmt14_pairs.txt` |
| lstm | `lstm` | `beijing_pm25/` | `beijing_pm25_forecasting.csv` |
| footprint | — | — | measures the framework before any data exists |

The subcommand names match `run.py --family` exactly. They did not always: the
preparer called the recurrent family `recurrent` while the runner called it
`lstm`, so `prepare.py lstm` — the command the README tells you to run — failed
with an argparse error.

Every step is skipped when its output exists, so re-running is cheap and an
interrupted download resumes.

## The one committed artefact

`imagenet_subset.manifest` (5 MB) pins exactly which 50,000 images the CNN
family measures on. It is committed so a subset can be *verified* rather than
trusted: the preparer picks deterministically from sorted synsets with a seeded
draw, so the same arguments reproduce the same subset on any machine, and the
manifest is what proves it did.

## Rules

- Do not commit datasets, prepared CSVs, image folders, binary caches, or
  downloaded archives.
- Do commit the code that recreates them from documented sources.
- Do not commit compiled binaries, ONNX files, or generated CSVs.
- Result artifacts are **not** committed either. `.gitignore` keeps
  `results/*` out and admits only `results/README.md`; the artifacts live on
  disk and a claim cites their `run_id`. See
  [`results/README.md`](results/README.md).

These rules are not currently enforced by a script. An earlier version of this
file said a `tools/validate_benchmarks.py` check would fail a commit that broke
them; no such file exists, and saying otherwise was worse than saying nothing,
because it invited trusting a gate that was never there. The `.gitignore`
patterns are the actual enforcement, and they only cover what someone thought
to list.

## Stale data is not this repository's problem, but it is someone's

The data root accumulates. `cifar10/` is still there from the suite this one
replaced and no current family reads it; so are several HIGGS variants
(`higgs_train_1m`, `_2m`, `_dup32`, `_pad32`) left from capacity experiments.
Nothing here deletes them, because a benchmark tool that deletes data is a
benchmark tool nobody runs twice. Check `du -sh $OPENNN_BENCH_DATA/*` when disk
matters.
