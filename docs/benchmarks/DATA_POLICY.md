# Benchmark Data Policy

Large datasets and generated benchmark caches must stay outside the git
working tree. The repository should contain benchmark code, small metadata,
result JSON artifacts, and documentation only.

## Data Root

Set `OPENNN_BENCH_DATA` to a local directory outside the repository:

```powershell
$env:OPENNN_BENCH_DATA = "C:\OpenNNBenchmarks\data"
```

```bash
export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
```

If the variable is not set, benchmark helpers default to:

```text
~/opennn-benchmark-data
```

## Layout

Use one subdirectory per dataset:

```text
$OPENNN_BENCH_DATA/
  higgs/
    raw/
    higgs_train.csv
    higgs_test.csv
    higgs_metadata.json
  imagenet/
  cifar10/
  cifar100/
  imagenet_like/
```

## Rules

- Do not commit raw datasets, prepared CSVs, image folders, binary caches, or
  downloaded archives.
- Do commit scripts that can recreate prepared data from documented sources.
- Do commit small metadata and result JSON artifacts when they support published
  benchmark numbers.
- Result JSON should record the dataset path, row counts, command line, git
  commit, dirty status, machine metadata, and raw benchmark output.
- Tiny synthetic fixtures are allowed only when they are intentionally small and
  used for tests or smoke checks.
