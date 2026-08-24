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
  higgs/                     # prepare_higgs.py  (throughput/higgs)
    raw/
    higgs_train.csv
    higgs_test.csv
    higgs_metadata.json
  cifar10/                   # prepare_cifar10.py  (throughput/resnet50)
  cifar100/                  # prepare_cifar100.py
  imagenet_like/             # prepare_imagenet_like.py
  chat/                      # prepare_chat.py  (energy/transformer-energy)
    chat_pairs.txt
    chat_metadata.json
  beijing_pm25/              # prepare_beijing_pm25.py  (quality/recurrent-lstm-forecasting)
    beijing_pm25_forecasting.csv
  wmt14/                     # prepare_wmt14.py  (capacity/transformer-max-batch)
    wmt14_en_de_pairs.txt
```

Every dataset has exactly ONE `prepare_<dataset>.py` that downloads/normalizes it
into its `$OPENNN_BENCH_DATA/<dataset>/` subdirectory; benchmarks read it from
there via `$OPENNN_BENCH_DATA`, never from a benchmark folder or a machine-specific
absolute path.

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
- Do not commit compiled binaries, ONNX files, or generated CSVs. Regenerate
  them outside the repo, or replace them with a documented result JSON. The
  `tools/validate_benchmarks.py` check fails if any such artifact is committed.
