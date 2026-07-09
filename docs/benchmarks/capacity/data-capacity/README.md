# Data Capacity Benchmark (HIGGS)

Purpose: measure how many tabular samples fit and train under a fixed RAM budget
— OpenNN's compact in-RAM `float32` matrix (`TabularDataset` default `Matrix`
storage mode) versus the `pandas.read_csv` path used by PyTorch and TensorFlow.

The rows come from the [HIGGS dense benchmark
contract](../../throughput/higgs/README.md): the prepared training file
`higgs_train.csv` (28 numeric features + 1 label per row, headerless,
comma-separated). To probe a RAM budget larger than the source file, the rows
are **tiled**: output row `i` is HIGGS row `i % file_rows`, so every value is a
real HIGGS row while the sample count can grow past the file's own row count.
All three engines load a byte-identical tiled CSV, so the crash point reflects
each engine's memory efficiency, not the data.

This benchmark only makes sense in `Matrix` storage mode, where the whole dataset
lives in RAM: it compares the size of each framework's default in-memory
representation. In `BinaryFile` storage mode `TabularDataset` keeps the data
matrix empty and streams batches from an on-disk `.bin` cache, so RAM no longer
scales with the sample count and the RAM ceiling this benchmark measures does not
exist.

This is a Windows-CPU benchmark: it reads the process working set through
`psapi` and caps committed memory with a Windows Job Object.

## Prepare the data

Large HIGGS files must live outside the repository. Set `OPENNN_BENCH_DATA` and
run the shared preparer once (see [`../DATA_POLICY.md`](../../DATA_POLICY.md) and
the [HIGGS contract](../../throughput/higgs/README.md)):

```powershell
$env:OPENNN_BENCH_DATA = "C:\OpenNNBenchmarks\data"
python ..\..\throughput\higgs\prepare_higgs.py --raw C:\path\to\HIGGS.csv.gz
```

That writes `$OPENNN_BENCH_DATA\higgs\higgs_train.csv`, which the drivers tile.

## Build the helpers

```powershell
cl /O2 tile_higgs.c        # tiles higgs_train.csv up to a target row count
cl /O2 run_capped.c        # runs a child under a hard committed-memory cap
```

The OpenNN engine binary is `opennn_capacity` (`opennn_capacity.exe` under
`build\bin\Release`).

## Run

Capped search — largest sample count each engine survives under one budget:

```powershell
.\capacity_search.ps1 -CapGB 8
```

Uncapped sweep — record RESULT and peak RAM per sample count for each engine:

```powershell
.\run_sweep.ps1
```

Both point `-HiggsCsv` at `$env:OPENNN_BENCH_DATA\higgs\higgs_train.csv` by
default. They report the maximum number of samples each framework loads and
trains before it runs out of the memory budget.
