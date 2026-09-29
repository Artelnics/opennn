# OpenNN benchmarks

These programs compare OpenNN (C++) with PyTorch (Python) on the same models.
Each run measures speed, peak memory and energy, and first checks that both
engines did the same work.

## Families

| Family | Model | Data | Measures |
| --- | --- | --- | --- |
| `dense` | 28 → 1024 × 2 → 1 classifier | HIGGS | Training and inference throughput |
| `cnn` | ResNet-50 v1.5 | ImageNet subset, 1,000 classes × 50 images | Training and inference throughput |
| `transformer` | Base encoder-decoder, d512, 6 layers | WMT14 English–German | Training and inference throughput |
| `lstm` | LSTM(15 → 128) → Linear | Beijing PM2.5 | Training and inference throughput |
| `startup` | Small dense, LSTM, CNN and Transformer applications | Constructed inputs | Time to the first prediction (Linux) |
| `deployment` | The same small applications | Constructed inputs | Installed size (Linux) |
| `quality` | Dense, LSTM, ResNet and Transformer | Held-out splits of the datasets above | Prediction quality after training |

## Quick start

Build the OpenNN drivers in Release, outside the checkout. Add
`-DOpenNN_DISABLE_CUDA=ON` for a CPU-only build:

```sh
cmake -S . -B ../opennn-bench-build -DCMAKE_BUILD_TYPE=Release -DOpenNN_BUILD_TESTS=OFF -DOpenNN_BUILD_EXAMPLES=OFF -DOpenNN_BUILD_BENCHMARKS=ON
cmake --build ../opennn-bench-build --config Release --target benchmarks --parallel
```

Install PyTorch for your device (CPU or CUDA) in the Python environment you run
the scripts with. Then set the data folder and the driver to compare:

```sh
export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
export OPENNN_BIN="$PWD/../opennn-bench-build/bin/dense_opennn"
```

```powershell
$env:OPENNN_BENCH_DATA = Join-Path $env:USERPROFILE 'opennn-benchmark-data'
$env:OPENNN_BIN = (Resolve-Path '../opennn-bench-build/bin/Release/dense_opennn.exe').Path
```

Download the family's data once, then run it:

```sh
python benchmarks/prepare.py dense
python benchmarks/run.py --family dense --mode train --device cpu --precision fp32 --batch 8192
```

The run prints throughput, peak memory and energy for each engine and writes a
JSON file with every launch. Change `OPENNN_BIN` when you switch family.

## Main options

| Option | Values |
| --- | --- |
| `--family` | `dense`, `cnn`, `transformer`, `lstm` |
| `--mode` | `train` or `infer` |
| `--device` | `cpu` or `cuda` |
| `--precision` | `fp32`, `bf16` or `strict` |
| `--batch` | `8192` for one size, `1024,8192,65536` for a curve, `1024:OOM` to double until memory runs out |

Run `python benchmarks/run.py --help` for the rest. `startup`, `deployment` and
`quality` have their own procedures in [PROTOCOL.md](PROTOCOL.md).

## Checks before a result counts

- **Same work:** both engines must report the same samples, shapes and parameter
  count.
- **Same quality:** where the family reports accuracy (dense training), it must
  be comparable.
- **Quiet machine:** CPU activity is sampled before, during and after each run.

A run that fails a check, runs on a busy machine, has uncommitted changes, or
uses the GPU with unlocked clocks is written to `scratch/` as a diagnostic.

For GPU results, lock the clocks first. On Linux use
`sudo benchmarks/tools/gpu_clocks.sh lock <MHz>`. On Windows lock them with
`nvidia-smi` and set `OPENNN_BENCH_CLOCKS_LOCKED=1`.

## Results

Results go to `../opennn-benchmark-results/`, or to the folder set in
`OPENNN_BENCH_RESULTS`. Keep your own reports in `benchmarks/reports/`, which
Git ignores. Results and reports are never committed.

To compare two runs from the same machine, for example before and after a code
change:

```sh
python benchmarks/compare.py baseline.json candidate.json
```

It accepts up to 5% variation in throughput and memory and fails if the largest
batch that fits gets smaller.

## Files

| Path | Contents |
| --- | --- |
| `run.py` | Runs a family on both engines |
| `prepare.py` | Downloads and prepares the datasets |
| `compare.py` | Compares two result files |
| `families/` | The OpenNN (`.cpp`) and PyTorch (`.py`) version of each family |
| `tools/` | Shared helpers: monitoring, startup, deployment and quality runners |
| `manifests/` | The pinned ImageNet subset and Python package versions |
| [`PROTOCOL.md`](PROTOCOL.md) | The complete measurement rules |
