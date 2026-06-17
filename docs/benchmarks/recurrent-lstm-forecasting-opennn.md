# Recurrent vs LSTM forecasting in OpenNN

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/).
Last updated 2026-06-15. Harness ready; Linux reference run pending.*

**Status:** this benchmark is now packaged as part of the benchmark suite, but
the current headline table should wait for a fresh Linux run with the JSON
artifact stored under [`results/`](results/).

## What This Measures

This is an OpenNN internal architecture benchmark: **Recurrent layer vs LSTM
layer** on the same time-series forecasting task, measured on both GPU and CPU.

It answers three questions:

- Does the LSTM improve forecasting accuracy over the simpler recurrent layer?
- What is the training-time cost of that accuracy?
- How much does the GPU path accelerate recurrent and LSTM workloads versus CPU?

This is **not** a PyTorch/TensorFlow comparison. It is a sequence-model coverage
benchmark for OpenNN itself.

## Method

The driver reuses the existing forecasting example target:
[`examples/no2_forecasting/main.cpp`](../../examples/no2_forecasting/main.cpp).
The executable is still named `no2_forecasting` for CMake compatibility, but
the benchmark runner selects the `beijing_pm25` dataset profile through
`OPENNN_FORECASTING_DATASET`.

It trains the same scenarios twice:

- Phase 1: GPU, `Configuration::set(Device::CUDA, Type::FP32)`.
- Phase 2: CPU, `Configuration::set(Device::CPU, Type::FP32)`.

For each scenario, it trains:

- `ForecastingNetwork`, built from OpenNN `Recurrent` layers.
- `ForecastingLstmNetwork`, built from OpenNN `LongShortTermMemory` layers.

The driver reports:

- Test RMSE.
- Relative test RMSE, when target range is available.
- Mean training time.
- Mean epochs.
- Parameter count.
- CPU/GPU speedup for each architecture.
- Winner by test RMSE for each scenario and device.

The scenario set uses the
[UCI Beijing PM2.5 dataset](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data).
It contains 43,824 hourly rows from 2010-01-01 through 2014-12-31. The
preparation script writes a numeric CSV with calendar/weather inputs, one-hot
wind direction, and `pm2_5` as the last column/forecast target. UCI marks
missing values as `NA`; the prep step linearly interpolates missing numeric
values so the hourly sequence remains continuous.

The current scenarios are:

- B1: 24-hour input window, 1-hour forecast horizon.
- B2: 48-hour input window, 1-hour forecast horizon.
- B3: 72-hour input window, 24-hour multi-target horizon.
- B4: 168-hour input window, 24-hour multi-target horizon.

## MLPerf-Style Protocol

This is an MLPerf-inspired OpenNN benchmark, not an official MLPerf result.

| Field | Value |
|---|---|
| Benchmark class | `training_throughput_with_quality` for the first Linux run; upgrade to `training_time_to_quality` after calibration |
| Division | `closed`: fixed dataset preparation, scenarios, model builders, optimizer, precision, seed count |
| Quality metric | test RMSE and relative test RMSE by scenario/device |
| Quality target | pending calibration from the first reference Linux run |
| Performance metric | mean training wall time, CPU/GPU speedup, and winner by test RMSE |
| Measurement rule | one seed per scenario today; repeat runner artifacts before headline use |
| Artifact rule | JSON under `docs/benchmarks/results/` with protocol metadata, command, machine data, parsed metrics, and raw output |

The MLPerf-style upgrade path is to convert each scenario from fixed maximum
epochs to time-to-target-RMSE once the reference Linux run establishes stable
quality targets.

## Reproducibility

Build:

```bash
cmake -S . -B build-gpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenNN_BUILD_EXAMPLES=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native

cmake --build build-gpu --target no2_forecasting -j"$(nproc)"
```

Run:

```bash
cd docs/benchmarks/recurrent-lstm-forecasting
REPO_ROOT="$(git rev-parse --show-toplevel)"
export OPENNN_FORECASTING_BIN="$REPO_ROOT/build-gpu/bin/no2_forecasting"
python run_forecasting.py --gpu-index 0
```

The runner prepares `data/beijing_pm25_forecasting.csv` automatically if it is
not already present. To prepare it explicitly:

```bash
python prepare_beijing_pm25.py --output-dir data
```

The runner writes:

```text
docs/benchmarks/results/recurrent-lstm-forecasting-<run_id>.json
```

The JSON artifact stores command line, commit hash, machine metadata,
machine-readable metrics, speedups, and raw output.

## Notes

- GPU Recurrent uses OpenNN's CUDA/cuBLAS recurrent path.
- GPU LSTM uses cuDNN RNN/LSTM.
- The benchmark currently uses one seed per scenario in the C++ driver. For a
  publishable headline, increase the seed count or repeat the runner and report
  median/stdev across artifacts.
- Use `--raw-path /path/to/PRSA_data_2010.1.1-2014.12.31.csv` or a local UCI
  ZIP when running without network access on the Linux benchmark machine.
