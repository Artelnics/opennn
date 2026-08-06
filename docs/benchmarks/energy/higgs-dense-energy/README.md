# HIGGS Dense Energy (GPU)

Fixed-work GPU energy benchmark for the canonical HIGGS dense classifier:
OpenNN vs PyTorch vs TensorFlow, fp32 and bf16. It replaces the historical
Rosenbrock dense energy run for new dense energy claims.

For the full benchmark index, start with [the benchmarks README](../../README.md).
For the dataset contract, use [`../../throughput/higgs/README.md`](../../throughput/higgs/README.md).
Large datasets must live outside the repository; see
[`../../DATA_POLICY.md`](../../DATA_POLICY.md).

## Protocol — `gpu-higgs-dense-energy`

Every engine trains the identical model (`28 -> hidden -> hidden -> 1`, ReLU
hidden, sigmoid output, binary cross-entropy, Adam defaults) for the identical
fixed work: same prepared CSVs, same epochs, same batch, same per-epoch
GPU-resident reshuffle. The engine programs are the dense speed drivers in
[`../../throughput/higgs-gpu/`](../../throughput/higgs-gpu/README.md); each
prints `TRAIN_START_UNIX` / `TRAIN_END_UNIX` around its timed training loop
(warmup epochs excluded).

The runner samples `power.draw` and `clocks.current.sm` at 20 Hz with
`nvidia-smi` and integrates power (trapezoid) only inside the train window. It
reports total and active energy (idle baseline subtracted), microjoules per
nominal epoch-sample (`energy / (train_rows x epochs)`, same divisor for every
engine), average power, median SM clock, wall time, and the engine's speed and
test-quality metrics.

GPU board energy only (sampled power, not a hardware joule counter). Run on a
quiet GPU.

## Run

```bash
cmake -S ../../.. -B ../../../build-benchmarks -DOpenNN_BUILD_EXAMPLES=OFF -DOpenNN_BUILD_BENCHMARKS=ON
cmake --build ../../../build-benchmarks --config Release --target opennn_speed

python run_higgs_dense_energy.py \
  --train "$OPENNN_BENCH_DATA/higgs/higgs_train.csv" \
  --test  "$OPENNN_BENCH_DATA/higgs/higgs_test.csv" \
  --epochs 20 --batch 7000 --hidden 1024 --hidden-layers 2 \
  --activation relu --precision both --runs 3
```

Writes `../../results/gpu-higgs-dense-energy-<run_id>.json` with per-run and
aggregate metrics, framework versions, GPU state, git commit, and dirty status.

## Result metrics

| Metric | Meaning |
|---|---|
| `energy_total_j` / `energy_active_j` | GPU board energy inside the train window; active subtracts the idle baseline |
| `uj_per_sample_total` / `uj_per_sample_active` | Energy per nominal epoch-sample, identical divisor for every engine |
| `avg_power_w`, `sm_clock_median_mhz` | Where the energy went: power level and sustained SM clock |
| `train_window_s`, `samples_per_sec` | Fixed-work wall time and throughput |
| `test_accuracy`, `test_log_loss`, `test_roc_auc` | Quality gate: proof the fixed work trained a real classifier |
