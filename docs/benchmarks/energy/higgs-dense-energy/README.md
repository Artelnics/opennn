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

## Latest result (2026-08-10, RTX 4080, commit 52e21e15d)

20 epochs × 10.5M rows, batch 7000, median of 3 runs
(artifact `results/gpu-higgs-dense-energy-20260810T125413Z.json`):

| Precision | Engine | Energy (J) | µJ/sample | Avg power | Train window |
|---|---|---:|---:|---:|---:|
| fp32 | **OpenNN** | **10,839** | **51.6** | 243 W | 44.5 s |
| fp32 | PyTorch | 19,786 | 94.2 | 320 W | 61.9 s |
| fp32 | TensorFlow | 19,991 | 95.2 | 297 W | 67.4 s |
| bf16 | **OpenNN** | **5,232** | **24.9** | 232 W | 22.6 s |
| bf16 | PyTorch | 5,590 | 26.6 | 234 W | 23.9 s |
| bf16 | TensorFlow | 6,347 | 30.2 | 225 W | 28.2 s |

OpenNN spends **1.83× less energy than either engine in fp32** and 1.07×/1.21×
less in bf16 for the identical fixed workload.
