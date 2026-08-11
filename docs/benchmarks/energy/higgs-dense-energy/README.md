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

## Latest result (2026-08-11, RTX 4080, commit 6a721ddc8)

20 epochs × 10.5M rows, batch 7000, median of 3 runs
(artifact `results/gpu-higgs-dense-energy-20260811T125222Z.json`):

| Precision | Engine | Energy (J) | µJ/sample | Avg power | Train window |
|---|---|---:|---:|---:|---:|
| fp32 (TF32) | **OpenNN** | **10,627** | **50.6** | 258 W | **41.2 s** |
| fp32 (TF32) | PyTorch | 11,243 | 53.5 | 259 W | 43.4 s |
| fp32 (TF32) | TensorFlow | 11,638 | 55.4 | 238 W | 49.0 s |
| bf16 | **OpenNN** | **4,957** | **23.6** | 260 W | **19.1 s** |
| bf16 | PyTorch | 5,574 | 26.5 | 235 W | 23.8 s |
| bf16 | TensorFlow | 6,342 | 30.2 | 224 W | 28.4 s |

OpenNN spends the least energy in every cell: **1.12× less than PyTorch and
1.28× less than TensorFlow in bf16**, 1.06×/1.10× in fp32. Versus the
2026-08-10 snapshot the bf16 window shrank 22.6 → 19.1 s (the asynchronous
batch-list prefetch keeps the GPU fed between epochs), and the fp32 gap
narrowed because "fp32" now means TF32 in all three engines — the earlier
1.83× fp32 figure compared OpenNN-TF32 against strict-fp32 competitors and is
retired.
