# HIGGS Dense Training Speed

This is the active dense MLP benchmark track. It replaces the older Rosenbrock
dense speed tests for new dense claims.

For the full benchmark index, start with [the benchmarks README](../../README.md).
For the dataset contract, use [`../higgs/README.md`](../higgs/README.md).
Large datasets must live outside the repository; see
[`../DATA_POLICY.md`](../../DATA_POLICY.md).

## Build

```bash
cmake -S ../../.. -B ../../../build-benchmarks \
  -DOpenNN_BUILD_EXAMPLES=OFF \
  -DOpenNN_BUILD_BENCHMARKS=ON

cmake --build ../../../build-benchmarks --config Release --target opennn_speed
```

## Publication Run

Use the Python harness for evidence artifacts:

```bash
python run_higgs_dense.py \
  --train "$OPENNN_BENCH_DATA/higgs/higgs_train.csv" \
  --test "$OPENNN_BENCH_DATA/higgs/higgs_test.csv" \
  --epochs 5 \
  --batch 7000 \
  --hidden 1024 \
  --activation relu \
  --hidden-layers 2 \
  --shuffle shuffle \
  --precision bf16 \
  --runs 5
```

It writes:

```text
../../results/gpu-dense-higgs-training-speed-<run_id>.json
```

The JSON captures speed, quality metrics, raw output, commands, framework
versions, CUDA/GPU metadata, git commit, dirty status, and dataset metadata.

## Quick Console Run

```bash
./run_speed.sh \
  "$OPENNN_BENCH_DATA/higgs/higgs_train.csv" \
  "$OPENNN_BENCH_DATA/higgs/higgs_test.csv" \
  5 7000 1024 relu 2 shuffle
```

Both runners report speed and quality:

- `samples_per_sec`
- `median_epoch_s`
- `test_accuracy`
- `test_log_loss`
- `test_roc_auc`
- `quality_gate`

Optional hard thresholds:

```bash
HIGGS_MIN_ACCURACY=0.70 \
HIGGS_MAX_LOG_LOSS=0.65 \
HIGGS_MIN_AUC=0.75 \
python run_higgs_dense.py \
  --train "$OPENNN_BENCH_DATA/higgs/higgs_train.csv" \
  --test "$OPENNN_BENCH_DATA/higgs/higgs_test.csv"
```

## Notes

The scripts named `analyze_*.py`, `gemm_floor.py`, and `pytorch_profile.py` are
engineering probes, not part of the published run.
