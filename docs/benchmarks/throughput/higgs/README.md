# HIGGS Dense Benchmark Contract

This directory defines the shared dataset contract for dense-network benchmarks.
Use it for dense MLP training speed, inference speed, capacity, energy, and
quality gates so the numbers all describe the same real tabular problem.

Large HIGGS files must live outside the repository. Set
`OPENNN_BENCH_DATA` first; see [`../DATA_POLICY.md`](../../DATA_POLICY.md).

## Dataset

Use the HIGGS dataset from the UCI Machine Learning Repository:

- 11,000,000 rows
- 28 numeric input features
- binary target: `1` for signal, `0` for background
- raw file layout: `label,feature_0,...,feature_27`
- canonical split: first 10,500,000 rows for training, last 500,000 rows for test

Reference: https://archive.ics.uci.edu/dataset/280/higgs

## Prepared Files

The benchmark runners expect prepared CSV files with this layout:

```text
feature_0,feature_1,...,feature_27,label
```

There is no header row by default. The target is last because OpenNN's
`TabularDataset` defaults the last column to `Target`, while PyTorch and
TensorFlow slice the same file as `x = data[:, :-1]`, `y = data[:, -1:]`.

Prepare the full split:

```bash
export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
python docs/benchmarks/throughput/higgs/prepare_higgs.py \
  --raw /path/to/HIGGS.csv.gz
```

Prepare a small smoke split:

```bash
python docs/benchmarks/throughput/higgs/prepare_higgs.py \
  --raw /path/to/HIGGS.csv.gz \
  --out "$OPENNN_BENCH_DATA/higgs-smoke" \
  --train-rows 10000 \
  --test-rows 2000
```

The script writes:

- `higgs_train.csv`
- `higgs_test.csv`
- `higgs_metadata.json`

By default it standardizes each feature using training-set mean and standard
deviation, then applies the same transform to the test set. Disable that with
`--no-normalize` only when the benchmark explicitly wants raw HIGGS values.

## Canonical Dense MLP

For dense speed and capacity benchmarks, use:

```text
28 -> hidden -> hidden -> 1
```

Default hidden width: `1024`.

Hidden activation: `ReLU`.

Output/loss: sigmoid output with binary cross entropy.

Batch, precision, and residency policy remain benchmark-specific, but every
published dense result should state the HIGGS file, split size, hidden width,
batch size, precision, and whether the dataset was GPU-resident.

## Quality Gate

Dense speed runs must also report:

- `test_accuracy`
- `test_log_loss`
- `test_roc_auc`

The shared training harness requires both train and test files:

```bash
cmake -S ../../.. -B ../../../build-benchmarks \
  -DOpenNN_BUILD_EXAMPLES=OFF \
  -DOpenNN_BUILD_BENCHMARKS=ON
cmake --build ../../../build-benchmarks --config Release --target opennn_speed opennn_higgs_cpu

cd docs/benchmarks/throughput/higgs-gpu
python run_higgs_dense.py \
  --train "$OPENNN_BENCH_DATA/higgs/higgs_train.csv" \
  --test "$OPENNN_BENCH_DATA/higgs/higgs_test.csv" \
  --epochs 5 --batch 7000 --runs 5 --precision bf16
```

Optional hard thresholds can be set through:

```bash
HIGGS_MIN_ACCURACY=0.70 \
HIGGS_MAX_LOG_LOSS=0.65 \
HIGGS_MIN_AUC=0.75 \
python run_higgs_dense.py \
  --train "$OPENNN_BENCH_DATA/higgs/higgs_train.csv" \
  --test "$OPENNN_BENCH_DATA/higgs/higgs_test.csv"
```

## Presentation Rule

Rosenbrock dense numbers are historical stress tests. New dense headline numbers
should come from this HIGGS contract, with raw logs and result JSON attached.
