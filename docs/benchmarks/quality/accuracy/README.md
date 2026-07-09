# Accuracy Parity Benchmark (HIGGS)

Purpose: quality diagnostic — do OpenNN, PyTorch, and TensorFlow reach the same
predictive quality on the HIGGS classification task at a fixed training budget?

Each engine trains the canonical HIGGS dense classifier and reports its test-set
quality. Parity means the three engines land on the same numbers, not that any
one is fastest — speed lives in the throughput benchmarks.

## Model

Follows the shared [HIGGS Dense Benchmark Contract](../../throughput/higgs/README.md):

```text
28 -> 1024 -> 1024 -> 1
```

- Hidden activation: ReLU
- Output: sigmoid
- Loss: binary cross entropy
- Optimizer: Adam
- Fixed epochs, CPU, fp32

## Metrics

Every engine prints, and the runner aggregates (median over `--runs`):

- `test_accuracy`
- `test_log_loss`
- `test_roc_auc`

The metric code is identical across engines (`metrics.py`, kept in sync with the
throughput sibling), so no framework's internal loss reduction can bias the
comparison.

## Data

Large HIGGS files live outside the repository. Set `OPENNN_BENCH_DATA` and
prepare the split as documented in [`../DATA_POLICY.md`](../../DATA_POLICY.md) and
the [dataset contract](../../throughput/higgs/README.md):

```bash
export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
python ../../throughput/higgs/prepare_higgs.py --raw /path/to/HIGGS.csv.gz
```

The runners read `$OPENNN_BENCH_DATA/higgs/higgs_train.csv` and
`$OPENNN_BENCH_DATA/higgs/higgs_test.csv` by default.

## Build

The OpenNN engine builds as the `opennn_accuracy` target:

```bash
cmake -S ../../.. -B ../../../build-benchmarks \
  -DOpenNN_BUILD_EXAMPLES=OFF \
  -DOpenNN_BUILD_BENCHMARKS=ON
cmake --build ../../../build-benchmarks --config Release --target opennn_accuracy
```

## Run

```bash
python run_accuracy.py --engines opennn,pytorch,tensorflow --epochs 5 --runs 3
```

Override paths and budget as needed:

```bash
python run_accuracy.py \
  --train "$OPENNN_BENCH_DATA/higgs/higgs_train.csv" \
  --test  "$OPENNN_BENCH_DATA/higgs/higgs_test.csv" \
  --epochs 5 --batch 1024 --runs 3
```

The runner writes an immutable result JSON to `../../results/` named
`accuracy-higgs-<run_id>.json` (`benchmark_id: "accuracy-higgs"`) with the
per-engine quality metrics, a parity summary, framework versions, and the git
commit. It refuses to overwrite an existing result file.

Each engine script can also be run directly for a single measurement, e.g.:

```bash
python pytorch_accuracy.py --epochs 5
python tensorflow_accuracy.py --epochs 5
./opennn_accuracy "$OPENNN_BENCH_DATA/higgs/higgs_train.csv" \
                  "$OPENNN_BENCH_DATA/higgs/higgs_test.csv" 5
```
