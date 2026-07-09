# HIGGS Dense Speed (GPU)

The active dense-MLP GPU benchmark track: **training** and **inference** throughput
of the canonical HIGGS dense classifier, OpenNN vs PyTorch vs TensorFlow, fp32 and
bf16. It replaces the older Rosenbrock dense speed tests for new dense claims.

For the full benchmark index, start with [the benchmarks README](../../README.md).
For the dataset contract, use [`../higgs/README.md`](../higgs/README.md).
Large datasets must live outside the repository; see
[`../DATA_POLICY.md`](../../DATA_POLICY.md).

## Model

Canonical HIGGS dense classifier (see the contract):

```text
28 -> hidden -> hidden -> 1
```

ReLU hidden activations, sigmoid output, binary cross-entropy. Default hidden
width `1024`, two hidden layers. CSV layout `feature_0,...,feature_27,label`.
Each engine runs its fair fast path: OpenNN GPU-resident data + CUDA graph,
PyTorch `torch.compile` + AMP + TF32, TensorFlow XLA + `mixed_bfloat16`.

## Training — `gpu-higgs-dense-training-speed`

```bash
cmake -S ../../.. -B ../../../build-benchmarks -DOpenNN_BUILD_EXAMPLES=OFF -DOpenNN_BUILD_BENCHMARKS=ON
cmake --build ../../../build-benchmarks --config Release --target opennn_speed

python run_higgs_dense.py \
  --train "$OPENNN_BENCH_DATA/higgs/higgs_train.csv" \
  --test  "$OPENNN_BENCH_DATA/higgs/higgs_test.csv" \
  --epochs 5 --batch 7000 --hidden 1024 --activation relu --hidden-layers 2 \
  --shuffle shuffle --precision both --runs 5
```

Writes `../../results/gpu-dense-higgs-training-speed-<run_id>.json`. Reports
`samples_per_sec`, `median_epoch_s`, and the quality gate `test_accuracy`,
`test_log_loss`, `test_roc_auc`, `quality_gate`. Optional hard thresholds:

```bash
HIGGS_MIN_ACCURACY=0.70 HIGGS_MAX_LOG_LOSS=0.65 HIGGS_MIN_AUC=0.75 \
python run_higgs_dense.py --train "$OPENNN_BENCH_DATA/higgs/higgs_train.csv" --test "$OPENNN_BENCH_DATA/higgs/higgs_test.csv"
```

## Inference — `gpu-higgs-dense-inference-speed`

Forward-only twin of the training benchmark; the label column is ignored for the
speed measurement.

```bash
cmake --build ../../../build-benchmarks --config Release --target opennn_higgs_infer

python run_higgs_infer.py \
  --test "$OPENNN_BENCH_DATA/higgs/higgs_test.csv" \
  --batch 8192 --hidden 1024 --activation relu --hidden-layers 2 \
  --precision both --runs 5
```

Writes `../../results/gpu-higgs-dense-inference-speed-<run_id>.json`. Reports
`samples_per_sec` and `ms_per_batch`. Each engine can also be driven directly
with the shared CLI `<test_csv> [batch] [runs] [fp32|bf16] [hidden] [hidden_layers] [activation]`.

## Result metrics

Every result JSON records the dataset path, row counts, command line, framework
versions, CUDA/GPU metadata, git commit, and dirty status alongside the numbers.
