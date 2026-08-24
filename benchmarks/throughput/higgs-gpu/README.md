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

## Building with CUTLASS

The first layer of this network contracts 28 features, and cuBLASLt is poor at
that shape - it can only promise two-element alignment on the input and picks an
`align2` kernel. A CUTLASS kernel instantiated for the shape is 1.03x to 1.48x
faster with bit-identical output, worth 2-3% of the whole bf16 batch between
4,096 and 65,536 rows and nothing at all below.

CUTLASS is header-only and opt-in. Without it the path compiles to a stub that
declines every shape and cuBLASLt runs everything, exactly as before:

```bash
cmake -S ../../.. -B ../../../build-benchmarks \
  -DOpenNN_BUILD_BENCHMARKS=ON \
  -DOpenNN_CUTLASS_INCLUDE_DIR=/path/to/cutlass/include
```

`OPENNN_CUTLASS_NARROW_K=0` keeps cuBLASLt at runtime for the A/B.

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
with the shared CLI `<test_csv> [batch[,batch...]] [runs] [fp32|bf16] [hidden] [hidden_layers] [activation]`.

## Inference across a batch ladder — `run_higgs_infer_sweep.py`

Batch size decides which cost dominates - a batch of 256 is 18 microseconds and
mostly kernel launches, a batch of 65,536 is two milliseconds and entirely GEMM -
so the inference claim is measured over a ladder, not at one point:

```bash
sudo ../../tools/gpu_clocks.sh lock 2700

python run_higgs_infer_sweep.py \
  --batches 256,1024,4096,8192,16384,65536 \
  --runs 5 --rounds 6 --soak 1 --precision both
```

Each engine sweeps the whole ladder inside one process, the engine order and the
batch order rotate every round, the first round is discarded as a soak, and the
per-pass times are kept in the artifact in temporal order. The same runner takes
`--arm engine:label:KEY=VAL,...` repeatedly, which alternates two configurations
of one engine against each other under the same protocol - that is how every
lever in the note is decided:

```bash
python run_higgs_infer_sweep.py --batches 256,1024,8192 --rounds 4 \
  --arm "opennn:on:" --arm "opennn:off:OPENNN_SINGLE_OUTPUT_ACTIVATION=0"
```

Each engine picks its own fastest path at each rung and names it: `pt_path`
(hand-captured CUDA graph vs `torch.compile`) and `tf_path` (per-batch dispatch
vs the batch loop compiled into one `tf.function`). Neither is the same at every
batch size, which is why neither is pinned.

## Result metrics

Every result JSON records the dataset path, row counts, command line, framework
versions, CUDA/GPU metadata, git commit, and dirty status alongside the numbers.
