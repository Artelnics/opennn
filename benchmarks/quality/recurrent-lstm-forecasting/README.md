# Recurrent/LSTM Forecasting Benchmark

Purpose: compare OpenNN's recurrent layer against its LSTM layer on UCI Beijing
PM2.5 forecasting, on CPU and GPU, optionally alongside PyTorch and TensorFlow.

Build the OpenNN driver and run the harness:

```bash
cmake --build build-benchmarks --target recurrent_lstm_forecasting_benchmark
OPENNN_FORECASTING_BIN=../../../build-benchmarks/bin/recurrent_lstm_forecasting_benchmark \
  python run_forecasting.py --frameworks opennn,pytorch,tensorflow
```

`prepare_beijing_pm25.py` fetches and prepares the dataset. The harness reports
test RMSE, training time, and CPU/GPU speedups, and writes a result JSON under
`../../results/`.

## Matched performance matrix

Use fixed epochs when comparing throughput. This makes every engine process the
same number of complete training windows and records both elapsed time and
samples per second for each batch size.

```bash
python run_forecasting.py \
  --frameworks opennn,pytorch,tensorflow \
  --python-cpu \
  --scenarios B1,B2,B3,B4 \
  --batch-sizes 32,64,128,256,512 \
  --epochs 5 \
  --seeds 3 \
  --precision fp32 \
  --pytorch-compile 0 \
  --tf-jit auto
```

The Python baselines use their native optimized recurrent implementations:
cuDNN plus fused Adam on PyTorch GPU, oneDNN/foreach Adam on PyTorch CPU, and
Keras' cuDNN-enabled RNN/LSTM plus compiled `fit` steps in TensorFlow. Set
`--cpu-threads` after tuning for the host CPU. `torch.compile` modes and
TensorFlow XLA can be selected explicitly; they should be retained only when
their measured aggregate is faster than eager/native execution.

Useful controls are also accepted as environment variables:

- `OPENNN_FORECASTING_BATCH_SIZES=32,64,128,256,512`
- `OPENNN_FORECASTING_SCENARIOS=B1,B2,B3,B4`
- `OPENNN_FORECASTING_EPOCHS=5`
- `OPENNN_FORECASTING_SEEDS=3`
- `OPENNN_FORECASTING_CPU_THREADS=4`
- `OPENNN_FORECASTING_PYTORCH_COMPILE=0`
- `OPENNN_FORECASTING_TF_JIT=auto`
- `OPENNN_FORECASTING_PRECISION=fp32` (`bf16` is an optional matched GPU mode)

Reported throughput is steady-state training throughput. Each engine warms the
exact full and remainder batch shapes (plus validation), then restores model,
optimizer, and random-generator state before timing. Validation still runs for
the quality/stopping protocol but is excluded from `time_s`; device setup,
cuDNN plan construction, and CUDA graph capture are outside the timed region.
OpenNN captures data staging, forward, loss, backward, and Adam in the replayed
training graph. Its MSE path also fuses epoch-metric accumulation with output
gradient generation in one CUDA kernel.
Use at least 10 fixed epochs and multiple seeds for a published comparison.

OpenNN recurrent CUDA tuning can be reproduced with
`OPENNN_RNN_GRAPH_GROUP=1..8`, `OPENNN_RNN_PERSIST=0|1`,
`OPENNN_RNN_PAD_FEATURES=0|1`,
`OPENNN_RNN_PACKED_LAYOUT=0|1`, `OPENNN_RNN_DOUBLE_BIAS=0|1`, and
`OPENNN_RNN_DEFAULT_MATH=0|1`. Defaults are selected from the measured cell and
shape matrix; overrides are intended for hardware-specific validation, not for
using different settings between compared engines.

Set `OPENNN_FORECASTING_PROFILE_STEPS=N` to collect a short PyTorch CUDA
operator/kernel profile after warmup and state reset. OpenNN's synchronized
per-stage profile uses its standard profiler controls and is intended for
attribution rather than throughput reporting.
