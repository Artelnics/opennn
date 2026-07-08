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
