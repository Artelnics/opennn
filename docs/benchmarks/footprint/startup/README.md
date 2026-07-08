# Startup Latency Benchmark

Purpose: compare time-to-first-prediction / import-startup overhead — a native
OpenNN binary versus `import torch` / `import tensorflow`.

Runners (each prints its time to first prediction):

- `opennn_startup.cpp` — compile against OpenNN, then run
- `pytorch_startup.py`
- `tensorflow_startup.py`

Measure cold start; record the warmup protocol and versions with the result.
