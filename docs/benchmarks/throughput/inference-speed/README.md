# CPU Inference Speed Benchmark

Purpose: CPU batch-inference speed comparison for a tuned dense MLP — OpenNN
vs PyTorch vs TensorFlow.

Build the OpenNN driver, then run:

```bash
cmake --build build-benchmarks --target opennn_inference
./run_inference.sh rosenbrock.csv 8000 1000 1000 30
```

Reports samples/s and ms per batch. Use identical data/weights (or a checksum
gate) across frameworks.
