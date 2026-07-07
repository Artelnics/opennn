# CPU Inference Speed Benchmark

Purpose: CPU batch-inference speed comparison for a tuned dense MLP.

Top-level note:
[`inference-speed-opennn-vs-pytorch-vs-tensorflow.md`](inference-speed-opennn-vs-pytorch-vs-tensorflow.md)

Run:

```bash
./run_inference.sh rosenbrock.csv 8000 1000 1000 30
```

Lifecycle: hold back. The current result is a valid Windows tuned run, but it
needs a reference Linux rerun and archived result JSON before it becomes a
public headline.
