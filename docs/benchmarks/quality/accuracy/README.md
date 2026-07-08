# Accuracy Benchmark

Purpose: controlled quality diagnostic — does OpenNN reach the same predictive
accuracy (R²) as PyTorch and TensorFlow on the Rosenbrock regression task?

Run:

```bash
./run_accuracy.sh
```

The same shared normalized train/test split feeds all three frameworks; a single
neutral scorer (`score.py`) computes MSE and R² from each framework's test-set
predictions so internal loss/reduction differences do not affect the comparison.
