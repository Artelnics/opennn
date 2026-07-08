# Precision Benchmark

Purpose: compare the best error floor each framework reaches with the optimizers
it ships, on the Rosenbrock approximation task — OpenNN's second-order
quasi-Newton and Levenberg-Marquardt versus first-order Adam (all three) and
PyTorch's LBFGS.

Run:

```bash
./run_precision.sh
```

Reports the final MSE and wall-clock time per optimizer.
