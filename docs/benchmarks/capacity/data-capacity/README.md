# Data Capacity Benchmark

Purpose: measure how many tabular samples fit and train under a fixed RAM budget
— OpenNN's memory-mapped, compact-matrix loader versus the `pandas.read_csv`
path used by PyTorch and TensorFlow.

Generate the data (`generate_rosenbrock.c`) and run the capped sweep on Windows:

```powershell
.\run_sweep.ps1
```

Reports the maximum number of samples each framework loads and trains before it
runs out of the memory budget.
