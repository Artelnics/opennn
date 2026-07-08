# Data Capacity Benchmark

Purpose: measure how many tabular samples fit and train under a fixed RAM budget
— OpenNN's compact in-RAM `float32` matrix (`TabularDataset` default `Matrix`
storage mode) versus the `pandas.read_csv` path used by PyTorch and TensorFlow.

This benchmark only makes sense in `Matrix` storage mode, where the whole dataset
lives in RAM: it compares the size of each framework's default in-memory
representation. In `BinaryFile` storage mode `TabularDataset` keeps the data
matrix empty and streams batches from an on-disk `.bin` cache, so RAM no longer
scales with the sample count and the RAM ceiling this benchmark measures does not
exist.

Generate the data (`generate_rosenbrock.c`) and run the capped sweep on Windows:

```powershell
.\run_sweep.ps1
```

Reports the maximum number of samples each framework loads and trains before it
runs out of the memory budget.
