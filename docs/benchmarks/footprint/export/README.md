# Standalone Export Benchmark

Purpose: demonstrate that OpenNN can export a trained model as standalone source
code (C, Python, JavaScript, PHP) instead of a runtime-dependent model file.

First generate the tiny synthetic training set (`generate_sum.py` writes
`sum.csv`: 100 inputs whose target is their sum), then compile
`opennn_export.cpp` against the OpenNN library and run it from this folder; it
trains a small model and writes it out as source in each target language.
Archive the generated source plus a compile/run check of it as the result.

```bash
cd docs/benchmarks/footprint/export
python generate_sum.py            # writes sum.csv
./opennn_export                   # reads sum.csv, writes model.c and model.py
```

`sum.csv` and the exported `model.*` sources are generated artifacts and are not
committed (see `.gitignore` and ../../DATA_POLICY.md).
