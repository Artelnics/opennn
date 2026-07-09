# OpenNN benchmarks: OpenNN vs PyTorch vs TensorFlow

This directory holds a **reproducible** benchmark suite comparing OpenNN with
PyTorch and TensorFlow. It ships the **code and the instructions to run each
benchmark**, not the results: no measured numbers, result tables, or run
artifacts are committed. You (or an AI agent) clone the repository, build the
OpenNN drivers, prepare the data, and run each benchmark to produce your own
numbers on your own hardware.

> Why no baked-in numbers? A benchmark result is only meaningful with its
> hardware, framework versions, and commit. Rather than ship stale numbers, the
> repository ships the recipe; every run writes its own result JSON under
> [`results/`](results/) with full provenance.

## What is measured

The suite compares deployment and performance characteristics across three
frameworks on matched tasks: quality (accuracy, precision, convergence),
throughput (CPU/GPU training and inference), capacity (largest batch / most data
under a memory cap), energy, and footprint (application code size, startup,
memory, standalone export).

## How to run

1. **Read the data policy.** Large datasets never live in git. Set the data root
   before preparing any dataset — see [`DATA_POLICY.md`](DATA_POLICY.md):

   ```bash
   export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
   ```

2. **Build the OpenNN benchmark drivers** (registered in
   [`CMakeLists.txt`](CMakeLists.txt)):

   ```bash
   cmake -S . -B build-benchmarks \
     -DOpenNN_BUILD_EXAMPLES=OFF \
     -DOpenNN_BUILD_BENCHMARKS=ON
   cmake --build build-benchmarks --config Release
   ```

3. **Pick a benchmark below, open its `README.md`, and follow it.** Each folder
   is self-contained: it names its runner script, its OpenNN/PyTorch/TensorFlow
   sources, how to prepare the data, and the exact command to run.

4. **Collect the result.** Runners that emit a result artifact write it to
   [`results/`](results/); the required schema is in
   [`results/README.md`](results/README.md).

## Benchmark index

Benchmarks live one level below these metric buckets. Open the folder `README.md`
for the runner and command.

### quality/
| Benchmark | What it runs |
|---|---|
| [accuracy](quality/accuracy/README.md) | Predictive quality parity (accuracy / log-loss / ROC-AUC) on the HIGGS dense classifier |
| [precision](quality/precision/README.md) | Best error floor per optimizer on the Rosenbrock task (the one documented regression exception to the HIGGS rule) |
| [convergence](quality/convergence/README.md) | Wall-clock time to a fixed held-out quality target on the HIGGS dense classifier |
| [recurrent-lstm-forecasting](quality/recurrent-lstm-forecasting/README.md) | Recurrent vs LSTM forecasting on UCI Beijing PM2.5 |

### throughput/
Each folder hosts a training and an inference benchmark for one model. GPU folders measure fp32 and bf16; the CPU folder measures fp32. All compare OpenNN vs PyTorch vs TensorFlow.

| Benchmark folder | What it runs |
|---|---|
| [attention-speed](throughput/attention-speed/README.md) | Encoder-decoder Transformer training and inference (GPU) |
| [resnet50](throughput/resnet50/README.md) | ResNet-50 training and inference on CIFAR (GPU) |
| [higgs-gpu](throughput/higgs-gpu/README.md) | HIGGS dense training and inference (GPU) |
| [higgs](throughput/higgs/README.md) | HIGGS dense training and inference (CPU) |
| [precision-sweep](throughput/precision-sweep/README.md) | OpenNN fp32-vs-bf16 sweep (supporting/internal) |

### capacity/
| Benchmark | What it runs |
|---|---|
| [data-capacity](capacity/data-capacity/README.md) | Most tabular samples that fit and train under a fixed RAM cap |
| [higgs-max-batch](capacity/higgs-max-batch/README.md) | Largest HIGGS dense batch that completes one step (GPU + CPU) |
| [resnet50-max-batch](capacity/resnet50-max-batch/README.md) | Largest ResNet-50 training batch that fits |
| [transformer-max-batch](capacity/transformer-max-batch/README.md) | Largest Transformer batch that fits (train + infer) |

### energy/
| Benchmark | What it runs |
|---|---|
| [transformer-energy](energy/transformer-energy/README.md) | GPU energy to train a Transformer to a fixed quality target |

### footprint/
| Benchmark | What it runs |
|---|---|
| [application-loc](footprint/application-loc/README.md) | Logical lines of code for the same Iris workflow |
| [export](footprint/export/README.md) | Exporting a trained model as standalone source code |
| [memory](footprint/memory/README.md) | Baseline RAM and GPU-ready VRAM after empty objects |
| [startup](footprint/startup/README.md) | Time-to-first-prediction / import-startup overhead |

## Files in this directory

| File | Purpose |
|---|---|
| [`README.md`](README.md) | This index and run guide. |
| [`DATA_POLICY.md`](DATA_POLICY.md) | Where datasets live; what must stay out of git. |
| [`benchmark_manifest.json`](benchmark_manifest.json) | Machine-readable inventory: each benchmark's folder, comparison, metric names, and runner commands. |
| [`CMakeLists.txt`](CMakeLists.txt) | Builds the OpenNN benchmark drivers. |
| [`results/`](results/) | Where runners write result JSON; empty in a clean checkout. |
| [`tools/validate_benchmarks.py`](tools/validate_benchmarks.py) | Checks the inventory stays consistent and that no results/binaries get committed. |

Run the validator after adding, renaming, or retiring a benchmark:

```bash
cd docs/benchmarks
python tools/validate_benchmarks.py
```
