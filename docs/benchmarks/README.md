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
| [accuracy](quality/accuracy/README.md) | Predictive accuracy (R²) on the Rosenbrock regression task |
| [precision](quality/precision/README.md) | Best error floor per optimizer on the Rosenbrock task |
| [convergence](quality/convergence/README.md) | Convergence diagnostic on the Rosenbrock task |
| [recurrent-lstm-forecasting](quality/recurrent-lstm-forecasting/README.md) | Recurrent vs LSTM forecasting on UCI Beijing PM2.5 |

### throughput/
| Benchmark | What it runs |
|---|---|
| [cnn-training-speed](throughput/cnn-training-speed/README.md) | Small CNN training on MNIST (GPU) |
| [resnet50-training-speed](throughput/resnet50-training-speed/README.md) | ResNet-50 training on CIFAR / ImageNet-geometry (GPU) |
| [attention-speed](throughput/attention-speed/README.md) | Encoder-decoder Transformer inference and training (GPU) |
| [inference-speed](throughput/inference-speed/README.md) | Dense MLP batch inference (CPU) |
| [higgs](throughput/higgs/README.md) | HIGGS dense classifier training and inference (CPU) |
| [training-speed](throughput/training-speed/README.md) | HIGGS dense training speed with quality gate (GPU) |
| [precision-sweep](throughput/precision-sweep/README.md) | Precision sweep driver |

### capacity/
| Benchmark | What it runs |
|---|---|
| [data-capacity](capacity/data-capacity/README.md) | Most tabular samples that fit and train under a fixed RAM cap |
| [higgs-max-batch](capacity/higgs-max-batch/README.md) | Largest HIGGS dense batch that completes one step (GPU + CPU) |
| [resnet50-max-batch](capacity/resnet50-max-batch/README.md) | Largest ResNet-50 training batch that fits |
| [image-classification-max-batch](capacity/image-classification-max-batch/README.md) | Low-level image-classification capacity probes |
| [rosenbrock-max-batch](capacity/rosenbrock-max-batch/README.md) | Dense MLP max batch and speed on the GPU |
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
