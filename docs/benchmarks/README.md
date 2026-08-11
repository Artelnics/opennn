# OpenNN benchmarks: OpenNN vs PyTorch vs TensorFlow

This directory holds a **reproducible** benchmark suite comparing OpenNN with
PyTorch and TensorFlow. It ships code, run instructions, immutable result JSON,
and the historical measurement reports that explain published engineering
claims. Every result must retain its hardware, framework versions, commit, and
methodology; reruns create new artifacts rather than overwriting old evidence.

> Historical reports are evidence from the machine and commit stated in each
> document, not measurements of the current checkout. Use the active runners to
> reproduce or supersede them on current hardware.

## Historical result reports

The detailed reports removed during the July 2026 benchmark cleanup have been
restored alongside their benchmark folders. The central claim/status matrix is
[`PRESENTATION_CLAIMS.md`](PRESENTATION_CLAIMS.md), and the machine-readable
artifacts are under [`results/`](results/).

Headline results from the 2026-08-10 full-matrix run (RTX 4080 / i9-12900K,
commit 52e21e15d; every number backed by an artifact in `results/`):

| Area | OpenNN | PyTorch | TensorFlow |
|---|---:|---:|---:|
| Transformer bf16 inference, seq 512 (GPU) | **588,435 tok/s** | 429,233 | 308,685 |
| Transformer bf16 inference max batch (GPU) | **1,987** | 951 | 563 |
| ResNet-50 fp32 training max batch (GPU) | **18,050** | 9,216 | 11,036 |
| ResNet-50 bf16 inference (GPU) | **185,903 samples/s** | 125,592 | 83,683 |
| HIGGS bf16 inference (GPU) | **34.6M samples/s** | 31.9M | 32.4M |
| HIGGS CPU training (MKL) | **107,121 samples/s** | 99,923 | 102,040 |
| Transformer fixed-work energy, 10 epochs (GPU) | **25.6 Wh** | 34.8 Wh | 39.7 Wh |
| Baseline RAM | **235.6 MB** | 816.0 MB | 982.5 MB |

Known open issue: the CUDA-graph *training* path currently adds only ~14%
over eager (the pre-2026-08-09 graph path was ~90% faster but a graph-off A/B
proved it numerically non-equivalent to eager — its speedup was partly
artifact; the current graph path is correct). Eager training, CPU training,
and every inference/capacity/energy cell are unaffected. The open task is
recovering the mega-launch speedup on the correct path. The former ResNet-50
max-batch deficit (4,752 vs TensorFlow's 2.47× lead, June 2026) is resolved —
OpenNN now leads every capacity cell.

## What is measured

The core of the suite is a **matrix of three model families times four
metrics**, every cell comparing OpenNN vs PyTorch vs TensorFlow on the
identical model, data, and workload. Energy cells all use the same fixed-work
semantics: identical epochs for every engine, GPU board power integrated over
each engine's timed training window. Around the matrix sit the quality
benchmarks (same HIGGS contract, CPU) and the footprint benchmarks.

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

### The matrix — three model families × four metrics

Folders stay grouped by metric bucket; the table maps every cell to its folder.
Open the folder `README.md` for the runner and command.

| | Training | Inference | Max batch | Energy (fixed work) |
|---|---|---|---|---|
| **Dense MLP — HIGGS, CPU** | [higgs](throughput/higgs/README.md) | [higgs](throughput/higgs/README.md) | [higgs-max-batch `--device cpu`](capacity/higgs-max-batch/README.md) | — |
| **Dense MLP — HIGGS, GPU** | [higgs-gpu](throughput/higgs-gpu/README.md) | [higgs-gpu](throughput/higgs-gpu/README.md) | [higgs-max-batch](capacity/higgs-max-batch/README.md) | [higgs-dense-energy](energy/higgs-dense-energy/README.md) |
| **CNN — ResNet-50, GPU** | [resnet50](throughput/resnet50/README.md) | [resnet50](throughput/resnet50/README.md) | [resnet50-max-batch](capacity/resnet50-max-batch/README.md) | [resnet50-energy](energy/resnet50-energy/README.md) |
| **Transformer, GPU** | [attention-speed](throughput/attention-speed/README.md) | [attention-speed](throughput/attention-speed/README.md) (seq 128/256/512) | [transformer-max-batch](capacity/transformer-max-batch/README.md) | [transformer-energy](energy/transformer-energy/README.md) |

GPU cells measure fp32 and bf16; CPU cells measure fp32. CPU is not measured
for ResNet-50 or the Transformer (impractically slow), and CPU energy is not
measured (GPU board power is the only clean sensor available to all engines).

One benchmark crosses the Training and Max-batch columns:
[peak-batch-speed](throughput/peak-batch-speed/README.md) sweeps the batch
upward per engine and reports training throughput at **each engine's own best
batch** (curve, peak, OOM frontier) for all three GPU families, reusing the
matrix's speed drivers. Scaffolding added 2026-08-11; not yet executed.

### quality/
| Benchmark | What it runs |
|---|---|
| [accuracy](quality/accuracy/README.md) | Predictive quality parity (accuracy / log-loss / ROC-AUC) on the HIGGS dense classifier |
| [precision](quality/precision/README.md) | Best error floor per optimizer on the Rosenbrock task (the one documented regression exception to the HIGGS rule) |
| [convergence](quality/convergence/README.md) | Wall-clock time to a fixed held-out quality target on the HIGGS dense classifier |
| [recurrent-lstm-forecasting](quality/recurrent-lstm-forecasting/README.md) | Recurrent vs LSTM forecasting on UCI Beijing PM2.5 |

### footprint/
| Benchmark | What it runs |
|---|---|
| [application-loc](footprint/application-loc/README.md) | Logical lines of code for the same Iris workflow |
| [export](footprint/export/README.md) | Exporting a trained model as standalone source code |
| [memory](footprint/memory/README.md) | Baseline RAM and GPU-ready VRAM after empty objects |
| [startup](footprint/startup/README.md) | Time-to-first-prediction / import-startup overhead |

The static capability/size notes (`footprint/gpu-on-windows-…`, `size-…`,
`loc-…`, `dependencies-…`) are prose comparisons with no runner; they live
alongside the executable footprint benchmarks.

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
