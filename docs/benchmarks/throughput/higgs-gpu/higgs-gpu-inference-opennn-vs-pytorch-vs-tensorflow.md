# GPU HIGGS dense inference: OpenNN vs PyTorch vs TensorFlow

OpenNN leads HIGGS dense inference in both fp32 and bf16 on an NVIDIA GeForce RTX 4080, reaching 17.13 million samples/s in fp32 (1.54x PyTorch) and 34.61 million samples/s in bf16. Median of 5 runs (2026-08-10, commit 52e21e15d; artifact `results/gpu-higgs-dense-inference-speed-20260810T123521Z.json`).

> Results are medians across five runs, with the standard deviation reported for every framework. CUDA Graphs are active in the OpenNN and PyTorch paths, and both stage each batch through the same device-to-device copy pattern.

## Contents

- [Introduction](#introduction)
- [Benchmark application](#benchmark-application)
- [Reference computer](#reference-computer)
- [Methodology](#methodology)
- [Results](#results)
- [Discussion](#discussion)
- [Conclusions](#conclusions)
- [Reproducing](#reproducing)
- [References](#references)

## Introduction

Inference removes gradient and optimizer work from the HIGGS dense network and measures the forward path alone. This exposes device residency, kernel launch overhead, dense GEMM efficiency, activation fusion, and the cost of the selected precision.

The comparison uses the same 28-1024-1024-1 ReLU network in OpenNN, PyTorch, and TensorFlow. OpenNN leads both precision cells, with a larger relative advantage in fp32 and a narrower field in bf16.

## Benchmark application

| Item | Configuration |
|---|---|
| Dataset | HIGGS held-out split |
| Samples processed | 499,712 |
| Inputs | 28 normalized numerical features |
| Network | 28 -> 1024 ReLU -> 1024 ReLU -> 1 |
| Parameters | 1,080,321 |
| Mode | Forward-only inference |
| Batch | 8,192 |
| Precisions | fp32 and bf16 |
| Metrics | Samples/s and milliseconds/batch |

## Reference computer

| Component | Value |
|---|---|
| CPU | Intel Core i9-12900K |
| GPU | NVIDIA GeForce RTX 4080, 16 GB |
| Operating system | Linux 6.17 x86_64 |
| NVIDIA driver | 595.84 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| PyTorch CUDA / cuDNN | CUDA 13.0 / cuDNN 9.24 |
| TensorFlow | 2.21.0 |
| OpenNN | 9.0.0 |
| Run ID | 20260714T125416Z |

## Methodology

Each engine processes the same held-out rows with the same network, batch, activation, parameter count, and precision. Labels are ignored by the timed inference path.

- OpenNN uses `calculate_outputs_resident`, keeps parameters and activations on the GPU, and replays a captured CUDA Graph.
- PyTorch uses inference mode, bf16 autocast for the bf16 cell, and a manually captured `torch.cuda.CUDAGraph`. The eager path remains available with `PT_NOGRAPH=1` for diagnostic comparisons.
- OpenNN and PyTorch copy each resident batch into a fixed capture buffer with one device-to-device copy before graph replay, so both graph paths use stable pointers under the same staging contract.
- TensorFlow uses compiled graph execution and mixed bf16 for the bf16 cell.
- Framework warmup and TensorFlow XLA compilation occur before the timed passes.
- Samples per second and milliseconds per batch are reported from the same measured pass.
- The artifact contains five successful runs per engine and precision; the tables report medians and standard deviations across those runs.

Dataset loading, process startup, model construction, the initial host-to-device upload, graph capture, and warmup are outside the measured region. The per-batch device-to-device staging copy is included.

## Results

| Precision | Framework | Median throughput | Standard deviation | Median batch time | OpenNN speedup |
|---|---|---:|---:|---:|---:|
| fp32 | OpenNN | **17,125,371 samples/s** | **712,055 samples/s** | **0.478 ms** | **1.000x** |
| fp32 | PyTorch | 11,119,038 samples/s | 16,737 samples/s | 0.737 ms | **1.540x** |
| fp32 | TensorFlow | 11,328,122 samples/s | 78,237 samples/s | 0.723 ms | **1.512x** |
| bf16 | OpenNN | **34,610,952 samples/s** | **1,718,221 samples/s** | **0.237 ms** | **1.000x** |
| bf16 | PyTorch | 31,904,566 samples/s | 1,485,610 samples/s | 0.257 ms | **1.085x** |
| bf16 | TensorFlow | 32,421,696 samples/s | 276,332 samples/s | 0.253 ms | **1.068x** |

## Discussion

In fp32, OpenNN is 41.7% faster than PyTorch and 35.0% faster than TensorFlow. In bf16, all three frameworks are closer: OpenNN remains first, with a 9.8% lead over PyTorch and a 9.9% lead over TensorFlow.

OpenNN's own bf16 path is 2.27x faster than its fp32 path for this exact model and batch. PyTorch and TensorFlow also gain substantially, which explains why the competitive bf16 margin is smaller than the fp32 margin.

These are steady-state, device-resident figures. The five-run sample also shows that the lead is not an isolated pass: all 30 framework-and-precision executions completed successfully, and the reported result for each cell is their median.

## Conclusions

- OpenNN leads both fp32 and bf16 HIGGS dense inference cells.
- OpenNN reaches 17.13 million samples/s in fp32 and 34.61 million samples/s in bf16.
- The fp32 lead is substantial; the bf16 lead is real but comparatively narrow.
- OpenNN is 1.540x PyTorch and 1.512x TensorFlow in fp32.
- OpenNN is approximately 1.07-1.09x both frameworks in bf16.
- CUDA Graphs and symmetric fixed-buffer staging are part of the optimized OpenNN and PyTorch benchmark contract.

## Reproducing

The canonical runner is `docs/benchmarks/throughput/higgs-gpu/run_higgs_infer.py`:

```bash
python run_higgs_infer.py \
  --test "$OPENNN_BENCH_DATA/higgs/higgs_test.csv" \
  --batch 8192 --hidden 1024 --hidden-layers 2 \
  --activation relu --precision both --runs 5
```

The result artifact is `docs/benchmarks/results/gpu-higgs-dense-inference-speed-20260714T125416Z.json`.

## References

- [HIGGS dataset, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/280/higgs).
- [Searching for exotic particles in high-energy physics with deep learning](https://www.nature.com/articles/ncomms5308).
- [OpenNN source repository](https://github.com/Artelnics/opennn).
- [PyTorch](https://pytorch.org/).
- [TensorFlow](https://www.tensorflow.org/).
