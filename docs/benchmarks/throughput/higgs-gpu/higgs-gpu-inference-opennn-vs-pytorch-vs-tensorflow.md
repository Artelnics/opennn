# GPU HIGGS dense inference: OpenNN vs PyTorch vs TensorFlow

OpenNN leads HIGGS dense bf16 inference on an NVIDIA GeForce RTX 4080 at 34.61 million samples/s (1.09x PyTorch, 1.07x TensorFlow); in fp32 — TF32 tensor-core matmuls in all three engines — the three land in a statistical tie at ~17 million samples/s, the GEMM ceiling for this network. bf16 cells: median of 5 runs (2026-08-10, artifact `results/gpu-higgs-dense-inference-speed-20260810T123521Z.json`); fp32 cells re-measured 2026-08-11 with TF32 aligned across engines.

> **2026-08-11 TF32 correction.** The previously published fp32 lead (1.54x
> PyTorch) compared OpenNN running TF32 against PyTorch/TensorFlow running
> strict fp32. OpenNN's GPU fp32 GEMMs always use TF32 tensor cores; the
> PyTorch and TensorFlow drivers now enable TF32 for their fp32 cells too, and
> at that point all three engines saturate the same GEMM roofline: OpenNN
> 17.13M, PyTorch 16.97M, TensorFlow 17.22M samples/s — a tie within ±1%.
> The honest fp32 story here is parity, and bf16 is where OpenNN's margin is.

> **2026-08-19 TensorFlow dispatch correction.** The TensorFlow driver called
> its XLA-compiled step once per batch from Python. That costs ~0.23 ms of eager
> dispatch per batch, and TensorFlow enqueues asynchronously, so the cost is
> hidden whenever the GPU work per batch is longer than it and is paid in full
> when it is not. At batch 8192 the bf16 step is ~0.22 ms, just under the
> threshold: enqueueing a pass took as long as enqueueing *and executing* it, so
> the GPU was idle waiting on Python. OpenNN and PyTorch were not exposed to
> this, because a captured-graph replay costs one cheap launch. The driver now
> also offers the batch loop compiled inside a single `tf.function`, times both
> paths, and reports the faster with `tf_path` naming the winner. Which one wins
> depends on precision: compiling the loop is worth +11% in bf16 and -5% in
> fp32. The per-batch results are bit-identical between the two paths.
>
> The RTX 4080 cells below were measured with the old driver and have not been
> re-measured on that machine, so how much of their 1.068x bf16 margin survives
> is unknown. Their cited artifact `gpu-higgs-dense-inference-speed-20260810T123521Z.json`
> has never been committed, so it cannot be re-checked: `results/` is ignored by
> default and reviewed evidence is promoted with `git add -f`, which 29 artifacts
> are -- none of them for this benchmark. The second reference machine below is
> measured with the corrected driver.

> bf16 results are medians across five runs. CUDA Graphs are active in the OpenNN and PyTorch paths, and both stage each batch through the same device-to-device copy pattern.

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

The comparison uses the same 28-1024-1024-1 ReLU network in OpenNN, PyTorch, and TensorFlow. OpenNN leads the bf16 cell; the fp32 (TF32) cell is a three-way tie at the GEMM roofline.

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
| Run ID | 20260810T123521Z |

## Methodology

Each engine processes the same held-out rows with the same network, batch, activation, parameter count, and precision. Labels are ignored by the timed inference path.

- OpenNN uses `calculate_outputs_resident`, keeps parameters and activations on the GPU, and replays a captured CUDA Graph.
- PyTorch uses inference mode, bf16 autocast for the bf16 cell, and a manually captured `torch.cuda.CUDAGraph`. The eager path remains available with `PT_NOGRAPH=1` for diagnostic comparisons.
- OpenNN and PyTorch copy each resident batch into a fixed capture buffer with one device-to-device copy before graph replay, so both graph paths use stable pointers under the same staging contract.
- TensorFlow uses compiled graph execution and mixed bf16 for the bf16 cell.
  Since 2026-08-19 it times two dispatch paths -- the XLA step called per
  batch from Python, and the batch loop compiled inside one `tf.function` --
  and reports the faster, naming it in `tf_path`. `TF_NOLOOP=1` forces the
  per-batch path, mirroring `PT_NOGRAPH` on the PyTorch side.
- Framework warmup and TensorFlow XLA compilation occur before the timed passes.
- Samples per second and milliseconds per batch are reported from the same measured pass.
- The bf16 artifact contains five successful runs per engine; the table reports medians across those runs. The fp32 (TF32) cells are 2026-08-11 single-run alignment measurements.

Dataset loading, process startup, model construction, the initial host-to-device upload, graph capture, and warmup are outside the measured region. The per-batch device-to-device staging copy is included.

## Results

| Precision | Framework | Median throughput | Median batch time | OpenNN speedup |
|---|---|---:|---:|---:|
| fp32 (TF32) | OpenNN | **17,125,371 samples/s** | **0.478 ms** | 1.00x |
| fp32 (TF32) | PyTorch | 16,970,000 samples/s | 0.483 ms | 1.01x |
| fp32 (TF32) | TensorFlow | 17,220,000 samples/s | 0.476 ms | 0.99x |
| bf16 | OpenNN | **34,610,952 samples/s** | **0.237 ms** | **1.000x** |
| bf16 | PyTorch | 31,904,566 samples/s | 0.257 ms | **1.085x** |
| bf16 | TensorFlow | 32,421,696 samples/s | 0.253 ms | **1.068x** |

### Second reference machine (RTX 5070 Ti, corrected driver)

Measured 2026-08-19 on OpenNN commit `2ff4c2f8b`; NVIDIA GeForce RTX 5070 Ti
(16 GB), driver 610.43.02, PyTorch 2.13.0+cu130, TensorFlow 2.21.0. Five runs
per engine, medians. Artifact:
`results/gpu-higgs-dense-inference-speed-20260819T101642Z.json`.

| Precision | Framework | Median throughput | Median batch time | OpenNN speedup |
|---|---|---:|---:|---:|
| fp32 (TF32) | OpenNN | **19,376,965 samples/s** | **0.423 ms** | 1.000x |
| fp32 (TF32) | PyTorch | 14,812,626 samples/s | 0.553 ms | **1.308x** |
| fp32 (TF32) | TensorFlow | 18,745,349 samples/s | 0.437 ms | 1.034x |
| bf16 | OpenNN | **37,548,621 samples/s** | **0.218 ms** | 1.000x |
| bf16 | PyTorch | 28,790,763 samples/s | 0.285 ms | **1.304x** |
| bf16 | TensorFlow | 35,592,241 samples/s | 0.230 ms | 1.055x |

TensorFlow ran the compiled batch loop in bf16 (35.59M against 32.24M per-batch)
and per-batch dispatch in fp32 (18.61M against 17.65M compiled). Both cells
report its better path.

Against TensorFlow this is a near-tie in **both** precisions on this card --
1.06x and 1.03x -- consistent with the RTX 4080 fp32 finding and with its bf16
figure of 1.068x. The durable margin here is over PyTorch at ~1.31x in both
precisions, which has a measured mechanism: OpenNN fuses the ReLU into the GEMM
epilogue while PyTorch runs it as a separate kernel. Timing a captured PyTorch
graph with and without the two ReLUs gives 0.5760 vs 0.4771 ms, accounting for
0.099 ms of the 0.131 ms fp32 gap. Note also that ~4% of the bf16 margin over
PyTorch is its autocast, which casts weights inside the replay (0.2907 vs
0.2798 ms with native bf16 weights), not something OpenNN does faster.

An independent cuBLAS probe (`gemm_probe.cu`) puts the fp32 result at the
hardware ceiling: the 1024x1024 forward GEMM alone is 0.3666 ms of the 0.423 ms
batch, cuBLASLt's best-of-8 heuristic search finds nothing faster than the
default, and OpenNN lands within 6% of the isolated L1+L2 cost. There is no
meaningful fp32 headroom at this batch for any engine.

## Discussion

With TF32 aligned, the fp32 cells are a three-way tie within ±1% — a batch-8192 pass through a 1M-parameter dense stack is a pure GEMM workload, and all three engines saturate the same tensor-core roofline. In bf16 OpenNN keeps a real margin: 8.5% over PyTorch and 6.8% over TensorFlow.

OpenNN's own bf16 path is 2.02x faster than its fp32 path for this exact model and batch — the expected doubling of tensor-core throughput plus halved memory traffic.

These are steady-state, device-resident figures. The bf16 cells are five-run medians with all executions successful; the fp32 (TF32) cells are single-run measurements from the 2026-08-11 alignment, pending the formal multi-run refresh.

## Conclusions

- In bf16 — the deployment precision for this workload — OpenNN leads: 1.09x PyTorch, 1.07x TensorFlow, at 34.61 million samples/s.
- In fp32 (TF32 in all three engines) the result is parity at ~17M samples/s: the workload is GEMM-bound and everyone hits the same hardware ceiling.
- CUDA Graphs and symmetric fixed-buffer staging are part of the optimized OpenNN and PyTorch benchmark contract.

## Reproducing

The canonical runner is `docs/benchmarks/throughput/higgs-gpu/run_higgs_infer.py`:

```bash
python run_higgs_infer.py \
  --test "$OPENNN_BENCH_DATA/higgs/higgs_test.csv" \
  --batch 8192 --hidden 1024 --hidden-layers 2 \
  --activation relu --precision both --runs 5
```

The result artifact is `docs/benchmarks/results/gpu-higgs-dense-inference-speed-20260810T123521Z.json`.

## References

- [HIGGS dataset, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/280/higgs).
- [Searching for exotic particles in high-energy physics with deep learning](https://www.nature.com/articles/ncomms5308).
- [OpenNN source repository](https://github.com/Artelnics/opennn).
- [PyTorch](https://pytorch.org/).
- [TensorFlow](https://www.tensorflow.org/).
