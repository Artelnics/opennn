# GPU HIGGS dense inference: OpenNN vs PyTorch vs TensorFlow

OpenNN leads HIGGS dense inference on an NVIDIA GeForce RTX 5070 Ti in both precisions against PyTorch -- 37.55 million samples/s bf16 and 19.38 million fp32, 1.30x and 1.31x -- and ties TensorFlow, 1.06x and 1.03x. Medians of five runs, 2026-08-19, artifact `results/gpu-higgs-dense-inference-speed-20260819T101642Z.json`. The PyTorch margin has a measured mechanism: OpenNN fuses the ReLU into the GEMM epilogue while PyTorch runs it as a separate kernel. The TensorFlow tie is the honest result once its driver stops paying per-batch dispatch -- see the correction below. fp32 is at the hardware ceiling for every engine: one GEMM is 87% of the batch and cuBLASLt's own autotuner finds nothing faster.

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
> The reference machine moved to the RTX 5070 Ti with this correction. The RTX
> 4080 numbers this note used to lead with were taken on the old driver, on a
> machine no longer available, and their artifact
> `gpu-higgs-dense-inference-speed-20260810T123521Z.json` was never committed --
> `results/` is ignored by default and reviewed evidence is promoted with
> `git add -f`, which 29 artifacts are, none of them for this benchmark. They
> are kept below for provenance and should not be cited.

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
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| Operating system | Linux 7.0 x86_64 |
| NVIDIA driver | 610.43.02 |
| CUDA | 13.3 |
| PyTorch | 2.13.0+cu130 (cuDNN 9.23) |
| TensorFlow | 2.21.0 |
| OpenNN commit | `2ff4c2f8b` |
| Run ID | 20260819T101642Z |

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

## 2026-08-19 protocol correction: PyTorch's best configuration

The PyTorch cell of this benchmark was never eager - the driver builds a CUDA
graph by hand and replays it per batch - but it was missing two things a
deployment would use: `torch.compile` over the model, and bf16 *weights*
instead of weights cast inside `autocast` on every replay. Both are now knobs
(`PT_COMPILE_MODE`, `PT_BF16_WEIGHTS`) and the runner sets PyTorch's measured
best by default: `reduce-overhead` plus bf16 weights in bf16, `max-autotune` in
fp32 (`PYTORCH_PLAIN=1` reverts). The eager fallback now stages through the
same fixed buffer the graph paths use, so the engines are compared on equal
terms.

What it is worth to PyTorch on an RTX 3060 Laptop at batch 8,192: bf16
6.59 -> 7.30 M samples/s (+11%), fp32 3.85 -> 4.19 M (+9%).

Three engines alternated, three rounds each, medians (RTX 3060 Laptop, WSL2,
batch 8,192, 28-1024-1024-1):

| Precision | OpenNN | PyTorch best | TensorFlow | OpenNN / PyTorch | OpenNN / TF |
|---|---:|---:|---:|---:|---:|
| bf16 | **9,442,101 samples/s** | 7,617,578 | 8,219,183 | **1.24x** | **1.15x** |
| fp32 (TF32) | **4,666,954 samples/s** | 4,116,292 | 4,017,197 | **1.13x** | **1.16x** |

The RTX 4080 figures below predate this correction, and so does the RTX 5070 Ti
artifact `results/gpu-higgs-dense-inference-speed-20260819T101642Z.json`
(OpenNN 1.28x PyTorch bf16, 1.30x fp32): both compared against the
uncompiled-autocast path, so their PyTorch cells are ~10% low. They need a
re-measurement with this protocol before they are quoted again.

## Results

Five runs per engine, medians. Artifact:
`results/gpu-higgs-dense-inference-speed-20260819T101642Z.json`.

| Precision | Framework | Median throughput | Median batch time | OpenNN speedup |
|---|---|---:|---:|---:|
| fp32 (TF32) | OpenNN | **19,376,965 samples/s** | **0.423 ms** | 1.000x |
| fp32 (TF32) | PyTorch | 14,812,626 samples/s | 0.553 ms | **1.308x** |
| fp32 (TF32) | TensorFlow | 18,745,349 samples/s | 0.437 ms | 1.034x |
| bf16 | OpenNN | **37,548,621 samples/s** | **0.218 ms** | 1.000x |
| bf16 | PyTorch | 28,790,763 samples/s | 0.285 ms | **1.304x** |
| bf16 | TensorFlow | 35,592,241 samples/s | 0.230 ms | 1.055x |

TensorFlow ran the compiled batch loop in bf16 (35.59M against 32.24M
per-batch) and per-batch dispatch in fp32 (18.61M against 17.65M compiled).
Both cells report its better path.

### Superseded: RTX 4080, pre-dispatch-fix

Kept for provenance only. These were measured with the TensorFlow driver that
paid per-batch eager dispatch, on a machine no longer available, and their
artifact was never committed, so they cannot be re-checked or re-run. Do not
cite them.

| Precision | Framework | Median throughput | Median batch time | OpenNN speedup |
|---|---|---:|---:|---:|
| fp32 (TF32) | OpenNN | 17,125,371 samples/s | 0.478 ms | 1.00x |
| fp32 (TF32) | PyTorch | 16,970,000 samples/s | 0.483 ms | 1.01x |
| fp32 (TF32) | TensorFlow | 17,220,000 samples/s | 0.476 ms | 0.99x |
| bf16 | OpenNN | 34,610,952 samples/s | 0.237 ms | 1.000x |
| bf16 | PyTorch | 31,904,566 samples/s | 0.257 ms | 1.085x |
| bf16 | TensorFlow | 32,421,696 samples/s | 0.253 ms | 1.068x |

## Discussion

Against TensorFlow this is a near-tie in both precisions -- 1.06x and 1.03x. That
is the result once TensorFlow gets the same dispatch amortization the two
graph-replaying engines already had; before the driver fix the same machine
reported 1.21x bf16, and the difference was Python, not TensorFlow.

Against PyTorch the ~1.30x margin holds in both precisions and has a mechanism
rather than a shrug. Timing a captured PyTorch graph with and without its two
ReLU kernels gives 0.5760 vs 0.4771 ms, so unfused activation accounts for
0.099 ms of the 0.131 ms fp32 gap. Note that ~4% of the bf16 margin is PyTorch's
autocast casting weights inside the replay (0.2907 vs 0.2798 ms with native bf16
weights), which is a methodology difference rather than OpenNN being faster.

fp32 has no headroom left for anyone. A standalone cuBLAS probe (`gemm_probe.cu`)
puts the 1024x1024 forward GEMM at 0.3666 ms of the 0.423 ms batch, cuBLASLt's
best-of-8 heuristic search finds nothing faster than its default, and OpenNN
lands within 6% of the isolated L1+L2 cost. TF32 measures exactly half of BF16
throughput on this silicon, which is why the bf16 margin exists and the fp32 one
cannot be manufactured.

OpenNN's own bf16 path is 1.94x its fp32 path for this model and batch -- the
tensor-core ratio plus halved activation traffic.

These are steady-state, device-resident figures: five-run medians, all
executions successful, every engine on its captured-graph or compiled path.

## Conclusions

- OpenNN leads PyTorch by ~1.30x in both precisions, at 37.55M samples/s bf16 and
  19.38M fp32, and the margin is attributable to fused-activation epilogues.
- Against TensorFlow the result is a tie in both precisions (1.06x, 1.03x) once
  every engine gets one dispatch per pass rather than one per batch.
- fp32 is GEMM-bound at the hardware ceiling; there is no engineering headroom
  there for any of the three.
- Captured graphs and symmetric fixed-buffer staging are part of the benchmark
  contract for every engine, TensorFlow now included.

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
