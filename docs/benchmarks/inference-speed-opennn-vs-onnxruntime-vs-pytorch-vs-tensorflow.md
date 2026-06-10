# Inference speed: OpenNN vs ONNX Runtime vs PyTorch vs TensorFlow (CPU)

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-10. Numbers below measured on Windows x86_64; re-measure on the reference Linux x86_64 box before publishing.*

The [startup-latency benchmark](startup-latency-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md)
measures the cost of *starting* — time-to-first-prediction. This note measures the cost of
*running*: once the model is loaded and warm, how fast does each framework push samples through
a forward pass? That is the number that matters for a service that stays up and serves many
requests, or a batch job that scores millions of rows.

Inference is the workload all four can be compared on directly. ONNX Runtime belongs here as a
first-class entry: it is a dedicated *inference engine* — it cannot train, but running a trained
model is exactly what it is built for, so it is the natural CPU-deployment competitor to OpenNN.

## The model

The same network throughout, identical to the [training-speed benchmark](training-speed/) so the
two read together: a 2-layer MLP, `F -> F -> 1` (tanh on the hidden layer, linear output), over a
Rosenbrock dataset with `F` input features. Every framework runs the **forward pass only**, in
its native inference mode:

* **OpenNN** — `calculate_outputs()`, which runs forward propagation with `is_training=false`
  (dropout skipped, no gradient buffers built).
* **ONNX Runtime** — `InferenceSession.run()` on a model exported once from the same MLP.
* **PyTorch** — `model.eval()` under `torch.no_grad()` (no autograd graph).
* **TensorFlow** — `model(x, training=False)` traced into a single `tf.function` (`jit_compile`).

All four run **single-machine CPU, FP32**, batching the dataset and reporting the median over many
passes after warm-up. The metric is **samples/second** (throughput); we also report **ms/batch**
(latency).

## The numbers

Config: `F = 1000` features, batch `1000`, `8000` samples, median of 30 passes after warm-up.
Measured on Windows x86_64 (Intel, 20 logical cores): OpenNN built with MSVC 19.5x / Ninja,
forced to `Device::CPU`; the Python engines in a venv with torch 2.6.0+cu124 (CPU path),
onnxruntime 1.26.0, tensorflow 2.21.0.

| | OpenNN | ONNX Runtime | PyTorch | TensorFlow |
|---|---:|---:|---:|---:|
| **Throughput (samples/sec)** | 62,700 | **218,028** | 192,421 | 206,962 |
| **Latency (ms/batch of 1000)** | 15.9 | **4.59** | 5.20 | 4.83 |
| vs OpenNN | 1× | 3.5× | 3.1× | 3.3× |

On this dense-MLP workload OpenNN is **~3× slower** than the three frameworks, which cluster
tightly at 190k–220k samples/sec. The three all dispatch the dominant `1000×1000` GEMM to a
heavily-tuned, multithreaded math backend (oneDNN / MKL) that OpenNN's Eigen-based dense path does
not match for a single large matmul. This is the honest result for *steady-state throughput* — a
different axis from the [startup](startup-latency-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md),
[size](size-cpu-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md), and
[memory](peak-memory-opennn-vs-pytorch-vs-tensorflow.md) benchmarks, where OpenNN's lean native
binary wins. Throughput on a big GEMM is exactly where a mature BLAS earns its footprint.

## Why CPU, and why this model

* **CPU** is the deployment surface where the footprint story (size, startup, dependencies) and
  the throughput story meet: most inference still runs on CPUs, and that is where shipping a
  3 MB native binary instead of a multi-hundred-MB framework matters most.
* The MLP is **deliberately the same model the training-speed benchmark uses** — a dense GEMM
  workload all four express identically, so the comparison is about each framework's forward-pass
  machinery, not about who has the cleverest operator for an exotic layer.

## Caveats

* This is a **steady-state throughput** benchmark: it times the warm forward pass, *not*
  time-to-first-prediction (that is the [startup benchmark](startup-latency-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md))
  and *not* training throughput (that is the [training-speed benchmark](training-speed/)).
* The result is dominated by the `1000×1000` GEMM. PyTorch, TensorFlow, and ONNX Runtime each
  link a vendor-tuned multithreaded math backend (oneDNN / MKL) for that matmul; OpenNN uses its
  Eigen-based dense path. The ~3× gap is that backend difference, not per-call dispatch overhead —
  it is the same trade-off that makes OpenNN's binary small: it does not ship a large external BLAS.
* CPU thread counts are left at each framework's default (all default to the machine's core count
  — 20 here; OpenNN sizes its Eigen `ThreadPoolDevice` from `hardware_concurrency()` too). Absolute
  numbers vary with machine, math backend, and thread settings; the note states the config so a run
  is reproducible.
* The gap does **not** close on smaller models — it widens. Repeating the run at `F = 64`
  (a `64×64` GEMM) gives OpenNN 1.42M samples/sec vs PyTorch 3.3M and ONNX Runtime 13.1M
  (≈9× OpenNN). For tiny matmuls the cost is per-call overhead, and OpenNN's `calculate_outputs`
  rebuilds its forward-propagation buffer on every call, whereas ORT's `session.run` is built to
  amortize that. So OpenNN trails on dense-MLP inference throughput at both ends of the size range;
  its wins are on the footprint and startup axes, not this one.
* The weights are randomly initialized — irrelevant for a speed benchmark, which times the work,
  not the predictions. (Numerical-accuracy parity is covered separately in the
  [accuracy benchmark](accuracy-opennn-vs-pytorch-vs-tensorflow.md).)

## Reproducing

The equivalent programs are in [`docs/benchmarks/inference-speed/`](inference-speed/):
`opennn_inference.cpp` (build the `opennn_inference` target against the OpenNN library),
`onnxruntime_inference.py`, `pytorch_inference.py`, and `tensorflow_inference.py`. ONNX Runtime
needs a model file, generated once by `export_onnx.py`. The driver runs all four and prints the
comparison:

```bash
# OpenNN: build the benchmark target (registered as an example).
cmake --build build --target opennn_inference

# Run all four at the article's config (csv samples features batch reps).
docs/benchmarks/inference-speed/run_inference.sh rosenbrock.csv 8000 1000 1000 30
```

Each program prints `samples_per_sec=`, `ms_per_batch=`, and a `RESULT=OK` sentinel the driver
greps into the summary table.
