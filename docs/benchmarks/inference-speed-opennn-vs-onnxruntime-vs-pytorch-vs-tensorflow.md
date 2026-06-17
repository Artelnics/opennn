# Inference speed: OpenNN vs ONNX Runtime vs PyTorch vs TensorFlow (CPU)

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-11. Numbers below measured on Windows x86_64; re-measure on the reference Linux x86_64 box before publishing.*

**Status:** valid Windows tuned result, but not yet a final public headline. The
benchmark index treats this as a platform-specific result until the same workload is
re-measured on the reference Linux x86_64 machine with raw logs and version metadata.

The [startup-latency benchmark](startup-latency-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md)
measures the cost of starting - time-to-first-prediction. This note measures the cost of running:
once the model is loaded and warm, how fast does each framework push samples through a forward
pass?

Inference is the workload all four can be compared on directly. ONNX Runtime belongs here as a
first-class entry: it is a dedicated inference engine. It cannot train, but running a trained
model is exactly what it is built for, so it is the natural CPU-deployment competitor to OpenNN.

## The model

The same network throughout: a 2-layer MLP, `F -> F -> 1` (tanh on the hidden layer, linear
output), over a Rosenbrock-shaped dataset with `F` input features. Every framework runs the
forward pass only:

* **OpenNN** - a bare `NeuralNetwork` with two `Dense` layers, using reusable
  `ForwardPropagation` storage and caller-owned batch views.
* **ONNX Runtime** - `InferenceSession.run()` on a model exported once from the same MLP.
* **PyTorch** - `model.eval()` under `torch.no_grad()`.
* **TensorFlow** - `model(x, training=False)` traced into a single `tf.function` with XLA.

All four run single-machine CPU, FP32, batching the dataset and reporting the median over many
passes after warm-up. The metric is samples/second (throughput); we also report ms/batch
(latency).

## The numbers

Config: `F = 1000` features, batch `1000`, `8000` samples, median of 30 passes after warm-up.
Measured on Windows x86_64, Intel Core i7-14700F (20 cores, 28 logical processors).

OpenNN was built with MSVC 19.44 and oneMKL 2025.3.1 (`OpenNN_ENABLE_MKL=ON`). For this tuned
inference run, OpenNN used MKL packed GEMM for fixed dense weights and MKL VML fast tanh:

```powershell
$env:MKL_DYNAMIC = "FALSE"
$env:OMP_DYNAMIC = "FALSE"
$env:MKL_NUM_THREADS = "28"
$env:OMP_NUM_THREADS = "28"
$env:KMP_BLOCKTIME = "0"
$env:OPENNN_MKL_FAST_VML = "1"
$env:OPENNN_MKL_PACKED_GEMM = "1"
```

The Python engines were measured in a temporary Python 3.13 venv with `onnxruntime 1.26.0`,
`torch 2.10.0+cpu`, and `tensorflow 2.21.0`. Competitors were also checked with their best
observed thread settings on this machine: ONNX Runtime `ORT_INTRA_OP_THREADS=24`, PyTorch
`MKL_NUM_THREADS=28`, TensorFlow default.

| | OpenNN + MKL | ONNX Runtime | PyTorch | TensorFlow |
|---|---:|---:|---:|---:|
| **Throughput (samples/sec)** | **466,837** | 465,138 | 372,780 | 375,409 |
| **Latency (ms/batch of 1000)** | **2.14** | 2.15 | 2.68 | 2.66 |
| vs OpenNN | 1x | 1.00x | 0.80x | 0.80x |

The win is intentionally narrow and should be treated as a tuned CPU result, not a universal
property of every dense inference shape. The important change is that OpenNN is no longer asking
Eigen to handle this path alone: the dense forward path can dispatch to MKL GEMM, add bias with
an MKL BLAS update, reuse packed weights for inference, and use MKL VML for tanh. That puts OpenNN
in the same class of CPU math backend as the other frameworks while keeping the non-MKL path
available.

## Why CPU, and why this model

* **CPU** is the deployment surface where footprint, startup, dependencies, and throughput meet.
  Many inference services still run on CPUs, and that is where shipping a small native binary
  instead of a multi-hundred-MB framework matters.
* The MLP is deliberately dense and simple: the comparison is about each framework's forward-pass
  machinery and math backend, not about a special operator for an exotic layer.

## Caveats

* This is a steady-state throughput benchmark. It does not measure startup latency or training.
* The result is dominated by the `1000 x 1000` GEMM and the hidden tanh. Thread counts, CPU
  topology, MKL version, and framework version matter.
* `OPENNN_MKL_PACKED_GEMM=1` is an inference optimization for fixed weights. Leave it off while
  weights are changing.
* `OPENNN_MKL_FAST_VML=1` uses MKL's enhanced-performance tanh mode. Accuracy parity is covered
  separately in the [accuracy benchmark](accuracy-opennn-vs-pytorch-vs-tensorflow.md).
* ONNX Runtime remains extremely competitive: in this run the margin between OpenNN and ORT is
  about 0.4%.

## Reproducing

The equivalent programs are in [`docs/benchmarks/inference-speed/`](inference-speed/):
`opennn_inference.cpp`, `onnxruntime_inference.py`, `pytorch_inference.py`, and
`tensorflow_inference.py`. ONNX Runtime needs a model file, generated once by `export_onnx.py`.

```bash
# OpenNN: configure with MKL and build the benchmark target.
cmake -S . -B build-mkl -DOpenNN_DISABLE_CUDA=ON -DOpenNN_ENABLE_MKL=ON \
  -DOpenNN_MKL_ROOT="/path/to/oneapi/mkl/latest"
cmake --build build-mkl --config Release --target opennn_inference

# Run all four at the article's config (csv samples features batch reps).
docs/benchmarks/inference-speed/run_inference.sh rosenbrock.csv 8000 1000 1000 30
```

Each program prints `samples_per_sec=`, `ms_per_batch=`, and a `RESULT=OK` sentinel the driver
greps into the summary table.
