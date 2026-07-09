# ResNet-50 Max Batch Benchmark

This directory contains the ResNet-50/CIFAR-10 training-capacity benchmark in
both fp32 and bf16. It searches for the largest batch that completes one real
training step (forward, backward, and Adam update) on a GPU, per precision.

## Files

- `opennn_resnet50_maxbatch_trial.cpp`: OpenNN trial binary source.
- `pytorch_resnet50_maxbatch.py`: PyTorch eager and torch.compile trial.
- `tensorflow_resnet50_maxbatch.py`: TensorFlow XLA trial.
- `run_resnet50_maxbatch.py`: fresh-process search driver and JSON writer.

## Build

From the repository root:

```bash
cmake --build build-gpu --target opennn_resnet50_maxbatch_trial
```

If your local GPU build directory is named `build`, use that directory name
instead.

## Run

```bash
export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
cd docs/benchmarks/capacity/resnet50-max-batch
python run_resnet50_maxbatch.py \
  --dataset cifar10 \
  --precision both \
  --engines opennn,pytorch,tensorflow \
  --gpu-index 0 \
  --require-gpu-idle \
  --start-batch 128
```

`--precision` accepts `fp32`, `bf16`, or `both` (default `both`); the runner
measures every requested precision and threads it through all three engines
(OpenNN `Configuration::set(Device::CUDA, Type::BF16|FP32)`, PyTorch bf16
autocast, TensorFlow `mixed_bfloat16`).

Datasets live under `$OPENNN_BENCH_DATA` (see
[`../../DATA_POLICY.md`](../../DATA_POLICY.md)), never inside a benchmark
folder. The CIFAR-10 tree is prepared under `$OPENNN_BENCH_DATA/cifar10`
(falling back to `~/opennn-benchmark-data/cifar10` when the variable is unset)
by `docs/benchmarks/throughput/resnet50/prepare_cifar10.py`, which the runner
invokes automatically if the data is missing. The OpenNN trial uses
`ImageDataset` in its default `BinaryFile` storage mode, so pixels are read
from the image cache file for the batch. It does not enable GPU-resident data
(the `GPUPersistantData` storage mode).

## Protocol

A candidate batch is successful only if the child process exits with code 0,
prints `RESULT=OK`, completes a warmup/capture step and one training step,
reports a finite loss, and stays below the configured physical VRAM cap.

The search uses exponential growth followed by binary search. Every candidate
is executed in a fresh process so allocator state from one framework or failed
batch cannot influence the next candidate.
