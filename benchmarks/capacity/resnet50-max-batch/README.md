# ResNet-50 Max Batch Benchmark

This directory contains the ResNet-50/CIFAR-10 training-capacity benchmark in
both fp32 and bf16. It searches for the largest batch that completes one real
training step (forward, backward, and Adam update) on a GPU, per precision.

## Files

- `opennn_resnet50_maxbatch_trial.cpp`: OpenNN trial binary source.
- `pytorch_resnet50_maxbatch.py`: PyTorch eager and torch.compile trial.
- `tensorflow_resnet50_maxbatch.py`: TensorFlow XLA trial.
- `run_resnet50_maxbatch.py`: fresh-process search driver and JSON writer.
- `resnet50-max-batch-gpu-opennn-vs-pytorch-vs-tensorflow.md`: historical
  RTX 4080 report retained as the regression baseline.

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
cd benchmarks/capacity/resnet50-max-batch
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

OpenNN capacity trials default to bounded cuDNN convolution workspaces:

```bash
--opennn-workspace-modes 16,32,64,128,256,auto
```

For each batch candidate the runner tries these policies in separate fresh
processes and accepts the first one that completes. This prevents an
autotuned, throughput-oriented convolution plan from consuming the VRAM that
the benchmark is meant to make available to model state. Every attempted and
selected policy is recorded in the result JSON. Use `off` only for a separate
throughput/debug run; the capacity parser intentionally rejects it.

The OpenNN trial also enables training-activation recomputation. Raw
pre-batch-normalization convolution outputs share one transient buffer and are
recomputed immediately before their backward pass; all numerically sensitive
batch-normalization outputs and statistics remain persistent. This trades
extra convolution work for capacity and is disabled by default in the OpenNN
library. Pass a final `0` to the trial binary to obtain the non-recomputed
control.

Datasets live under `$OPENNN_BENCH_DATA` (see
[`../../DATA_POLICY.md`](../../DATA_POLICY.md)), never inside a benchmark
folder. The CIFAR-10 tree is prepared under `$OPENNN_BENCH_DATA/cifar10`
(falling back to `~/opennn-benchmark-data/cifar10` when the variable is unset)
by `benchmarks/throughput/resnet50/prepare_cifar10.py`, which the runner
invokes automatically if the data is missing. The OpenNN trial uses
`ImageDataset` in its default `BinaryFile` storage mode, so pixels are read
from the image cache file for the batch. It does not enable GPU-resident data
(the `GPUPersistantData` storage mode).

## Protocol

A candidate batch is successful only if the child process exits with code 0,
prints `RESULT=OK`, completes a warmup/capture step and one training step,
reports a finite loss, and stays below the configured physical VRAM cap.

The historical RTX 4080 OpenNN result stopped at batch 4,752 while a cuDNN
frontend plan failed and fell back to a legacy path. That result did not search
bounded workspace policies. It remains in the repository as evidence, but it
must not be presented as the capacity of the revised runner.

On an RTX 3060 Laptop GPU, the revised fp32 trial reduced the forward activation
allocation at batch 128 from 226.21 MiB to 123.46 MiB while preserving the
training error (`4.81168`). Reusing one consumer delta as the residual fan-out
accumulator then reduced the backward arena from 37 MiB to 29 MiB. A
compact offline placement for supported CNN graphs eliminates the remaining
fragmentation, reaching the 26 MiB live-memory lower bound. At batch 4,400
this lowered monitored peak VRAM from 5,775 MiB to 5,671 MiB. Batch 4,560
completed at 5,861 MiB under a 5,888 MiB cap.

The next phase places each convolution recomputation scratch slot in the
still-unused persistent suffix belonging to later layers. That suffix is also
dead when execution returns to the convolution during backward, so this changes
only static tensor offsets and adds no kernels, copies, or synchronization. It
eliminated the separate 8 MiB scratch block at batch 128, reducing the forward
allocation from 123.46 MiB to 115.46 MiB while preserving the training error
(`4.81168`). Batch 4,840 then completed twice at a monitored 5,887 MiB under
the same 5,888 MiB cap: a 280-sample or 6.14% increase over batch 4,560. The
official search stopped at that configured limit, so 4,840 is a demonstrated
lower bound rather than a failure-bounded exact maximum:
[`../../results/gpu-resnet50-max-batch-cifar10-20260802T103952Z.json`](../../results/gpu-resnet50-max-batch-cifar10-20260802T103952Z.json).

Training batches are already scaled by `ImageDataset`, so the next phase marks
the skipped leading `Scaling` layer as a layout passthrough. It also treats an
exact 1x1, stride-1, zero-padding pooling operation as a forward/backward
passthrough. The pooling output had previously hosted the final convolution
recomputation scratch, so removing it did not lower the arena peak; eliminating
the unused scaling output produced the net reduction from 115.46 MiB to
113.96 MiB at batch 128. The training error remained `4.81168`. Under the same
5,888 MiB cap, batch 4,893 completed at 5,887 MiB and batch 4,894 exceeded the
cap at 5,889 MiB. This is an exact, failure-bounded maximum and increases batch
capacity by 53 samples or 1.10%:
[`../../results/gpu-resnet50-max-batch-cifar10-20260802T114546Z.json`](../../results/gpu-resnet50-max-batch-cifar10-20260802T114546Z.json).

The search uses exponential growth followed by binary search. Every candidate
is executed in a fresh process so allocator state from one framework or failed
batch cannot influence the next candidate.

The residual-projection phase removes the backward dependency on projection
values. Fused batch-normalization/ReLU backward now derives the ReLU mask from
the final activated output, so each projection output can be released after
its residual consumer runs forward. The compact planner reuses all four such
outputs without joining the forward and backward arenas. At batch 128 this
reduced the forward allocation from 113.96 MiB to 98.96 MiB (15.00 MiB) while
preserving the training error (`4.81168`). Under the same 5,888 MiB cap, batch
5,471 completed at 5,887 MiB and batch 5,472 exceeded the cap at 5,889 MiB.
This is an exact, failure-bounded maximum and increases capacity by 578 samples
or 11.81% over batch 4,893:
[`../../results/gpu-resnet50-max-batch-cifar10-20260802T165941Z.json`](../../results/gpu-resnet50-max-batch-cifar10-20260802T165941Z.json).
