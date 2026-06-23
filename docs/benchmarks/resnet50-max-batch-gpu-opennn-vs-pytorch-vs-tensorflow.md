# ResNet-50 max training batch: OpenNN vs PyTorch vs TensorFlow

This benchmark asks a capacity question, not a speed question: what is the
largest fp32 training batch that completes one real ResNet-50/CIFAR-10 training
step on the same GPU?

Each engine is run in a fresh process. A candidate batch counts only if it
completes a warmup/capture step and one training step, reports a finite
cross-entropy loss, exits cleanly, and stays within the configured physical VRAM
limit. The runner finds the boundary by exponential growth followed by binary
search, then records both the largest passing batch and the next failing batch.

## The result

On this RTX 4080 run, TensorFlow XLA fits the largest batch, PyTorch
`torch.compile` is second, and OpenNN with `OPENNN_BATCH_POOL=1` is third. The
OpenNN number below is the corrected BinaryFile-cache run; PyTorch and
TensorFlow are from the same-machine full 3-way run because their protocol did
not change.

| Engine | Max batch | Peak VRAM at max | Next batch | Result |
|---|---:|---:|---:|---|
| OpenNN, BinaryFile cache, batch pool 1 | 4,752 | 15,865 MiB | 4,753 | failed |
| PyTorch torch.compile | 9,216 | 15,820 MiB | 9,217 | failed |
| PyTorch eager | 8,704 | 15,856 MiB | 8,705 | failed |
| TensorFlow XLA | 11,760 | 15,239 MiB | 11,761 | failed |

The headline is therefore:

| Comparison | Ratio |
|---|---:|
| TensorFlow XLA vs OpenNN | 2.47x larger batch |
| PyTorch best vs OpenNN | 1.94x larger batch |
| TensorFlow XLA vs PyTorch best | 1.28x larger batch |

This is useful evidence, but it is not an OpenNN marketing win. It shows that
for this exact ResNet-50/CIFAR-10 fp32 one-step capacity workload, TensorFlow's
XLA path and PyTorch's compiled path fit larger batches on the same GPU than
the current OpenNN implementation.

## What is measured

| Item | Configuration |
|---|---|
| Dataset | CIFAR-10 geometry, 32x32x3 images, 10 classes |
| Network | ResNet-50 v1.5 bottleneck, stages 3-4-6-3 |
| Loss / optimizer | Cross-entropy, Adam, learning rate 0.001 |
| Precision | fp32 |
| Capacity rule | Largest batch that completes forward, backward, and Adam update |
| Search rule | Fresh process per candidate, exponential search plus binary search |
| VRAM rule | Physical GPU memory cap with 256 MiB reserved |
| OpenNN mode | ImageDataset BinaryFile cache, CUDA graph enabled, batch pool set to 1 |

The CIFAR-10 data is used for shape and labels. If a candidate batch is larger
than the source dataset size, the Python frameworks repeat CIFAR-10 samples
deterministically by modulo indexing. The OpenNN trial builds a temporary image
tree of exactly the requested batch size from the same CIFAR-10 training images,
lets `ImageDataset` build/read its `BinaryFile` image cache, and then runs a
single full-batch training step. The trial does not enable
`OPENNN_GPU_RESIDENT_DATA`, so the whole dataset is not staged as a
GPU-resident matrix.

## Machine and software

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 |
| Driver | 595.71.05 |
| GPU memory | 16,376 MiB |
| Python | 3.12.3 |
| PyTorch | 2.12.0+cu130 |
| PyTorch CUDA / cuDNN | 13.0 / 92000 |
| TensorFlow | 2.21.0 |
| TensorFlow build CUDA / cuDNN | 12.5.1 / 9 |
| CUDA nvcc | 13.3 |
| OpenNN commit | b7d4255ca23d |
| Full 3-way result JSON | results/gpu-resnet50-max-batch-cifar10-20260622T133809Z.json |
| OpenNN BinaryFile rerun JSON | results/gpu-resnet50-max-batch-cifar10-20260622T135753Z.json |

## Why the result matters

The ResNet-50 training-speed benchmark measures throughput at a fixed batch.
This benchmark measures a different limit: how much batch state each framework
can hold while doing the real training work. That includes activations,
gradients, optimizer state, framework workspaces, graph/capture overhead, and
any allocator reserve that is needed for the step.

OpenNN's current result fails just above 4,752 samples because cuDNN workspace
allocation or graph capture runs out of usable device memory in the convolution
path. PyTorch and TensorFlow both keep going to larger batches on this card.
That makes this benchmark a good regression target for future OpenNN memory
work: the runner can be rerun after changing convolution workspaces, graph
capture strategy, batch buffering, or allocator behavior.

## Caveats

* This is a one-step capacity benchmark, not a time-to-quality training result.
  It proves that a batch completes a real optimizer update with finite loss; it
  does not say that the largest batch is the best batch for model quality.
* TensorFlow reserves a large amount of GPU memory up front in this environment.
  The runner records the observed peak and still gates every candidate by the
  configured physical VRAM cap.
* OpenNN is reported with `OPENNN_BATCH_POOL=1` because this benchmark is about
  capacity. The default prefetch pool is useful for throughput-oriented
  streaming workloads, but it holds extra batch buffers and lowers the maximum
  batch that fits. The dataset itself is not GPU-resident; it is read through
  the `ImageDataset` binary cache path.
* The test uses CIFAR-10 image geometry with a ResNet-50 model. At ImageNet
  resolution the memory balance changes substantially.
* The benchmark was measured once on the current machine. Before using it as a
  public headline, repeat the run and store another JSON artifact.

## Reproducing

Build the OpenNN trial target:

```bash
cmake --build build-gpu --target opennn_resnet50_maxbatch_trial
```

If your local GPU build directory is named `build`, use that directory name
instead.

Run the full benchmark from the repository root:

```bash
cd docs/benchmarks/resnet50-max-batch
python run_resnet50_maxbatch.py \
  --dataset cifar10 \
  --precision fp32 \
  --engines opennn,pytorch,tensorflow \
  --gpu-index 0 \
  --require-gpu-idle \
  --start-batch 128
```

The runner writes immutable JSON artifacts under `docs/benchmarks/results/`.
The run used for this note is
`gpu-resnet50-max-batch-cifar10-20260622T133809Z.json`.
