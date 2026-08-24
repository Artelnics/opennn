# ResNet-50 max training batch: OpenNN vs PyTorch vs TensorFlow

*Last updated 2026-08-11. OpenNN cells: [`results/gpu-resnet50-max-batch-cifar10-20260811T100851Z.json`](../../results/); PyTorch/TensorFlow cells: [`results/gpu-resnet50-max-batch-cifar10-20260810T120959Z.json`](../../results/) (their binaries are unchanged).*

This benchmark asks a capacity question, not a speed question: what is the
largest training batch that completes one real ResNet-50/CIFAR-10 training
step on the same GPU?

Each engine is run in a fresh process. A candidate batch counts only if it
completes a warmup/capture step and one training step, reports a finite
cross-entropy loss, exits cleanly, and stays within the configured physical VRAM
limit. The runner finds the boundary by exponential growth followed by binary
search, then records both the largest passing batch and the next failing batch.

## The result

**OpenNN fits the largest training batch in every cell** — 1.96× PyTorch's
best path and 1.64× TensorFlow XLA in fp32:

| Engine | fp32 max batch | bf16 max batch | Peak VRAM at fp32 max |
|---|---:|---:|---:|
| **OpenNN, batch pool 1** | **18,085** | **27,306** | 15,873 MiB |
| PyTorch torch.compile | 9,216 | 18,112 | 15,840 MiB |
| PyTorch eager | 8,704 | 16,896 | 15,856 MiB |
| TensorFlow XLA | 11,036 | 23,296 | 14,426 MiB |

| Comparison | fp32 | bf16 |
|---|---:|---:|
| OpenNN vs PyTorch best | **1.96×** | **1.51×** |
| OpenNN vs TensorFlow XLA | **1.64×** | **1.17×** |

The bf16 ceiling rose from 24,455 (2026-08-10) to 27,306 (+12%): the hybrid
batch-norm forward now runs native bf16 tensor IO in the cuDNN graph, which
frees the fp32 staging workspace that used to shadow every BN call.

Every boundary is a genuine out-of-memory limit (next batch fails, peak at the
budget). This same workload measured **4,752 for OpenNN in 2026-06** — a 2.47×
deficit against TensorFlow at the time. The June result is preserved below as
the regression baseline; the turnaround came from the shared cuDNN workspace
buffer, the bounded workspace-policy search, and the memory-planning work of
2026-07/08 (training activation recomputation trades step time for batch
state, which is exactly the right trade for a capacity benchmark).

## What is measured

| Item | Configuration |
|---|---|
| Dataset | CIFAR-10 geometry, 32x32x3 images, 10 classes |
| Network | ResNet-50 v1.5 bottleneck, stages 3-4-6-3 |
| Loss / optimizer | Cross-entropy, Adam, learning rate 0.001 |
| Precision | fp32 and bf16 |
| Capacity rule | Largest batch that completes forward, backward, and Adam update |
| Search rule | Fresh process per candidate, exponential search plus binary search |
| VRAM rule | Physical GPU memory cap with 256 MiB reserved |
| OpenNN mode | ImageDataset BinaryFile cache, CUDA graph enabled, batch pool set to 1 |

The CIFAR-10 data is used for shape and labels. If a candidate batch is larger
than the source dataset size, the Python frameworks repeat CIFAR-10 samples
deterministically by modulo indexing. The OpenNN trial builds a temporary image
tree of exactly the requested batch size from the same CIFAR-10 training images,
lets `ImageDataset` build/read its `BinaryFile` image cache, and then runs a
single full-batch training step. The trial does not enable GPU-resident
data, so the whole dataset is not staged as a GPU-resident matrix.

## Machine and software

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 |
| Driver | 595.84 |
| GPU memory | 16,376 MiB |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| TensorFlow | 2.21.0 |
| CUDA nvcc | 13.3 |
| OpenNN commit | c63275648 |
| Result JSON (OpenNN cells) | results/gpu-resnet50-max-batch-cifar10-20260811T100851Z.json |
| Result JSON (PyTorch/TensorFlow cells) | results/gpu-resnet50-max-batch-cifar10-20260810T120959Z.json |
| June 2026 baseline (OpenNN 4,752) | results/gpu-resnet50-max-batch-cifar10-20260622T133809Z.json |

## Why the result matters

The ResNet-50 training-speed benchmark measures throughput at a fixed batch.
This benchmark measures a different limit: how much batch state each framework
can hold while doing the real training work. That includes activations,
gradients, optimizer state, framework workspaces, graph/capture overhead, and
any allocator reserve that is needed for the step.

The June 2026 run failed just above 4,752 samples because cuDNN workspace
allocation ran out of usable device memory in the convolution path while both
Python engines kept going. The 2026-08 result flips that: bounding the cuDNN
workspace policy per candidate, sharing one scratch buffer across graphs, and
planning the forward/backward arena jointly (with training activation
recomputation) almost quadrupled OpenNN's ceiling on the same card. The
benchmark remains the regression target for that memory work.

## Caveats

* This is a one-step capacity benchmark, not a time-to-quality training result.
  It proves that a batch completes a real optimizer update with finite loss; it
  does not say that the largest batch is the best batch for model quality.
* TensorFlow reserves a large amount of GPU memory up front in this environment.
  The runner records the observed peak and still gates every candidate by the
  configured physical VRAM cap.
* OpenNN is reported with prefetch-pool depth 1 (`set_batch_pool_size(1)`,
  the pool1 engine) because this benchmark is about
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
cd benchmarks/capacity/resnet50-max-batch
python run_resnet50_maxbatch.py \
  --dataset cifar10 \
  --precision fp32 \
  --engines opennn,pytorch,tensorflow \
  --gpu-index 0 \
  --require-gpu-idle \
  --start-batch 128
```

The runner writes immutable JSON artifacts under `benchmarks/results/`.
The run used for this note is
`gpu-resnet50-max-batch-cifar10-20260622T133809Z.json`.
