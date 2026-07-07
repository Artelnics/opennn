# Memory footprint: OpenNN vs PyTorch vs TensorFlow

Disk size is only one deployment cost. A process also has a minimum resident
footprint before it loads real data or runs a real model. On a memory-capped
edge device, service container, or multi-worker server, that baseline matters:
it is the fixed cost paid by every model process.

This benchmark now measures two footprint states:

- **Baseline RAM**: current resident set size after loading the library and
  constructing the minimum training object graph, before GPU work.
- **GPU-ready VRAM**: GPU memory reported for the same process after one tiny
  `32x32` GPU matrix multiply, before moving model or dataset buffers to the
  GPU.

## Contents

- [The numbers](#the-numbers)
- [What is constructed](#what-is-constructed)
- [How VRAM is measured](#how-vram-is-measured)
- [Why it matters](#why-it-matters)
- [Caveats](#caveats)
- [Reproducing](#reproducing)
- [References](#references)

## The numbers

Run date: 2026-07-05, WSL2 Ubuntu on an NVIDIA GeForce RTX 3060 Laptop GPU
(6 GB, driver 555.85). OpenNN was built with CUDA 12.0 and cuDNN 9.24; PyTorch
was CUDA-enabled (`2.5.1+cu121`) and TensorFlow was `2.21.0`.

|  | OpenNN | PyTorch | TensorFlow |
| --- | --- | --- | --- |
| **Baseline RAM** | **195.2 MB** | 516.2 MB | 871.2 MB |
| **GPU-ready VRAM** | **119.0 MB** | 155.0 MB | 121.0 MB |

Raw runner output:

```
OpenNN
baseline_ram_mb 195.2
gpu_ready_vram_mb 119.0

PyTorch
baseline_ram_mb 516.2
gpu_ready_vram_mb 155.0

TensorFlow
baseline_ram_mb 871.2
gpu_ready_vram_mb 121.0
```

The old training-peak RSS table has been retired because it mixed framework
baseline memory with data, model, optimizer, and training-step allocations.

## What is constructed

Each runner creates the minimum objects expected in a trainable application, but
does not load a dataset or run a training step.

**OpenNN**

- `TabularDataset(0, {1}, {1})`
- `ApproximationNetwork({1}, {}, {1})`
- `TrainingStrategy`, `MeanSquaredError`, `AdaptiveMomentEstimation`
- one `32x32` CUDA SGEMM through the OpenNN CUDA backend for GPU-ready VRAM.

**PyTorch**

- empty `TensorDataset`
- `nn.Sequential(nn.Linear(1, 1))`
- `MSELoss`
- `Adam`
- one `32x32` CUDA matrix multiply for GPU-ready VRAM.

**TensorFlow**

- empty `tf.data.Dataset`
- one-layer Keras `Sequential` model
- `MeanSquaredError`
- `Adam`
- memory growth enabled, then one `32x32` CUDA matrix multiply for GPU-ready
  VRAM.

## How VRAM is measured

Framework tensor-memory counters are not enough for this benchmark because they
usually exclude CUDA context, driver, cuBLAS/cuDNN handles, and other backend
state. The runners therefore query:

```
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
```

and report the memory assigned to the current process. If no CUDA device,
CUDA-enabled framework, or `nvidia-smi` process row is available, the runner
falls back to the before/after total GPU memory delta. If neither reading is
available, it prints `gpu_ready_vram_mb NA`.

## Why it matters

- **RAM baseline** is the minimum host memory paid by every deployed worker.
- **GPU-ready VRAM** is the fixed GPU memory cost once the math backend is ready,
  before model weights, activations, workspaces, or batches are allocated.
- Keeping these baselines separate prevents CUDA context overhead from being
  confused with model memory or training-step memory.

## Caveats

- GPU-ready VRAM is sensitive to driver, CUDA, cuDNN, framework version, and
  whether a framework preallocates GPU memory. TensorFlow is run with memory
  growth enabled to avoid full-device reservation.
- The benchmark intentionally avoids data and training. Larger models add
  parameter, activation, optimizer, and workspace memory on top of this fixed
  baseline.
- Run each framework in a fresh process. CUDA contexts and framework allocators
  are process-local and should not be compared from a shared process.
- The current output is line-oriented; archive the raw command output and then
  convert it to result JSON before publication.

## Reproducing

Build the OpenNN runner:

```
cmake -S . -B build-benchmarks -DOpenNN_BUILD_EXAMPLES=OFF -DOpenNN_BUILD_BENCHMARKS=ON -DOpenNN_DISABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-benchmarks --target opennn_memory -j
```

Run each framework in a fresh shell:

```
cd docs/benchmarks/footprint/memory
../../../build-benchmarks/examples/memory_benchmark/opennn_memory
python pytorch_memory.py
TF_FORCE_GPU_ALLOW_GROWTH=true python tensorflow_memory.py
```

Archive the runner output, `nvidia-smi`, framework versions, CUDA/cuDNN
versions, commit hash, and dirty status with the published result. Current
artifact: `results/baseline-memory-footprint-wsl2-20260705T110753Z.json`.

## References

- [OpenNN](https://www.opennn.net/).
- [PyTorch](https://pytorch.org/).
- [TensorFlow](https://www.tensorflow.org/).
- [Linux procfs process status](https://man7.org/linux/man-pages/man5/proc_pid_status.5.html).
- [NVIDIA System Management Interface](https://developer.nvidia.com/nvidia-system-management-interface).
