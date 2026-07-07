# Baseline Memory Footprint Benchmark

Purpose: compare baseline RAM after constructing empty training objects and
GPU-ready VRAM after one tiny `32x32` matrix multiply, before loading data or
running a real model.

Top-level note:
[`../peak-memory-opennn-vs-pytorch-vs-tensorflow.md`](../peak-memory-opennn-vs-pytorch-vs-tensorflow.md)

Runners:

- `opennn_memory.cpp` emits `baseline_ram_mb` and `gpu_ready_vram_mb`
- `pytorch_memory.py` emits `baseline_ram_mb` and `gpu_ready_vram_mb`
- `tensorflow_memory.py` emits `baseline_ram_mb` and `gpu_ready_vram_mb`

Current run: 2026-07-05, WSL2 Ubuntu, RTX 3060 Laptop GPU.

| Framework | Baseline RAM | GPU-ready VRAM |
|---|---:|---:|
| OpenNN | 195.2 MB | 119.0 MB |
| PyTorch | 516.2 MB | 155.0 MB |
| TensorFlow | 871.2 MB | 121.0 MB |

Lifecycle: archive raw commands, versions, `nvidia-smi`, and result JSON with
each published run.
