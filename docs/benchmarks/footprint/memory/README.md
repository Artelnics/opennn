# Baseline Memory Footprint Benchmark

Purpose: compare baseline RAM after constructing empty training objects and
GPU-ready VRAM after one tiny `32x32` matrix multiply, before loading data or
running a real model — OpenNN vs PyTorch vs TensorFlow.

Runners (each emits `baseline_ram_mb` and `gpu_ready_vram_mb`):

- `opennn_memory.cpp` — build the `opennn_memory` target
- `pytorch_memory.py`
- `tensorflow_memory.py`

Run:

```bash
cmake --build build-benchmarks --target opennn_memory
cd docs/benchmarks/footprint/memory
CUDA_MODULE_LOADING=LAZY ./opennn_memory
CUDA_MODULE_LOADING=LAZY python pytorch_memory.py
CUDA_MODULE_LOADING=LAZY TF_FORCE_GPU_ALLOW_GROWTH=true python tensorflow_memory.py
```

GPU-ready VRAM mostly reflects the CUDA runtime/context and first math-backend
setup before tensors are resident. Archive the raw output, framework versions,
and `nvidia-smi` with each run.
