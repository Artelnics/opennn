# Deployment size on GPU (CNN): OpenNN vs PyTorch vs TensorFlow

Plenty of models do run on a GPU — but not always a rack of them in a data center. Increasingly the
GPU is at the edge too: a Jetson module in a robot or smart camera, a single workstation card in a
clinic or a factory cell, one GPU instance in a tightly-budgeted container. On those targets you
still want CUDA acceleration, but the **disk and image budget is finite** — a 16 GB eMMC, a base
container you don't want to balloon, an update you have to push over a network.

## Contents

- [The numbers](#the-numbers)
- [What each number is](#what-each-number-is)
- [Why the gap is far smaller than on CPU](#why-the-gap-is-far-smaller-than-on-cpu)
- [Caveats](#caveats)
- [Reproducing the OpenNN number](#reproducing-the-opennn-number)
- [References](#references)

## The numbers

|  | OpenNN | ONNX Runtime | PyTorch | TensorFlow |
| --- | --- | --- | --- | --- |
| **GPU (CNN) deployment size** | **~1.3 GB** | **~2.0 GB** | **~5.0 GB** | **~6.2 GB** |
| vs OpenNN | 1× | ≈1.5× | ≈4× | ≈5× |

All figures are dominated by NVIDIA's GPU libraries, which they *all* depend on — so the gaps are
far smaller than the CPU one (≈138×). The difference is how much each ships: OpenNN ships only the
NVIDIA libraries its CNN actually loads, plus a ~5 MB binary; ONNX Runtime adds its ~438 MB GPU
package on top of the CUDA stack; PyTorch and TensorFlow bundle the full NVIDIA set plus their own
multi-gigabyte runtime.

## What each number is

**OpenNN — ~1.3 GB (measured).** OpenNN's CUDA build is a small statically-linked executable plus
the NVIDIA shared libraries it links. We built the convolutional image-classification example
(`examples/mnist`) with CUDA on Linux x86_64 (CUDA 12.0, cuDNN 9.10.2) and listed the binary's real
dependencies with `ldd`, then sized exactly those `.so` files (resolving symlinks):

| Component | Size |
| --- | --- |
| `mnist` executable (CNN, OpenNN statically linked, stripped) | 5.2 MB |
| `libcudart.so.12` (CUDA runtime) | 0.6 MB |
| cuBLAS — `libcublas.so.12` + `libcublasLt.so.12` | 587 MB |
| cuDNN — loader + the libraries a CNN loads (`graph`, `ops`, `cnn`, `heuristic`, `engines_precompiled`, `engines_runtime_compiled`) | 728 MB |
| **Total** | **~1.3 GB** |

The key point is what is **not** there: OpenNN links only the cuDNN components a convolutional
network calls, so it omits `libcudnn_adv.so.9` — **271 MB** of RNN/advanced kernels — entirely,
because a CNN never touches them. The NVIDIA libraries, not OpenNN, are essentially the whole
footprint; OpenNN's own contribution is the 5 MB binary.

**PyTorch — ~5.0 GB (primary source).** The official LibTorch CUDA 12.6 C++ distribution
(`libtorch-shared-with-deps-2.8.0+cu126.zip`) unpacks to **5.01 GB on disk** (measured exactly from
the archive's directory). Its largest parts:

| Component | Size |
| --- | --- |
| `libtorch_cuda.so` (PyTorch's own CUDA engine) | 1.53 GB |
| bundled cuDNN (`libcudnn*.so.9`, all components) | 0.98 GB |
| bundled cuBLAS (`libcublas*.so.12`) | 0.56 GB |
| `libtorch_cpu.so`, `libcusparseLt.so`, and the rest | ~1.9 GB |

PyTorch's cuDNN bundle (~0.98 GB) is larger than OpenNN's (~0.73 GB) because it ships **every**
cuDNN component — including the RNN/advanced library a CNN never uses — for any model. On top of
those shared NVIDIA libraries, PyTorch adds its own ~1.53 GB `libtorch_cuda.so`, which has no OpenNN
counterpart: OpenNN's GPU kernels live in its 5 MB binary.

**TensorFlow — ~6.2 GB (measured).** A `pip install tensorflow[and-cuda]` (TF 2.21.0, Linux x86_64)
lands **6.2 GB** in `site-packages`. Its largest parts:

| Component | Size |
| --- | --- |
| bundled NVIDIA CUDA packages (`nvidia-*`: cuDNN, cuBLAS, cuSPARSE, cuSOLVER, cuFFT, NCCL, …) | ~4.1 GB |
| `libtensorflow_cc.so.2` (TF's own CUDA-enabled C++ engine) | ~1.05 GB |
| `libtensorflow_framework.so.2` and the rest | ~1.0 GB |

Like PyTorch, TensorFlow bundles the full NVIDIA CUDA stack regardless of the model, plus its own
multi-gigabyte runtime — the largest GPU footprint of the three.

**ONNX Runtime — ~2.0 GB (measured).** The `onnxruntime-gpu` 1.26.0 package is **438 MB** (its
`libonnxruntime_providers_cuda.so` alone is 382 MB), and it bundles *no* CUDA libraries — `ldd` of
the CUDA provider shows it loads the system `libcublas`, `libcublasLt`, `libcudart`, `libcudnn`,
`libcufft`, and `libcurand`. Sizing those (cuDNN's loader pulls in its engine sub-libs) gives ~1.55
GB, for a real deployment of **~2.0 GB**:

| Component | Size |
| --- | --- |
| `onnxruntime-gpu` package (incl. 382 MB CUDA provider) | ~438 MB |
| cuBLAS (`libcublas` + `libcublasLt`) | ~586 MB |
| cuDNN (loader + engine libraries) | ~725 MB |
| cuFFT + cuRand | ~238 MB |

ONNX Runtime is the lightest GPU deployment after OpenNN — it has no multi-gigabyte framework engine
— but at ~1.5× OpenNN it links cuFFT and cuRand that OpenNN's CNN never loads, plus its own 438 MB
package.

## Why the gap is far smaller than on CPU

On CPU the comparison was ~138×, because PyTorch ships a large math/tensor runtime and OpenNN ships
almost nothing. On GPU, **both** must ship NVIDIA's cuBLAS and cuDNN, which are inherently large
(precompiled kernels for many GPU architectures), so a big, irreducible chunk of the footprint is
common to both. OpenNN comes out ~4× smaller for two reasons: it ships only the NVIDIA component
libraries the model actually loads (skipping, e.g., the 271 MB cuDNN RNN library for a CNN), and it
has no separate multi-gigabyte framework runtime — its own CUDA code is in the 5 MB executable,
versus PyTorch's 1.53 GB `libtorch_cuda.so`.

## Caveats

- This is a **size** benchmark, not a capability or speed one. PyTorch is a general-purpose GPU
  framework; OpenNN is a focused native library. The numbers reflect that difference in scope, not
  GPU performance.
- Both numbers are measured on Linux x86_64. OpenNN: a CUDA build of `examples/mnist`, sized from
  `ldd` plus the on-disk `.so` files it loads (cuDNN 9.10.2). PyTorch: the on-disk size of the
  official libtorch 2.8.0+cu126 archive, read from its directory.
- CUDA packaging is in flux. We use the established, self-contained CUDA-12 libtorch (2.8.0+cu126,
  ~5.0 GB on disk) as the representative deploy-ready figure. PyTorch's newest CUDA-13 packaging
  externalizes some CUDA libraries out of the distribution, lowering the *download* size, but
  those libraries must still be present to run.
- Both figures are the minimum "what you must ship to run the model." OpenNN's already includes
  the application binary; PyTorch's is the runtime before your own code.

## Reproducing the OpenNN number

```
# Build a CNN example with CUDA (the default when a CUDA toolkit + cuDNN >= 9 are found)
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DOpenNN_BUILD_EXAMPLES=ON
cmake --build build-gpu --config Release --target mnist -j

# List the CUDA libraries the binary actually loads, then size exactly those:
ldd build-gpu/examples/mnist/mnist | grep -E 'cudnn|cublas|cudart'
# cuDNN's loader (libcudnn.so.9) pulls in its component libs at runtime; for a CNN that is
# graph + ops + cnn + heuristic + engines_precompiled + engines_runtime_compiled
# (NOT libcudnn_adv.so.9, the RNN/advanced library). Sum the executable + those .so files.
```

## References

- [OpenNN](https://www.opennn.net/).
- [PyTorch / LibTorch installation resources](https://pytorch.org/get-started/locally/).
- [TensorFlow installation guide](https://www.tensorflow.org/install).
- [ONNX Runtime installation documentation](https://onnxruntime.ai/docs/install/).
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn).
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
