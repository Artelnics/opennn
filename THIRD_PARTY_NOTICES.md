# Third-party software

OpenNN is licensed under the GNU Lesser General Public License v3
(`LICENSE.txt`). The build uses or can use the following third-party
components, each under its own license. Components marked *fetched* are
downloaded by CMake at configure time from the pinned version given;
components marked *optional* are used only when the corresponding CMake
option or system library is present; nothing marked *not redistributed* is
included in OpenNN's sources or binaries.

`LICENSE-GPL-3.0.txt` supplies the GNU GPL v3 text incorporated by LGPL v3;
it is copied verbatim from the [GCC project's COPYING3](https://github.com/gcc-mirror/gcc/blob/master/COPYING3).
OpenNN's existing LGPL licence remains in `LICENSE.txt`.

| component | version | license | how it is used |
|---|---|---|---|
| [Eigen](https://eigen.tuxfamily.org) | 5.0.1 (fetched if not found) | MPL-2.0 | linear algebra on the CPU; a public dependency of the `opennn` target |
| [libjpeg-turbo](https://libjpeg-turbo.org) | 3.1.4 (fetched, SHA-256 pinned) | IJG, BSD-3-Clause and zlib licenses | JPEG decoding for image datasets; built as a static library and installed beside OpenNN |
| [zlib](https://zlib.net) | 1.3.2 (fetched, SHA-256 pinned) | zlib license | compression for datasets and model files |
| [cuDNN frontend](https://github.com/NVIDIA/cudnn-frontend/blob/v1.27.0/LICENSING.md) | v1.27.0 (fetched; CUDA builds) | Apache-2.0 and MIT, per-file SPDX tags | graph-API access to cuDNN engines (convolution, attention, matmul); upstream licensing and attribution files accompany the installation |
| [GoogleTest](https://github.com/google/googletest) | v1.18.0 (fetched; tests only) | BSD-3-Clause | unit tests; not installed |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | v2.8.3 (fetched; optional, `OpenNN_WITH_FLASH_ATTENTION`) | BSD-3-Clause | FlashAttention-2 kernels, compiled against OpenNN's own minimal `ATen`/`c10` shim (`opennn/core/cuda/flash_attention_shim`, which contains no PyTorch code) |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | user-provided (optional, `OpenNN_CUTLASS_INCLUDE_DIR`) | BSD-3-Clause | one narrow-contraction GEMM kernel; header-only, not redistributed |
| [oneDNN](https://github.com/uxlfoundation/oneDNN) | system (optional) | Apache-2.0 | LSTM primitives on the CPU |
| [Intel oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) | system (optional, `OpenNN_ENABLE_MKL`) | Intel Simplified Software License | BLAS on the CPU; linked, not redistributed |
| [oneTBB](https://github.com/uxlfoundation/oneTBB) | system (optional) | Apache-2.0 | parallel standard algorithms |
| NVIDIA CUDA Toolkit, cuBLAS, cuDNN, NVML | system (CUDA builds) | NVIDIA software license agreements | GPU execution; linked (NVML loaded at run time), not redistributed |
| OpenMP runtime (libgomp / libomp) | system | GPL-3.0 with runtime exception / Apache-2.0 with LLVM exceptions | CPU threading |

The JSON support in `opennn/core/json.h` is OpenNN's own. Example datasets
under `examples/*/data` carry the terms of their original publishers; see
DATASETS.md for verified sources and unresolved redistribution records.
