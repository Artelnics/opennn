// Stand-in for the two launch-check macros FlashAttention-2 uses, so its
// kernels build without PyTorch: both turn a CUDA error into an abort naming
// the file and line, which is what the originals do in a release build.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define C10_CUDA_CHECK(expr)                                            \
    do {                                                                \
        const cudaError_t opennn_cuda_status = (expr);                  \
        if (opennn_cuda_status != cudaSuccess) {                        \
            fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(opennn_cuda_status));  \
            abort();                                                    \
        }                                                               \
    } while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
