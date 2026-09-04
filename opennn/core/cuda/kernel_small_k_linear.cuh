#ifndef KERNEL_SMALL_K_LINEAR_CUH
#define KERNEL_SMALL_K_LINEAR_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

// output = relu?(input * weights + bias) for a bf16 linear whose contraction
// is at most 32 - a first layer over tabular features. Returns false for a
// shape it does not cover, and the caller runs cuBLASLt instead.
//
// input is rows x contraction, weights contraction x out_features, output
// rows x out_features, all row-major bf16. bias has out_features entries,
// bf16 or fp32 (bias_fp32), or is null.
bool small_k_linear_forward_cuda(Index rows, Index contraction, Index out_features,
                                 const void* input, const void* weights,
                                 const void* bias, bool bias_fp32,
                                 void* output, bool relu, cudaStream_t stream);

#endif

#endif
