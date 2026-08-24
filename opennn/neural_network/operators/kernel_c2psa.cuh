#ifndef KERNEL_C2PSA_CUH
#define KERNEL_C2PSA_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

// Rows of x are (BT, C) with C = 2H: the left half feeds the attention (xa,
// compact (BT, H)), the right half is carried into cat unchanged.
void c2psa_gather_left_cuda(const void* x, void* xa, int BT, int C, int H, cudaDataType_t dtype);
void c2psa_split_cuda(const void* x, void* xa, void* cat, int BT, int C, int H, cudaDataType_t dtype);
void c2psa_fill_cat_left_cuda(const void* attn_v, void* cat, int BT, int C, int H, cudaDataType_t dtype);
void c2psa_scatter_dx_cuda(const void* d_xa, const void* d_cat, void* din, int BT, int C, int H, cudaDataType_t dtype);

#endif

#endif
