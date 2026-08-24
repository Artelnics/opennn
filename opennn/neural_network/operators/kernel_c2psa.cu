//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C 2 P S A   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/opennn_types.h"
#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/neural_network/layers/kernel_concat.cuh"
#include "opennn/neural_network/operators/kernel_c2psa.cuh"

template<typename T>
__global__ void c2psa_row_softmax_kernel(const int rows, T* __restrict__ A, int T_sz)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    T* p = A + row * T_sz;
    row_softmax<T>(p, p, T_sz, 0.0f);
}

template<typename T>
__global__ void c2psa_softmax_bwd_kernel(
    const int rows,
    const T* __restrict__ A,
    T* __restrict__ dA,
    float scale,
    int T_sz)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const T* Ap  = A  + row * T_sz;
    T*       dAp = dA + row * T_sz;
    row_softmax_backward<T>(Ap, dAp, dAp, T_sz, scale);
}

void c2psa_gather_left_cuda(
    const void* x, void* xa,
    int BT, int C, int H, cudaDataType_t dtype)
{
    dispatch_float_bf16(dtype != CUDA_R_32F, [&]<typename T>() {
        slice_channels_cuda<T, false>(BT, 1, 1, H, C, 0, (const T*)x, (T*)xa);
    });
}

void c2psa_split_cuda(
    const void* x, void* xa, void* cat,
    int BT, int C, int H, cudaDataType_t dtype)
{
    c2psa_gather_left_cuda(x, xa, BT, C, H, dtype);

    // The right half is a strided column copy, which cudaMemcpy2DAsync
    // expresses directly: BT rows of (C - H) elements, pitch C on both sides.
    dispatch_float_bf16(dtype != CUDA_R_32F, [&]<typename T>() {
        const size_t pitch = size_t(C) * sizeof(T);
        CHECK_CUDA(cudaMemcpy2DAsync((T*)cat + H, pitch,
                                     (const T*)x + H, pitch,
                                     size_t(C - H) * sizeof(T), size_t(BT),
                                     cudaMemcpyDeviceToDevice,
                                     opennn::device::get_compute_stream()));
    });
}

void c2psa_fill_cat_left_cuda(
    const void* attn_v, void* cat,
    int BT, int C, int H, cudaDataType_t dtype)
{
    dispatch_float_bf16(dtype != CUDA_R_32F, [&]<typename T>() {
        slice_channels_cuda<T, true>(BT, 1, 1, H, C, 0, (const T*)attn_v, (T*)cat);
    });
}

void c2psa_row_softmax_cuda(void* A, int rows, int T_sz, cudaDataType_t dtype)
{
    dispatch_float_bf16(dtype != CUDA_R_32F, [&]<typename T>() {
        launch_elementwise(Index(rows), c2psa_row_softmax_kernel<T>, (T*)A, T_sz);
    });
}

void c2psa_softmax_bwd_cuda(const void* A, void* dA, float scale, int rows, int T_sz, cudaDataType_t dtype)
{
    dispatch_float_bf16(dtype != CUDA_R_32F, [&]<typename T>() {
        launch_elementwise(Index(rows), c2psa_softmax_bwd_kernel<T>, (const T*)A, (T*)dA, scale, T_sz);
    });
}

void c2psa_scatter_dx_cuda(
    const void* d_xa, const void* d_cat, void* din,
    int BT, int C, int H, cudaDataType_t dtype)
{
    // Left half from the attention gradient, right half straight through: a
    // channel scatter and a pitched copy, the same two shapes as the split.
    dispatch_float_bf16(dtype != CUDA_R_32F, [&]<typename T>() {
        slice_channels_cuda<T, true>(BT, 1, 1, H, C, 0, (const T*)d_xa, (T*)din);

        const size_t pitch = size_t(C) * sizeof(T);
        CHECK_CUDA(cudaMemcpy2DAsync((T*)din + H, pitch,
                                     (const T*)d_cat + H, pitch,
                                     size_t(C - H) * sizeof(T), size_t(BT),
                                     cudaMemcpyDeviceToDevice,
                                     opennn::device::get_compute_stream()));
    });
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
