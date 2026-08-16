//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C 2 P S A   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/neural_network/layers/kernel_concat.cuh"
#include "opennn/neural_network/operators/kernel_c2psa.cuh"

// x and cat share the (BT, C) layout; copies channels [H, 2H) of every row.
template<typename T>
__global__ void c2psa_copy_right_kernel(
    const int n,
    const T* __restrict__ x,
    T* __restrict__ cat,
    int C, int H)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int row = i / H;
        const int col = i % H;
        cat[row * C + H + col] = x[row * C + H + col];
    }
}

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

template<typename T>
__global__ void c2psa_scatter_dx_kernel(
    const int n,
    const T* __restrict__ d_xa,
    const T* __restrict__ d_cat,
    T* __restrict__ din,
    int C, int H)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int row = i / C;
        const int col = i % C;
        din[i] = (col < H) ? d_xa[row * H + col] : d_cat[i];
    }
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
    dispatch_float_bf16(dtype != CUDA_R_32F, [&]<typename T>() {
        launch_elementwise_strided(Index(BT) * H, c2psa_copy_right_kernel<T>,
            (const T*)x, (T*)cat, C, H);
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
    dispatch_float_bf16(dtype != CUDA_R_32F, [&]<typename T>() {
        launch_elementwise_strided(Index(BT) * C, c2psa_scatter_dx_kernel<T>,
            (const T*)d_xa, (const T*)d_cat, (T*)din, C, H);
    });
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
