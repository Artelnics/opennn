//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// Shape- and reduction-level kernels that belong to no single layer family:
// a 2D transpose used when materializing transposed weights, and the bias
// gradient reduction behind linear_backward.

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/core/cuda/kernel_tensor.cuh"

template<typename T>
__global__ void transpose_2d_kernel(const int total,
                                    const int rows,
                                    const int cols,
                                    const T* __restrict__ src,
                                    T* __restrict__ dst)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int r = idx / cols;
    const int c = idx - r * cols;
    dst[c * rows + r] = src[r * cols + c];
}

template<typename T>
void transpose_2d_cuda(const Index rows,
                       const Index cols,
                       const T* src,
                       T* dst)
{
    launch_elementwise(rows * cols, transpose_2d_kernel<T>, checked_int(rows), checked_int(cols), src, dst);
}

template<typename T>
__global__ void bias_grad_sum_kernel(const int batch, const int features, const int chunk,
                                     const T* __restrict__ delta, float* __restrict__ bias_grad)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= features) return;
    const long long b0 = (long long)blockIdx.y * chunk;
    const long long b1 = min((long long)batch, b0 + chunk);
    float acc = 0.0f;
    for (long long b = b0; b < b1; ++b)
        acc += static_cast<float>(delta[b * features + f]);
    atomicAdd(bias_grad + f, acc);
}

template<typename T>
void bias_grad_sum_cuda(const Index batch, const Index features, const T* delta, float* bias_grad)
{
    if (batch == 0 || features == 0) return;
    const int f = checked_int(features);

    const int f_blocks = ceil_div(f, block_size);
    const int desired_chunks = f_blocks < 256 ? 256 / f_blocks : 1;
    int chunk = checked_int((batch + desired_chunks - 1) / desired_chunks);
    if (chunk < 64) chunk = 64;
    const int n_chunks = int((batch + chunk - 1) / chunk);
    const dim3 grid(f_blocks, n_chunks);
    OPENNN_CUDA_LAUNCH(bias_grad_sum_kernel<T><<<grid, block_size, 0,
                                         opennn::device::get_compute_stream()>>>(
        checked_int(batch), f, chunk, delta, bias_grad));
}

#define INSTANTIATE(T) \
    template void transpose_2d_cuda<T>(const Index, const Index, const T*, T*); \
    template void bias_grad_sum_cuda<T>(const Index, const Index, const T*, float*);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE

template void transpose_2d_cuda<int8_t>(const Index, const Index, const int8_t*, int8_t*);


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
