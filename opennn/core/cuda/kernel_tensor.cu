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
    int chunk = checked_int(ceil_div(batch, Index(desired_chunks)));
    if (chunk < 64) chunk = 64;
    const int n_chunks = checked_int(ceil_div(batch, Index(chunk)));
    const dim3 grid(f_blocks, n_chunks);
    OPENNN_CUDA_LAUNCH(bias_grad_sum_kernel<T><<<grid, block_size, 0,
                                         opennn::device::get_compute_stream()>>>(
        checked_int(batch), f, chunk, delta, bias_grad));
}


// One warp per row. Each lane walks the row in 16-byte steps holding its share
// of the weight vector, and a shuffle tree folds the 32 partial sums; lane 0
// adds the bias on the way out. Accumulation is fp32 whatever T is and the
// reduction order is fixed, so the result is deterministic.
template<typename T>
__global__ void linear_forward_single_output_kernel(const int rows,
                                                    const int features,
                                                    const T* __restrict__ input,
                                                    const T* __restrict__ weights,
                                                    const T* __restrict__ bias,
                                                    T* __restrict__ output)
{
    constexpr int VEC = int(sizeof(uint4) / sizeof(T));

    const int row = int((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (row >= rows) return;

    const int lane = int(threadIdx.x & 31);
    const T* row_data = input + size_t(row) * size_t(features);

    float sum = 0.0f;
    for (int base = lane * VEC; base < features; base += 32 * VEC)
    {
        float a[VEC];
        float w[VEC];
        VecIO<T, VEC>::load_float(row_data + base, a);
        VecIO<T, VEC>::load_float(weights + base, w);

        #pragma unroll
        for (int k = 0; k < VEC; ++k) sum += a[k] * w[k];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);

    if (lane != 0) return;

    if (bias) sum += element_to_float(bias[0]);
    element_from_float(sum, output[row]);
}

template<typename T>
void linear_forward_single_output_cuda(const Index rows,
                                       const Index features,
                                       const T* input,
                                       const T* weights,
                                       const T* bias,
                                       T* output)
{
    constexpr int block_size = 256;
    const int row_count = checked_int(rows);
    const int blocks = (row_count * 32 + block_size - 1) / block_size;

    OPENNN_CUDA_LAUNCH(linear_forward_single_output_kernel<T>
                       <<<blocks, block_size, 0, opennn::device::get_compute_stream()>>>(
                           row_count, checked_int(features), input, weights, bias, output));
}

#define INSTANTIATE(T) \
    template void transpose_2d_cuda<T>(const Index, const Index, const T*, T*); \
    template void bias_grad_sum_cuda<T>(const Index, const Index, const T*, float*); \
    template void linear_forward_single_output_cuda<T>(const Index, const Index, \
                                                       const T*, const T*, const T*, T*);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE

template void transpose_2d_cuda<int8_t>(const Index, const Index, const int8_t*, int8_t*);


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
