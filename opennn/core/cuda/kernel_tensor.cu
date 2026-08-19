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
#include "opennn/core/device_backend.h"
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


// Backward of the same single-output layer. The three operations it needs -
// the input delta (a rank-1 product of the output delta and the weight
// vector), the weight gradient (the input's columns weighted by the output
// delta) and the bias gradient (the output delta's sum) - all walk the same
// rows, so one pass over the input does all three: cuBLAS runs them as two
// GEMVs that read the input twice and reach a third of the bandwidth.
//
// One warp per row, each lane holding its slice of the weight vector and of
// the weight-gradient accumulator across the rows the warp visits. The block
// folds its warps in warp order through shared memory and writes one row of
// partials; single_output_gradient_finalize_kernel sums those in block order,
// so the result does not depend on how the rows were scheduled.
template<typename T, int CHUNKS>
__global__ void linear_backward_single_output_kernel(const int rows,
                                                     const int features,
                                                     const T* __restrict__ output_delta,
                                                     const T* __restrict__ input,
                                                     const T* __restrict__ weights,
                                                     T* __restrict__ input_delta,
                                                     const bool fuse_input_relu,
                                                     float* __restrict__ weight_gradient_partials,
                                                     float* __restrict__ bias_gradient_partials)
{
    constexpr int VEC = int(sizeof(uint4) / sizeof(T));

    const int lane  = int(threadIdx.x & 31);
    const int warp  = int(threadIdx.x >> 5);
    const int warps = int(blockDim.x >> 5);

    float weight[CHUNKS][VEC];
    float weight_gradient[CHUNKS][VEC];
    #pragma unroll
    for (int chunk = 0; chunk < CHUNKS; ++chunk)
    {
        VecIO<T, VEC>::load_float(weights + (chunk * 32 + lane) * VEC, weight[chunk]);
        #pragma unroll
        for (int k = 0; k < VEC; ++k) weight_gradient[chunk][k] = 0.0f;
    }
    float bias_gradient = 0.0f;

    for (int row = int(blockIdx.x) * warps + warp; row < rows; row += int(gridDim.x) * warps)
    {
        const float delta = element_to_float(output_delta[row]);
        if (lane == 0) bias_gradient += delta;

        const T* input_row = input + size_t(row) * size_t(features);
        T* input_delta_row = input_delta ? input_delta + size_t(row) * size_t(features) : nullptr;

        #pragma unroll
        for (int chunk = 0; chunk < CHUNKS; ++chunk)
        {
            const int base = (chunk * 32 + lane) * VEC;
            float values[VEC];
            VecIO<T, VEC>::load_float(input_row + base, values);

            float deltas[VEC];
            #pragma unroll
            for (int k = 0; k < VEC; ++k)
            {
                weight_gradient[chunk][k] += values[k] * delta;
                deltas[k] = (fuse_input_relu && !(values[k] > 0.0f))
                    ? 0.0f
                    : delta * weight[chunk][k];
            }
            if (input_delta_row) VecIO<T, VEC>::store_float(input_delta_row + base, deltas);
        }
    }

    extern __shared__ float block_gradient[];   // features + 1
    for (int i = int(threadIdx.x); i <= features; i += int(blockDim.x)) block_gradient[i] = 0.0f;
    __syncthreads();

    for (int owner = 0; owner < warps; ++owner)
    {
        if (owner == warp)
        {
            #pragma unroll
            for (int chunk = 0; chunk < CHUNKS; ++chunk)
            {
                const int base = (chunk * 32 + lane) * VEC;
                #pragma unroll
                for (int k = 0; k < VEC; ++k) block_gradient[base + k] += weight_gradient[chunk][k];
            }
            if (lane == 0) block_gradient[features] += bias_gradient;
        }
        __syncthreads();
    }

    float* partials = weight_gradient_partials + size_t(blockIdx.x) * size_t(features);
    for (int i = int(threadIdx.x); i < features; i += int(blockDim.x)) partials[i] = block_gradient[i];
    if (threadIdx.x == 0 && bias_gradient_partials)
        bias_gradient_partials[blockIdx.x] = block_gradient[features];
}

__global__ void single_output_gradient_finalize_kernel(const int blocks,
                                                       const int features,
                                                       const float* __restrict__ weight_partials,
                                                       const float* __restrict__ bias_partials,
                                                       float* __restrict__ weight_gradient,
                                                       float* __restrict__ bias_gradient)
{
    const int feature = int(blockIdx.x * blockDim.x + threadIdx.x);

    if (feature < features)
    {
        float sum = 0.0f;
        for (int block = 0; block < blocks; ++block)
            sum += weight_partials[size_t(block) * size_t(features) + feature];
        weight_gradient[feature] = sum;
    }

    if (feature == 0 && bias_gradient)
    {
        float sum = 0.0f;
        for (int block = 0; block < blocks; ++block) sum += bias_partials[block];
        bias_gradient[0] = sum;
    }
}

// Rows per warp beyond which more blocks stop paying: enough warps to fill the
// device several times, few enough that the finalize sum stays short.
static constexpr int single_output_maximum_blocks = 240;

template<typename T>
bool linear_backward_single_output_cuda(const Index rows,
                                        const Index features,
                                        const T* output_delta,
                                        const T* input,
                                        const T* weights,
                                        T* input_delta,
                                        bool fuse_input_relu,
                                        float* weight_gradient,
                                        float* bias_gradient)
{
    constexpr int VEC = int(sizeof(uint4) / sizeof(T));
    constexpr int block_size = 256;
    constexpr int warps = block_size / 32;

    const int feature_count = checked_int(features);
    const int chunks = feature_count / (32 * VEC);
    if (chunks * 32 * VEC != feature_count) return false;

    const int row_count = checked_int(rows);
    const int needed = (row_count + warps - 1) / warps;
    const int blocks = needed < single_output_maximum_blocks ? needed : single_output_maximum_blocks;
    if (blocks <= 0) return false;

    float* const partials = opennn::ensure_workspace<float>(
        opennn::device::GraphWorkspaceKind::GradientPartials,
        Index(blocks) * (features + 1));
    float* const bias_partials = partials + size_t(blocks) * size_t(feature_count);

    const cudaStream_t stream = opennn::device::get_compute_stream();
    const size_t shared_bytes = size_t(feature_count + 1) * sizeof(float);

    const auto launch = [&]<int CHUNKS>()
    {
        OPENNN_CUDA_LAUNCH((linear_backward_single_output_kernel<T, CHUNKS>
                            <<<blocks, block_size, shared_bytes, stream>>>(
                                row_count, feature_count, output_delta, input, weights,
                                input_delta, fuse_input_relu,
                                partials, bias_gradient ? bias_partials : nullptr)));
    };

    switch (chunks)
    {
    case 1: launch.template operator()<1>(); break;
    case 2: launch.template operator()<2>(); break;
    case 4: launch.template operator()<4>(); break;
    case 8: launch.template operator()<8>(); break;
    default: return false;
    }

    OPENNN_CUDA_LAUNCH((single_output_gradient_finalize_kernel
                        <<<(feature_count + block_size - 1) / block_size, block_size, 0, stream>>>(
                            blocks, feature_count, partials, bias_partials,
                            weight_gradient, bias_gradient)));
    return true;
}

#define INSTANTIATE(T) \
    template void transpose_2d_cuda<T>(const Index, const Index, const T*, T*); \
    template void bias_grad_sum_cuda<T>(const Index, const Index, const T*, float*); \
    template void linear_forward_single_output_cuda<T>(const Index, const Index, \
                                                       const T*, const T*, const T*, T*); \
    template bool linear_backward_single_output_cuda<T>(const Index, const Index, \
                                                        const T*, const T*, const T*, T*, bool, float*, float*);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE

template void transpose_2d_cuda<int8_t>(const Index, const Index, const int8_t*, int8_t*);


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
