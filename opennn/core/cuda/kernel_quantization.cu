//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Q U A N T I Z A T I O N   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// INT8 weight-only quantized linear, dequantization and embedding

#include "opennn/core/cuda/kernel_common.cuh"

template<typename T, int WARPS_PER_ROW>
__global__ void w8a16_linear_out_major_kernel(
    const int m, const int in_features, const int out_features,
    const T* __restrict__ x, const int8_t* __restrict__ w,
    const float* __restrict__ scales, const T* __restrict__ bias,
    T* __restrict__ y)
{
    constexpr int ROWS_PER_BLOCK = 8 / WARPS_PER_ROW;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row_in_block = warp / WARPS_PER_ROW;
    const int part = warp % WARPS_PER_ROW;
    const int j = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
    const bool active = j < out_features;

    float acc[W8A16_MAX_M];
    for (int r = 0; r < m; ++r) acc[r] = 0.0f;

    if (active)
    {
        const int8_t* __restrict__ row = w + size_t(j) * in_features;
        const int stride = 32 * WARPS_PER_ROW;

        if ((in_features & 3) == 0)
        {
            const char4* __restrict__ row4 = reinterpret_cast<const char4*>(row);
            const int k4 = in_features >> 2;
            for (int k = lane + part * 32; k < k4; k += stride)
            {
                const char4 wv = row4[k];
                const int kk = k << 2;
                for (int r = 0; r < m; ++r)
                {
                    const T* __restrict__ xr = x + size_t(r) * in_features + kk;
                    acc[r] += float(wv.x) * static_cast<float>(xr[0])
                            + float(wv.y) * static_cast<float>(xr[1])
                            + float(wv.z) * static_cast<float>(xr[2])
                            + float(wv.w) * static_cast<float>(xr[3]);
                }
            }
        }
        else
            for (int k = lane + part * 32; k < in_features; k += stride)
            {
                const float wv = float(row[k]);
                for (int r = 0; r < m; ++r)
                    acc[r] += wv * static_cast<float>(x[size_t(r) * in_features + k]);
            }
    }

    for (int r = 0; r < m; ++r)
        for (int offset = 16; offset > 0; offset >>= 1)
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], offset);

    const auto store = [&](const int r, const float sum)
    {
        y[size_t(r) * out_features + j] = static_cast<T>(
            sum * scales[j] + (bias ? static_cast<float>(bias[j]) : 0.0f));
    };

    if constexpr (WARPS_PER_ROW == 1)
    {
        if (!active || lane != 0) return;
        for (int r = 0; r < m; ++r) store(r, acc[r]);
    }
    else
    {
        __shared__ float partials[8][W8A16_MAX_M];
        if (lane == 0)
            for (int r = 0; r < m; ++r) partials[warp][r] = acc[r];
        __syncthreads();
        if (!active || part != 0 || lane != 0) return;
        for (int r = 0; r < m; ++r)
        {
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < WARPS_PER_ROW; ++i)
                sum += partials[row_in_block * WARPS_PER_ROW + i][r];
            store(r, sum);
        }
    }
}

template<typename T>
__global__ void w8a16_linear_in_major_kernel(
    const int m, const int in_features, const int out_features,
    const T* __restrict__ x, const int8_t* __restrict__ w,
    const float* __restrict__ scales, const T* __restrict__ bias,
    T* __restrict__ y)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_features) return;

    float acc[W8A16_MAX_M];
    for (int r = 0; r < m; ++r) acc[r] = 0.0f;

    for (int k = 0; k < in_features; ++k)
    {
        const float wv = float(w[size_t(k) * out_features + j]);
        for (int r = 0; r < m; ++r)
            acc[r] += wv * static_cast<float>(x[size_t(r) * in_features + k]);
    }

    const float scale = scales[j];
    const float bias_value = bias ? static_cast<float>(bias[j]) : 0.0f;
    for (int r = 0; r < m; ++r)
        y[size_t(r) * out_features + j] = static_cast<T>(acc[r] * scale + bias_value);
}

template<typename T>
void w8a16_linear_cuda(const int m, const int in_features, const int out_features,
                       const bool weights_out_major,
                       const T* x, const int8_t* w, const float* scales,
                       const T* bias, T* y)
{
    checked_host_condition(m <= 0 || m > W8A16_MAX_M,
                           "w8a16_linear_cuda: m out of range.");
    if (out_features == 0) return;
    cudaStream_t stream = opennn::device::get_compute_stream();

    if (!weights_out_major)
    {
        OPENNN_CUDA_LAUNCH(w8a16_linear_in_major_kernel<T>
            <<<grid_size_for(out_features), block_size, 0, stream>>>(
                m, in_features, out_features, x, w, scales, bias, y));
        return;
    }

    if (w8a16_out_major_warps(out_features) == 8)
        OPENNN_CUDA_LAUNCH((w8a16_linear_out_major_kernel<T, 8>
            <<<out_features, block_size, 0, stream>>>(
                m, in_features, out_features, x, w, scales, bias, y)));
    else
        OPENNN_CUDA_LAUNCH((w8a16_linear_out_major_kernel<T, 1>
            <<<ceil_div(out_features, 8), block_size, 0, stream>>>(
                m, in_features, out_features, x, w, scales, bias, y)));
}

template<typename T, bool SCALE_BY_ROW>
__global__ void w8_dequant_kernel(const int rows,
                                  const int row_length,
                                  const int8_t* __restrict__ q,
                                  const float* __restrict__ scales,
                                  T* __restrict__ out)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= row_length) return;

    for (int row = blockIdx.y; row < rows; row += gridDim.y)
    {
        const Index i = Index(row) * row_length + column;
        const float scale = SCALE_BY_ROW ? scales[row] : scales[column];
        out[i] = static_cast<T>(float(q[i]) * scale);
    }
}

template<typename T>
void w8_dequant_cuda(const Index rows, const Index row_length, const bool scale_by_row,
                     const int8_t* q, const float* scales, T* out)
{
    if (rows == 0 || row_length == 0) return;
    constexpr int max_grid_y = 65535;
    const dim3 grid(unsigned(grid_size_for(checked_int(row_length))),
                    unsigned(min(checked_int(rows), max_grid_y)));
    cudaStream_t stream = opennn::device::get_compute_stream();
    if (scale_by_row)
        OPENNN_CUDA_LAUNCH((w8_dequant_kernel<T, true>
            <<<grid, block_size, 0, stream>>>(checked_int(rows), checked_int(row_length), q, scales, out)));
    else
        OPENNN_CUDA_LAUNCH((w8_dequant_kernel<T, false>
            <<<grid, block_size, 0, stream>>>(checked_int(rows), checked_int(row_length), q, scales, out)));
}

template<typename T>
__global__ void embedding_forward_w8_kernel(
    const int n, const float* __restrict__ inputs, const int8_t* __restrict__ weights,
    const float* __restrict__ weight_scales, const float* __restrict__ positional_encoding,
    T* __restrict__ outputs, const int sequence_length, const int embedding_dimension,
    const int vocabulary_size, const bool scale_embedding)
{
    const float scale = scale_embedding ? sqrtf(static_cast<float>(embedding_dimension)) : 1.0f;

    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += Index(blockDim.x) * gridDim.x)
    {
        const int token_index = i / embedding_dimension;
        const int dim_index = i % embedding_dimension;
        const int token_id = static_cast<int>(inputs[token_index]);

        float val = (token_id > 0 && token_id < vocabulary_size)
            ? scale * weight_scales[token_id]
                * float(weights[size_t(token_id) * embedding_dimension + dim_index])
            : 0.0f;

        if (positional_encoding != nullptr && token_id > 0)
        {
            const int seq_index = token_index % sequence_length;
            val += positional_encoding[seq_index * embedding_dimension + dim_index];
        }

        outputs[i] = static_cast<T>(val);
    }
}

template<typename T>
void embedding_forward_w8_cuda(const Index n, const float* inputs, const int8_t* weights,
                               const float* weight_scales, const float* positional_encoding,
                               T* outputs, const int sequence_length,
                               const int embedding_dimension, const int vocabulary_size,
                               const bool scale_embedding)
{
    launch_elementwise_strided(n, embedding_forward_w8_kernel<T>, inputs, weights,
                       weight_scales, positional_encoding, outputs,
                       sequence_length, embedding_dimension, vocabulary_size, scale_embedding);
}

#define INSTANTIATE(T) \
    template void w8a16_linear_cuda<T>(const int, const int, const int, const bool, const T*, const int8_t*, const float*, const T*, T*); \
    template void w8_dequant_cuda<T>(const Index, const Index, const bool, const int8_t*, const float*, T*); \
    template void embedding_forward_w8_cuda<T>(const Index, const float*, const int8_t*, const float*, const float*, T*, const int, const int, const int, const bool);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
