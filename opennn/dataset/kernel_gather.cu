// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/dataset/kernel_gather.cuh"

#include <stdexcept>

struct ImageAffine
{
    float scale[4];
    float offset[4];
};

template <typename T>
__global__ void unpack_image_pixels_kernel(const uint8_t* input, T* output,
                                            int count, int channels, ImageAffine affine)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int channel = i % channels;
    // Match the separate multiply and add used by the host scaling path.
    output[i] = static_cast<T>(__fadd_rn(__fmul_rn(float(input[i]), affine.scale[channel]),
                                        affine.offset[channel]));
}

void unpack_image_pixels_cuda(const uint8_t* input, void* output, bool output_bf16,
                               Index count, int channels, const float* scale,
                               const float* offset, cudaStream_t stream)
{
    if (count == 0) return;
    if (channels < 1 || channels > 4)
        throw std::invalid_argument("Packed image input requires one to four channels.");
    ImageAffine affine{};
    for (int i = 0; i < channels; ++i)
    {
        affine.scale[i] = scale[i];
        affine.offset[i] = offset[i];
    }
    const int n = checked_int(count);
    dispatch_float_bf16(output_bf16, [&]<typename T>()
    {
        OPENNN_CUDA_LAUNCH(unpack_image_pixels_kernel<T>
            <<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
                input, static_cast<T*>(output), n, channels, affine));
    });
}

static inline int row_threads(const int cols)
{
    return cols < block_size ? ((cols + 31) / 32) * 32 : block_size;
}

template<typename TDst>
__global__ void gather_rows_kernel(const float* __restrict__ matrix,
                                   const int* __restrict__ row_indices,
                                   TDst* __restrict__ out,
                                   const int n_rows,
                                   const int n_cols,
                                   const int matrix_cols,
                                   const int col_offset)
{
    const int row = blockIdx.x * blockDim.y + threadIdx.y;

    if (row >= n_rows) return;

    const float* __restrict__ src = matrix + size_t(row_indices[row]) * matrix_cols + col_offset;
    TDst* __restrict__ dst = out + size_t(row) * n_cols;

    for (int j = threadIdx.x; j < n_cols; j += blockDim.x)
        dst[j] = static_cast<TDst>(src[j]);
}

void gather_rows_cuda(const float* matrix, const int* row_indices, void* out, const bool out_bf16,
                      const Index n_rows, const Index n_cols,
                      const Index matrix_cols, const Index col_offset,
                      cudaStream_t stream)
{
    if (n_rows == 0 || n_cols == 0) return;
    if (stream == nullptr) stream = opennn::device::get_compute_stream();

    const int rows = checked_int(n_rows);
    const int cols = checked_int(n_cols);

    constexpr int block_threads = 256;
    int col_threads = 1;
    while (col_threads < cols && col_threads < 32) col_threads *= 2;
    const int rows_per_block = block_threads / col_threads;
    const int blocks = (rows + rows_per_block - 1) / rows_per_block;

    dispatch_float_bf16(out_bf16, [&]<typename TDst>()
    {
        OPENNN_CUDA_LAUNCH(gather_rows_kernel<TDst>
            <<<blocks, dim3(col_threads, rows_per_block), 0, stream>>>(
            matrix, row_indices, static_cast<TDst*>(out), rows, cols,
            checked_int(matrix_cols), checked_int(col_offset)));
    });
}

__global__ void gather_window_inputs_kernel(const float* __restrict__ matrix,
                                            const int* __restrict__ start_rows,
                                            float* __restrict__ out,
                                            const int features,
                                            const int matrix_cols,
                                            const int matrix_rows,
                                            const int col_offset)
{
    const int s = blockIdx.x;
    const int t = blockIdx.y;

    const long long row = (long long)start_rows[s] + t;
    float* __restrict__ dst = out + (size_t(s) * gridDim.y + t) * features;

    if (row >= matrix_rows)
    {
        for (int f = threadIdx.x; f < features; f += blockDim.x) dst[f] = 0.0f;
        return;
    }

    const float* __restrict__ src = matrix + size_t(row) * matrix_cols + col_offset;
    for (int f = threadIdx.x; f < features; f += blockDim.x)
        dst[f] = src[f];
}

void gather_window_inputs_cuda(const float* matrix, const int* start_rows, float* out,
                               const WindowLayout& window,
                               const Index features, const Index col_offset,
                               cudaStream_t stream)
{
    if (window.batch == 0 || window.past == 0 || features == 0) return;
    if (stream == nullptr) stream = opennn::device::get_compute_stream();

    const int batch = checked_int(window.batch);
    const int past = checked_int(window.past);
    const int cols = checked_int(features);

    OPENNN_CUDA_LAUNCH(gather_window_inputs_kernel<<<dim3(batch, past), row_threads(cols), 0, stream>>>(
        matrix, start_rows, out, cols,
        checked_int(window.matrix_cols), checked_int(window.matrix_rows), checked_int(col_offset)));
}

template<bool MultiTarget>
__global__ void gather_window_targets_kernel(const float* __restrict__ matrix,
                                             const int* __restrict__ start_rows,
                                             float* __restrict__ out,
                                             const int past,
                                             const int future,
                                             const int target_cols,
                                             const int matrix_cols,
                                             const int matrix_rows,
                                             const int col_offset)
{
    const int s = blockIdx.x;

    const long long base = (long long)start_rows[s] + past;

    if constexpr (MultiTarget)
    {
        const int width = target_cols * future;
        float* __restrict__ dst = out + size_t(s) * width;
        for (int i = threadIdx.x; i < width; i += blockDim.x)
        {
            const long long row = base + i % future;
            dst[i] = (row < matrix_rows)
                ? matrix[size_t(row) * matrix_cols + col_offset + i / future]
                : 0.0f;
        }
    }
    else
    {
        const long long row = base + future - 1;
        float* __restrict__ dst = out + size_t(s) * target_cols;
        for (int j = threadIdx.x; j < target_cols; j += blockDim.x)
            dst[j] = (row < matrix_rows)
                ? matrix[size_t(row) * matrix_cols + col_offset + j]
                : 0.0f;
    }
}

void gather_window_targets_cuda(const float* matrix, const int* start_rows, float* out,
                                const WindowLayout& window,
                                const Index future, const Index target_cols,
                                const bool multi_target, const Index col_offset,
                                cudaStream_t stream)
{
    if (window.batch == 0 || future == 0 || target_cols == 0) return;
    if (stream == nullptr) stream = opennn::device::get_compute_stream();

    const int batch = checked_int(window.batch);
    const int width = checked_int(multi_target ? target_cols * future : target_cols);

    auto launch = [&](auto kernel)
    {
        OPENNN_CUDA_LAUNCH(kernel<<<batch, row_threads(width), 0, stream>>>(
            matrix, start_rows, out, checked_int(window.past), checked_int(future),
            checked_int(target_cols),
            checked_int(window.matrix_cols), checked_int(window.matrix_rows), checked_int(col_offset)));
    };

    if (multi_target) launch(gather_window_targets_kernel<true>);
    else              launch(gather_window_targets_kernel<false>);
}
