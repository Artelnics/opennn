//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   G A T H E R   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/dataset/kernel_gather.cuh"

template<typename TDst>
__global__ void gather_rows_kernel(const float* __restrict__ matrix,
                                   const int* __restrict__ row_indices,
                                   TDst* __restrict__ out,
                                   const int n_rows,
                                   const int n_cols,
                                   const int matrix_cols,
                                   const int col_offset)
{
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const float* __restrict__ src = matrix + size_t(row_indices[row]) * matrix_cols + col_offset;
    TDst* __restrict__ dst = out + size_t(row) * n_cols;

    for (int j = threadIdx.x; j < n_cols; j += blockDim.x)
        dst[j] = static_cast<TDst>(src[j]);
}

template<typename TDst>
static void gather_rows_launch(const float* matrix, const int* row_indices, TDst* out,
                               const Index n_rows, const Index n_cols,
                               const Index matrix_cols, const Index col_offset,
                               cudaStream_t stream)
{
    if (n_rows == 0 || n_cols == 0) return;
    if (stream == nullptr) stream = opennn::device::get_compute_stream();

    const int rows = checked_int(n_rows);
    const int cols = checked_int(n_cols);
    const int threads = cols < block_size ? ((cols + 31) / 32) * 32 : block_size;

    OPENNN_CUDA_LAUNCH(gather_rows_kernel<TDst><<<rows, threads, 0, stream>>>(
        matrix, row_indices, out, rows, cols, checked_int(matrix_cols), checked_int(col_offset)));
}

void gather_rows_cuda(const float* matrix, const int* row_indices, float* out,
                      const Index n_rows, const Index n_cols,
                      const Index matrix_cols, const Index col_offset,
                      cudaStream_t stream)
{
    gather_rows_launch<float>(matrix, row_indices, out, n_rows, n_cols, matrix_cols, col_offset, stream);
}

void gather_rows_bf16_cuda(const float* matrix, const int* row_indices, __nv_bfloat16* out,
                           const Index n_rows, const Index n_cols,
                           const Index matrix_cols, const Index col_offset,
                           cudaStream_t stream)
{
    gather_rows_launch<__nv_bfloat16>(matrix, row_indices, out, n_rows, n_cols, matrix_cols, col_offset, stream);
}

__global__ void gather_window_rows_kernel(const float* __restrict__ matrix,
                                          const int* __restrict__ start_rows,
                                          float* __restrict__ out,
                                          const int batch,
                                          const int past,
                                          const int features,
                                          const int matrix_cols,
                                          const int matrix_rows,
                                          const int col_offset)
{
    const int s = blockIdx.x;
    const int t = blockIdx.y;
    if (s >= batch || t >= past) return;

    const long long row = (long long)start_rows[s] + t;
    float* __restrict__ dst = out + (size_t(s) * past + t) * features;

    if (row >= matrix_rows)
    {
        for (int f = threadIdx.x; f < features; f += blockDim.x) dst[f] = 0.0f;
        return;
    }

    const float* __restrict__ src = matrix + size_t(row) * matrix_cols + col_offset;
    for (int f = threadIdx.x; f < features; f += blockDim.x)
        dst[f] = src[f];
}

void gather_window_rows_cuda(const float* matrix, const int* start_rows, float* out,
                             const Index batch, const Index past, const Index features,
                             const Index matrix_cols, const Index matrix_rows,
                             const Index col_offset, cudaStream_t stream)
{
    if (batch == 0 || past == 0 || features == 0) return;
    if (stream == nullptr) stream = opennn::device::get_compute_stream();

    const int cols = checked_int(features);
    const int threads = cols < block_size ? ((cols + 31) / 32) * 32 : block_size;
    const dim3 grid(checked_int(batch), checked_int(past));

    OPENNN_CUDA_LAUNCH(gather_window_rows_kernel<<<grid, threads, 0, stream>>>(
        matrix, start_rows, out, checked_int(batch), checked_int(past), cols,
        checked_int(matrix_cols), checked_int(matrix_rows), checked_int(col_offset)));
}

__global__ void gather_window_targets_kernel(const float* __restrict__ matrix,
                                             const int* __restrict__ start_rows,
                                             float* __restrict__ out,
                                             const int batch,
                                             const int past,
                                             const int future,
                                             const int target_cols,
                                             const bool multi_target,
                                             const int matrix_cols,
                                             const int matrix_rows,
                                             const int col_offset)
{
    const int s = blockIdx.x;
    if (s >= batch) return;

    const long long base = (long long)start_rows[s] + past;

    if (multi_target)
    {
        const int width = target_cols * future;
        float* __restrict__ dst = out + size_t(s) * width;
        for (int i = threadIdx.x; i < width; i += blockDim.x)
        {
            const int j = i / future;
            const int k = i % future;
            const long long row = base + k;
            dst[j * future + k] = (row < matrix_rows)
                ? matrix[size_t(row) * matrix_cols + col_offset + j]
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
                                const Index batch, const Index past, const Index future,
                                const Index target_cols, const bool multi_target,
                                const Index matrix_cols, const Index matrix_rows,
                                const Index col_offset, cudaStream_t stream)
{
    if (batch == 0 || future == 0 || target_cols == 0) return;
    if (stream == nullptr) stream = opennn::device::get_compute_stream();

    const int width = checked_int(multi_target ? target_cols * future : target_cols);
    const int threads = width < block_size ? ((width + 31) / 32) * 32 : block_size;

    OPENNN_CUDA_LAUNCH(gather_window_targets_kernel<<<checked_int(batch), threads, 0, stream>>>(
        matrix, start_rows, out, checked_int(batch), checked_int(past), checked_int(future),
        checked_int(target_cols), multi_target,
        checked_int(matrix_cols), checked_int(matrix_rows), checked_int(col_offset)));
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
