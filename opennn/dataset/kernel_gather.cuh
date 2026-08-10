#ifndef KERNEL_GATHER_CUH
#define KERNEL_GATHER_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

void gather_rows_cuda(const float* matrix, const int* row_indices, float* out,
                      const Index n_rows, const Index n_cols,
                      const Index matrix_cols, const Index col_offset,
                      cudaStream_t stream = nullptr);

void gather_rows_bf16_cuda(const float* matrix, const int* row_indices, __nv_bfloat16* out,
                           const Index n_rows, const Index n_cols,
                           const Index matrix_cols, const Index col_offset,
                           cudaStream_t stream = nullptr);

void gather_window_rows_cuda(const float* matrix, const int* start_rows, float* out,
                             const Index batch, const Index past, const Index features,
                             const Index matrix_cols, const Index matrix_rows,
                             const Index col_offset, cudaStream_t stream = nullptr);

void gather_window_targets_cuda(const float* matrix, const int* start_rows, float* out,
                                const Index batch, const Index past, const Index future,
                                const Index target_cols, const bool multi_target,
                                const Index matrix_cols, const Index matrix_rows,
                                const Index col_offset, cudaStream_t stream = nullptr);

#endif

#endif
