#ifndef KERNEL_GATHER_CUH
#define KERNEL_GATHER_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

// out points to __nv_bfloat16 when out_bf16 is set, to float otherwise.
void gather_rows_cuda(const float* matrix, const int* row_indices, void* out, const bool out_bf16,
                      const Index n_rows, const Index n_cols,
                      const Index matrix_cols, const Index col_offset,
                      cudaStream_t stream = nullptr);

// Source matrix shape and the window over it, shared by both window gathers.
struct WindowLayout
{
    Index batch = 0;
    Index past = 0;
    Index matrix_cols = 0;
    Index matrix_rows = 0;
};

void gather_window_inputs_cuda(const float* matrix, const int* start_rows, float* out,
                               const WindowLayout& window,
                               const Index features, const Index col_offset,
                               cudaStream_t stream = nullptr);

void gather_window_targets_cuda(const float* matrix, const int* start_rows, float* out,
                                const WindowLayout& window,
                                const Index future, const Index target_cols,
                                const bool multi_target, const Index col_offset,
                                cudaStream_t stream = nullptr);

#endif

#endif
