#ifndef KERNEL_OPTIMIZERS_CUH
#define KERNEL_OPTIMIZERS_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

void adam_update_cuda(const Index, float*, float*, float*, const float*,
                      const float, const float, const float, const float,
                      const float, const float,
                      __nv_bfloat16* parameters_bf16_mirror = nullptr);

void sgd_update_cuda(const Index, float*, float*, const float*,
                     const float, const float, const bool,
                     __nv_bfloat16* parameters_bf16_mirror = nullptr);

void sgd_update_capturable_cuda(
    const Index n, float* parameters, float* velocity, const float* gradients,
    const float* learning_rate_device, const float momentum, const bool nesterov,
    __nv_bfloat16* parameters_bf16_mirror = nullptr, cudaStream_t stream = nullptr);

void set_scalar_device_cuda(float* dst, const float value, cudaStream_t stream = nullptr);

void adam_update_capturable_cuda(
    const Index n, float* parameters, float* m, float* v, const float* gradients,
    const float beta_1, const float beta_2,
    const float learning_rate, const float epsilon,
    int* step_device, float* effective_lr_device, float* effective_eps_device,
    __nv_bfloat16* parameters_bf16_mirror = nullptr, cudaStream_t stream = nullptr);

void clip_gradient_norm_cuda(const Index n, float* gradient, const float* squared_norm, const float max_norm, const float eps);

void cast_fp32_to_bf16(const Index n, const float* src, __nv_bfloat16* dst,
                            cudaStream_t stream = nullptr);
void cast_bf16_to_fp32(const Index n, const __nv_bfloat16* src, float* dst);

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
