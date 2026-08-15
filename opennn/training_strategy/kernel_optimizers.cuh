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
    const float* learning_rate_device, const float epsilon,
    int* step_device, float* effective_lr_device, float* effective_eps_device,
    __nv_bfloat16* parameters_bf16_mirror = nullptr, cudaStream_t stream = nullptr);

void clip_gradient_norm_cuda(const Index n, float* gradient, const float* squared_norm, const float max_norm, const float eps);

#endif

#endif
