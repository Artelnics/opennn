//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z E R   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/training_strategy/kernel_optimizers.cuh"

__device__ __forceinline__ void adam_update_one(
    float& p,
    float& m,
    float& v,
    float g,
    float beta_1,
    float one_minus_beta_1,
    float beta_2,
    float one_minus_beta_2,
    float lr,
    float eps)
{
    m = fmaf(beta_1, m, one_minus_beta_1 * g);
    v = fmaf(beta_2, v, one_minus_beta_2 * g * g);
    p -= lr * m / (sqrtf(v) + eps);
}

__global__ void adam_update_kernel(
    const int n_vec,
    const int n,
    float* __restrict__ parameters,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ gradients,
    __nv_bfloat16* __restrict__ parameters_bf16_mirror,
    const float beta_1,
    const float one_minus_beta_1,
    const float beta_2,
    const float one_minus_beta_2,
    const float lr_scalar,
    const float eps_scalar,
    const float* __restrict__ effective_lr,
    const float* __restrict__ effective_eps)
{
    const float lr  = effective_lr  ? *effective_lr  : lr_scalar;
    const float eps = effective_eps ? *effective_eps : eps_scalar;

    const Index tid = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = Index(blockDim.x) * gridDim.x;

    float4* __restrict__ const       p4 = reinterpret_cast<float4*>(parameters);
    float4* __restrict__ const       m4 = reinterpret_cast<float4*>(m);
    float4* __restrict__ const       v4 = reinterpret_cast<float4*>(v);
    const float4* __restrict__ const g4 = reinterpret_cast<const float4*>(gradients);
    __nv_bfloat162* __restrict__ const bf2 = reinterpret_cast<__nv_bfloat162*>(parameters_bf16_mirror);

    for (Index i = tid; i < n_vec; i += stride)
    {
        float4 P = p4[i];
        float4 M = m4[i];
        float4 V = v4[i];
        const float4 G = g4[i];

        adam_update_one(P.x, M.x, V.x, G.x, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, lr, eps);
        adam_update_one(P.y, M.y, V.y, G.y, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, lr, eps);
        adam_update_one(P.z, M.z, V.z, G.z, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, lr, eps);
        adam_update_one(P.w, M.w, V.w, G.w, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, lr, eps);

        p4[i] = P;
        m4[i] = M;
        v4[i] = V;

        if (bf2)
        {
            bf2[i * 2 + 0] = __floats2bfloat162_rn(P.x, P.y);
            bf2[i * 2 + 1] = __floats2bfloat162_rn(P.z, P.w);
        }
    }

    const int tail_start = n_vec * 4;
    for (Index i = tail_start + tid; i < n; i += stride)
    {
        adam_update_one(parameters[i], m[i], v[i], gradients[i],
                        beta_1, one_minus_beta_1, beta_2, one_minus_beta_2,
                        lr, eps);

        if (parameters_bf16_mirror)
            parameters_bf16_mirror[i] = __float2bfloat16(parameters[i]);
    }
}

void adam_update_cuda(
    const Index n,
    float* parameters,
    float* m,
    float* v,
    const float* gradients,
    const float beta_1,
    const float beta_2,
    const float learning_rate,
    const float epsilon,
    const float bias_correction_1,
    const float bias_correction_2,
    __nv_bfloat16* parameters_bf16_mirror)
{
    const float sqrt_bias_correction_2 = sqrtf(bias_correction_2);

    const float effective_lr = learning_rate * sqrt_bias_correction_2 / bias_correction_1;
    const float effective_eps = epsilon * sqrt_bias_correction_2;

    const bool aligned = are_float4_aligned(parameters, m, v, gradients)
        && is_bfloat162_aligned(parameters_bf16_mirror);

    launch_vec4_on(opennn::device::get_compute_stream(), n, aligned, adam_update_kernel,
                   parameters, m, v, gradients, parameters_bf16_mirror,
                   beta_1, 1.0f - beta_1, beta_2, 1.0f - beta_2,
                   effective_lr, effective_eps,
                   static_cast<const float*>(nullptr), static_cast<const float*>(nullptr));
}

__global__ void adam_prepare_kernel(int* __restrict__ step,
                                    float beta_1, float beta_2,
                                    float learning_rate, float epsilon,
                                    float* __restrict__ effective_lr,
                                    float* __restrict__ effective_eps)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const int t = (*step) + 1;
    *step = t;

    const float bias_correction_1 = 1.0f - powf(beta_1, float(t));
    const float bias_correction_2 = 1.0f - powf(beta_2, float(t));
    const float sqrt_bc2 = sqrtf(bias_correction_2);

    *effective_lr = learning_rate * sqrt_bc2 / bias_correction_1;
    *effective_eps = epsilon * sqrt_bc2;
}

void adam_update_capturable_cuda(
    const Index n,
    float* parameters, float* m, float* v, const float* gradients,
    const float beta_1, const float beta_2,
    const float learning_rate, const float epsilon,
    int* step_device, float* effective_lr_device, float* effective_eps_device,
    __nv_bfloat16* parameters_bf16_mirror,
    cudaStream_t stream)
{
    if (n == 0) return;
    if (stream == nullptr) stream = opennn::device::get_compute_stream();

    OPENNN_CUDA_LAUNCH(adam_prepare_kernel<<<1, 1, 0, stream>>>(
        step_device, beta_1, beta_2, learning_rate, epsilon,
        effective_lr_device, effective_eps_device));

    const bool aligned = are_float4_aligned(parameters, m, v, gradients)
        && is_bfloat162_aligned(parameters_bf16_mirror);

    launch_vec4_on(stream, n, aligned, adam_update_kernel,
                   parameters, m, v, gradients, parameters_bf16_mirror,
                   beta_1, 1.0f - beta_1, beta_2, 1.0f - beta_2,
                   0.0f, 0.0f,
                   static_cast<const float*>(effective_lr_device),
                   static_cast<const float*>(effective_eps_device));
}

__device__ __forceinline__ void sgd_update_one(
    float& p,
    float& v,
    float g,
    float lr,
    float momentum,
    bool nesterov)
{
    const float lr_g = lr * g;
    if (momentum <= 0.0f) { p -= lr_g; return; }

    const float v_new = fmaf(momentum, v, -lr_g);
    v = v_new;
    p += nesterov ? fmaf(momentum, v_new, -lr_g) : v_new;
}

__global__ void sgd_update_kernel(
    const int n_vec,
    const int n,
    float* __restrict__ parameters,
    float* __restrict__ velocity,
    const float* __restrict__ gradients,
    __nv_bfloat16* __restrict__ parameters_bf16_mirror,
    const float learning_rate_scalar,
    const float* __restrict__ learning_rate_device,
    const float momentum,
    const bool nesterov)
{
    const float lr = learning_rate_device ? *learning_rate_device : learning_rate_scalar;

    const Index tid = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = Index(blockDim.x) * gridDim.x;
    const bool has_momentum = momentum > 0.0f;

    float4* __restrict__ const       p4 = reinterpret_cast<float4*>(parameters);
    float4* __restrict__ const       v4 = reinterpret_cast<float4*>(velocity);
    const float4* __restrict__ const g4 = reinterpret_cast<const float4*>(gradients);
    __nv_bfloat162* __restrict__ const bf2 = reinterpret_cast<__nv_bfloat162*>(parameters_bf16_mirror);

    if (has_momentum)
    {
        for (Index i = tid; i < n_vec; i += stride)
        {
            float4 P = p4[i];
            float4 V = v4[i];
            const float4 G = g4[i];

            sgd_update_one(P.x, V.x, G.x, lr, momentum, nesterov);
            sgd_update_one(P.y, V.y, G.y, lr, momentum, nesterov);
            sgd_update_one(P.z, V.z, G.z, lr, momentum, nesterov);
            sgd_update_one(P.w, V.w, G.w, lr, momentum, nesterov);

            p4[i] = P;
            v4[i] = V;

            if (bf2)
            {
                bf2[i * 2 + 0] = __floats2bfloat162_rn(P.x, P.y);
                bf2[i * 2 + 1] = __floats2bfloat162_rn(P.z, P.w);
            }
        }

        const int tail_start = n_vec * 4;
        for (Index i = tail_start + tid; i < n; i += stride)
        {
            sgd_update_one(parameters[i], velocity[i], gradients[i],
                           lr, momentum, nesterov);
            if (parameters_bf16_mirror)
                parameters_bf16_mirror[i] = __float2bfloat16(parameters[i]);
        }
    }
    else
    {
        for (Index i = tid; i < n_vec; i += stride)
        {
            float4 P = p4[i];
            const float4 G = g4[i];

            P.x -= lr * G.x;
            P.y -= lr * G.y;
            P.z -= lr * G.z;
            P.w -= lr * G.w;

            p4[i] = P;

            if (bf2)
            {
                bf2[i * 2 + 0] = __floats2bfloat162_rn(P.x, P.y);
                bf2[i * 2 + 1] = __floats2bfloat162_rn(P.z, P.w);
            }
        }

        const int tail_start = n_vec * 4;
        for (Index i = tail_start + tid; i < n; i += stride)
        {
            parameters[i] -= lr * gradients[i];
            if (parameters_bf16_mirror)
                parameters_bf16_mirror[i] = __float2bfloat16(parameters[i]);
        }
    }
}

void sgd_update_cuda(
    const Index n,
    float* parameters,
    float* velocity,
    const float* gradients,
    const float learning_rate,
    const float momentum,
    const bool nesterov,
    __nv_bfloat16* parameters_bf16_mirror)
{
    if (learning_rate == 0.0f) return;

    const bool aligned = are_float4_aligned(parameters, gradients)
        && (velocity == nullptr || is_float4_aligned(velocity))
        && is_bfloat162_aligned(parameters_bf16_mirror);

    launch_vec4_on(opennn::device::get_compute_stream(), n, aligned, sgd_update_kernel,
                   parameters, velocity, gradients, parameters_bf16_mirror,
                   learning_rate, static_cast<const float*>(nullptr), momentum, nesterov);
}

__global__ void set_scalar_kernel(float* __restrict__ dst, const float value)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *dst = value;
}

void set_scalar_device_cuda(float* dst, const float value, cudaStream_t stream)
{
    if (stream == nullptr) stream = opennn::device::get_compute_stream();
    OPENNN_CUDA_LAUNCH(set_scalar_kernel<<<1, 1, 0, stream>>>(dst, value));
}

void sgd_update_capturable_cuda(
    const Index n,
    float* parameters,
    float* velocity,
    const float* gradients,
    const float* learning_rate_device,
    const float momentum,
    const bool nesterov,
    __nv_bfloat16* parameters_bf16_mirror,
    cudaStream_t stream)
{
    if (stream == nullptr) stream = opennn::device::get_compute_stream();

    const bool aligned = are_float4_aligned(parameters, gradients)
        && (velocity == nullptr || is_float4_aligned(velocity))
        && is_bfloat162_aligned(parameters_bf16_mirror);

    launch_vec4_on(stream, n, aligned, sgd_update_kernel,
                   parameters, velocity, gradients, parameters_bf16_mirror,
                   0.0f, learning_rate_device, momentum, nesterov);
}

__global__ void clip_apply_kernel(const int n,
                                  const float* __restrict__ squared_norm,
                                  const float max_norm,
                                  const float eps,
                                  float* __restrict__ gradient)
{
    const float norm = sqrtf(*squared_norm);
    if (norm <= max_norm) return;
    const float scale = max_norm / (norm + eps);

    const Index tid = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = Index(blockDim.x) * gridDim.x;
    for (Index i = tid; i < n; i += stride)
        gradient[i] *= scale;
}

void clip_gradient_norm_cuda(const Index n,
                             float* gradient,
                             const float* squared_norm,
                             const float max_norm,
                             const float eps)
{
    launch_elementwise(n, clip_apply_kernel, squared_norm, max_norm, eps, gradient);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
