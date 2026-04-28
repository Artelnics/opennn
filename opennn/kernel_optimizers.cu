// Adam and SGD parameter-update kernels.
//
// Both follow the same vec+tail layout: caller computes n_vec from a runtime
// alignment check on the parameter pointers, then this TU's __global__ kernel
// vectorises [0, n_vec) as float4 chunks and falls back to scalar for the
// [n_vec*4, n) tail.

#include "kernel_common.cuh"

// Adam scalar update for a single element. Used by adam_update_kernel for
// both the float4 vector phase (called 4× per chunk) and the scalar tail.
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

// Adam optimizer step. Element-wise: m,v <- bias-corrected moments of g; then
// parameters[i] -= lr * m[i]/(sqrt(v[i]) + eps). Vectorised via float4 over
// [0, n_vec); the [n_vec*4, n) tail runs scalar. Caller decides n_vec based on
// pointer alignment (0 = scalar-only).
__global__ void adam_update_kernel(
    const int n_vec,
    const int n,
    float* __restrict__ parameters,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ gradients,
    const float beta_1,
    const float one_minus_beta_1,
    const float beta_2,
    const float one_minus_beta_2,
    const float lr,
    const float eps)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float4* __restrict__ const       p4 = reinterpret_cast<float4*>(parameters);
    float4* __restrict__ const       m4 = reinterpret_cast<float4*>(m);
    float4* __restrict__ const       v4 = reinterpret_cast<float4*>(v);
    const float4* __restrict__ const g4 = reinterpret_cast<const float4*>(gradients);

    for (int i = tid; i < n_vec; i += stride)
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
    }

    const int tail_start = n_vec * 4;
    for (int i = tail_start + tid; i < n; i += stride)
        adam_update_one(parameters[i], m[i], v[i], gradients[i],
                        beta_1, one_minus_beta_1, beta_2, one_minus_beta_2,
                        lr, eps);
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
    const float bias_correction_2)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);
    const float sqrt_bias_correction_2 = sqrtf(bias_correction_2);

    const float effective_lr  = learning_rate * sqrt_bias_correction_2 / bias_correction_1;
    const float effective_eps = epsilon * sqrt_bias_correction_2;

    const float one_minus_beta_1 = 1.0f - beta_1;
    const float one_minus_beta_2 = 1.0f - beta_2;

    const bool aligned = are_float4_aligned(parameters, m, v, gradients);

    const int n_vec = aligned ? (total / 4) : 0;
    const int grid_size = grid_size_for(vector_work_size(total, n_vec, 4));

    adam_update_kernel<<<grid_size, block_size>>>(
        n_vec,
        total,
        parameters,
        m,
        v,
        gradients,
        beta_1,
        one_minus_beta_1,
        beta_2,
        one_minus_beta_2,
        effective_lr,
        effective_eps);
}

// SGD scalar update for a single element. v is left untouched when momentum<=0
// (caller can skip the velocity write-back in that case).
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

// SGD step with optional momentum and Nesterov correction. Same vec+tail layout
// as adam_update_kernel. Velocity write-back is skipped when momentum<=0.
__global__ void sgd_update_kernel(
    const int n_vec,
    const int n,
    float* __restrict__ parameters,
    float* __restrict__ velocity,
    const float* __restrict__ gradients,
    const float learning_rate,
    const float momentum,
    const bool nesterov)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const bool has_momentum = momentum > 0.0f;

    float4* __restrict__ const       p4 = reinterpret_cast<float4*>(parameters);
    float4* __restrict__ const       v4 = reinterpret_cast<float4*>(velocity);
    const float4* __restrict__ const g4 = reinterpret_cast<const float4*>(gradients);

    for (int i = tid; i < n_vec; i += stride)
    {
        float4 P = p4[i];
        float4 V = v4[i];
        const float4 G = g4[i];

        sgd_update_one(P.x, V.x, G.x, learning_rate, momentum, nesterov);
        sgd_update_one(P.y, V.y, G.y, learning_rate, momentum, nesterov);
        sgd_update_one(P.z, V.z, G.z, learning_rate, momentum, nesterov);
        sgd_update_one(P.w, V.w, G.w, learning_rate, momentum, nesterov);

        p4[i] = P;

        if (has_momentum) v4[i] = V;
    }

    const int tail_start = n_vec * 4;
    for (int i = tail_start + tid; i < n; i += stride)
        sgd_update_one(parameters[i], velocity[i], gradients[i],
                       learning_rate, momentum, nesterov);
}

void sgd_update_cuda(
    const Index n,
    float* parameters,
    float* velocity,
    const float* gradients,
    const float learning_rate,
    const float momentum,
    const bool nesterov)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    const bool aligned = are_float4_aligned(parameters, velocity, gradients);

    const int n_vec = aligned ? (total / 4) : 0;
    const int grid_size = grid_size_for(vector_work_size(total, n_vec, 4));

    sgd_update_kernel<<<grid_size, block_size>>>(
        n_vec,
        total,
        parameters,
        velocity,
        gradients,
        learning_rate, momentum, nesterov);
}

// Element-wise FP32 → BF16 cast. Used to refresh the BF16 working copy of
// network parameters after each Adam step (master FP32 stays the source of
// truth, BF16 mirror feeds GEMMs that hit BF16 Tensor Cores).
//
// Vec phase: each thread reads one float4 (4 fp32) and writes one
// __nv_bfloat162-pair sequence (4 bf16 packed into 2× 32-bit stores via
// __nv_bfloat162). Scalar tail handles the remainder when `n` isn't a
// multiple of 4. Aligned-input path is enabled by `aligned_in_out` from the
// host wrapper; otherwise we fall back to scalar over the whole range.
__global__ void cast_fp32_to_bf16_kernel(const int n_vec,
                                         const int n,
                                         const float* __restrict__ src,
                                         __nv_bfloat16* __restrict__ dst)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const float4* __restrict__ const src4 = reinterpret_cast<const float4*>(src);
    __nv_bfloat162* __restrict__ const dst2 = reinterpret_cast<__nv_bfloat162*>(dst);

    for (int i = tid; i < n_vec; i += stride)
    {
        const float4 in = src4[i];
        // Pack each pair of floats into one __nv_bfloat162 (one 32-bit store).
        dst2[i * 2 + 0] = __floats2bfloat162_rn(in.x, in.y);
        dst2[i * 2 + 1] = __floats2bfloat162_rn(in.z, in.w);
    }

    const int tail_start = n_vec * 4;
    for (int i = tail_start + tid; i < n; i += stride)
        dst[i] = __float2bfloat16(src[i]);
}

void cast_fp32_to_bf16_cuda(const Index n, const float* src, __nv_bfloat16* dst)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);
    const bool aligned = are_float4_aligned(src);
    const int n_vec = aligned ? (total / 4) : 0;
    const int grid_size = grid_size_for(vector_work_size(total, n_vec, 4));

    cast_fp32_to_bf16_kernel<<<grid_size, block_size>>>(n_vec, total, src, dst);
}
