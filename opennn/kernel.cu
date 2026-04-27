#include <cstdint>
#include <algorithm>

#include "kernel.cuh"

static constexpr int block_size = 256;

static inline int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

static inline int grid_size_for(int n)
{
    return ceil_div(n, block_size);
}

static inline int vector_work_size(int total, int n_vec, int vec_width)
{
    const int n_tail = total - n_vec * vec_width;
    return n_vec > n_tail ? n_vec : n_tail;
}

static inline bool is_float4_aligned(const void* ptr)
{
    return (reinterpret_cast<std::uintptr_t>(ptr) & 0xF) == 0;
}

template<typename... Ptrs>
static inline bool are_float4_aligned(const Ptrs*... ptrs)
{
    return (is_float4_aligned(ptrs) && ...);
}

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

// Per-element BCE forward term: tgt*log(out+eps) + (1-tgt)*log(1-out+eps).
// Reduced into a scalar by the host (cublasSasum on term_results).
template<typename T>
__global__ void binary_cross_entropy_kernel(const int n, float* __restrict__ term_results, const T* __restrict__ targets, const T* __restrict__ outputs, const float epsilon)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float out = static_cast<float>(outputs[i]);
        const float tgt = static_cast<float>(targets[i]);

        const float log_pos = logf(out + epsilon);
        const float log_neg = logf(1.0f - out + epsilon);

        term_results[i] = fmaf(tgt, log_pos - log_neg, log_neg);
    }
}

template<typename T>
void binary_cross_entropy_cuda(const Index n, float* term_results, const T* targets, const T* outputs, const float epsilon)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    binary_cross_entropy_kernel<T><<<grid_size_for(total), block_size>>>(
        total, term_results, targets, outputs, epsilon);
}

template void binary_cross_entropy_cuda<float>        (const Index, float*, const float*,         const float*,         const float);
template void binary_cross_entropy_cuda<__nv_bfloat16>(const Index, float*, const __nv_bfloat16*, const __nv_bfloat16*, const float);

template<typename T>
__device__ __forceinline__ void bce_gradient_one(T& d, T target, T output, float epsilon, float scaling_factor)
{
    const float out = static_cast<float>(output);
    const float tgt = static_cast<float>(target);
    d = static_cast<T>(((1.0f - tgt) / (1.0f - out + epsilon) - tgt / (out + epsilon)) * scaling_factor);
}

// BCE gradient: delta[i] = ((1 - tgt) / (1 - out + eps) - tgt / (out + eps)) * scale.
// Vec+tail layout (see adam_update_kernel for the convention).
template<typename T>
__global__ void binary_cross_entropy_gradient_kernel(
    const int n_vec, const int n,
    T* __restrict__ deltas,
    const T* __restrict__ targets,
    const T* __restrict__ outputs,
    const float epsilon, const float scaling_factor)
{
    constexpr int vec_width = 16 / sizeof(T);

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float4* const       d_v = reinterpret_cast<float4*>(deltas);
    const float4* const t_v = reinterpret_cast<const float4*>(targets);
    const float4* const o_v = reinterpret_cast<const float4*>(outputs);

    for (int i = tid; i < n_vec; i += stride)
    {
        float4 d_chunk;
        float4 t_chunk = t_v[i];
        float4 o_chunk = o_v[i];
        T* d_lanes = reinterpret_cast<T*>(&d_chunk);
        const T* t_lanes = reinterpret_cast<const T*>(&t_chunk);
        const T* o_lanes = reinterpret_cast<const T*>(&o_chunk);

        #pragma unroll
        for (int k = 0; k < vec_width; ++k)
            bce_gradient_one(d_lanes[k], t_lanes[k], o_lanes[k], epsilon, scaling_factor);

        d_v[i] = d_chunk;
    }

    const int tail_start = n_vec * vec_width;
    for (int i = tail_start + tid; i < n; i += stride)
        bce_gradient_one(deltas[i], targets[i], outputs[i], epsilon, scaling_factor);
}

template<typename T>
void binary_cross_entropy_gradient_cuda(const Index n, T* deltas, const T* targets, const T* outputs, const float epsilon, const float scaling_factor)
{
    if (n == 0) return;

    constexpr int vec_width = 16 / sizeof(T);
    const int total = static_cast<int>(n);

    const bool aligned = are_float4_aligned(deltas, targets, outputs);

    const int n_vec = aligned ? (total / vec_width) : 0;
    const int grid_size = grid_size_for(vector_work_size(total, n_vec, vec_width));

    binary_cross_entropy_gradient_kernel<T><<<grid_size, block_size>>>(
        n_vec, total, deltas, targets, outputs, epsilon, scaling_factor);
}

template void binary_cross_entropy_gradient_cuda<float>        (const Index, float*,         const float*,         const float*,         const float, const float);
template void binary_cross_entropy_gradient_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float, const float);

// Per-element multi-class CE forward term: tgt > 0 ? tgt*log(out+eps) : 0.
// Reduced into a scalar by the host.
template<typename T>
__global__ void multiple_cross_entropy_kernel(const int n, float* __restrict__ term_results, const T* __restrict__ targets, const T* __restrict__ outputs, const float epsilon)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float tgt = static_cast<float>(targets[i]);
        term_results[i] = (tgt > 0.0f) ? tgt * logf(static_cast<float>(outputs[i]) + epsilon) : 0.0f;
    }
}

template<typename T>
void multiple_cross_entropy_cuda(const Index n, float* term_results, const T* targets, const T* outputs, const float epsilon)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    multiple_cross_entropy_kernel<T><<<grid_size_for(total), block_size>>>(
        total, term_results, targets, outputs, epsilon);
}

template void multiple_cross_entropy_cuda<float>        (const Index, float*, const float*,         const float*,         const float);
template void multiple_cross_entropy_cuda<__nv_bfloat16>(const Index, float*, const __nv_bfloat16*, const __nv_bfloat16*, const float);

template<typename T>
__device__ __forceinline__ void mce_gradient_one(T& d, T target, T output, float scaling_factor)
{
    d = static_cast<T>((static_cast<float>(output) - static_cast<float>(target)) * scaling_factor);
}

// Multi-class CE gradient: delta[i] = (out[i] - tgt[i]) * scale.
// Vec+tail layout.
template<typename T>
__global__ void multiple_cross_entropy_gradient_kernel(
    const int n_vec, const int n,
    T* __restrict__ deltas,
    const T* __restrict__ targets,
    const T* __restrict__ outputs,
    const float scaling_factor)
{
    constexpr int vec_width = 16 / sizeof(T);

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float4* const       d_v = reinterpret_cast<float4*>(deltas);
    const float4* const t_v = reinterpret_cast<const float4*>(targets);
    const float4* const o_v = reinterpret_cast<const float4*>(outputs);

    for (int i = tid; i < n_vec; i += stride)
    {
        float4 d_chunk;
        float4 t_chunk = t_v[i];
        float4 o_chunk = o_v[i];
        T* d_lanes = reinterpret_cast<T*>(&d_chunk);
        const T* t_lanes = reinterpret_cast<const T*>(&t_chunk);
        const T* o_lanes = reinterpret_cast<const T*>(&o_chunk);

        #pragma unroll
        for (int k = 0; k < vec_width; ++k)
            mce_gradient_one(d_lanes[k], t_lanes[k], o_lanes[k], scaling_factor);

        d_v[i] = d_chunk;
    }

    const int tail_start = n_vec * vec_width;
    for (int i = tail_start + tid; i < n; i += stride)
        mce_gradient_one(deltas[i], targets[i], outputs[i], scaling_factor);
}

template<typename T>
void multiple_cross_entropy_gradient_cuda(const Index n, T* deltas, const T* targets, const T* outputs, const float scaling_factor)
{
    if (n == 0) return;

    constexpr int vec_width = 16 / sizeof(T);
    const int total = static_cast<int>(n);

    const bool aligned = are_float4_aligned(deltas, targets, outputs);

    const int n_vec = aligned ? (total / vec_width) : 0;
    const int grid_size = grid_size_for(vector_work_size(total, n_vec, vec_width));

    multiple_cross_entropy_gradient_kernel<T><<<grid_size, block_size>>>(
        n_vec, total, deltas, targets, outputs, scaling_factor);
}

template void multiple_cross_entropy_gradient_cuda<float>        (const Index, float*,         const float*,         const float*,         const float);
template void multiple_cross_entropy_gradient_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float);

// Per-element weighted squared error: (out - tgt)^2 * (tgt == 0 ? neg_w : pos_w).
// Reduced into a scalar by the host.
template<typename T>
__global__ void weighted_squared_error_kernel(const int n, float* __restrict__ term_results, const T* __restrict__ targets, const T* __restrict__ outputs, const float positives_weight, const float negatives_weight)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float tgt = static_cast<float>(targets[i]);
        const float diff = static_cast<float>(outputs[i]) - tgt;
        const float weight = (tgt == 0.0f) ? negatives_weight : positives_weight;

        term_results[i] = diff * diff * weight;
    }
}

template<typename T>
void weighted_squared_error_cuda(const Index n, float* term_results, const T* targets, const T* outputs, const float positives_weight, const float negatives_weight)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    weighted_squared_error_kernel<T><<<grid_size_for(total), block_size>>>(
        total, term_results, targets, outputs, positives_weight, negatives_weight);
}

template void weighted_squared_error_cuda<float>        (const Index, float*, const float*,         const float*,         const float, const float);
template void weighted_squared_error_cuda<__nv_bfloat16>(const Index, float*, const __nv_bfloat16*, const __nv_bfloat16*, const float, const float);

template<typename T>
__device__ __forceinline__ void wse_gradient_one(T& d, T target, T output,
                                                 float positives_weight,
                                                 float negatives_weight,
                                                 float scaling_factor)
{
    const float tgt = static_cast<float>(target);
    const float diff = static_cast<float>(output) - tgt;
    const float weight = (tgt == 0.0f) ? negatives_weight : positives_weight;
    d = static_cast<T>(diff * weight * scaling_factor);
}

// Weighted squared-error gradient: delta[i] = (out - tgt) * weight * scale,
// with weight = (tgt == 0 ? neg_w : pos_w). Vec+tail layout.
template<typename T>
__global__ void weighted_squared_error_gradient_kernel(
    const int n_vec, const int n,
    T* __restrict__ deltas,
    const T* __restrict__ targets,
    const T* __restrict__ outputs,
    const float positives_weight,
    const float negatives_weight,
    const float scaling_factor)
{
    constexpr int vec_width = 16 / sizeof(T);

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float4* const       d_v = reinterpret_cast<float4*>(deltas);
    const float4* const t_v = reinterpret_cast<const float4*>(targets);
    const float4* const o_v = reinterpret_cast<const float4*>(outputs);

    for (int i = tid; i < n_vec; i += stride)
    {
        float4 d_chunk;
        float4 t_chunk = t_v[i];
        float4 o_chunk = o_v[i];
        T* d_lanes = reinterpret_cast<T*>(&d_chunk);
        const T* t_lanes = reinterpret_cast<const T*>(&t_chunk);
        const T* o_lanes = reinterpret_cast<const T*>(&o_chunk);

        #pragma unroll
        for (int k = 0; k < vec_width; ++k)
            wse_gradient_one(d_lanes[k], t_lanes[k], o_lanes[k],
                             positives_weight, negatives_weight, scaling_factor);

        d_v[i] = d_chunk;
    }

    const int tail_start = n_vec * vec_width;
    for (int i = tail_start + tid; i < n; i += stride)
        wse_gradient_one(deltas[i], targets[i], outputs[i],
                         positives_weight, negatives_weight, scaling_factor);
}

template<typename T>
void weighted_squared_error_gradient_cuda(const Index n, T* deltas, const T* targets, const T* outputs, const float positives_weight, const float negatives_weight, const float scaling_factor)
{
    if (n == 0) return;

    constexpr int vec_width = 16 / sizeof(T);
    const int total = static_cast<int>(n);

    const bool aligned = are_float4_aligned(deltas, targets, outputs);

    const int n_vec = aligned ? (total / vec_width) : 0;
    const int grid_size = grid_size_for(vector_work_size(total, n_vec, vec_width));

    weighted_squared_error_gradient_kernel<T><<<grid_size, block_size>>>(
        n_vec, total, deltas, targets, outputs, positives_weight, negatives_weight, scaling_factor);
}

template void weighted_squared_error_gradient_cuda<float>        (const Index, float*,         const float*,         const float*,         const float, const float, const float);
template void weighted_squared_error_gradient_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float, const float, const float);

// Per-token CE for [batch, seq, vocab] outputs. Writes per-token error,
// valid-token mask (target_class > 0), and correct-prediction mask to
// host-allocated FP32 scratch; host reduces all three with cublasSasum.
template<typename T>
__global__ void cross_entropy_3d_multiple_forward_kernel(const int total_tokens,
                                                         const int vocab_size,
                                                         const T* __restrict__ outputs,
                                                         const float* __restrict__ targets,
                                                         float* __restrict__ errors,
                                                         float* __restrict__ valid_mask,
                                                         float* __restrict__ correct_mask,
                                                         const float epsilon)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_tokens; idx += blockDim.x * gridDim.x)
    {
        const int target_class = static_cast<int>(targets[idx]);
        const bool valid = target_class > 0 && target_class < vocab_size;

        errors[idx] = valid ? -logf(static_cast<float>(outputs[idx * vocab_size + target_class]) + epsilon) : 0.0f;
        if (valid_mask) valid_mask[idx] = valid ? 1.0f : 0.0f;

        if (correct_mask)
        {
            float best_match = 0.0f;
            if (valid)
            {
                const T* row = outputs + idx * vocab_size;
                float best_value = static_cast<float>(row[0]);
                int best_index = 0;
                for (int k = 1; k < vocab_size; ++k)
                {
                    const float value = static_cast<float>(row[k]);
                    if (value > best_value) { best_value = value; best_index = k; }
                }
                best_match = (best_index == target_class) ? 1.0f : 0.0f;
            }
            correct_mask[idx] = best_match;
        }
    }
}

template<typename T>
void cross_entropy_3d_multiple_forward_cuda(const Index n,
                                            const int vocab_size,
                                            const T* outputs,
                                            const float* targets,
                                            float* errors,
                                            float* valid_mask,
                                            float* correct_mask,
                                            const float epsilon)
{
    if(n == 0) return;

    const int total = static_cast<int>(n);

    cross_entropy_3d_multiple_forward_kernel<T><<<grid_size_for(total), block_size>>>(
        total, vocab_size, outputs, targets, errors, valid_mask, correct_mask, epsilon);
}

template void cross_entropy_3d_multiple_forward_cuda<float>        (const Index, const int, const float*,         const float*, float*, float*, float*, const float);
template void cross_entropy_3d_multiple_forward_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, float*, float*, float*, const float);

// CE gradient for [batch, seq, vocab]: delta = (output - one_hot(target)) * scale,
// zero where target_class is invalid (<=0 or >=vocab).
template<typename T>
__global__ void cross_entropy_3d_multiple_backward_kernel(const int n,
                                                          const int vocab_size,
                                                          const T* __restrict__ outputs,
                                                          const float* __restrict__ targets,
                                                          T* __restrict__ output_deltas,
                                                          const float scale_factor)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        const int token_idx = idx / vocab_size;
        const int class_idx = idx % vocab_size;
        const int target_class = static_cast<int>(targets[token_idx]);

        if (target_class <= 0 || target_class >= vocab_size)
        {
            output_deltas[idx] = static_cast<T>(0.0f);
            continue;
        }

        output_deltas[idx] = static_cast<T>((static_cast<float>(outputs[idx]) - (class_idx == target_class ? 1.0f : 0.0f)) * scale_factor);
    }
}

template<typename T>
void cross_entropy_3d_multiple_backward_cuda(const Index n,
                                             const int vocab_size,
                                             const T* outputs,
                                             const float* targets,
                                             T* output_deltas,
                                             const float scale_factor)
{
    if(n == 0) return;

    const int total = static_cast<int>(n);

    cross_entropy_3d_multiple_backward_kernel<T><<<grid_size_for(total), block_size>>>(
        total, vocab_size, outputs, targets, output_deltas, scale_factor);
}

template void cross_entropy_3d_multiple_backward_cuda<float>        (const Index, const int, const float*,         const float*, float*,         const float);
template void cross_entropy_3d_multiple_backward_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, __nv_bfloat16*, const float);

template<typename T>
__device__ __forceinline__ void l1_gradient_one(T& d, T p, float weight)
{
    const float pf = static_cast<float>(p);
    const float s = (pf > 0.0f) ? 1.0f : ((pf < 0.0f) ? -1.0f : 0.0f);
    d = static_cast<T>(static_cast<float>(d) + weight * s);
}

// L1 regularisation gradient: deltas[i] += weight * sign(parameters[i]).
// Vec+tail layout. Accumulates onto existing deltas (does not overwrite).
template<typename T>
__global__ void l1_gradient_kernel(
    const int n_vec, const int n,
    T* __restrict__ deltas,
    const T* __restrict__ parameters,
    const float weight)
{
    constexpr int vec_width = 16 / sizeof(T);

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float4* const       d_v = reinterpret_cast<float4*>(deltas);
    const float4* const p_v = reinterpret_cast<const float4*>(parameters);

    for (int i = tid; i < n_vec; i += stride)
    {
        float4 d_chunk = d_v[i];
        float4 p_chunk = p_v[i];
        T* d_lanes = reinterpret_cast<T*>(&d_chunk);
        T* p_lanes = reinterpret_cast<T*>(&p_chunk);

        #pragma unroll
        for (int k = 0; k < vec_width; ++k)
            l1_gradient_one(d_lanes[k], p_lanes[k], weight);

        d_v[i] = d_chunk;
    }

    const int tail_start = n_vec * vec_width;
    for (int i = tail_start + tid; i < n; i += stride)
        l1_gradient_one(deltas[i], parameters[i], weight);
}

template<typename T>
void l1_gradient_cuda(const Index n, T* deltas, const T* parameters, const float weight)
{
    if (n == 0) return;

    constexpr int vec_width = 16 / sizeof(T);
    const int total = static_cast<int>(n);

    const bool aligned = are_float4_aligned(deltas, parameters);

    const int n_vec = aligned ? (total / vec_width) : 0;
    const int grid_size = grid_size_for(vector_work_size(total, n_vec, vec_width));

    l1_gradient_kernel<T><<<grid_size, block_size>>>(n_vec, total, deltas, parameters, weight);
}

template void l1_gradient_cuda<float>        (const Index, float*,         const float*,         const float);
template void l1_gradient_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const __nv_bfloat16*, const float);

// Per-feature clip: output[i] = clamp(input[i], lower[f], upper[f]) where f = i % features.
template<typename T>
__global__ void bounding_kernel(const int n, const int features,
                                const T* __restrict__ input,
                                const float* __restrict__ lower,
                                const float* __restrict__ upper,
                                T* __restrict__ output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int f = i % features;
        const float x = static_cast<float>(input[i]);
        output[i] = static_cast<T>(fminf(fmaxf(x, lower[f]), upper[f]));
    }
}

template<typename T>
void bounding_cuda(const Index n, const int features,
                   const T* input, const float* lower, const float* upper,
                   T* output)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    bounding_kernel<T><<<grid_size_for(total), block_size>>>(total, features, input, lower, upper, output);
}

template void bounding_cuda<float>        (const Index, const int, const float*,         const float*, const float*, float*);
template void bounding_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, const float*, __nv_bfloat16*);

#define SCALER_EPSILON 1e-7f

// Per-feature input scaling. Method per feature is encoded in `scalers` (cast from
// ScalerMethod enum stored as float): MinMax, MeanStd, StdDev, Logarithm, ImageMinMax.
template<typename T>
__global__ void scale_kernel(const int n, const int features,
                             const T* __restrict__ input,
                             const float* __restrict__ minimums,
                             const float* __restrict__ maximums,
                             const float* __restrict__ means,
                             const float* __restrict__ stds,
                             const float* __restrict__ scalers,
                             const float min_range, const float max_range,
                             T* __restrict__ output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int f = i % features;
        const int code = static_cast<int>(scalers[f]);
        const float x = static_cast<float>(input[i]);
        float y = x;

        switch(code)
        {
        case 1: // MinimumMaximum
            y = (x - minimums[f]) / ((maximums[f] - minimums[f]) + SCALER_EPSILON)
                * (max_range - min_range) + min_range;
            break;
        case 2: // MeanStandardDeviation
            y = (x - means[f]) / (stds[f] + SCALER_EPSILON);
            break;
        case 3: // StandardDeviation
            y = x / (stds[f] + SCALER_EPSILON);
            break;
        case 4: // Logarithm
            y = logf(x);
            break;
        case 5: // ImageMinMax
            y = x / 255.0f;
            break;
        default:
            break;
        }

        output[i] = static_cast<T>(y);
    }
}

template<typename T>
void scale_cuda(const Index n, const int features,
                const T* input,
                const float* minimums, const float* maximums,
                const float* means, const float* stds,
                const float* scalers,
                float min_range, float max_range,
                T* output)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    scale_kernel<T><<<grid_size_for(total), block_size>>>(total, features,
                                                          input, minimums, maximums, means, stds, scalers,
                                                          min_range, max_range, output);
}

template void scale_cuda<float>        (const Index, const int, const float*,         const float*, const float*, const float*, const float*, const float*, float, float, float*);
template void scale_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, const float*, const float*, const float*, const float*, float, float, __nv_bfloat16*);

// Inverse of scale_kernel: per-feature output unscaling.
template<typename T>
__global__ void unscale_kernel(const int n, const int features,
                               const T* __restrict__ input,
                               const float* __restrict__ minimums,
                               const float* __restrict__ maximums,
                               const float* __restrict__ means,
                               const float* __restrict__ stds,
                               const float* __restrict__ scalers,
                               const float min_range, const float max_range,
                               T* __restrict__ output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int f = i % features;
        const int code = static_cast<int>(scalers[f]);
        const float x = static_cast<float>(input[i]);
        float y = x;

        switch(code)
        {
        case 1: // MinimumMaximum
            y = (x - min_range) / (max_range - min_range)
                * (maximums[f] - minimums[f]) + minimums[f];
            break;
        case 2: // MeanStandardDeviation
            y = means[f] + x * stds[f];
            break;
        case 3: // StandardDeviation
            y = x * stds[f];
            break;
        case 4: // Logarithm
            y = expf(x);
            break;
        case 5: // ImageMinMax
            y = x * 255.0f;
            break;
        default:
            break;
        }

        output[i] = static_cast<T>(y);
    }
}

template<typename T>
void unscale_cuda(const Index n, const int features,
                  const T* input,
                  const float* minimums, const float* maximums,
                  const float* means, const float* stds,
                  const float* scalers,
                  float min_range, float max_range,
                  T* output)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    unscale_kernel<T><<<grid_size_for(total), block_size>>>(total, features,
                                                            input, minimums, maximums, means, stds, scalers,
                                                            min_range, max_range, output);
}

template void unscale_cuda<float>        (const Index, const int, const float*,         const float*, const float*, const float*, const float*, const float*, float, float, float*);
template void unscale_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, const float*, const float*, const float*, const float*, float, float, __nv_bfloat16*);

// Token embedding lookup: outputs[i] = scale * weights[token_id, dim] + positional_encoding.
// `inputs` carries integer token ids stored as float. Out-of-vocab tokens get 0.
// `scale_embedding` applies sqrt(D) scaling (Transformer convention).
template<typename T>
__global__ void embedding_forward_kernel(const int n, const float* __restrict__ inputs, const float* __restrict__ weights, const float* __restrict__ positional_encoding, T* __restrict__ outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding, const bool add_positional_encoding)
{
    const float scale = scale_embedding ? sqrtf(static_cast<float>(embedding_dimension)) : 1.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int token_idx = i / embedding_dimension;
        const int dim_idx = i % embedding_dimension;
        const int token_id = static_cast<int>(inputs[token_idx]);

        float val = (token_id >= 0 && token_id < vocabulary_size)
            ? scale * weights[token_id * embedding_dimension + dim_idx]
            : 0.0f;

        if (add_positional_encoding && positional_encoding != nullptr && token_id > 0)
        {
            const int seq_idx = token_idx % sequence_length;
            val += positional_encoding[seq_idx * embedding_dimension + dim_idx];
        }

        outputs[i] = static_cast<T>(val);
    }
}

template<typename T>
void embedding_forward_cuda(const Index n, const float* inputs, const float* weights, const float* positional_encoding, T* outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding, const bool add_positional_encoding)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    embedding_forward_kernel<T><<<grid_size_for(total), block_size>>>(
        total, inputs, weights, positional_encoding, outputs,
        sequence_length, embedding_dimension, vocabulary_size, scale_embedding, add_positional_encoding);
}

template void embedding_forward_cuda<float>        (const Index, const float*, const float*, const float*, float*,         const int, const int, const int, const bool, const bool);
template void embedding_forward_cuda<__nv_bfloat16>(const Index, const float*, const float*, const float*, __nv_bfloat16*, const int, const int, const int, const bool, const bool);

// Embedding weight gradient via atomicAdd into the vocabulary table.
// Padding tokens (id 0 or out-of-range) contribute nothing.
template<typename T>
__global__ void embedding_backward_kernel(const int n, const float* __restrict__ inputs, const T* __restrict__ output_deltas, float* __restrict__ weight_gradients, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    const float scale = scale_embedding ? sqrtf(static_cast<float>(embedding_dimension)) : 1.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int token_idx = i / embedding_dimension;
        const int dim_idx = i % embedding_dimension;
        const int token_id = static_cast<int>(inputs[token_idx]);

        if (token_id <= 0 || token_id >= vocabulary_size) continue;

        atomicAdd(&weight_gradients[token_id * embedding_dimension + dim_idx], scale * static_cast<float>(output_deltas[i]));
    }
}

template<typename T>
void embedding_backward_cuda(const Index n, const float* inputs, const T* output_deltas, float* weight_gradients, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    embedding_backward_kernel<T><<<grid_size_for(total), block_size>>>(
        total, inputs, output_deltas, weight_gradients,
        embedding_dimension, vocabulary_size, scale_embedding);
}

template void embedding_backward_cuda<float>        (const Index, const float*, const float*,         float*, const int, const int, const bool);
template void embedding_backward_cuda<__nv_bfloat16>(const Index, const float*, const __nv_bfloat16*, float*, const int, const int, const bool);

// [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim], scalar path.
template<typename T>
__global__ void split_heads_scalar_kernel(const int n, const T* __restrict__ in, T* __restrict__ out, const int S, const int H, const int D)
{
    // [Batch, Seq, Heads, HeadDim] -> [Batch, Heads, Seq, HeadDim]
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int d = i % D;
        const int h = (i / D) % H;
        const int s = (i / (D * H)) % S;
        const int b = i / (D * H * S);

        out[((b * H + h) * S + s) * D + d] = in[i];
    }
}

// split_heads using float4 lanes when head_dim*sizeof(T) is a multiple of 16 bytes.
// D_vec = head_dim / vec_width.
template<typename T>
__global__ void split_heads_vec_kernel(const int n_vec, const T* __restrict__ in, T* __restrict__ out, const int S, const int H, const int D_vec)
{
    const float4* const in_v  = reinterpret_cast<const float4*>(in);
    float4* const       out_v = reinterpret_cast<float4*>(out);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += blockDim.x * gridDim.x)
    {
        const int d = i % D_vec;
        const int h = (i / D_vec) % H;
        const int s = (i / (D_vec * H)) % S;
        const int b = i / (D_vec * H * S);

        out_v[((b * H + h) * S + s) * D_vec + d] = in_v[i];
    }
}

template<typename T>
void split_heads_cuda(const Index n, const T* in, T* out, const int S, const int H, const int D)
{
    if (n == 0) return;

    if ((static_cast<size_t>(D) * sizeof(T)) % 16 == 0)
    {
        const int vec_width = static_cast<int>(16 / sizeof(T));
        const int D_vec     = D / vec_width;
        const int n_vec     = static_cast<int>(n) / vec_width;
        split_heads_vec_kernel<T><<<grid_size_for(n_vec), block_size>>>(n_vec, in, out, S, H, D_vec);
    }
    else
    {
        const int total = static_cast<int>(n);
        split_heads_scalar_kernel<T><<<grid_size_for(total), block_size>>>(total, in, out, S, H, D);
    }
}

template void split_heads_cuda<float>        (const Index, const float*,         float*,         const int, const int, const int);
template void split_heads_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const int, const int, const int);

// Inverse of split_heads: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim].
template<typename T>
__global__ void merge_heads_scalar_kernel(const int n, const T* __restrict__ in, T* __restrict__ out, const int S, const int H, const int D)
{
    // [Batch, Heads, Seq, HeadDim] -> [Batch, Seq, Heads, HeadDim]
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int d = i % D;
        const int s = (i / D) % S;
        const int h = (i / (D * S)) % H;
        const int b = i / (D * S * H);

        out[((b * S + s) * H + h) * D + d] = in[i];
    }
}

// merge_heads using float4 lanes when head_dim*sizeof(T) is a multiple of 16 bytes.
template<typename T>
__global__ void merge_heads_vec_kernel(const int n_vec, const T* __restrict__ in, T* __restrict__ out, const int S, const int H, const int D_vec)
{
    const float4* const in_v  = reinterpret_cast<const float4*>(in);
    float4* const       out_v = reinterpret_cast<float4*>(out);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += blockDim.x * gridDim.x)
    {
        const int d = i % D_vec;
        const int s = (i / D_vec) % S;
        const int h = (i / (D_vec * S)) % H;
        const int b = i / (D_vec * S * H);

        out_v[((b * S + s) * H + h) * D_vec + d] = in_v[i];
    }
}

template<typename T>
void merge_heads_cuda(const Index n, const T* in, T* out, const int S, const int H, const int D)
{
    if (n == 0) return;

    if ((static_cast<size_t>(D) * sizeof(T)) % 16 == 0)
    {
        const int vec_width = static_cast<int>(16 / sizeof(T));
        const int D_vec     = D / vec_width;
        const int n_vec     = static_cast<int>(n) / vec_width;
        merge_heads_vec_kernel<T><<<grid_size_for(n_vec), block_size>>>(n_vec, in, out, S, H, D_vec);
    }
    else
    {
        const int total = static_cast<int>(n);
        merge_heads_scalar_kernel<T><<<grid_size_for(total), block_size>>>(total, in, out, S, H, D);
    }
}

template void merge_heads_cuda<float>        (const Index, const float*,         float*,         const int, const int, const int);
template void merge_heads_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const int, const int, const int);

// Per-token validity for attention: padding_mask[i] = 1 if all embedding dims of
// token i are ~0 (treated as PAD), else 0.
template<typename T>
__global__ void padding_mask_kernel(const int num_tokens, const T* __restrict__ source_input, T* __restrict__ padding_mask, const int embedding_dimension)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_tokens; i += blockDim.x * gridDim.x)
    {
        const T* token = source_input + i * embedding_dimension;
        bool is_pad = true;
        for (int e = 0; e < embedding_dimension; ++e)
            if (fabsf(static_cast<float>(token[e])) > 1e-7f) { is_pad = false; break; }
        padding_mask[i] = static_cast<T>(is_pad ? 1.0f : 0.0f);
    }
}

// Apply padding + optional causal mask to pre-softmax attention weights by
// setting masked positions to -1e9 (so softmax produces ~0). Operates on
// [batch, heads, seq_q, seq_k] flattened to n.
template<typename T>
__global__ void fused_masks_kernel(const int n, T* __restrict__ attention_weights, const T* __restrict__ padding_mask,
                                         const int heads_number, const int query_sequence_length,
                                         const int source_sequence_length, const bool use_causal_mask)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int sk = i % source_sequence_length;
        const int sq = (i / source_sequence_length) % query_sequence_length;
        const int b  = i / (source_sequence_length * query_sequence_length * heads_number);

        if((use_causal_mask && sk > sq) || static_cast<float>(padding_mask[b * source_sequence_length + sk]) > 0.5f)
            attention_weights[i] = static_cast<T>(-1e9f);
    }
}

template<typename T>
void attention_masks_cuda(const int batch_size, const int heads_number,
                          const int query_sequence_length, const int source_sequence_length,
                          const int embedding_dimension, const T* source_input,
                          T* attention_weights, T* padding_mask, const bool use_causal_mask)
{
    const int num_tokens = batch_size * source_sequence_length;
    if(num_tokens > 0)
        padding_mask_kernel<T><<<grid_size_for(num_tokens), block_size>>>(
            num_tokens, source_input, padding_mask, embedding_dimension);

    const int n = batch_size * heads_number * query_sequence_length * source_sequence_length;
    if(n > 0)
        fused_masks_kernel<T><<<grid_size_for(n), block_size>>>(
            n, attention_weights, padding_mask, heads_number,
            query_sequence_length, source_sequence_length, use_causal_mask);
}

template void attention_masks_cuda<float>        (int, int, int, int, int, const float*,         float*,         float*,         bool);
template void attention_masks_cuda<__nv_bfloat16>(int, int, int, int, int, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, bool);

// Max pooling over the seq axis of [batch, seq, features]. Saves the argmax
// position per (batch, feature) into `indices` for the backward pass.
template<typename T>
__global__ void max_pooling_3d_forward_kernel(const int n, const T* __restrict__ in, T* __restrict__ out, float* __restrict__ indices, const int S, const int F)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        const int f = idx % F;
        const int b = idx / F;

        float max_val = -1e20f;
        int max_idx = 0;

        for (int s = 0; s < S; ++s)
        {
            const float val = static_cast<float>(in[(b * S + s) * F + f]);
            if (val > max_val) { max_val = val; max_idx = s; }
        }

        out[idx] = static_cast<T>(max_val);
        if (indices != nullptr) indices[idx] = static_cast<float>(max_idx);
    }
}

template<typename T>
void max_pooling_3d_forward_cuda(const Index n, const T* in, T* out, float* indices, const int S, const int F)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    max_pooling_3d_forward_kernel<T><<<grid_size_for(total), block_size>>>(total, in, out, indices, S, F);
}

template void max_pooling_3d_forward_cuda<float>        (const Index, const float*,         float*,         float*, const int, const int);
template void max_pooling_3d_forward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, float*, const int, const int);

// Scatter delta back to argmax positions saved by max_pooling_3d_forward.
// Caller must zero-initialise in_gradient first (host does cudaMemset).
template<typename T>
__global__ void max_pooling_3d_backward_kernel(const int n, const T* __restrict__ delta, T* __restrict__ in_gradient, const float* __restrict__ indices, const int S, const int F)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        const int f = idx % F;
        const int b = idx / F;
        const int max_s = static_cast<int>(indices[idx]);

        in_gradient[(b * S + max_s) * F + f] = delta[idx];
    }
}

template<typename T>
void max_pooling_3d_backward_cuda(const Index n, const T* delta, T* in_gradient, const float* indices, const int S, const int F)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    max_pooling_3d_backward_kernel<T><<<grid_size_for(total), block_size>>>(total, delta, in_gradient, indices, S, F);
}

template void max_pooling_3d_backward_cuda<float>        (const Index, const float*,         float*,         const float*, const int, const int);
template void max_pooling_3d_backward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const float*, const int, const int);

namespace { float* pooling_scratch_ = nullptr; size_t pooling_scratch_size_ = 0; }

static float* get_pooling_scratch(size_t floats_needed)
{
    if (floats_needed > pooling_scratch_size_)
    {
        if (pooling_scratch_) cudaFree(pooling_scratch_);
        cudaMalloc(&pooling_scratch_, floats_needed * sizeof(float));
        pooling_scratch_size_ = floats_needed;
    }
    return pooling_scratch_;
}

// Helper for average pooling: writes per-token validity (1 if any feature is
// ~nonzero, else 0) and accumulates per-batch valid counts via atomicAdd.
template<typename T>
__global__ void pooling_3d_valid_mask_kernel(const int BS, const int S, const int F,
                                             const T* __restrict__ in,
                                             float* __restrict__ valid_mask,
                                             float* __restrict__ counts)
{
    const int bs = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs >= BS) return;

    const T* token = in + bs * F;
    bool valid = false;
    for (int f = 0; f < F; ++f)
        if (fabsf(static_cast<float>(token[f])) > 1e-7f) { valid = true; break; }

    valid_mask[bs] = valid ? 1.0f : 0.0f;
    if (valid) atomicAdd(&counts[bs / S], 1.0f);
}

// Average pooling over the seq axis, masking invalid (~zero) tokens. Output
// is sum-of-valid-tokens / valid_count; zero rows when no valid tokens.
template<typename T>
__global__ void average_pooling_3d_forward_kernel(const int n, const T* __restrict__ in, T* __restrict__ out,
                                                  const int S, const int F,
                                                  const float* __restrict__ valid_mask,
                                                  const float* __restrict__ counts)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        const int f = idx % F;
        const int b = idx / F;

        const float count = counts[b];
        if (count == 0.0f) { out[idx] = static_cast<T>(0.0f); continue; }

        float sum = 0.0f;
        for (int s = 0; s < S; ++s)
        {
            const int bs = b * S + s;
            sum += valid_mask[bs] * static_cast<float>(in[bs * F + f]);
        }

        out[idx] = static_cast<T>(sum / count);
    }
}

template<typename T>
void average_pooling_3d_forward_cuda(const Index n, const T* in, T* out, const int S, const int F)
{
    if (n == 0) return;

    const int B  = static_cast<int>(n) / F;
    const int BS = B * S;

    float* scratch    = get_pooling_scratch(static_cast<size_t>(BS) + B);
    float* valid_mask = scratch;
    float* counts     = scratch + BS;
    cudaMemset(counts, 0, B * sizeof(float));

    pooling_3d_valid_mask_kernel<T><<<grid_size_for(BS), block_size>>>(BS, S, F, in, valid_mask, counts);

    const int total = static_cast<int>(n);
    average_pooling_3d_forward_kernel<T><<<grid_size_for(total), block_size>>>(total, in, out, S, F, valid_mask, counts);
}

template void average_pooling_3d_forward_cuda<float>        (const Index, const float*,         float*,         const int, const int);
template void average_pooling_3d_forward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const int, const int);

// Distribute output delta uniformly across the batch's valid tokens
// (delta_per_token = delta / valid_count, zero where invalid).
template<typename T>
__global__ void average_pooling_3d_backward_kernel(const int n, const T* __restrict__ delta, T* __restrict__ in_gradient,
                                                   const int S, const int F,
                                                   const float* __restrict__ valid_mask,
                                                   const float* __restrict__ counts)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        const int f = idx % F;
        const int b = idx / F;

        const float count = counts[b];
        if (count == 0.0f) continue;

        const float gradient_val = static_cast<float>(delta[idx]) / count;
        for (int s = 0; s < S; ++s)
        {
            const int bs = b * S + s;
            in_gradient[bs * F + f] = static_cast<T>(valid_mask[bs] * gradient_val);
        }
    }
}

template<typename T>
void average_pooling_3d_backward_cuda(const Index n, const T* in, const T* delta, T* in_gradient, const int S, const int F)
{
    if (n == 0) return;

    const int B  = static_cast<int>(n) / F;
    const int BS = B * S;

    float* scratch    = get_pooling_scratch(static_cast<size_t>(BS) + B);
    float* valid_mask = scratch;
    float* counts     = scratch + BS;
    cudaMemset(counts, 0, B * sizeof(float));

    pooling_3d_valid_mask_kernel<T><<<grid_size_for(BS), block_size>>>(BS, S, F, in, valid_mask, counts);

    const int total = static_cast<int>(n);
    average_pooling_3d_backward_kernel<T><<<grid_size_for(total), block_size>>>(total, delta, in_gradient, S, F, valid_mask, counts);
}

template void average_pooling_3d_backward_cuda<float>        (const Index, const float*,         const float*,         float*,         const int, const int);
template void average_pooling_3d_backward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, const int, const int);

__device__ __forceinline__ void warp_reduce_sum2(float& a, float& b)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        a += __shfl_down_sync(0xffffffff, a, offset);
        b += __shfl_down_sync(0xffffffff, b, offset);
    }
}

// Per-row layer norm, one block per row of [N, D]: y = gamma*(x-mean)/sqrt(var+eps) + beta.
// Saves mean and 1/sqrt(var+eps) for the backward pass. Variance computed from the
// online moments E[X], E[X^2] reduced via warp_reduce_sum2.
template<typename T>
__global__ void layernorm_forward_kernel(const int N, const int D, const T* __restrict__ X, T* __restrict__ Y, float* __restrict__ means, float* __restrict__ inv_vars, const float* __restrict__ gamma, const float* __restrict__ beta, const float eps)
{
    const int idx = blockIdx.x;
    if (idx >= N) return;

    const T* x_row = X + idx * D;
    T* y_row = Y + idx * D;

    // Per-thread accumulation. Variance derived as E[X^2] - E[X]^2 after the block reduction.
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float x = static_cast<float>(x_row[i]);
        local_sum    += x;
        local_sum_sq += x * x;
    }

    warp_reduce_sum2(local_sum, local_sum_sq);

    __shared__ float warp_sum[32];
    __shared__ float warp_sum_sq[32];
    __shared__ float s_mean;
    __shared__ float s_inv_var;

    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane == 0)
    {
        warp_sum[warp_id]    = local_sum;
        warp_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + 31) >> 5;
    if (warp_id == 0)
    {
        float s    = (threadIdx.x < num_warps) ? warp_sum[threadIdx.x]    : 0.0f;
        float s_sq = (threadIdx.x < num_warps) ? warp_sum_sq[threadIdx.x] : 0.0f;
        warp_reduce_sum2(s, s_sq);

        if (threadIdx.x == 0)
        {
            const float inv_D = 1.0f / static_cast<float>(D);
            const float mean = s * inv_D;
            const float variance = s_sq * inv_D - mean * mean;
            const float inv_var = rsqrtf(variance + eps);
            s_mean    = mean;
            s_inv_var = inv_var;
            means[idx]    = mean;
            inv_vars[idx] = inv_var;
        }
    }
    __syncthreads();

    const float mean    = s_mean;
    const float inv_var = s_inv_var;

    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float x_hat = (static_cast<float>(x_row[i]) - mean) * inv_var;
        y_row[i] = static_cast<T>(fmaf(gamma[i], x_hat, beta[i]));
    }
}

template<typename T>
void layernorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps)
{
    if (N == 0 || D == 0) return;

    int threads = 256;
    if (D <= 32) threads = 32;
    else if (D <= 64) threads = 64;
    else if (D <= 128) threads = 128;

    layernorm_forward_kernel<T><<<N, threads>>>(N, D, X, Y, means, inv_vars, gamma, beta, eps);
}

template void layernorm_forward_cuda<float>        (const int, const int, const float*,         float*,         float*, float*, const float*, const float*, const float);
template void layernorm_forward_cuda<__nv_bfloat16>(const int, const int, const __nv_bfloat16*, __nv_bfloat16*, float*, float*, const float*, const float*, const float);

// Per-row dX for layer norm, one block per row of [N, D]. Uses cached mean and
// inv_var saved by layernorm_forward_kernel.
template<typename T>
__global__ void layernorm_backward_kernel(const int N, const int D, const T* __restrict__ dY, const T* __restrict__ X, const float* __restrict__ means, const float* __restrict__ inv_vars, const float* __restrict__ gamma, T* __restrict__ dX)
{
    const int idx = blockIdx.x;
    if (idx >= N) return;

    const T* dy_row = dY + idx * D;
    const T* x_row = X + idx * D;
    T* dx_row = dX + idx * D;

    const float mean = means[idx];
    const float inv_var = inv_vars[idx];

    float local_sum_D      = 0.0f;
    float local_sum_D_xhat = 0.0f;

    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float d     = static_cast<float>(dy_row[i]) * gamma[i];
        const float x_hat = (static_cast<float>(x_row[i]) - mean) * inv_var;
        local_sum_D      += d;
        local_sum_D_xhat += d * x_hat;
    }

    warp_reduce_sum2(local_sum_D, local_sum_D_xhat);

    __shared__ float warp_sum_D[32];
    __shared__ float warp_sum_D_xhat[32];
    __shared__ float s_mean_D;
    __shared__ float s_mean_D_xhat;

    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane == 0)
    {
        warp_sum_D[warp_id]      = local_sum_D;
        warp_sum_D_xhat[warp_id] = local_sum_D_xhat;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + 31) >> 5;
    if (warp_id == 0)
    {
        float s  = (threadIdx.x < num_warps) ? warp_sum_D[threadIdx.x]      : 0.0f;
        float sx = (threadIdx.x < num_warps) ? warp_sum_D_xhat[threadIdx.x] : 0.0f;
        warp_reduce_sum2(s, sx);

        if (threadIdx.x == 0)
        {
            const float inv_D = 1.0f / static_cast<float>(D);
            s_mean_D      = s  * inv_D;
            s_mean_D_xhat = sx * inv_D;
        }
    }
    __syncthreads();

    const float mean_D      = s_mean_D;
    const float mean_D_xhat = s_mean_D_xhat;

    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float d     = static_cast<float>(dy_row[i]) * gamma[i];
        const float x_hat = (static_cast<float>(x_row[i]) - mean) * inv_var;
        dx_row[i] = static_cast<T>((d - mean_D - x_hat * mean_D_xhat) * inv_var);
    }
}

// Per-feature dGamma/dBeta accumulation for layer norm, one block per feature
// (column of [N, D]). Reduces across rows via warp shuffle.
template<typename T>
__global__ void layernorm_gamma_beta_gradient_kernel(const int N, const int D, const T* __restrict__ dY, const T* __restrict__ X, const float* __restrict__ means, const float* __restrict__ inv_vars, float* __restrict__ dGamma, float* __restrict__ dBeta)
{
    const int d = blockIdx.x;
    if (d >= D) return;

    float local_gamma = 0.0f;
    float local_beta  = 0.0f;

    for (int n = threadIdx.x; n < N; n += blockDim.x)
    {
        const float dy    = static_cast<float>(dY[n * D + d]);
        const float x_hat = (static_cast<float>(X[n * D + d]) - means[n]) * inv_vars[n];
        local_gamma += dy * x_hat;
        local_beta  += dy;
    }

    warp_reduce_sum2(local_gamma, local_beta);

    __shared__ float warp_gamma[32];
    __shared__ float warp_beta[32];

    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane == 0)
    {
        warp_gamma[warp_id] = local_gamma;
        warp_beta[warp_id]  = local_beta;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + 31) >> 5;
    if (warp_id == 0)
    {
        float g = (threadIdx.x < num_warps) ? warp_gamma[threadIdx.x] : 0.0f;
        float b = (threadIdx.x < num_warps) ? warp_beta[threadIdx.x]  : 0.0f;
        warp_reduce_sum2(g, b);

        if (threadIdx.x == 0)
        {
            dGamma[d] = g;
            dBeta[d]  = b;
        }
    }
}

template<typename T>
void layernorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* means, const float* inv_vars, const float* gamma, T* dX, float* dGamma, float* dBeta)
{
    if (N == 0 || D == 0) return;

    int dx_threads = 256;
    if (D <= 32) dx_threads = 32;
    else if (D <= 64) dx_threads = 64;
    else if (D <= 128) dx_threads = 128;

    layernorm_backward_kernel<T><<<N, dx_threads>>>(N, D, dY, X, means, inv_vars, gamma, dX);

    const int gb_threads = 256;
    layernorm_gamma_beta_gradient_kernel<T><<<D, gb_threads>>>(N, D, dY, X, means, inv_vars, dGamma, dBeta);
}

template void layernorm_backward_cuda<float>        (const int, const int, const float*,         const float*,         const float*, const float*, const float*, float*,         float*, float*);
template void layernorm_backward_cuda<__nv_bfloat16>(const int, const int, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, const float*, __nv_bfloat16*, float*, float*);
