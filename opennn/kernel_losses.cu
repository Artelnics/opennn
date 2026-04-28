// Loss-function kernels: BCE / MCE / WSE forward + gradient, the per-token
// 3D cross-entropy used by language-model heads, and the L1 regularisation
// gradient. All host wrappers are templated over T = float / __nv_bfloat16.

#include "kernel_common.cuh"

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

// Single-element BCE gradient. Used by binary_cross_entropy_gradient_kernel
// for both the float4 vector phase (vec_width × per chunk) and the scalar tail.
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

// Single-element multi-class CE gradient (vec+tail callee).
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

// Single-element weighted squared-error gradient (vec+tail callee).
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

// Single-element L1 gradient: d += weight * sign(p). Vec+tail callee.
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
