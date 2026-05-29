#include "kernel_common.cuh"

template<typename T>
__global__ void binary_cross_entropy_kernel(const int n, float* __restrict__ term_results, const float* __restrict__ targets, const T* __restrict__ outputs, const float epsilon)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float out = static_cast<float>(outputs[i]);
        const float tgt = targets[i];

        const float log_pos = logf(out + epsilon);
        const float log_neg = logf(1.0f - out + epsilon);

        term_results[i] = fmaf(tgt, log_pos - log_neg, log_neg);
    }
}

template<typename T>
void binary_cross_entropy_cuda(const Index n, float* term_results, const float* targets, const T* outputs, const float epsilon)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    binary_cross_entropy_kernel<T><<<grid_size_for(total), block_size, 0, opennn::Backend::get_compute_stream()>>>(
        total, term_results, targets, outputs, epsilon);
}

template void binary_cross_entropy_cuda<float>        (const Index, float*, const float*, const float*,         const float);
template void binary_cross_entropy_cuda<__nv_bfloat16>(const Index, float*, const float*, const __nv_bfloat16*, const float);

template<typename T>
__global__ void binary_cross_entropy_gradient_kernel(
    const int n,
    T* __restrict__ deltas,
    const float* __restrict__ targets,
    const T* __restrict__ outputs,
    const float epsilon, const float scaling_factor)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride)
    {
        const float out = static_cast<float>(outputs[i]);
        const float tgt = targets[i];
        deltas[i] = static_cast<T>(((1.0f - tgt) / (1.0f - out + epsilon) - tgt / (out + epsilon)) * scaling_factor);
    }
}

template<typename T>
void binary_cross_entropy_gradient_cuda(const Index n, T* deltas, const float* targets, const T* outputs, const float epsilon, const float scaling_factor)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    binary_cross_entropy_gradient_kernel<T><<<grid_size_for(total), block_size, 0, opennn::Backend::get_compute_stream()>>>(
        total, deltas, targets, outputs, epsilon, scaling_factor);
}

template void binary_cross_entropy_gradient_cuda<float>        (const Index, float*,         const float*, const float*,         const float, const float);
template void binary_cross_entropy_gradient_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const float*, const __nv_bfloat16*, const float, const float);

template<typename T>
__global__ void multiple_cross_entropy_kernel(const int n, float* __restrict__ term_results, const float* __restrict__ targets, const T* __restrict__ outputs, const float epsilon)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float tgt = targets[i];
        term_results[i] = (tgt > 0.0f) ? tgt * logf(static_cast<float>(outputs[i]) + epsilon) : 0.0f;
    }
}

template<typename T>
void multiple_cross_entropy_cuda(const Index n, float* term_results, const float* targets, const T* outputs, const float epsilon)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    multiple_cross_entropy_kernel<T><<<grid_size_for(total), block_size, 0, opennn::Backend::get_compute_stream()>>>(
        total, term_results, targets, outputs, epsilon);
}

template void multiple_cross_entropy_cuda<float>        (const Index, float*, const float*, const float*,         const float);
template void multiple_cross_entropy_cuda<__nv_bfloat16>(const Index, float*, const float*, const __nv_bfloat16*, const float);

template<typename T>
__global__ void multiple_cross_entropy_gradient_kernel(
    const int n,
    T* __restrict__ deltas,
    const float* __restrict__ targets,
    const T* __restrict__ outputs,
    const float scaling_factor)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride)
        deltas[i] = static_cast<T>((static_cast<float>(outputs[i]) - targets[i]) * scaling_factor);
}

template<typename T>
void multiple_cross_entropy_gradient_cuda(const Index n, T* deltas, const float* targets, const T* outputs, const float scaling_factor)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    multiple_cross_entropy_gradient_kernel<T><<<grid_size_for(total), block_size, 0, opennn::Backend::get_compute_stream()>>>(
        total, deltas, targets, outputs, scaling_factor);
}

template void multiple_cross_entropy_gradient_cuda<float>        (const Index, float*,         const float*, const float*,         const float);
template void multiple_cross_entropy_gradient_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const float*, const __nv_bfloat16*, const float);

template<typename T>
__global__ void weighted_squared_error_kernel(const int n, float* __restrict__ term_results, const float* __restrict__ targets, const T* __restrict__ outputs, const float positives_weight, const float negatives_weight)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float tgt = targets[i];
        const float diff = static_cast<float>(outputs[i]) - tgt;
        const float weight = (tgt == 0.0f) ? negatives_weight : positives_weight;

        term_results[i] = diff * diff * weight;
    }
}

template<typename T>
void weighted_squared_error_cuda(const Index n, float* term_results, const float* targets, const T* outputs, const float positives_weight, const float negatives_weight)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    weighted_squared_error_kernel<T><<<grid_size_for(total), block_size, 0, opennn::Backend::get_compute_stream()>>>(
        total, term_results, targets, outputs, positives_weight, negatives_weight);
}

template void weighted_squared_error_cuda<float>        (const Index, float*, const float*, const float*,         const float, const float);
template void weighted_squared_error_cuda<__nv_bfloat16>(const Index, float*, const float*, const __nv_bfloat16*, const float, const float);

template<typename T>
__global__ void weighted_squared_error_gradient_kernel(
    const int n,
    T* __restrict__ deltas,
    const float* __restrict__ targets,
    const T* __restrict__ outputs,
    const float positives_weight,
    const float negatives_weight,
    const float scaling_factor)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride)
    {
        const float tgt = targets[i];
        const float diff = static_cast<float>(outputs[i]) - tgt;
        const float weight = (tgt == 0.0f) ? negatives_weight : positives_weight;
        deltas[i] = static_cast<T>(diff * weight * scaling_factor);
    }
}

template<typename T>
void weighted_squared_error_gradient_cuda(const Index n, T* deltas, const float* targets, const T* outputs, const float positives_weight, const float negatives_weight, const float scaling_factor)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    weighted_squared_error_gradient_kernel<T><<<grid_size_for(total), block_size, 0, opennn::Backend::get_compute_stream()>>>(
        total, deltas, targets, outputs, positives_weight, negatives_weight, scaling_factor);
}

template void weighted_squared_error_gradient_cuda<float>        (const Index, float*,         const float*, const float*,         const float, const float, const float);
template void weighted_squared_error_gradient_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const float*, const __nv_bfloat16*, const float, const float, const float);

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
    if (n == 0) return;

    const int total = static_cast<int>(n);

    cross_entropy_3d_multiple_forward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::Backend::get_compute_stream()>>>(
        total, vocab_size, outputs, targets, errors, valid_mask, correct_mask, epsilon);
}

template void cross_entropy_3d_multiple_forward_cuda<float>        (const Index, const int, const float*,         const float*, float*, float*, float*, const float);
template void cross_entropy_3d_multiple_forward_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, float*, float*, float*, const float);

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
        const int token_index = idx / vocab_size;
        const int class_index = idx % vocab_size;
        const int target_class = static_cast<int>(targets[token_index]);

        if (target_class <= 0 || target_class >= vocab_size)
        {
            output_deltas[idx] = static_cast<T>(0.0f);
            continue;
        }

        output_deltas[idx] = static_cast<T>((static_cast<float>(outputs[idx]) - (class_index == target_class ? 1.0f : 0.0f)) * scale_factor);
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
    if (n == 0) return;

    const int total = static_cast<int>(n);

    cross_entropy_3d_multiple_backward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::Backend::get_compute_stream()>>>(
        total, vocab_size, outputs, targets, output_deltas, scale_factor);
}

template void cross_entropy_3d_multiple_backward_cuda<float>        (const Index, const int, const float*,         const float*, float*,         const float);
template void cross_entropy_3d_multiple_backward_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, __nv_bfloat16*, const float);

template<typename T>
__global__ void cross_entropy_3d_multiple_backward_device_count_kernel(const int n,
                                                                       const int vocab_size,
                                                                       const T* __restrict__ outputs,
                                                                       const float* __restrict__ targets,
                                                                       T* __restrict__ output_deltas,
                                                                       const float* __restrict__ active_count_device)
{
    const float active_count = active_count_device ? active_count_device[0] : 0.0f;
    const float scale_factor = active_count > 0.0f ? 1.0f / active_count : 0.0f;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        const int token_index = idx / vocab_size;
        const int class_index = idx % vocab_size;
        const int target_class = static_cast<int>(targets[token_index]);

        if (target_class <= 0 || target_class >= vocab_size)
        {
            output_deltas[idx] = static_cast<T>(0.0f);
            continue;
        }

        output_deltas[idx] = static_cast<T>((static_cast<float>(outputs[idx]) - (class_index == target_class ? 1.0f : 0.0f)) * scale_factor);
    }
}

template<typename T>
void cross_entropy_3d_multiple_backward_device_count_cuda(const Index n,
                                                          const int vocab_size,
                                                          const T* outputs,
                                                          const float* targets,
                                                          T* output_deltas,
                                                          const float* active_count_device)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    cross_entropy_3d_multiple_backward_device_count_kernel<T><<<grid_size_for(total), block_size, 0, opennn::Backend::get_compute_stream()>>>(
        total, vocab_size, outputs, targets, output_deltas, active_count_device);
}

template void cross_entropy_3d_multiple_backward_device_count_cuda<float>        (const Index, const int, const float*,         const float*, float*,         const float*);
template void cross_entropy_3d_multiple_backward_device_count_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, __nv_bfloat16*, const float*);

__global__ void accumulate_scaled_metric_kernel(const float* __restrict__ value,
                                                const float scale,
                                                float* __restrict__ error_sum)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        error_sum[0] += value[0] * scale;
}

void accumulate_scaled_metric_cuda(const float* value, float scale, float* error_sum)
{
    accumulate_scaled_metric_kernel<<<1, 1, 0, opennn::Backend::get_compute_stream()>>>(value, scale, error_sum);
}

__global__ void accumulate_cross_entropy_3d_metrics_kernel(const float* __restrict__ values,
                                                           float* __restrict__ error_sum,
                                                           float* __restrict__ accuracy_sum)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const float loss_sum = values[0];
    const float active_count = values[1];
    const float correct_count = values[2];

    if (active_count > 0.0f)
    {
        error_sum[0] += loss_sum / active_count;
        if (accuracy_sum) accuracy_sum[0] += correct_count / active_count;
    }
}

void accumulate_cross_entropy_3d_metrics_cuda(const float* values,
                                              float* error_sum,
                                              float* accuracy_sum)
{
    accumulate_cross_entropy_3d_metrics_kernel<<<1, 1, 0, opennn::Backend::get_compute_stream()>>>(
        values, error_sum, accuracy_sum);
}

template<typename T>
__device__ __forceinline__ void l1_gradient_one(T& d, T p, float weight)
{
    const float pf = static_cast<float>(p);
    const float s = (pf > 0.0f) ? 1.0f : ((pf < 0.0f) ? -1.0f : 0.0f);
    d = static_cast<T>(static_cast<float>(d) + weight * s);
}

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

    l1_gradient_kernel<T><<<grid_size, block_size, 0, opennn::Backend::get_compute_stream()>>>(n_vec, total, deltas, parameters, weight);
}

template void l1_gradient_cuda<float>        (const Index, float*,         const float*,         const float);
template void l1_gradient_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const __nv_bfloat16*, const float);
