#include "kernel.cuh"
#include <cuda_bf16.h>

__global__ void adam_update_scalar_kernel(
    const int n,
    float* __restrict__ parameters,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ gradients,
    const float beta_1,
    const float one_minus_beta_1,
    const float beta_2,
    const float one_minus_beta_2,
    const float effective_learning_rate,
    const float effective_epsilon)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float gradient = gradients[i];

        const float new_m = fmaf(beta_1, m[i], one_minus_beta_1 * gradient);
        m[i] = new_m;

        const float new_v = fmaf(beta_2, v[i], one_minus_beta_2 * gradient * gradient);
        v[i] = new_v;

        parameters[i] -= effective_learning_rate * new_m / (sqrtf(new_v) + effective_epsilon);
    }
}

__global__ void adam_update_vec_kernel(
    const int n_vec,
    float4* __restrict__ parameters,
    float4* __restrict__ m,
    float4* __restrict__ v,
    const float4* __restrict__ gradients,
    const float beta_1,
    const float one_minus_beta_1,
    const float beta_2,
    const float one_minus_beta_2,
    const float effective_learning_rate,
    const float effective_epsilon)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += blockDim.x * gridDim.x)
    {
        float4 g  = gradients[i];
        float4 mi = m[i];
        float4 vi = v[i];
        float4 p  = parameters[i];

        float* gp  = reinterpret_cast<float*>(&g);
        float* mip = reinterpret_cast<float*>(&mi);
        float* vip = reinterpret_cast<float*>(&vi);
        float* pp  = reinterpret_cast<float*>(&p);

        #pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            const float gk = gp[k];
            mip[k] = fmaf(beta_1, mip[k], one_minus_beta_1 * gk);
            vip[k] = fmaf(beta_2, vip[k], one_minus_beta_2 * gk * gk);
            pp[k] -= effective_learning_rate * mip[k] / (sqrtf(vip[k]) + effective_epsilon);
        }

        m[i]          = mi;
        v[i]          = vi;
        parameters[i] = p;
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
    const float bias_correction_2)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);

    const float s = sqrtf(bias_correction_2);
    const float effective_lr  = learning_rate * s / bias_correction_1;
    const float effective_eps = epsilon * s;

    if ((total & 3) == 0)
    {
        const int n_vec = total / 4;
        const int grid_size = (n_vec + block_size - 1) / block_size;
        adam_update_vec_kernel<<<grid_size, block_size>>>(
            n_vec,
            reinterpret_cast<float4*>(parameters),
            reinterpret_cast<float4*>(m),
            reinterpret_cast<float4*>(v),
            reinterpret_cast<const float4*>(gradients),
            beta_1, 1.0f - beta_1,
            beta_2, 1.0f - beta_2,
            effective_lr, effective_eps);
    }
    else
    {
        const int grid_size = (total + block_size - 1) / block_size;
        adam_update_scalar_kernel<<<grid_size, block_size>>>(
            total, parameters, m, v, gradients,
            beta_1, 1.0f - beta_1,
            beta_2, 1.0f - beta_2,
            effective_lr, effective_eps);
    }
}

__global__ void sgd_update_scalar_kernel(
    const int n,
    float* __restrict__ parameters,
    float* __restrict__ velocity,
    const float* __restrict__ gradients,
    const float learning_rate,
    const float momentum,
    const bool nesterov)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float lr_g = learning_rate * gradients[i];

        if (momentum <= 0.0f)
        {
            parameters[i] -= lr_g;
            continue;
        }

        const float v_new = fmaf(momentum, velocity[i], -lr_g);
        velocity[i] = v_new;
        parameters[i] += nesterov ? fmaf(momentum, v_new, -lr_g) : v_new;
    }
}

__global__ void sgd_update_vec_kernel(
    const int n_vec,
    float4* __restrict__ parameters,
    float4* __restrict__ velocity,
    const float4* __restrict__ gradients,
    const float learning_rate,
    const float momentum,
    const bool nesterov)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += blockDim.x * gridDim.x)
    {
        float4 g    = gradients[i];
        float4 velv = velocity[i];
        float4 p    = parameters[i];

        float* gp  = reinterpret_cast<float*>(&g);
        float* vp  = reinterpret_cast<float*>(&velv);
        float* pp  = reinterpret_cast<float*>(&p);

        #pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            const float lr_g = learning_rate * gp[k];
            if (momentum <= 0.0f)
            {
                pp[k] -= lr_g;
            }
            else
            {
                const float v_new = fmaf(momentum, vp[k], -lr_g);
                vp[k] = v_new;
                pp[k] += nesterov ? fmaf(momentum, v_new, -lr_g) : v_new;
            }
        }

        if (momentum > 0.0f) velocity[i] = velv;
        parameters[i] = p;
    }
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

    const int block_size = 256;
    const int total = static_cast<int>(n);

    if ((total & 3) == 0)
    {
        const int n_vec = total / 4;
        const int grid_size = (n_vec + block_size - 1) / block_size;
        sgd_update_vec_kernel<<<grid_size, block_size>>>(
            n_vec,
            reinterpret_cast<float4*>(parameters),
            reinterpret_cast<float4*>(velocity),
            reinterpret_cast<const float4*>(gradients),
            learning_rate, momentum, nesterov);
    }
    else
    {
        const int grid_size = (total + block_size - 1) / block_size;
        sgd_update_scalar_kernel<<<grid_size, block_size>>>(
            total, parameters, velocity, gradients,
            learning_rate, momentum, nesterov);
    }
}

// term_results is always FP32 (reduction scratch); only inputs are dtype T.
// Matches the AMP pattern: compute in T, reduce in FP32.
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

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    binary_cross_entropy_kernel<T><<<grid_size, block_size>>>(
        total, term_results, targets, outputs, epsilon);
}

template void binary_cross_entropy_cuda<float>        (const Index, float*, const float*,         const float*,         const float);
template void binary_cross_entropy_cuda<__nv_bfloat16>(const Index, float*, const __nv_bfloat16*, const __nv_bfloat16*, const float);

template<typename T>
__global__ void binary_cross_entropy_gradient_scalar_kernel(const int n, T* __restrict__ deltas, const T* __restrict__ targets, const T* __restrict__ outputs, const float epsilon, const float scaling_factor)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float out = static_cast<float>(outputs[i]);
        const float tgt = static_cast<float>(targets[i]);

        deltas[i] = static_cast<T>(((1.0f - tgt) / (1.0f - out + epsilon) - tgt / (out + epsilon)) * scaling_factor);
    }
}

template<typename T>
__global__ void binary_cross_entropy_gradient_vec_kernel(const int n_vec, T* __restrict__ deltas, const T* __restrict__ targets, const T* __restrict__ outputs, const float epsilon, const float scaling_factor)
{
    constexpr int vec_width = 16 / sizeof(T);
    float4*       d_v = reinterpret_cast<float4*>(deltas);
    const float4* t_v = reinterpret_cast<const float4*>(targets);
    const float4* o_v = reinterpret_cast<const float4*>(outputs);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += blockDim.x * gridDim.x)
    {
        float4 d_chunk;
        float4 t_chunk = t_v[i];
        float4 o_chunk = o_v[i];
        T* d_lanes = reinterpret_cast<T*>(&d_chunk);
        const T* t_lanes = reinterpret_cast<const T*>(&t_chunk);
        const T* o_lanes = reinterpret_cast<const T*>(&o_chunk);

        #pragma unroll
        for (int k = 0; k < vec_width; ++k)
        {
            const float out = static_cast<float>(o_lanes[k]);
            const float tgt = static_cast<float>(t_lanes[k]);
            d_lanes[k] = static_cast<T>(((1.0f - tgt) / (1.0f - out + epsilon) - tgt / (out + epsilon)) * scaling_factor);
        }

        d_v[i] = d_chunk;
    }
}


template<typename T>
void binary_cross_entropy_gradient_cuda(const Index n, T* deltas, const T* targets, const T* outputs, const float epsilon, const float scaling_factor)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);

    if ((static_cast<size_t>(total) * sizeof(T)) % 16 == 0)
    {
        const int vec_width = static_cast<int>(16 / sizeof(T));
        const int n_vec     = total / vec_width;
        const int grid_size = (n_vec + block_size - 1) / block_size;
        binary_cross_entropy_gradient_vec_kernel<T><<<grid_size, block_size>>>(
            n_vec, deltas, targets, outputs, epsilon, scaling_factor);
    }
    else
    {
        const int grid_size = (total + block_size - 1) / block_size;
        binary_cross_entropy_gradient_scalar_kernel<T><<<grid_size, block_size>>>(
            total, deltas, targets, outputs, epsilon, scaling_factor);
    }
}

template void binary_cross_entropy_gradient_cuda<float>        (const Index, float*,         const float*,         const float*,         const float, const float);
template void binary_cross_entropy_gradient_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float, const float);

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

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    multiple_cross_entropy_kernel<T><<<grid_size, block_size>>>(
        total, term_results, targets, outputs, epsilon);
}

template void multiple_cross_entropy_cuda<float>        (const Index, float*, const float*,         const float*,         const float);
template void multiple_cross_entropy_cuda<__nv_bfloat16>(const Index, float*, const __nv_bfloat16*, const __nv_bfloat16*, const float);

template<typename T>
__global__ void multiple_cross_entropy_gradient_scalar_kernel(const int n, T* __restrict__ deltas, const T* __restrict__ targets, const T* __restrict__ outputs, const float scaling_factor)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        deltas[i] = static_cast<T>((static_cast<float>(outputs[i]) - static_cast<float>(targets[i])) * scaling_factor);
}

template<typename T>
__global__ void multiple_cross_entropy_gradient_vec_kernel(const int n_vec, T* __restrict__ deltas, const T* __restrict__ targets, const T* __restrict__ outputs, const float scaling_factor)
{
    constexpr int vec_width = 16 / sizeof(T);
    float4*       d_v = reinterpret_cast<float4*>(deltas);
    const float4* t_v = reinterpret_cast<const float4*>(targets);
    const float4* o_v = reinterpret_cast<const float4*>(outputs);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += blockDim.x * gridDim.x)
    {
        float4 d_chunk;
        float4 t_chunk = t_v[i];
        float4 o_chunk = o_v[i];
        T* d_lanes = reinterpret_cast<T*>(&d_chunk);
        const T* t_lanes = reinterpret_cast<const T*>(&t_chunk);
        const T* o_lanes = reinterpret_cast<const T*>(&o_chunk);

        #pragma unroll
        for (int k = 0; k < vec_width; ++k)
            d_lanes[k] = static_cast<T>((static_cast<float>(o_lanes[k]) - static_cast<float>(t_lanes[k])) * scaling_factor);

        d_v[i] = d_chunk;
    }
}


template<typename T>
void multiple_cross_entropy_gradient_cuda(const Index n, T* deltas, const T* targets, const T* outputs, const float scaling_factor)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);

    if ((static_cast<size_t>(total) * sizeof(T)) % 16 == 0)
    {
        const int vec_width = static_cast<int>(16 / sizeof(T));
        const int n_vec     = total / vec_width;
        const int grid_size = (n_vec + block_size - 1) / block_size;
        multiple_cross_entropy_gradient_vec_kernel<T><<<grid_size, block_size>>>(
            n_vec, deltas, targets, outputs, scaling_factor);
    }
    else
    {
        const int grid_size = (total + block_size - 1) / block_size;
        multiple_cross_entropy_gradient_scalar_kernel<T><<<grid_size, block_size>>>(
            total, deltas, targets, outputs, scaling_factor);
    }
}

template void multiple_cross_entropy_gradient_cuda<float>        (const Index, float*,         const float*,         const float*,         const float);
template void multiple_cross_entropy_gradient_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float);

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

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    weighted_squared_error_kernel<T><<<grid_size, block_size>>>(
        total, term_results, targets, outputs, positives_weight, negatives_weight);
}

template void weighted_squared_error_cuda<float>        (const Index, float*, const float*,         const float*,         const float, const float);
template void weighted_squared_error_cuda<__nv_bfloat16>(const Index, float*, const __nv_bfloat16*, const __nv_bfloat16*, const float, const float);

template<typename T>
__global__ void weighted_squared_error_gradient_scalar_kernel(const int n, T* __restrict__ deltas, const T* __restrict__ targets, const T* __restrict__ outputs, const float positives_weight, const float negatives_weight, const float scaling_factor)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float tgt = static_cast<float>(targets[i]);
        const float diff = static_cast<float>(outputs[i]) - tgt;
        const float weight = (tgt == 0.0f) ? negatives_weight : positives_weight;

        deltas[i] = static_cast<T>(diff * weight * scaling_factor);
    }
}

template<typename T>
__global__ void weighted_squared_error_gradient_vec_kernel(const int n_vec, T* __restrict__ deltas, const T* __restrict__ targets, const T* __restrict__ outputs, const float positives_weight, const float negatives_weight, const float scaling_factor)
{
    constexpr int vec_width = 16 / sizeof(T);
    float4*       d_v = reinterpret_cast<float4*>(deltas);
    const float4* t_v = reinterpret_cast<const float4*>(targets);
    const float4* o_v = reinterpret_cast<const float4*>(outputs);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += blockDim.x * gridDim.x)
    {
        float4 d_chunk;
        float4 t_chunk = t_v[i];
        float4 o_chunk = o_v[i];
        T* d_lanes = reinterpret_cast<T*>(&d_chunk);
        const T* t_lanes = reinterpret_cast<const T*>(&t_chunk);
        const T* o_lanes = reinterpret_cast<const T*>(&o_chunk);

        #pragma unroll
        for (int k = 0; k < vec_width; ++k)
        {
            const float tgt = static_cast<float>(t_lanes[k]);
            const float diff = static_cast<float>(o_lanes[k]) - tgt;
            const float weight = (tgt == 0.0f) ? negatives_weight : positives_weight;
            d_lanes[k] = static_cast<T>(diff * weight * scaling_factor);
        }

        d_v[i] = d_chunk;
    }
}


template<typename T>
void weighted_squared_error_gradient_cuda(const Index n, T* deltas, const T* targets, const T* outputs, const float positives_weight, const float negatives_weight, const float scaling_factor)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);

    if ((static_cast<size_t>(total) * sizeof(T)) % 16 == 0)
    {
        const int vec_width = static_cast<int>(16 / sizeof(T));
        const int n_vec     = total / vec_width;
        const int grid_size = (n_vec + block_size - 1) / block_size;
        weighted_squared_error_gradient_vec_kernel<T><<<grid_size, block_size>>>(
            n_vec, deltas, targets, outputs, positives_weight, negatives_weight, scaling_factor);
    }
    else
    {
        const int grid_size = (total + block_size - 1) / block_size;
        weighted_squared_error_gradient_scalar_kernel<T><<<grid_size, block_size>>>(
            total, deltas, targets, outputs, positives_weight, negatives_weight, scaling_factor);
    }
}

template void weighted_squared_error_gradient_cuda<float>        (const Index, float*,         const float*,         const float*,         const float, const float, const float);
template void weighted_squared_error_gradient_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float, const float, const float);

// `targets` holds integer class IDs stored as float — kept FP32 so BF16 conversion doesn't
// round large vocab indices. `outputs` is activation-dtype (T); `errors` and `valid_mask`
// are FP32 reduction scratch (AMP pattern: compute in T, reduce in FP32).
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

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    cross_entropy_3d_multiple_forward_kernel<T><<<grid_size, block_size>>>(
        total, vocab_size, outputs, targets, errors, valid_mask, correct_mask, epsilon);
}

template void cross_entropy_3d_multiple_forward_cuda<float>        (const Index, const int, const float*,         const float*, float*, float*, float*, const float);
template void cross_entropy_3d_multiple_forward_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, float*, float*, float*, const float);

template<typename T>
__global__ void cross_entropy_3d_multiple_backward_kernel(const int n,
                                                          const int vocab_size,
                                                          const T* __restrict__ outputs,
                                                          const float* __restrict__ targets,
                                                          T* __restrict__ output_gradients,
                                                          const float scale_factor)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        const int token_idx = idx / vocab_size;
        const int class_idx = idx % vocab_size;
        const int target_class = static_cast<int>(targets[token_idx]);

        if (target_class <= 0 || target_class >= vocab_size)
        {
            output_gradients[idx] = static_cast<T>(0.0f);
            continue;
        }

        output_gradients[idx] = static_cast<T>((static_cast<float>(outputs[idx]) - (class_idx == target_class ? 1.0f : 0.0f)) * scale_factor);
    }
}


template<typename T>
void cross_entropy_3d_multiple_backward_cuda(const Index n,
                                             const int vocab_size,
                                             const T* outputs,
                                             const float* targets,
                                             T* output_gradients,
                                             const float scale_factor)
{
    if(n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    cross_entropy_3d_multiple_backward_kernel<T><<<grid_size, block_size>>>(
        total, vocab_size, outputs, targets, output_gradients, scale_factor);
}

template void cross_entropy_3d_multiple_backward_cuda<float>        (const Index, const int, const float*,         const float*, float*,         const float);
template void cross_entropy_3d_multiple_backward_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, __nv_bfloat16*, const float);

template<typename T>
__global__ void l1_gradient_scalar_kernel(const int n, T* __restrict__ deltas, const T* __restrict__ parameters, const float weight)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float p = static_cast<float>(parameters[i]);
        const float s = (p > 0.0f) ? 1.0f : ((p < 0.0f) ? -1.0f : 0.0f);
        deltas[i] = static_cast<T>(static_cast<float>(deltas[i]) + weight * s);
    }
}

template<typename T>
__global__ void l1_gradient_vec_kernel(const int n_vec, T* __restrict__ deltas, const T* __restrict__ parameters, const float weight)
{
    constexpr int vec_width = 16 / sizeof(T);
    float4*       d_v = reinterpret_cast<float4*>(deltas);
    const float4* p_v = reinterpret_cast<const float4*>(parameters);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += blockDim.x * gridDim.x)
    {
        float4 d_chunk = d_v[i];
        float4 p_chunk = p_v[i];
        T* d_lanes = reinterpret_cast<T*>(&d_chunk);
        T* p_lanes = reinterpret_cast<T*>(&p_chunk);

        #pragma unroll
        for (int k = 0; k < vec_width; ++k)
        {
            const float p = static_cast<float>(p_lanes[k]);
            const float s = (p > 0.0f) ? 1.0f : ((p < 0.0f) ? -1.0f : 0.0f);
            d_lanes[k] = static_cast<T>(static_cast<float>(d_lanes[k]) + weight * s);
        }

        d_v[i] = d_chunk;
    }
}

template<typename T>
void l1_gradient_cuda(const Index n, T* deltas, const T* parameters, const float weight)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);

    if ((static_cast<size_t>(total) * sizeof(T)) % 16 == 0)
    {
        const int vec_width = static_cast<int>(16 / sizeof(T));
        const int n_vec     = total / vec_width;
        const int grid_size = (n_vec + block_size - 1) / block_size;
        l1_gradient_vec_kernel<T><<<grid_size, block_size>>>(n_vec, deltas, parameters, weight);
    }
    else
    {
        const int grid_size = (total + block_size - 1) / block_size;
        l1_gradient_scalar_kernel<T><<<grid_size, block_size>>>(total, deltas, parameters, weight);
    }
}

template void l1_gradient_cuda<float>        (const Index, float*,         const float*,         const float);
template void l1_gradient_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const __nv_bfloat16*, const float);

// Broadcasts a 1-D FP32 bias across a (total_rows × bias_dim) output. Output is
// dtype T (activation dtype). Used by Dense's combination — replaces cudnnAddTensor
// so FP32 biases can be added to BF16 outputs (AMP recipe).
template<typename T>
__global__ void add_bias_scalar_kernel(const int total_elements, T* __restrict__ output, const float* __restrict__ bias, const int bias_dim)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += blockDim.x * gridDim.x)
    {
        const int bias_idx = i % bias_dim;
        const float val = static_cast<float>(output[i]) + bias[bias_idx];
        output[i] = static_cast<T>(val);
    }
}

// Vec path is only safe when bias_dim is a multiple of vec_width — that keeps every
// 16-byte output chunk entirely inside one row, so the bias broadcast is still just
// vec_width consecutive FP32 loads starting at (linear_index % bias_dim).
template<typename T>
__global__ void add_bias_vec_kernel(const int n_vec, T* __restrict__ output, const float* __restrict__ bias, const int bias_dim)
{
    constexpr int vec_width = 16 / sizeof(T);
    float4* out_v = reinterpret_cast<float4*>(output);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += blockDim.x * gridDim.x)
    {
        const int bias_start = (i * vec_width) % bias_dim;

        float4 chunk = out_v[i];
        T* lanes = reinterpret_cast<T*>(&chunk);

        #pragma unroll
        for (int k = 0; k < vec_width; ++k)
            lanes[k] = static_cast<T>(static_cast<float>(lanes[k]) + bias[bias_start + k]);

        out_v[i] = chunk;
    }
}


template<typename T>
void add_bias_cuda(const Index n, T* output, const float* bias, const int bias_dim)
{
    if (n == 0) return;
    const int block_size = 256;
    const int total = static_cast<int>(n);
    constexpr int vec_width = 16 / sizeof(T);

    if ((bias_dim % vec_width) == 0)
    {
        const int n_vec = total / vec_width;
        const int grid_size = (n_vec + block_size - 1) / block_size;
        add_bias_vec_kernel<T><<<grid_size, block_size>>>(n_vec, output, bias, bias_dim);
    }
    else
    {
        const int grid_size = (total + block_size - 1) / block_size;
        add_bias_scalar_kernel<T><<<grid_size, block_size>>>(total, output, bias, bias_dim);
    }
}

template void add_bias_cuda<float>        (const Index, float*,         const float*, const int);
template void add_bias_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const float*, const int);

template<typename T>
__global__ void addition_scalar_kernel(const int n, const T* __restrict__ input1, const T* __restrict__ input2, T* __restrict__ output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        output[i] = static_cast<T>(static_cast<float>(input1[i]) + static_cast<float>(input2[i]));
}

template<typename T>
__global__ void addition_vec_kernel(const int n_vec, const T* __restrict__ input1, const T* __restrict__ input2, T* __restrict__ output)
{
    constexpr int vec_width = 16 / sizeof(T);
    const float4* in1_v = reinterpret_cast<const float4*>(input1);
    const float4* in2_v = reinterpret_cast<const float4*>(input2);
    float4*       out_v = reinterpret_cast<float4*>(output);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += blockDim.x * gridDim.x)
    {
        float4 c1 = in1_v[i];
        float4 c2 = in2_v[i];
        float4 co;
        const T* c1_lanes = reinterpret_cast<const T*>(&c1);
        const T* c2_lanes = reinterpret_cast<const T*>(&c2);
        T*       co_lanes = reinterpret_cast<T*>(&co);

        #pragma unroll
        for (int k = 0; k < vec_width; ++k)
            co_lanes[k] = static_cast<T>(static_cast<float>(c1_lanes[k]) + static_cast<float>(c2_lanes[k]));

        out_v[i] = co;
    }
}


template<typename T>
void addition_cuda(const Index n, const T* input1, const T* input2, T* output)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);

    if ((static_cast<size_t>(total) * sizeof(T)) % 16 == 0)
    {
        const int vec_width = static_cast<int>(16 / sizeof(T));
        const int n_vec     = total / vec_width;
        const int grid_size = (n_vec + block_size - 1) / block_size;
        addition_vec_kernel<T><<<grid_size, block_size>>>(n_vec, input1, input2, output);
    }
    else
    {
        const int grid_size = (total + block_size - 1) / block_size;
        addition_scalar_kernel<T><<<grid_size, block_size>>>(total, input1, input2, output);
    }
}

template void addition_cuda<float>        (const Index, const float*,         const float*,         float*);
template void addition_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);

// Template parameter T applies only to the `outputs` activation buffer. Inputs (token IDs),
// weights, and positional encoding stay FP32 (see get_forward_dtypes for Embedding).
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

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    embedding_forward_kernel<T> << <grid_size, block_size >> > (
        total, inputs, weights, positional_encoding, outputs,
        sequence_length, embedding_dimension, vocabulary_size, scale_embedding, add_positional_encoding);
}

template void embedding_forward_cuda<float>        (const Index, const float*, const float*, const float*, float*,         const int, const int, const int, const bool, const bool);
template void embedding_forward_cuda<__nv_bfloat16>(const Index, const float*, const float*, const float*, __nv_bfloat16*, const int, const int, const int, const bool, const bool);

// Template parameter T applies only to `output_gradients`. Inputs (IDs) and weight_gradients stay FP32
// — weight gradients are FP32 master and atomicAdd is safest on float.
template<typename T>
__global__ void embedding_backward_kernel(const int n, const float* __restrict__ inputs, const T* __restrict__ output_gradients, float* __restrict__ weight_gradients, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    const float scale = scale_embedding ? sqrtf(static_cast<float>(embedding_dimension)) : 1.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int token_idx = i / embedding_dimension;
        const int dim_idx = i % embedding_dimension;
        const int token_id = static_cast<int>(inputs[token_idx]);

        if (token_id <= 0 || token_id >= vocabulary_size) continue;

        atomicAdd(&weight_gradients[token_id * embedding_dimension + dim_idx], scale * static_cast<float>(output_gradients[i]));
    }
}


template<typename T>
void embedding_backward_cuda(const Index n, const float* inputs, const T* output_gradients, float* weight_gradients, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    embedding_backward_kernel<T> << <grid_size, block_size >> > (
        total, inputs, output_gradients, weight_gradients,
        embedding_dimension, vocabulary_size, scale_embedding);
}

template void embedding_backward_cuda<float>        (const Index, const float*, const float*,         float*, const int, const int, const bool);
template void embedding_backward_cuda<__nv_bfloat16>(const Index, const float*, const __nv_bfloat16*, float*, const int, const int, const bool);

// Vectorized split/merge use 16-byte (float4) loads/stores when D·sizeof(T) is
// 16-aligned — standard head_dims (32/64/128) satisfy this for both FP32 and BF16.
// Falls back to scalar for non-aligned D.

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

template<typename T>
__global__ void split_heads_vec_kernel(const int n_vec, const T* __restrict__ in, T* __restrict__ out, const int S, const int H, const int D_vec)
{
    const float4* in_v  = reinterpret_cast<const float4*>(in);
    float4*       out_v = reinterpret_cast<float4*>(out);

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

    const int block_size = 256;

    if ((static_cast<size_t>(D) * sizeof(T)) % 16 == 0)
    {
        const int vec_width = static_cast<int>(16 / sizeof(T));
        const int D_vec     = D / vec_width;
        const int n_vec     = static_cast<int>(n) / vec_width;
        const int grid_size = (n_vec + block_size - 1) / block_size;
        split_heads_vec_kernel<T><<<grid_size, block_size>>>(n_vec, in, out, S, H, D_vec);
    }
    else
    {
        const int total     = static_cast<int>(n);
        const int grid_size = (total + block_size - 1) / block_size;
        split_heads_scalar_kernel<T><<<grid_size, block_size>>>(total, in, out, S, H, D);
    }
}

template void split_heads_cuda<float>        (const Index, const float*,         float*,         const int, const int, const int);
template void split_heads_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const int, const int, const int);

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

template<typename T>
__global__ void merge_heads_vec_kernel(const int n_vec, const T* __restrict__ in, T* __restrict__ out, const int S, const int H, const int D_vec)
{
    const float4* in_v  = reinterpret_cast<const float4*>(in);
    float4*       out_v = reinterpret_cast<float4*>(out);

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

    const int block_size = 256;

    if ((static_cast<size_t>(D) * sizeof(T)) % 16 == 0)
    {
        const int vec_width = static_cast<int>(16 / sizeof(T));
        const int D_vec     = D / vec_width;
        const int n_vec     = static_cast<int>(n) / vec_width;
        const int grid_size = (n_vec + block_size - 1) / block_size;
        merge_heads_vec_kernel<T><<<grid_size, block_size>>>(n_vec, in, out, S, H, D_vec);
    }
    else
    {
        const int total     = static_cast<int>(n);
        const int grid_size = (total + block_size - 1) / block_size;
        merge_heads_scalar_kernel<T><<<grid_size, block_size>>>(total, in, out, S, H, D);
    }
}

template void merge_heads_cuda<float>        (const Index, const float*,         float*,         const int, const int, const int);
template void merge_heads_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const int, const int, const int);

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
    const int block_size = 256;

    const int num_tokens = batch_size * source_sequence_length;
    if(num_tokens > 0)
    {
        const int grid_size = (num_tokens + block_size - 1) / block_size;
        padding_mask_kernel<T><<<grid_size, block_size>>>(
            num_tokens, source_input, padding_mask, embedding_dimension);
    }

    const int n = batch_size * heads_number * query_sequence_length * source_sequence_length;

    if(n > 0)
    {
        const int grid_size = (n + block_size - 1) / block_size;
        fused_masks_kernel<T><<<grid_size, block_size>>>(
            n, attention_weights, padding_mask, heads_number,
            query_sequence_length, source_sequence_length, use_causal_mask);
    }
}

template void attention_masks_cuda<float>        (int, int, int, int, int, const float*,         float*,         float*,         bool);
template void attention_masks_cuda<__nv_bfloat16>(int, int, int, int, int, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, bool);

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

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    max_pooling_3d_forward_kernel<T><<<grid_size, block_size>>>(total, in, out, indices, S, F);
}

template void max_pooling_3d_forward_cuda<float>        (const Index, const float*,         float*,         float*, const int, const int);
template void max_pooling_3d_forward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, float*, const int, const int);

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

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    max_pooling_3d_backward_kernel<T><<<grid_size, block_size>>>(total, delta, in_gradient, indices, S, F);
}

template void max_pooling_3d_backward_cuda<float>        (const Index, const float*,         float*,         const float*, const int, const int);
template void max_pooling_3d_backward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const float*, const int, const int);

// Scratch pool for average-pooling: valid_mask[B*S] + counts[B] packed as floats.
// Persistent across calls, grown on demand — matches the pattern used by sum_abs_cuda.
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

// Stage 1: per-token validity. valid[b, s] = 1.0 if any |in[b, s, f]| > eps, else 0.0.
// Also writes per-batch counts via atomicAdd (caller must zero counts first).
// Previously this scan was duplicated inside the inner loop of both forward and backward,
// giving O(B·S·F²) total work; now it runs once at O(B·S·F).
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

// Stage 2 forward: out[b, f] = Σ_s valid_mask[b, s] · in[b, s, f] / counts[b].
// Inner loop is O(S); padding detection is already baked into valid_mask.
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

    const int block_size = 256;
    {
        const int grid = (BS + block_size - 1) / block_size;
        pooling_3d_valid_mask_kernel<T><<<grid, block_size>>>(BS, S, F, in, valid_mask, counts);
    }
    {
        const int total = static_cast<int>(n);
        const int grid = (total + block_size - 1) / block_size;
        average_pooling_3d_forward_kernel<T><<<grid, block_size>>>(total, in, out, S, F, valid_mask, counts);
    }
}

template void average_pooling_3d_forward_cuda<float>        (const Index, const float*,         float*,         const int, const int);
template void average_pooling_3d_forward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const int, const int);

// Stage 2 backward: in_gradient[b, s, f] = valid_mask[b, s] · delta[b, f] / counts[b].
// Writing the masked value (zero at padding positions) lets us drop the branch vs the
// original "check padding per element" pattern; in_gradient was zeroed by the caller so
// we can also skip the whole batch when counts[b] == 0.
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

    const int block_size = 256;
    {
        const int grid = (BS + block_size - 1) / block_size;
        pooling_3d_valid_mask_kernel<T><<<grid, block_size>>>(BS, S, F, in, valid_mask, counts);
    }
    {
        const int total = static_cast<int>(n);
        const int grid = (total + block_size - 1) / block_size;
        average_pooling_3d_backward_kernel<T><<<grid, block_size>>>(total, delta, in_gradient, S, F, valid_mask, counts);
    }
}

template void average_pooling_3d_backward_cuda<float>        (const Index, const float*,         const float*,         float*,         const int, const int);
template void average_pooling_3d_backward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, const int, const int);

// Reduce two floats across the 32 lanes of a warp via shuffle — no shared mem.
// After the loop, lane 0 holds (a + b) summed over all lanes; other lanes hold partial sums.
__device__ __forceinline__ void warp_reduce_sum2(float& a, float& b)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        a += __shfl_down_sync(0xffffffff, a, offset);
        b += __shfl_down_sync(0xffffffff, b, offset);
    }
}

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

// Computes dX only. One block per row; warp-shuffle reduction for per-row sums.
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


template<typename T>
__global__ void layernorm_gamma_beta_gradient_kernel(const int N, const int D, const T* __restrict__ dY, const T* __restrict__ X, const float* __restrict__ means, const float* __restrict__ inv_vars, float* __restrict__ dGamma, float* __restrict__ dBeta)
{
    // Computes dGamma and dBeta. One block per dim; reduces across all N rows via warp shuffles.

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

    layernorm_backward_kernel<T> << <N, dx_threads >> > (N, D, dY, X, means, inv_vars, gamma, dX);

    const int gb_threads = 256;
    layernorm_gamma_beta_gradient_kernel<T> << <D, gb_threads >> > (N, D, dY, X, means, inv_vars, dGamma, dBeta);
}

template void layernorm_backward_cuda<float>        (const int, const int, const float*,         const float*,         const float*, const float*, const float*, float*,         float*, float*);
template void layernorm_backward_cuda<__nv_bfloat16>(const int, const int, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, const float*, __nv_bfloat16*, float*, float*);
