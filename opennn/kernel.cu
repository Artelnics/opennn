#include "kernel.cuh"
#include <cuda_bf16.h>

__global__ void adam_update_kernel(
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
    const int grid_size = (total + block_size - 1) / block_size;

    const float s = sqrtf(bias_correction_2);

    adam_update_kernel << <grid_size, block_size >> > (
        total, parameters, m, v, gradients,
        beta_1, 1.0f - beta_1,
        beta_2, 1.0f - beta_2,
        learning_rate * s / bias_correction_1,
        epsilon * s);
    CUDA_CHECK_KERNEL();
}

__global__ void sgd_update_kernel(
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
    const int grid_size = (total + block_size - 1) / block_size;

    sgd_update_kernel << <grid_size, block_size >> > (
        total, parameters, velocity, gradients,
        learning_rate, momentum, nesterov);
    CUDA_CHECK_KERNEL();
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
    CUDA_CHECK_KERNEL();
}

template void binary_cross_entropy_cuda<float>        (const Index, float*, const float*,         const float*,         const float);
template void binary_cross_entropy_cuda<__nv_bfloat16>(const Index, float*, const __nv_bfloat16*, const __nv_bfloat16*, const float);

template<typename T>
__global__ void binary_cross_entropy_gradient_kernel(const int n, T* __restrict__ deltas, const T* __restrict__ targets, const T* __restrict__ outputs, const float epsilon, const float scaling_factor)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float out = static_cast<float>(outputs[i]);
        const float tgt = static_cast<float>(targets[i]);

        deltas[i] = static_cast<T>(((1.0f - tgt) / (1.0f - out + epsilon) - tgt / (out + epsilon)) * scaling_factor);
    }
}


template<typename T>
void binary_cross_entropy_gradient_cuda(const Index n, T* deltas, const T* targets, const T* outputs, const float epsilon, const float scaling_factor)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    binary_cross_entropy_gradient_kernel<T><<<grid_size, block_size>>>(
        total, deltas, targets, outputs, epsilon, scaling_factor);
    CUDA_CHECK_KERNEL();
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
    CUDA_CHECK_KERNEL();
}

template void multiple_cross_entropy_cuda<float>        (const Index, float*, const float*,         const float*,         const float);
template void multiple_cross_entropy_cuda<__nv_bfloat16>(const Index, float*, const __nv_bfloat16*, const __nv_bfloat16*, const float);

template<typename T>
__global__ void multiple_cross_entropy_gradient_kernel(const int n, T* __restrict__ deltas, const T* __restrict__ targets, const T* __restrict__ outputs, const float scaling_factor)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        deltas[i] = static_cast<T>((static_cast<float>(outputs[i]) - static_cast<float>(targets[i])) * scaling_factor);
}


template<typename T>
void multiple_cross_entropy_gradient_cuda(const Index n, T* deltas, const T* targets, const T* outputs, const float scaling_factor)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    multiple_cross_entropy_gradient_kernel<T><<<grid_size, block_size>>>(
        total, deltas, targets, outputs, scaling_factor);
    CUDA_CHECK_KERNEL();
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
    CUDA_CHECK_KERNEL();
}

template void weighted_squared_error_cuda<float>        (const Index, float*, const float*,         const float*,         const float, const float);
template void weighted_squared_error_cuda<__nv_bfloat16>(const Index, float*, const __nv_bfloat16*, const __nv_bfloat16*, const float, const float);

template<typename T>
__global__ void weighted_squared_error_gradient_kernel(const int n, T* __restrict__ deltas, const T* __restrict__ targets, const T* __restrict__ outputs, const float positives_weight, const float negatives_weight, const float scaling_factor)
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
void weighted_squared_error_gradient_cuda(const Index n, T* deltas, const T* targets, const T* outputs, const float positives_weight, const float negatives_weight, const float scaling_factor)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    weighted_squared_error_gradient_kernel<T><<<grid_size, block_size>>>(
        total, deltas, targets, outputs, positives_weight, negatives_weight, scaling_factor);
    CUDA_CHECK_KERNEL();
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
                                                         const float epsilon)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_tokens; idx += blockDim.x * gridDim.x)
    {
        const int target_class = static_cast<int>(targets[idx]);
        const bool valid = target_class > 0 && target_class < vocab_size;

        errors[idx] = valid ? -logf(static_cast<float>(outputs[idx * vocab_size + target_class]) + epsilon) : 0.0f;
        if (valid_mask) valid_mask[idx] = valid ? 1.0f : 0.0f;
    }
}


template<typename T>
void cross_entropy_3d_multiple_forward_cuda(const Index n,
                                            const int vocab_size,
                                            const T* outputs,
                                            const float* targets,
                                            float* errors,
                                            float* valid_mask,
                                            const float epsilon)
{
    if(n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    cross_entropy_3d_multiple_forward_kernel<T><<<grid_size, block_size>>>(
        total, vocab_size, outputs, targets, errors, valid_mask, epsilon);
    CUDA_CHECK_KERNEL();
}

template void cross_entropy_3d_multiple_forward_cuda<float>        (const Index, const int, const float*,         const float*, float*, float*, const float);
template void cross_entropy_3d_multiple_forward_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, float*, float*, const float);

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
    CUDA_CHECK_KERNEL();
}

template void cross_entropy_3d_multiple_backward_cuda<float>        (const Index, const int, const float*,         const float*, float*,         const float);
template void cross_entropy_3d_multiple_backward_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, __nv_bfloat16*, const float);

template<typename T>
__global__ void l1_gradient_kernel(const int n, T* __restrict__ deltas, const T* __restrict__ parameters, const float weight)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float p = static_cast<float>(parameters[i]);
        const float s = (p > 0.0f) ? 1.0f : ((p < 0.0f) ? -1.0f : 0.0f);
        deltas[i] = static_cast<T>(static_cast<float>(deltas[i]) + weight * s);
    }
}

// Explicit instantiations — compile-check the BF16 path even though no caller uses it yet.

template<typename T>
void l1_gradient_cuda(const Index n, T* deltas, const T* parameters, const float weight)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    l1_gradient_kernel<T><<<grid_size, block_size>>>(total, deltas, parameters, weight);
    CUDA_CHECK_KERNEL();
}

template void l1_gradient_cuda<float>        (const Index, float*,         const float*,         const float);
template void l1_gradient_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const __nv_bfloat16*, const float);

// Broadcasts a 1-D FP32 bias across a (total_rows × bias_dim) output. Output is
// dtype T (activation dtype). Used by Dense's combination — replaces cudnnAddTensor
// so FP32 biases can be added to BF16 outputs (AMP recipe).
template<typename T>
__global__ void add_bias_kernel(const int total_elements, T* __restrict__ output, const float* __restrict__ bias, const int bias_dim)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += blockDim.x * gridDim.x)
    {
        const int bias_idx = i % bias_dim;
        const float val = static_cast<float>(output[i]) + bias[bias_idx];
        output[i] = static_cast<T>(val);
    }
}


template<typename T>
void add_bias_cuda(const Index n, T* output, const float* bias, const int bias_dim)
{
    if (n == 0) return;
    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;
    add_bias_kernel<T><<<grid_size, block_size>>>(total, output, bias, bias_dim);
    CUDA_CHECK_KERNEL();
}

template void add_bias_cuda<float>        (const Index, float*,         const float*, const int);
template void add_bias_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const float*, const int);

// Dtype-agnostic sum-of-absolutes. Reads dtype-T data, accumulates in FP32,
// returns FP32 scalar via atomicAdd across blocks. Needed for BF16/FP16 paths
// where cuBLAS has no asum variant. (FP32 callers stay on cublasSasum for perf.)
template<typename T>
__global__ void sum_abs_kernel(const int n, const T* __restrict__ data, float* __restrict__ result)
{
    __shared__ float shared[256];
    float acc = 0.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        acc += fabsf(static_cast<float>(data[i]));

    shared[threadIdx.x] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) atomicAdd(result, shared[0]);
}

template<typename T>
float sum_abs_cuda(const T* data, Index n)
{
    if (n == 0) return 0.0f;

    // Persistent scratch scalar — one cudaMalloc per process.
    static float* d_result = nullptr;
    if (!d_result) CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(float)));

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    sum_abs_kernel<T><<<grid_size, block_size>>>(total, data, d_result);
    CUDA_CHECK_KERNEL();

    float h_result = 0.0f;
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    return h_result;
}

template float sum_abs_cuda<__nv_bfloat16>(const __nv_bfloat16*, Index);

template<typename T>
__global__ void addition_kernel(const int n, const T* __restrict__ input1, const T* __restrict__ input2, T* __restrict__ output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        output[i] = static_cast<T>(static_cast<float>(input1[i]) + static_cast<float>(input2[i]));
}


template<typename T>
void addition_cuda(const Index n, const T* input1, const T* input2, T* output)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    addition_kernel<T><<<grid_size, block_size>>>(total, input1, input2, output);
    CUDA_CHECK_KERNEL();
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
    CUDA_CHECK_KERNEL();
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
    CUDA_CHECK_KERNEL();
}

template void embedding_backward_cuda<float>        (const Index, const float*, const float*,         float*, const int, const int, const bool);
template void embedding_backward_cuda<__nv_bfloat16>(const Index, const float*, const __nv_bfloat16*, float*, const int, const int, const bool);

template<typename T>
__global__ void split_heads_kernel(const int n, const T* __restrict__ in, T* __restrict__ out, const int S, const int H, const int D)
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
void split_heads_cuda(const Index n, const T* in, T* out, const int S, const int H, const int D)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    split_heads_kernel<T><<<grid_size, block_size>>>(total, in, out, S, H, D);
    CUDA_CHECK_KERNEL();
}

template void split_heads_cuda<float>        (const Index, const float*,         float*,         const int, const int, const int);
template void split_heads_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const int, const int, const int);

template<typename T>
__global__ void merge_heads_kernel(const int n, const T* __restrict__ in, T* __restrict__ out, const int S, const int H, const int D)
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
void merge_heads_cuda(const Index n, const T* in, T* out, const int S, const int H, const int D)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    merge_heads_kernel<T><<<grid_size, block_size>>>(total, in, out, S, H, D);
    CUDA_CHECK_KERNEL();
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
        CUDA_CHECK_KERNEL();
    }

    const int n = batch_size * heads_number * query_sequence_length * source_sequence_length;

    if(n > 0)
    {
        const int grid_size = (n + block_size - 1) / block_size;
        fused_masks_kernel<T><<<grid_size, block_size>>>(
            n, attention_weights, padding_mask, heads_number,
            query_sequence_length, source_sequence_length, use_causal_mask);
        CUDA_CHECK_KERNEL();
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
    CUDA_CHECK_KERNEL();
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
    CUDA_CHECK_KERNEL();
}

template void max_pooling_3d_backward_cuda<float>        (const Index, const float*,         float*,         const float*, const int, const int);
template void max_pooling_3d_backward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const float*, const int, const int);

template<typename T>
__global__ void average_pooling_3d_forward_kernel(const int n, const T* __restrict__ in, T* __restrict__ out, const int S, const int F)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        const int f = idx % F;
        const int b = idx / F;

        float sum = 0.0f;
        int valid_count = 0;

        for (int s = 0; s < S; ++s)
        {
            bool is_padding = true;
            for (int check_f = 0; check_f < F; ++check_f)
                if (fabsf(static_cast<float>(in[(b * S + s) * F + check_f])) > 1e-7f)
                {
                    is_padding = false;
                    break;
                }

            if (!is_padding)
            {
                sum += static_cast<float>(in[(b * S + s) * F + f]);
                ++valid_count;
            }
        }

        out[idx] = static_cast<T>((valid_count > 0) ? (sum / static_cast<float>(valid_count)) : 0.0f);
    }
}


template<typename T>
void average_pooling_3d_forward_cuda(const Index n, const T* in, T* out, const int S, const int F)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    average_pooling_3d_forward_kernel<T><<<grid_size, block_size>>>(total, in, out, S, F);
    CUDA_CHECK_KERNEL();
}

template void average_pooling_3d_forward_cuda<float>        (const Index, const float*,         float*,         const int, const int);
template void average_pooling_3d_forward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const int, const int);

template<typename T>
__global__ void average_pooling_3d_backward_kernel(const int n, const T* __restrict__ in, const T* __restrict__ delta, T* __restrict__ in_gradient, const int S, const int F)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        const int f = idx % F;
        const int b = idx / F;

        int valid_count = 0;
        for (int s = 0; s < S; ++s)
        {
            bool is_padding = true;
            for (int check_f = 0; check_f < F; ++check_f)
                if (fabsf(static_cast<float>(in[(b * S + s) * F + check_f])) > 1e-7f)
                {
                    is_padding = false;
                    break;
                }

            if (!is_padding)
                ++valid_count;
        }

        if (valid_count == 0) continue;

        const float gradient_val = static_cast<float>(delta[idx]) / static_cast<float>(valid_count);
        for (int s = 0; s < S; ++s)
        {
            bool is_padding = true;
            for (int check_f = 0; check_f < F; ++check_f)
                if (fabsf(static_cast<float>(in[(b * S + s) * F + check_f])) > 1e-7f) { is_padding = false; break; }
            if (!is_padding) in_gradient[(b * S + s) * F + f] = static_cast<T>(gradient_val);
        }
    }
}


template<typename T>
void average_pooling_3d_backward_cuda(const Index n, const T* in, const T* delta, T* in_gradient, const int S, const int F)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    average_pooling_3d_backward_kernel<T><<<grid_size, block_size>>>(total, in, delta, in_gradient, S, F);
    CUDA_CHECK_KERNEL();
}

template void average_pooling_3d_backward_cuda<float>        (const Index, const float*,         const float*,         float*,         const int, const int);
template void average_pooling_3d_backward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, const int, const int);

template<typename T>
__global__ void layernorm_forward_kernel(const int N, const int D, const T* __restrict__ X, T* __restrict__ Y, float* __restrict__ means, float* __restrict__ inv_vars, const float* __restrict__ gamma, const float* __restrict__ beta, const float eps)
{
    const int idx = blockIdx.x;
    if (idx >= N) return;

    const T* x_row = X + idx * D;
    T* y_row = Y + idx * D;

    // Single-pass accumulate sum and sum-of-squares, then derive variance as E[X^2] - E[X]^2.
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float x = static_cast<float>(x_row[i]);
        sum += x;
        sum_sq += x * x;
    }

    __shared__ float shared_sum[256];
    __shared__ float shared_sum_sq[256];
    shared_sum[threadIdx.x] = sum;
    shared_sum_sq[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared_sum[threadIdx.x]    += shared_sum[threadIdx.x + stride];
            shared_sum_sq[threadIdx.x] += shared_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inv_D = 1.0f / static_cast<float>(D);
    const float mean = shared_sum[0] * inv_D;
    const float variance = shared_sum_sq[0] * inv_D - mean * mean;
    const float inv_var = rsqrtf(variance + eps);

    if (threadIdx.x == 0)
    {
        means[idx]    = mean;
        inv_vars[idx] = inv_var;
    }

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
    CUDA_CHECK_KERNEL();
}

template void layernorm_forward_cuda<float>        (const int, const int, const float*,         float*,         float*, float*, const float*, const float*, const float);
template void layernorm_forward_cuda<__nv_bfloat16>(const int, const int, const __nv_bfloat16*, __nv_bfloat16*, float*, float*, const float*, const float*, const float);

// Computes dX only. One block per row; shared-memory reduction for per-row sums.
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

    float sum_D = 0.0f;
    float sum_D_xhat = 0.0f;

    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float d = static_cast<float>(dy_row[i]) * gamma[i];
        const float x_hat = (static_cast<float>(x_row[i]) - mean) * inv_var;
        sum_D += d;
        sum_D_xhat += d * x_hat;
    }

    __shared__ float s_sum_D[256];
    __shared__ float s_sum_D_xhat[256];

    s_sum_D[threadIdx.x] = sum_D;
    s_sum_D_xhat[threadIdx.x] = sum_D_xhat;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            s_sum_D[threadIdx.x] += s_sum_D[threadIdx.x + stride];
            s_sum_D_xhat[threadIdx.x] += s_sum_D_xhat[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float mean_D = s_sum_D[0] / static_cast<float>(D);
    const float mean_D_xhat = s_sum_D_xhat[0] / static_cast<float>(D);

    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float d = static_cast<float>(dy_row[i]) * gamma[i];
        const float x_hat = (static_cast<float>(x_row[i]) - mean) * inv_var;
        dx_row[i] = static_cast<T>((d - mean_D - x_hat * mean_D_xhat) * inv_var);
    }
}


template<typename T>
__global__ void layernorm_gamma_beta_gradient_kernel(const int N, const int D, const T* __restrict__ dY, const T* __restrict__ X, const float* __restrict__ means, const float* __restrict__ inv_vars, float* __restrict__ dGamma, float* __restrict__ dBeta)
{
    // Computes dGamma and dBeta. One block per dim; reduces across all N rows in shared memory, no atomics.

    const int d = blockIdx.x;
    if (d >= D) return;

    float local_gamma = 0.0f;
    float local_beta = 0.0f;

    for (int n = threadIdx.x; n < N; n += blockDim.x)
    {
        const float dy = static_cast<float>(dY[n * D + d]);
        const float x_hat = (static_cast<float>(X[n * D + d]) - means[n]) * inv_vars[n];
        local_gamma += dy * x_hat;
        local_beta += dy;
    }

    __shared__ float s_gamma[256];
    __shared__ float s_beta[256];

    s_gamma[threadIdx.x] = local_gamma;
    s_beta[threadIdx.x] = local_beta;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            s_gamma[threadIdx.x] += s_gamma[threadIdx.x + stride];
            s_beta[threadIdx.x]  += s_beta[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        dGamma[d] = s_gamma[0];
        dBeta[d]  = s_beta[0];
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
    CUDA_CHECK_KERNEL();

    const int gb_threads = 256;
    layernorm_gamma_beta_gradient_kernel<T> << <D, gb_threads >> > (N, D, dY, X, means, inv_vars, dGamma, dBeta);
    CUDA_CHECK_KERNEL();
}

template void layernorm_backward_cuda<float>        (const int, const int, const float*,         const float*,         const float*, const float*, const float*, float*,         float*, float*);
template void layernorm_backward_cuda<__nv_bfloat16>(const int, const int, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, const float*, __nv_bfloat16*, float*, float*);
