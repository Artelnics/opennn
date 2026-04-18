#include "kernel.cuh"

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

__global__ void binary_cross_entropy_kernel(const int n, float* __restrict__ term_results, const float* __restrict__ targets, const float* __restrict__ outputs, const float epsilon)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float out = outputs[i];
        const float tgt = targets[i];

        const float log_pos = logf(out + epsilon);
        const float log_neg = logf(1.0f - out + epsilon);

        term_results[i] = fmaf(tgt, log_pos - log_neg, log_neg);
    }
}

void binary_cross_entropy_cuda(const Index n, type* term_results, const type* targets, const type* outputs, const type epsilon)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    binary_cross_entropy_kernel << <grid_size, block_size >> > (
        total, term_results, targets, outputs, epsilon);
    CUDA_CHECK_KERNEL();
}

__global__ void binary_cross_entropy_gradient_kernel(const int n, type* __restrict__ deltas, const type* __restrict__ targets, const type* __restrict__ outputs, const type epsilon, const type scaling_factor)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const type out = outputs[i];
        const type tgt = targets[i];

        deltas[i] = ((1.0f - tgt) / (1.0f - out + epsilon) - tgt / (out + epsilon)) * scaling_factor;
    }
}

void binary_cross_entropy_gradient_cuda(const Index n, type* deltas, const type* targets, const type* outputs, const type epsilon, const type scaling_factor)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    binary_cross_entropy_gradient_kernel << <grid_size, block_size >> > (
        total, deltas, targets, outputs, epsilon, scaling_factor);
    CUDA_CHECK_KERNEL();
}

__global__ void multiple_cross_entropy_kernel(const int n, type* __restrict__ term_results, const type* __restrict__ targets, const type* __restrict__ outputs, const type epsilon)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const type tgt = targets[i];
        term_results[i] = (tgt > 0.0f) ? tgt * logf(outputs[i] + epsilon) : 0.0f;
    }
}

void multiple_cross_entropy_cuda(const Index n, type* term_results, const type* targets, const type* outputs, const type epsilon)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    multiple_cross_entropy_kernel << <grid_size, block_size >> > (
        total, term_results, targets, outputs, epsilon);
    CUDA_CHECK_KERNEL();
}

__global__ void multiple_cross_entropy_gradient_kernel(const int n, type* __restrict__ deltas, const type* __restrict__ targets, const type* __restrict__ outputs, const type scaling_factor)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        deltas[i] = (outputs[i] - targets[i]) * scaling_factor;
}

void multiple_cross_entropy_gradient_cuda(const Index n, type* deltas, const type* targets, const type* outputs, const type scaling_factor)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    multiple_cross_entropy_gradient_kernel << <grid_size, block_size >> > (
        total, deltas, targets, outputs, scaling_factor);
    CUDA_CHECK_KERNEL();
}

__global__ void weighted_squared_error_kernel(const int n, type* __restrict__ term_results, const type* __restrict__ targets, const type* __restrict__ outputs, const type positives_weight, const type negatives_weight)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const type tgt = targets[i];
        const type diff = outputs[i] - tgt;
        const type weight = (tgt == 0.0f) ? negatives_weight : positives_weight;

        term_results[i] = diff * diff * weight;
    }
}

void weighted_squared_error_cuda(const Index n, type* term_results, const type* targets, const type* outputs, const type positives_weight, const type negatives_weight)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    weighted_squared_error_kernel << <grid_size, block_size >> > (
        total, term_results, targets, outputs, positives_weight, negatives_weight);
    CUDA_CHECK_KERNEL();
}

__global__ void weighted_squared_error_gradient_kernel(const int n, type* __restrict__ deltas, const type* __restrict__ targets, const type* __restrict__ outputs, const type positives_weight, const type negatives_weight, const type scaling_factor)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const type tgt = targets[i];
        const type diff = outputs[i] - tgt;
        const type weight = (tgt == 0.0f) ? negatives_weight : positives_weight;

        deltas[i] = diff * weight * scaling_factor;
    }
}

void weighted_squared_error_gradient_cuda(const Index n, type* deltas, const type* targets, const type* outputs, const type positives_weight, const type negatives_weight, const type scaling_factor)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    weighted_squared_error_gradient_kernel << <grid_size, block_size >> > (
        total, deltas, targets, outputs, positives_weight, negatives_weight, scaling_factor);
    CUDA_CHECK_KERNEL();
}

__global__ void cross_entropy_3d_multiple_forward_kernel(const int total_tokens,
                                                         const int vocab_size,
                                                         const float* __restrict__ outputs,
                                                         const float* __restrict__ targets,
                                                         float* __restrict__ errors,
                                                         const float epsilon)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_tokens; idx += blockDim.x * gridDim.x)
    {
        const int target_class = static_cast<int>(targets[idx]);
        const bool valid = target_class > 0 && target_class < vocab_size;

        errors[idx] = valid ? -logf(outputs[idx * vocab_size + target_class] + epsilon) : 0.0f;
    }
}

void cross_entropy_3d_multiple_forward_cuda(const Index n,
                                            const int vocab_size,
                                            const float* outputs,
                                            const float* targets,
                                            float* errors,
                                            const float epsilon)
{
    if(n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    cross_entropy_3d_multiple_forward_kernel << <grid_size, block_size >> > (
        total, vocab_size, outputs, targets, errors, epsilon);
    CUDA_CHECK_KERNEL();
}

__global__ void cross_entropy_3d_multiple_backward_kernel(const int n,
                                                          const int vocab_size,
                                                          const float* __restrict__ outputs,
                                                          const float* __restrict__ targets,
                                                          float* __restrict__ output_gradients,
                                                          const float scale_factor)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        const int token_idx = idx / vocab_size;
        const int class_idx = idx % vocab_size;
        const int target_class = static_cast<int>(targets[token_idx]);

        if (target_class <= 0 || target_class >= vocab_size)
        {
            output_gradients[idx] = 0.0f;
            continue;
        }

        output_gradients[idx] = (outputs[idx] - (class_idx == target_class ? 1.0f : 0.0f)) * scale_factor;
    }
}

void cross_entropy_3d_multiple_backward_cuda(const Index n,
                                             const int vocab_size,
                                             const float* outputs,
                                             const float* targets,
                                             float* output_gradients,
                                             const float scale_factor)
{
    if(n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    cross_entropy_3d_multiple_backward_kernel << <grid_size, block_size >> > (
        total, vocab_size, outputs, targets, output_gradients, scale_factor);
    CUDA_CHECK_KERNEL();
}

__global__ void l1_gradient_kernel(const int n, float* __restrict__ deltas, const float* __restrict__ parameters, const float weight)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float p = parameters[i];
        const float s = (p > 0.0f) ? 1.0f : ((p < 0.0f) ? -1.0f : 0.0f);
        deltas[i] += weight * s;
    }
}

void l1_gradient_cuda(const Index n, float* deltas, const float* parameters, const float weight)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    l1_gradient_kernel << <grid_size, block_size >> > (total, deltas, parameters, weight);
    CUDA_CHECK_KERNEL();
}

__global__ void addition_kernel(const int n, const float* __restrict__ input1, const float* __restrict__ input2, float* __restrict__ output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        output[i] = input1[i] + input2[i];
}

void addition_cuda(const Index n, const float* input1, const float* input2, float* output)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    addition_kernel << <grid_size, block_size >> > (total, input1, input2, output);
    CUDA_CHECK_KERNEL();
}

__global__ void embedding_forward_kernel(const int n, const float* __restrict__ inputs, const float* __restrict__ weights, const float* __restrict__ positional_encoding, float* __restrict__ outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding, const bool add_positional_encoding)
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

        outputs[i] = val;
    }
}

void embedding_forward_cuda(const Index n, const float* inputs, const float* weights, const float* positional_encoding, float* outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding, const bool add_positional_encoding)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    embedding_forward_kernel << <grid_size, block_size >> > (
        total, inputs, weights, positional_encoding, outputs,
        sequence_length, embedding_dimension, vocabulary_size, scale_embedding, add_positional_encoding);
    CUDA_CHECK_KERNEL();
}

__global__ void embedding_backward_kernel(const int n, const float* __restrict__ inputs, const float* __restrict__ output_gradients, float* __restrict__ weight_gradients, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    const float scale = scale_embedding ? sqrtf(static_cast<float>(embedding_dimension)) : 1.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int token_idx = i / embedding_dimension;
        const int dim_idx = i % embedding_dimension;
        const int token_id = static_cast<int>(inputs[token_idx]);

        if (token_id <= 0 || token_id >= vocabulary_size) continue;

        atomicAdd(&weight_gradients[token_id * embedding_dimension + dim_idx], scale * output_gradients[i]);
    }
}

void embedding_backward_cuda(const Index n, const float* inputs, const float* output_gradients, float* weight_gradients, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    embedding_backward_kernel << <grid_size, block_size >> > (
        total, inputs, output_gradients, weight_gradients,
        embedding_dimension, vocabulary_size, scale_embedding);
    CUDA_CHECK_KERNEL();
}

__global__ void split_heads_kernel(const int n, const float* __restrict__ in, float* __restrict__ out, const int S, const int H, const int D)
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

void split_heads_cuda(const Index n, const float* in, float* out, const int S, const int H, const int D)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    split_heads_kernel << <grid_size, block_size >> > (total, in, out, S, H, D);
    CUDA_CHECK_KERNEL();
}

__global__ void merge_heads_kernel(const int n, const float* __restrict__ in, float* __restrict__ out, const int S, const int H, const int D)
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

void merge_heads_cuda(const Index n, const float* in, float* out, const int S, const int H, const int D)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    merge_heads_kernel << <grid_size, block_size >> > (total, in, out, S, H, D);
    CUDA_CHECK_KERNEL();
}

__global__ void padding_mask_kernel(const int num_tokens, const float* __restrict__ source_input, float* __restrict__ padding_mask, const int embedding_dimension)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_tokens; i += blockDim.x * gridDim.x)
    {
        const float* token = source_input + i * embedding_dimension;
        bool is_pad = true;
        for (int e = 0; e < embedding_dimension; ++e)
            if (fabsf(token[e]) > 1e-7f) { is_pad = false; break; }
        padding_mask[i] = is_pad ? 1.0f : 0.0f;
    }
}

__global__ void fused_masks_kernel(const int n, float* __restrict__ attention_weights, const float* __restrict__ padding_mask,
                                         const int heads_number, const int query_sequence_length,
                                         const int source_sequence_length, const bool use_causal_mask)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int sk = i % source_sequence_length;
        const int sq = (i / source_sequence_length) % query_sequence_length;
        const int b  = i / (source_sequence_length * query_sequence_length * heads_number);

        if((use_causal_mask && sk > sq) || padding_mask[b * source_sequence_length + sk] > 0.5f)
            attention_weights[i] = -1e9f;
    }
}

void attention_masks_cuda(const int batch_size, const int heads_number,
                          const int query_sequence_length, const int source_sequence_length,
                          const int embedding_dimension, const float* source_input,
                          float* attention_weights, float* padding_mask, const bool use_causal_mask)
{
    const int block_size = 256;

    const int num_tokens = batch_size * source_sequence_length;
    if(num_tokens > 0)
    {
        const int grid_size = (num_tokens + block_size - 1) / block_size;
        padding_mask_kernel << <grid_size, block_size >> > (
            num_tokens, source_input, padding_mask, embedding_dimension);
        CUDA_CHECK_KERNEL();
    }

    const int n = batch_size * heads_number * query_sequence_length * source_sequence_length;

    if(n > 0)
    {
        const int grid_size = (n + block_size - 1) / block_size;
        fused_masks_kernel << <grid_size, block_size >> > (
            n, attention_weights, padding_mask, heads_number,
            query_sequence_length, source_sequence_length, use_causal_mask);
        CUDA_CHECK_KERNEL();
    }
}

__global__ void max_pooling_3d_forward_kernel(const int n, const float* __restrict__ in, float* __restrict__ out, float* __restrict__ indices, const int S, const int F)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        const int f = idx % F;
        const int b = idx / F;

        float max_val = -1e20f;
        int max_idx = 0;

        for (int s = 0; s < S; ++s)
        {
            const float val = in[(b * S + s) * F + f];
            if (val > max_val) { max_val = val; max_idx = s; }
        }

        out[idx] = max_val;
        if (indices != nullptr) indices[idx] = static_cast<float>(max_idx);
    }
}

void max_pooling_3d_forward_cuda(const Index n, const float* in, float* out, float* indices, const int S, const int F)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    max_pooling_3d_forward_kernel << <grid_size, block_size >> > (total, in, out, indices, S, F);
    CUDA_CHECK_KERNEL();
}

__global__ void max_pooling_3d_backward_kernel(const int n, const float* __restrict__ delta, float* __restrict__ in_gradient, const float* __restrict__ indices, const int S, const int F)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        const int f = idx % F;
        const int b = idx / F;
        const int max_s = static_cast<int>(indices[idx]);

        in_gradient[(b * S + max_s) * F + f] = delta[idx];
    }
}

void max_pooling_3d_backward_cuda(const Index n, const float* delta, float* in_gradient, const float* indices, const int S, const int F)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    max_pooling_3d_backward_kernel << <grid_size, block_size >> > (total, delta, in_gradient, indices, S, F);
    CUDA_CHECK_KERNEL();
}

__global__ void average_pooling_3d_forward_kernel(const int n, const float* __restrict__ in, float* __restrict__ out, const int S, const int F)
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
                if (fabsf(in[(b * S + s) * F + check_f]) > 1e-7f) 
                { 
                    is_padding = false; 
                    break; 
                }

            if (!is_padding) 
            { 
                sum += in[(b * S + s) * F + f]; 
                ++valid_count; 
            }
        }

        out[idx] = (valid_count > 0) ? (sum / static_cast<float>(valid_count)) : 0.0f;
    }
}

void average_pooling_3d_forward_cuda(const Index n, const float* in, float* out, const int S, const int F)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    average_pooling_3d_forward_kernel << <grid_size, block_size >> > (total, in, out, S, F);
    CUDA_CHECK_KERNEL();
}

__global__ void average_pooling_3d_backward_kernel(const int n, const float* __restrict__ in, const float* __restrict__ delta, float* __restrict__ in_gradient, const int S, const int F)
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
                if (fabsf(in[(b * S + s) * F + check_f]) > 1e-7f) 
                { 
                    is_padding = false; 
                    break; 
                }

            if (!is_padding) 
                ++valid_count;
        }

        if (valid_count == 0) continue;

        const float gradient_val = delta[idx] / static_cast<float>(valid_count);
        for (int s = 0; s < S; ++s)
        {
            bool is_padding = true;
            for (int check_f = 0; check_f < F; ++check_f)
                if (fabsf(in[(b * S + s) * F + check_f]) > 1e-7f) { is_padding = false; break; }
            if (!is_padding) in_gradient[(b * S + s) * F + f] = gradient_val;
        }
    }
}

void average_pooling_3d_backward_cuda(const Index n, const float* in, const float* delta, float* in_gradient, const int S, const int F)
{
    if (n == 0) return;

    const int block_size = 256;
    const int total = static_cast<int>(n);
    const int grid_size = (total + block_size - 1) / block_size;

    average_pooling_3d_backward_kernel << <grid_size, block_size >> > (total, in, delta, in_gradient, S, F);
    CUDA_CHECK_KERNEL();
}

__global__ void layernorm_forward_kernel(const int N, const int D, const float* __restrict__ X, float* __restrict__ Y, float* __restrict__ means, float* __restrict__ inv_vars, const float* __restrict__ gamma, const float* __restrict__ beta, const float eps)
{
    const int idx = blockIdx.x;
    if (idx >= N) return;

    const float* x_row = X + idx * D;
    float* y_row = Y + idx * D;

    // Single-pass accumulate sum and sum-of-squares, then derive variance as E[X^2] - E[X]^2.
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float x = x_row[i];
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
        const float x_hat = (x_row[i] - mean) * inv_var;
        y_row[i] = fmaf(gamma[i], x_hat, beta[i]);
    }
}

void layernorm_forward_cuda(const int N, const int D, const float* X, float* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps)
{
    if (N == 0 || D == 0) return;

    int threads = 256;
    if (D <= 32) threads = 32;
    else if (D <= 64) threads = 64;
    else if (D <= 128) threads = 128;

    layernorm_forward_kernel<<<N, threads>>>(N, D, X, Y, means, inv_vars, gamma, beta, eps);
    CUDA_CHECK_KERNEL();
}

// Computes dX only. One block per row; shared-memory reduction for per-row sums.
__global__ void layernorm_backward_kernel(const int N, const int D, const float* __restrict__ dY, const float* __restrict__ X, const float* __restrict__ means, const float* __restrict__ inv_vars, const float* __restrict__ gamma, float* __restrict__ dX)
{
    const int idx = blockIdx.x;
    if (idx >= N) return;

    const float* dy_row = dY + idx * D;
    const float* x_row = X + idx * D;
    float* dx_row = dX + idx * D;

    const float mean = means[idx];
    const float inv_var = inv_vars[idx];

    float sum_D = 0.0f;
    float sum_D_xhat = 0.0f;

    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float d = dy_row[i] * gamma[i];
        const float x_hat = (x_row[i] - mean) * inv_var;
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
        const float d = dy_row[i] * gamma[i];
        const float x_hat = (x_row[i] - mean) * inv_var;
        dx_row[i] = (d - mean_D - x_hat * mean_D_xhat) * inv_var;
    }
}

__global__ void layernorm_gamma_beta_gradient_kernel(const int N, const int D, const float* __restrict__ dY, const float* __restrict__ X, const float* __restrict__ means, const float* __restrict__ inv_vars, float* __restrict__ dGamma, float* __restrict__ dBeta)
{
    // Computes dGamma and dBeta. One block per dim; reduces across all N rows in shared memory, no atomics.

    const int d = blockIdx.x;
    if (d >= D) return;

    float local_gamma = 0.0f;
    float local_beta = 0.0f;

    for (int n = threadIdx.x; n < N; n += blockDim.x)
    {
        const float dy = dY[n * D + d];
        const float x_hat = (X[n * D + d] - means[n]) * inv_vars[n];
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

void layernorm_backward_cuda(const int N, const int D, const float* dY, const float* X, const float* means, const float* inv_vars, const float* gamma, float* dX, float* dGamma, float* dBeta)
{
    if (N == 0 || D == 0) return;

    int dx_threads = 256;
    if (D <= 32) dx_threads = 32;
    else if (D <= 64) dx_threads = 64;
    else if (D <= 128) dx_threads = 128;

    layernorm_backward_kernel << <N, dx_threads >> > (N, D, dY, X, means, inv_vars, gamma, dX);
    CUDA_CHECK_KERNEL();

    const int gb_threads = 256;
    layernorm_gamma_beta_gradient_kernel << <D, gb_threads >> > (N, D, dY, X, means, inv_vars, dGamma, dBeta);
    CUDA_CHECK_KERNEL();
}
