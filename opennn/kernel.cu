#include "kernel.cuh"


// ADAM

__global__ void adam_update_kernel(
    const int n,
    float* __restrict__ params,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ grads,
    const float beta1,
    const float beta2,
    const float learning_rate,
    const float epsilon,
    const float bias_correction_1,
    const float bias_correction_2)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float grad = grads[i];

        const float new_m = beta1 * m[i] + (1.0f - beta1) * grad;
        m[i] = new_m;

        const float new_v = beta2 * v[i] + (1.0f - beta2) * grad * grad;
        v[i] = new_v;

        const float m_hat = new_m / bias_correction_1;
        const float v_hat = new_v / bias_correction_2;

        params[i] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}


void adam_update_device(
    const size_t n,
    float* params,
    float* m,
    float* v,
    const float* grads,
    const float beta1,
    const float beta2,
    const float learning_rate,
    const float epsilon,
    const float bias_correction_1,
    const float bias_correction_2)
{
    if (n == 0) return;

    const int block_size = 256;
    const int grid_size = (static_cast<int>(n) + block_size - 1) / block_size;

    adam_update_kernel << <grid_size, block_size >> > (
        static_cast<int>(n),
        params,
        m,
        v,
        grads,
        beta1,
        beta2,
        learning_rate,
        epsilon,
        bias_correction_1,
        bias_correction_2
        );
    CUDA_CHECK_KERNEL();
}

//SGD

__global__ void sgd_update_kernel(
    const int n,
    float* __restrict__ params,
    float* __restrict__ velocity,
    const float* __restrict__ grads,
    const float learning_rate,
    const float momentum,
    const bool nesterov)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float grad = grads[i];

        if (momentum <= 0.0f)
        {
            params[i] -= learning_rate * grad;
        }
        else
        {
            float v = velocity[i];
            float v_new = momentum * v - learning_rate * grad;
            velocity[i] = v_new;

            if (nesterov)
            {
                params[i] += momentum * v_new - learning_rate * grad;
            }
            else
            {
                params[i] += v_new;
            }
        }
    }
}


void sgd_update_device(
    const size_t n,
    float* params_d,
    float* velocity_d,
    const float* grads_d,
    const float learning_rate,
    const float momentum,
    const bool nesterov)
{
    if (n == 0) return;

    const int threads = 256;
    const int blocks = (static_cast<int>(n) + threads - 1) / threads;

    sgd_update_kernel << <blocks, threads >> > (
        static_cast<int>(n),
        params_d,
        velocity_d,
        grads_d,
        learning_rate,
        momentum,
        nesterov);
    CUDA_CHECK_KERNEL();
}

// Errors

__global__ void calculate_binary_cross_entropy_kernel(const int n, float* __restrict__ term_results, const float* __restrict__ targets, const float* __restrict__ outputs, const float epsilon)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float out = outputs[i];
        const float tgt = targets[i];

        term_results[i] = tgt * logf(out + epsilon) + (1.0f - tgt) * logf(1.0f - out + epsilon);
    }
}

void calculate_binary_cross_entropy_cuda(const size_t& n, type* term_results, const type* targets, const type* outputs, const type epsilon)
{
    const int threads_per_block = 256;
    const int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    calculate_binary_cross_entropy_kernel << <blocks_per_grid, threads_per_block >> > (n, term_results, targets, outputs, epsilon);
    CUDA_CHECK_KERNEL();
}


__global__ void calculate_binary_cross_entropy_delta_kernel(const int n, type* __restrict__ deltas, const type* __restrict__ targets, const type* __restrict__ outputs, const type epsilon, const type scaling_factor)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        const type out = outputs[i];
        const type tgt = targets[i];

        const type term1 = (1.0f - tgt) / (1.0f - out + epsilon);
        const type term2 = tgt / (out + epsilon);

        deltas[i] = (term1 - term2) * scaling_factor;
    }
}

void calculate_binary_cross_entropy_delta_cuda(const size_t& n, type* deltas, const type* targets, const type* outputs, const type epsilon, const type scaling_factor)
{
    const int threads_per_block = 256;
    const int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    calculate_binary_cross_entropy_delta_kernel << <blocks_per_grid, threads_per_block >> > (n, deltas, targets, outputs, epsilon, scaling_factor);
    CUDA_CHECK_KERNEL();
}


__global__ void calculate_multiple_cross_entropy_kernel(const int n, type* __restrict__ term_results, const type* __restrict__ targets, const type* __restrict__ outputs, const type epsilon)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        const type tgt = targets[i];

        if (tgt > 0.0f) {
            term_results[i] = tgt * logf(outputs[i] + epsilon);
        }
        else {
            term_results[i] = 0.0f;
        }
    }
}

void calculate_multiple_cross_entropy_cuda(const size_t& n, type* term_results, const type* targets, const type* outputs, const type epsilon)
{
    const int threads_per_block = 256;
    const int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    calculate_multiple_cross_entropy_kernel << <blocks_per_grid, threads_per_block >> > (n, term_results, targets, outputs, epsilon);
    CUDA_CHECK_KERNEL();
}


__global__ void calculate_multiple_cross_entropy_delta_kernel(const int n, type* __restrict__ deltas, const type* __restrict__ targets, const type* __restrict__ outputs, const type scaling_factor)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        deltas[i] = (outputs[i] - targets[i]) * scaling_factor;
    }
}

void calculate_multiple_cross_entropy_delta_cuda(const size_t& n, type* deltas, const type* targets, const type* outputs, const type scaling_factor)
{
    const int threads_per_block = 256;
    const int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    calculate_multiple_cross_entropy_delta_kernel << <blocks_per_grid, threads_per_block >> > (n, deltas, targets, outputs, scaling_factor);
    CUDA_CHECK_KERNEL();
}


__global__ void calculate_weighted_squared_error_kernel(const int n, type* __restrict__ term_results, const type* __restrict__ targets, const type* __restrict__ outputs, const type positives_weight, const type negatives_weight)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        const type tgt = targets[i];
        const type out = outputs[i];

        const type diff = out - tgt;
        const type weight = (tgt == 0.0f) ? negatives_weight : positives_weight;

        term_results[i] = diff * diff * weight;
    }
}

void calculate_weighted_squared_error_cuda(const size_t& n, type* term_results, const type* targets, const type* outputs, const type positives_weight, const type negatives_weight)
{
    if (n == 0) return;
    const int threads_per_block = 256;
    const int blocks_per_grid = (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;

    calculate_weighted_squared_error_kernel<<<blocks_per_grid, threads_per_block>>>(static_cast<int>(n), term_results, targets, outputs, positives_weight, negatives_weight);
    CUDA_CHECK_KERNEL();
}


__global__ void calculate_weighted_squared_error_delta_kernel(const int n, type* __restrict__ deltas, const type* __restrict__ targets, const type* __restrict__ outputs, const type positives_weight, const type negatives_weight, const type scaling_factor)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        const type tgt = targets[i];
        const type out = outputs[i];

        const type diff = out - tgt;
        const type weight = (tgt == 0.0f) ? negatives_weight : positives_weight;

        deltas[i] = diff * weight * scaling_factor;
    }
}

void calculate_weighted_squared_error_delta_cuda(const size_t& n, type* deltas, const type* targets, const type* outputs, const type positives_weight, const type negatives_weight, const type scaling_factor)
{
    if (n == 0) return;
    const int threads_per_block = 256;
    const int blocks_per_grid = (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;

    calculate_weighted_squared_error_delta_kernel<<<blocks_per_grid, threads_per_block>>>(static_cast<int>(n), deltas, targets, outputs, positives_weight, negatives_weight, scaling_factor);
    CUDA_CHECK_KERNEL();
}

// Cross Entropy 3D

__global__ void cross_entropy_3d_multiple_forward_kernel(const int total_tokens,
                                                         const int vocab_size,
                                                         const float* __restrict__ outputs,
                                                         const float* __restrict__ targets,
                                                         float* __restrict__ errors,
                                                         const float epsilon)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < total_tokens)
    {
        const int target_class = static_cast<int>(targets[idx]);

        if(target_class > 0 && target_class < vocab_size)
        {
            const float prob = outputs[idx * vocab_size + target_class];
            errors[idx] = -logf(prob + epsilon);
        }
        else
        {
            errors[idx] = 0.0f;
        }
    }
}

void cross_entropy_3d_multiple_forward_cuda(const size_t n,
                                            const int batch_size,
                                            const int seq_length,
                                            const int vocab_size,
                                            const float* outputs,
                                            const float* targets,
                                            float* errors,
                                            const float epsilon)
{
    (void)n;

    const int total_tokens = batch_size * seq_length;
    if(total_tokens == 0) return;

    const int block_size = 256;
    const int grid_size = (total_tokens + block_size - 1) / block_size;

    cross_entropy_3d_multiple_forward_kernel<<<grid_size, block_size>>>(
        total_tokens, vocab_size, outputs, targets, errors, epsilon);
    CUDA_CHECK_KERNEL();
}


__global__ void cross_entropy_3d_multiple_backward_kernel(const int n,
                                                          const int vocab_size,
                                                          const float* __restrict__ outputs,
                                                          const float* __restrict__ targets,
                                                          float* __restrict__ output_gradients,
                                                          const float scale_factor)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n)
    {
        const int token_idx = idx / vocab_size;
        const int class_idx = idx % vocab_size;
        const int target_class = static_cast<int>(targets[token_idx]);

        if(target_class > 0 && target_class < vocab_size)
        {
            if(class_idx == target_class)
                output_gradients[idx] = (outputs[idx] - 1.0f) * scale_factor;
            else
                output_gradients[idx] = outputs[idx] * scale_factor;
        }
        else
        {
            output_gradients[idx] = 0.0f;
        }
    }
}

void cross_entropy_3d_multiple_backward_cuda(const size_t n,
                                             const int batch_size,
                                             const int seq_length,
                                             const int vocab_size,
                                             const float* outputs,
                                             const float* targets,
                                             float* output_gradients,
                                             const float scale_factor)
{
    (void)batch_size;
    (void)seq_length;

    if(n == 0) return;

    const int block_size = 256;
    const int grid_size = (static_cast<int>(n) + block_size - 1) / block_size;

    cross_entropy_3d_multiple_backward_kernel<<<grid_size, block_size>>>(
        static_cast<int>(n), vocab_size, outputs, targets, output_gradients, scale_factor);
    CUDA_CHECK_KERNEL();
}


__global__ void cross_entropy_3d_multiple_counts_kernel(const int total_tokens, const int vocab_size,
                                                        const float* __restrict__ outputs, const float* __restrict__ targets, float* __restrict__ counts)
{
    const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(token_idx < total_tokens)
    {
        const int target_class = static_cast<int>(targets[token_idx]);
        if(target_class > 0 && target_class < vocab_size)
        {
            atomicAdd(&counts[0], 1.0f);

            int best_index = 0;
            float best_value = outputs[token_idx * vocab_size];
            for(int k = 1; k < vocab_size; ++k)
            {
                const float value = outputs[token_idx * vocab_size + k];
                if(value > best_value) { best_value = value; best_index = k; }
            }
            if(best_index == target_class) atomicAdd(&counts[1], 1.0f);
        }
    }
}

void cross_entropy_3d_multiple_counts_cuda(const size_t total_tokens, const int vocab_size,
                                           const float* outputs, const float* targets, float* counts)
{
    if(total_tokens == 0) return;
    const int block_size = 256;
    const int grid_size = (static_cast<int>(total_tokens) + block_size - 1) / block_size;
    cross_entropy_3d_multiple_counts_kernel<<<grid_size, block_size>>>(
        static_cast<int>(total_tokens), vocab_size, outputs, targets, counts);
    CUDA_CHECK_KERNEL();
}


__global__ void cross_entropy_3d_binary_forward_kernel(const int n,
                                                       const float* __restrict__ outputs,
                                                       const float* __restrict__ targets,
                                                       float* __restrict__ errors,
                                                       const float epsilon)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n)
    {
        const float target_val = targets[idx];

        if(target_val >= 0.0f)
        {
            const float prob = outputs[idx];
            errors[idx] = -(target_val * logf(prob + epsilon)
                            + (1.0f - target_val) * logf(1.0f - prob + epsilon));
        }
        else
        {
            errors[idx] = 0.0f;
        }
    }
}

void cross_entropy_3d_binary_forward_cuda(const size_t n,
                                          const int batch_size,
                                          const int seq_length,
                                          const float* outputs,
                                          const float* targets,
                                          float* errors,
                                          const float epsilon)
{
    (void)batch_size;
    (void)seq_length;

    if(n == 0) return;

    const int block_size = 256;
    const int grid_size = (static_cast<int>(n) + block_size - 1) / block_size;

    cross_entropy_3d_binary_forward_kernel<<<grid_size, block_size>>>(
        static_cast<int>(n), outputs, targets, errors, epsilon);
    CUDA_CHECK_KERNEL();
}


__global__ void cross_entropy_3d_binary_backward_kernel(const int n,
                                                        const float* __restrict__ outputs,
                                                        const float* __restrict__ targets,
                                                        float* __restrict__ output_gradients,
                                                        const float scale_factor)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n)
    {
        const float target_val = targets[idx];

        if(target_val >= 0.0f)
        {
            const float prob = outputs[idx];

            const float term1 = (1.0f - target_val) / (1.0f - prob + 1e-7f);
            const float term2 = target_val / (prob + 1e-7f);

            output_gradients[idx] = (term1 - term2) * scale_factor;
        }
        else
        {
            output_gradients[idx] = 0.0f;
        }
    }
}

void cross_entropy_3d_binary_backward_cuda(const size_t n,
                                           const int batch_size,
                                           const int seq_length,
                                           const float* outputs,
                                           const float* targets,
                                           float* output_gradients,
                                           const float scale_factor)
{
    (void)batch_size;
    (void)seq_length;

    if(n == 0) return;

    const int block_size = 256;
    const int grid_size = (static_cast<int>(n) + block_size - 1) / block_size;

    cross_entropy_3d_binary_backward_kernel<<<grid_size, block_size>>>(
        static_cast<int>(n), outputs, targets, output_gradients, scale_factor);
    CUDA_CHECK_KERNEL();
}

// Regularization

__global__ void apply_l1_gradient_kernel(const int n, float* __restrict__ deltas, const float* __restrict__ params, const float weight)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        const float p = params[i];
        const float s = (p > 0.0f) ? 1.0f : ((p < 0.0f) ? -1.0f : 0.0f);
        deltas[i] += weight * s;
    }
}

void apply_l1_gradient_cuda(const size_t n, float* deltas, const float* params, const float weight)
{
    if (n == 0) return;
    const int threads_per_block = 256;
    const int blocks_per_grid = (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;

    apply_l1_gradient_kernel << <blocks_per_grid, threads_per_block >> > (static_cast<int>(n), deltas, params, weight);
    CUDA_CHECK_KERNEL();
}


__global__ void apply_elastic_net_gradient_kernel(const int n, float* __restrict__ deltas, const float* __restrict__ params, const float weight, const float mix_factor)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float param_val = params[i];
        const float sign = (param_val > 0.0f) ? 1.0f : ((param_val < 0.0f) ? -1.0f : 0.0f);

        const float l1_grad = mix_factor * sign;
        const float l2_grad = (1.0f - mix_factor) * param_val;

        deltas[i] += weight * (l1_grad + l2_grad);
    }
}

void apply_elastic_net_gradient_cuda(const size_t n, float* deltas, const float* params, const float weight, const float mix_factor)
{
    if (n == 0) return;
    const int threads_per_block = 256;
    const int blocks_per_grid = (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;

    apply_elastic_net_gradient_kernel << <blocks_per_grid, threads_per_block >> > (static_cast<int>(n), deltas, params, weight, mix_factor);
    CUDA_CHECK_KERNEL();
}

// Scaling

#define EPSILON 1e-7f

__global__ void scale_2d_kernel(const int n, const int batch_size, const int outputs_number,
    const float* __restrict__ inputs_device, float* __restrict__ outputs_device,
    const int* __restrict__ scalers_device,
    const float* __restrict__ minimums_device, const float* __restrict__ maximums_device,
    const float* __restrict__ means_device, const float* __restrict__ std_devs_device,
    const float min_range, const float max_range)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        const int col = i % outputs_number;

        const int scaler_type = scalers_device[col];
        const float input_val = inputs_device[i];
        float output_val = input_val;

        switch (scaler_type)
        {
        case CudaScalerNone:
            break;

        case CudaScalerMinimumMaximum:
        {
            const float min_val = minimums_device[col];
            const float max_val = maximums_device[col];
            output_val = (input_val - min_val) / ((max_val - min_val) + EPSILON) * (max_range - min_range) + min_range;
            break;
        }
        case CudaScalerMeanStandardDeviation:
        {
            const float mean = means_device[col];
            const float std_dev = std_devs_device[col];
            output_val = (input_val - mean) / (std_dev + EPSILON);
            break;
        }
        case CudaScalerStandardDeviation:
        {
            const float std_dev = std_devs_device[col];
            output_val = input_val / (std_dev + EPSILON);
            break;
        }
        case CudaScalerLogarithm:
            output_val = logf(input_val);
            break;

        case CudaScalerImageMinMax:
            output_val = input_val / 255.0f;
            break;

        default:
            break;
        }

        outputs_device[i] = output_val;
    }
}

void scale_2d_cuda(const size_t n, const int batch_size, const int outputs_number,
    const float* inputs_device, float* outputs_device,
    const int* scalers_device,
    const float* minimums_device, const float* maximums_device,
    const float* means_device, const float* std_devs_device,
    const float min_range, const float max_range)
{
    if (n == 0) return;

    const int threads_per_block = 256;
    const int blocks_per_grid = (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;

    scale_2d_kernel << <blocks_per_grid, threads_per_block >> > (
        static_cast<int>(n),
        batch_size,
        outputs_number,
        inputs_device,
        outputs_device,
        scalers_device,
        minimums_device,
        maximums_device,
        means_device,
        std_devs_device,
        min_range,
        max_range
        );
    CUDA_CHECK_KERNEL();
}

// Addition

__global__ void addition_kernel(const int n, const float* __restrict__ input1, const float* __restrict__ input2, float* __restrict__ output)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        output[i] = input1[i] + input2[i];
    }
}

void addition_cuda(const size_t n, const float* input1, const float* input2, float* output)
{
    if (n == 0) return;

    const int threads_per_block = 256;
    const int blocks_per_grid = (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;

    addition_kernel << <blocks_per_grid, threads_per_block >> > (static_cast<int>(n), input1, input2, output);
    CUDA_CHECK_KERNEL();
}


// Embedding

__global__ void embedding_forward_kernel(const int n, const float* __restrict__ inputs, const float* __restrict__ weights, const float* __restrict__ positional_encoding, float* __restrict__ outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding, const bool add_positional_encoding)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        const int token_idx = i / embedding_dimension;
        const int dim_idx = i % embedding_dimension;
        const int seq_idx = token_idx % sequence_length;

        const int token_id = static_cast<int>(inputs[token_idx]);

        float val = 0.0f;

        if (token_id >= 0 && token_id < vocabulary_size)
        {
            val = weights[token_id * embedding_dimension + dim_idx];
        }

        if (scale_embedding)
        {
            val *= sqrtf(static_cast<float>(embedding_dimension));
        }

        if (add_positional_encoding && positional_encoding != nullptr)
        {
            if (token_id > 0)
            {
                val += positional_encoding[seq_idx * embedding_dimension + dim_idx];
            }
        }

        outputs[i] = val;
    }
}

void embedding_forward_cuda(const size_t n, const float* inputs, const float* weights, const float* positional_encoding, float* outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding, const bool add_positional_encoding)
{
    if (n == 0) return;

    const int threads_per_block = 256;
    const int blocks_per_grid = (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;

    embedding_forward_kernel<<<blocks_per_grid, threads_per_block>>>(
        static_cast<int>(n), inputs, weights, positional_encoding, outputs,
        sequence_length, embedding_dimension, vocabulary_size, scale_embedding, add_positional_encoding
        );
    CUDA_CHECK_KERNEL();
}


__global__ void embedding_backward_kernel(const int n, const float* __restrict__ inputs, const float* __restrict__ output_gradients, float* __restrict__ weight_gradients, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        const int token_idx = i / embedding_dimension;
        const int dim_idx = i % embedding_dimension;

        const int token_id = static_cast<int>(inputs[token_idx]);

        if (token_id > 0 && token_id < vocabulary_size)
        {
            float grad_val = output_gradients[i];

            if (scale_embedding)
            {
                grad_val *= sqrtf(static_cast<float>(embedding_dimension));
            }

            atomicAdd(&weight_gradients[token_id * embedding_dimension + dim_idx], grad_val);
        }
    }
}

void embedding_backward_cuda(const size_t n, const float* inputs, const float* output_gradients, float* weight_gradients, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    if (n == 0) return;

    const int threads_per_block = 256;
    const int blocks_per_grid = (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;

    embedding_backward_kernel<<<blocks_per_grid, threads_per_block>>>(
        static_cast<int>(n), inputs, output_gradients, weight_gradients,
        sequence_length, embedding_dimension, vocabulary_size, scale_embedding
        );
    CUDA_CHECK_KERNEL();
}

// Multihead

// [Batch, Seq, Heads, HeadDim] -> [Batch, Heads, Seq, HeadDim]
__global__ void mha_transpose_qkv_kernel(const int n, const float* __restrict__ in, float* __restrict__ out, const int S, const int H, const int D)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        const int d = i % D;
        const int h = (i / D) % H;
        const int s = (i / (D * H)) % S;
        const int b = i / (D * H * S);

        const int out_idx = ((b * H + h) * S + s) * D + d;
        out[out_idx] = in[i];
    }
}

void mha_transpose_qkv_cuda(const size_t n, const float* in, float* out, const int S, const int H, const int D)
{
    if (n == 0) return;
    const int threads = 256;
    mha_transpose_qkv_kernel<<<(n + threads - 1) / threads, threads>>>(n, in, out, S, H, D);
    CUDA_CHECK_KERNEL();
}

// [Batch, Heads, Seq, HeadDim] -> [Batch, Seq, Heads, HeadDim]
__global__ void mha_transpose_o_kernel(const int n, const float* __restrict__ in, float* __restrict__ out, const int S, const int H, const int D)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        const int d = i % D;
        const int s = (i / D) % S;
        const int h = (i / (D * S)) % H;
        const int b = i / (D * S * H);

        const int out_idx = ((b * S + s) * H + h) * D + d;
        out[out_idx] = in[i];
    }
}

void mha_transpose_o_cuda(const size_t n, const float* in, float* out, const int S, const int H, const int D)
{
    if (n == 0) return;
    const int threads = 256;
    mha_transpose_o_kernel<<<(n + threads - 1) / threads, threads>>>(n, in, out, S, H, D);
    CUDA_CHECK_KERNEL();
}


__global__ void mha_key_padding_mask_kernel(const int n, const float* __restrict__ source_input, float* __restrict__ attention_weights, const int H, const int Sq, const int Sk, const int E)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        const int sk = i % Sk;
        const int b  = i / (Sk * Sq * H);

        bool is_padding = true;
        const int source_token_offset = (b * Sk + sk) * E;

        for (int e = 0; e < E; ++e)
        {
            if (fabsf(source_input[source_token_offset + e]) > 1e-7f)
            {
                is_padding = false;
                break;
            }
        }

        if (is_padding)
        {
            attention_weights[i] = -1e9f;
        }
    }
}

void mha_key_padding_mask_cuda(const size_t n, const float* source_input, float* attention_weights, const int H, const int Sq, const int Sk, const int E)
{
    if (n == 0) return;
    const int threads = 256;
    mha_key_padding_mask_kernel<<<(n + threads - 1) / threads, threads>>>(static_cast<int>(n), source_input, attention_weights, H, Sq, Sk, E);
    CUDA_CHECK_KERNEL();
}


__global__ void mha_causal_mask_kernel(const int n, float* __restrict__ scores, const int seq_q, const int seq_k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int c = i % seq_k;
        const int r = (i / seq_k) % seq_q;
        if (c > r) scores[i] = -1e9f;
    }
}

void mha_causal_mask_cuda(const size_t n, float* scores, const int seq_q, const int seq_k) {
    if (n == 0) return;
    const int threads = 256;
    mha_causal_mask_kernel<<<(n + threads - 1) / threads, threads>>>(n, scores, seq_q, seq_k);
    CUDA_CHECK_KERNEL();
}


__global__ void compute_padding_mask_kernel(const int num_tokens, const float* __restrict__ source_input, float* __restrict__ padding_mask, const int embedding_dimension)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < num_tokens)
    {
        const float* token = source_input + i * embedding_dimension;
        bool is_pad = true;
        for(int e = 0; e < embedding_dimension; ++e)
        {
            if(fabsf(token[e]) > 1e-7f) { is_pad = false; break; }
        }
        padding_mask[i] = is_pad ? 1.0f : 0.0f;
    }
}


__global__ void apply_fused_masks_kernel(const int n, float* __restrict__ attention_weights, const float* __restrict__ padding_mask,
                                         const int heads_number, const int query_sequence_length,
                                         const int source_sequence_length, const bool use_causal_mask)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        const int sk = i % source_sequence_length;
        const int sq = (i / source_sequence_length) % query_sequence_length;
        const int b  = i / (source_sequence_length * query_sequence_length * heads_number);

        if((use_causal_mask && sk > sq) || padding_mask[b * source_sequence_length + sk] > 0.5f)
            attention_weights[i] = -1e9f;
    }
}


void mha_fused_masks_cuda(const int batch_size, const int heads_number,
                          const int query_sequence_length, const int source_sequence_length,
                          const int embedding_dimension, const float* source_input,
                          float* attention_weights, float* padding_mask, const bool use_causal_mask)
{
    const int num_tokens = batch_size * source_sequence_length;
    if(num_tokens > 0)
    {
        const int threads = 256;
        compute_padding_mask_kernel<<<(num_tokens + threads - 1) / threads, threads>>>(
            num_tokens, source_input, padding_mask, embedding_dimension);
        CUDA_CHECK_KERNEL();
    }

    const int n = batch_size * heads_number * query_sequence_length * source_sequence_length;
    if(n > 0)
    {
        const int threads = 256;
        apply_fused_masks_kernel<<<(n + threads - 1) / threads, threads>>>(
            n, attention_weights, padding_mask, heads_number,
            query_sequence_length, source_sequence_length, use_causal_mask);
        CUDA_CHECK_KERNEL();
    }
}


// Pooling 3D

__global__ void pooling3d_max_forward_kernel(const int n, const float* __restrict__ in, float* __restrict__ out, float* __restrict__ indices, const int B, const int S, const int F)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int f = idx % F;
        const int b = idx / F;

        float max_val = -1e20f;
        int max_idx = 0;

        for (int s = 0; s < S; s++) {
            float val = in[(b * S + s) * F + f];
            if (val > max_val) {
                max_val = val;
                max_idx = s;
            }
        }
        out[idx] = max_val;

        if (indices != nullptr) indices[idx] = static_cast<float>(max_idx);
    }
}

void pooling3d_max_forward_cuda(const size_t n, const float* in, float* out, float* indices, const int B, const int S, const int F) {
    if (n == 0) return;
    const int threads = 256;
    pooling3d_max_forward_kernel<<<(n + threads - 1) / threads, threads>>>(static_cast<int>(n), in, out, indices, B, S, F);
    CUDA_CHECK_KERNEL();
}


__global__ void pooling3d_max_backward_kernel(const int n, const float* __restrict__ delta, float* __restrict__ in_grad, const float* __restrict__ indices, const int B, const int S, const int F)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int f = idx % F;
        const int b = idx / F;

        const int max_s = static_cast<int>(indices[idx]);

        in_grad[(b * S + max_s) * F + f] = delta[idx];
    }
}

void pooling3d_max_backward_cuda(const size_t n, const float* delta, float* in_grad, const float* indices, const int B, const int S, const int F) {
    if (n == 0) return;
    const int threads = 256;
    pooling3d_max_backward_kernel<<<(n + threads - 1) / threads, threads>>>(static_cast<int>(n), delta, in_grad, indices, B, S, F);
    CUDA_CHECK_KERNEL();
}


__global__ void pooling3d_avg_forward_kernel(const int n, const float* __restrict__ in, float* __restrict__ out, const int B, const int S, const int F)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        const int f = idx % F;
        const int b = idx / F;

        float sum = 0.0f;
        int valid_count = 0;

        for (int s = 0; s < S; s++)
        {
            bool is_padding = true;
            for (int check_f = 0; check_f < F; check_f++)
            {
                if (fabsf(in[(b * S + s) * F + check_f]) > 1e-7f)
                {
                    is_padding = false;
                    break;
                }
            }

            if (!is_padding)
            {
                sum += in[(b * S + s) * F + f];
                valid_count++;
            }
        }

        out[idx] = (valid_count > 0) ? (sum / (float)valid_count) : 0.0f;
    }
}

void pooling3d_avg_forward_cuda(const size_t n, const float* in, float* out, const int B, const int S, const int F) {
    if (n == 0) return;
    const int threads = 256;
    pooling3d_avg_forward_kernel<<<(n + threads - 1) / threads, threads>>>(static_cast<int>(n), in, out, B, S, F);
    CUDA_CHECK_KERNEL();
}


__global__ void pooling3d_avg_backward_kernel(const int n, const float* __restrict__ in, const float* __restrict__ delta, float* __restrict__ in_grad, const int B, const int S, const int F)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        const int f = idx % F;
        const int b = idx / F;

        int valid_count = 0;

        for (int s = 0; s < S; s++)
        {
            bool is_padding = true;
            for (int check_f = 0; check_f < F; check_f++)
            {
                if (fabsf(in[(b * S + s) * F + check_f]) > 1e-7f)
                {
                    is_padding = false;
                    break;
                }
            }
            if (!is_padding) valid_count++;
        }

        if (valid_count > 0)
        {
            float grad_val = delta[idx] / (float)valid_count;

            for (int s = 0; s < S; s++)
            {
                bool is_padding = true;
                for (int check_f = 0; check_f < F; check_f++)
                {
                    if (fabsf(in[(b * S + s) * F + check_f]) > 1e-7f)
                    {
                        is_padding = false;
                        break;
                    }
                }
                if (!is_padding)
                    in_grad[(b * S + s) * F + f] = grad_val;
            }
        }
    }
}

void pooling3d_avg_backward_cuda(const size_t n, const float* in, const float* delta, float* in_grad, const int B, const int S, const int F) {
    if (n == 0) return;
    const int threads = 256;
    pooling3d_avg_backward_kernel<<<(n + threads - 1) / threads, threads>>>(static_cast<int>(n), in, delta, in_grad, B, S, F);
    CUDA_CHECK_KERNEL();
}


// Normalization layer

__global__ void layernorm_forward_kernel(const int N, const int D, const float* __restrict__ X, float* __restrict__ Y, float* __restrict__ means, float* __restrict__ inv_vars, const float* __restrict__ gamma, const float* __restrict__ beta, const float eps)
{
    const int idx = blockIdx.x;
    if (idx >= N) return;

    const float* x_row = X + idx * D;
    float* y_row = Y + idx * D;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += x_row[i];
    }

    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean = shared_sum[0] / (float)D;
    if (threadIdx.x == 0) means[idx] = mean;

    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float diff = x_row[i] - mean;
        var_sum += diff * diff;
    }

    shared_sum[threadIdx.x] = var_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float var = shared_sum[0] / (float)D;
    float inv_var = rsqrtf(var + eps);
    if (threadIdx.x == 0) inv_vars[idx] = inv_var;

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float x_hat = (x_row[i] - mean) * inv_var;
        y_row[i] = gamma[i] * x_hat + beta[i];
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


__global__ void layernorm_backward_kernel(const int N, const int D, const float* __restrict__ dY, const float* __restrict__ X, const float* __restrict__ means, const float* __restrict__ inv_vars, const float* __restrict__ gamma, float* __restrict__ dX, float* __restrict__ dGamma, float* __restrict__ dBeta)
{
    const int idx = blockIdx.x;
    if (idx >= N) return;

    const float* dy_row = dY + idx * D;
    const float* x_row = X + idx * D;
    float* dx_row = dX + idx * D;

    float mean = means[idx];
    float inv_var = inv_vars[idx];

    float sum_D = 0.0f;
    float sum_D_xhat = 0.0f;

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float d = dy_row[i] * gamma[i];
        float x_hat = (x_row[i] - mean) * inv_var;

        sum_D += d;
        sum_D_xhat += d * x_hat;

        atomicAdd(&dGamma[i], dy_row[i] * x_hat);
        atomicAdd(&dBeta[i], dy_row[i]);
    }

    __shared__ float s_sum_D[256];
    __shared__ float s_sum_D_xhat[256];

    s_sum_D[threadIdx.x] = sum_D;
    s_sum_D_xhat[threadIdx.x] = sum_D_xhat;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum_D[threadIdx.x] += s_sum_D[threadIdx.x + stride];
            s_sum_D_xhat[threadIdx.x] += s_sum_D_xhat[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean_D = s_sum_D[0] / (float)D;
    float mean_D_xhat = s_sum_D_xhat[0] / (float)D;

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float d = dy_row[i] * gamma[i];
        float x_hat = (x_row[i] - mean) * inv_var;

        dx_row[i] = (d - mean_D - x_hat * mean_D_xhat) * inv_var;
    }
}

void layernorm_backward_cuda(const int N, const int D, const float* dY, const float* X, const float* means, const float* inv_vars, const float* gamma, float* dX, float* dGamma, float* dBeta)
{
    if (N == 0 || D == 0) return;

    int threads = 256;
    if (D <= 32) threads = 32;
    else if (D <= 64) threads = 64;
    else if (D <= 128) threads = 128;

    layernorm_backward_kernel<<<N, threads>>>(N, D, dY, X, means, inv_vars, gamma, dX, dGamma, dBeta);
    CUDA_CHECK_KERNEL();
}

