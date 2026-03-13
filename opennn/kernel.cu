#include "kernel.cuh"


// ADAM

__global__ void adam_update_kernel(
    const int n,
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
}

//SGD

__global__ void sgd_update_kernel(
    const int n,
    float* params,
    float* velocity,
    const float* grads,
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
}

// Errors

__global__ void calculate_binary_cross_entropy_kernel(const int n, float* term_results, const float* targets, const float* outputs, const float epsilon)
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
}


__global__ void calculate_binary_cross_entropy_delta_kernel(const int n, type* deltas, const type* targets, const type* outputs, const type epsilon, const type scaling_factor)
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
}


__global__ void calculate_multiple_cross_entropy_kernel(const int n, type* term_results, const type* targets, const type* outputs, const type epsilon)
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
}


__global__ void calculate_multiple_cross_entropy_delta_kernel(const int n, type* deltas, const type* targets, const type* outputs, const type scaling_factor)
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
}


__global__ void calculate_weighted_squared_error_kernel(const int n, type* term_results, const type* targets, const type* outputs, const type positives_weight, const type negatives_weight)
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
}


__global__ void calculate_weighted_squared_error_delta_kernel(const int n, type* deltas, const type* targets, const type* outputs, const type positives_weight, const type negatives_weight, const type scaling_factor)
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
}

// Regularization

__global__ void apply_l1_gradient_kernel(const int n, float* deltas, const float* params, const float weight)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) 
    {
        const float param_val = params[i];

        float sign = 0.0f;

        if (param_val > 0.0f) sign = 1.0f;
        else if (param_val < 0.0f) sign = -1.0f;

        deltas[i] += weight * sign;
    }
}

void apply_l1_gradient_cuda(const size_t n, float* deltas, const float* params, const float weight)
{
    if (n == 0) return;
    const int threads_per_block = 256;
    const int blocks_per_grid = (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;

    apply_l1_gradient_kernel << <blocks_per_grid, threads_per_block >> > (static_cast<int>(n), deltas, params, weight);
}


__global__ void apply_elastic_net_gradient_kernel(const int n, float* deltas, const float* params, const float weight, const float mix_factor)
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
}

// Scaling

#define EPSILON 1e-7f

__global__ void scale_2d_kernel(const int n, const int batch_size, const int outputs_number,
    const float* inputs_device, float* outputs_device,
    const int* scalers_device,
    const float* minimums_device, const float* maximums_device,
    const float* means_device, const float* std_devs_device,
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
}

// Addition

__global__ void addition_kernel(const int n, const float* input1, const float* input2, float* output)
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
}
