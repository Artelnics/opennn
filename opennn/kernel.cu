#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cudnn.h>
#include "kernel.cuh"


__global__ void reorder_inputs_kernel(const float* __restrict__ source, float* __restrict__ destination,
    const int batch_samples_number, const int channels, const int height, const int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_samples_number * channels * width * height;
    if (idx >= total) return;

    int tmp = idx;
    int h = tmp % height;    tmp /= height;
    int w = tmp % width;     tmp /= width;
    int c = tmp % channels;
    int b = tmp / channels;

    int in_idx = ((b * channels + c) * height + h) * width + w;

    destination[idx] = source[in_idx];
}


void reorder_inputs_cuda(const float* source, float* destination, int N,int C,int H,int W)
{
    int total = N * H * W * C;
    const int threads_per_block = 256;
    int blocks = (total + threads_per_block - 1) / threads_per_block;

    reorder_inputs_kernel << <blocks, threads_per_block >> > (source, destination, N, C, H, W);
}


__global__ void invert_reorder_inputs_kernel(const float* __restrict__ source, float* __restrict__ destination,
    const int batch_samples_number, const int channels, const int height, const int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_samples_number * channels * height * width;
    if (idx >= total) return;

    int tmp = idx;
    int h = tmp % height;    tmp /= height;
    int w = tmp % width;     tmp /= width;
    int c = tmp % channels;  tmp /= channels;
    int b = tmp;

    int dest_idx = ((b * channels + c) * height + h) * width + w;
    destination[dest_idx] = source[idx];
}


void invert_reorder_inputs_cuda(const float* source, float* destination, int N, int C, int H, int W)
{
    int total = N * C * H * W;
    const int threads_per_block = 256;
    int blocks = (total + threads_per_block - 1) / threads_per_block;
    invert_reorder_inputs_kernel << <blocks, threads_per_block >> > (source, destination, N, C, H, W);
}


__global__ void reverse_kernel(type* d_kernel, int width, int height, int channels) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int c = 0; c < channels; ++c) {
        if (x < width && y < height) {

            int channel_offset = c * width * height;
            int idx1 = channel_offset + y * width + x;
            int idx2 = channel_offset + (height - 1 - y) * width + (width - 1 - x);

            if (idx1 < idx2) {
                type temp = d_kernel[idx1];
                d_kernel[idx1] = d_kernel[idx2];
                d_kernel[idx2] = temp;
            }
        }
    }
}

void reverse_cuda(int width, int height, int channels, type* d_kernel) 
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    reverse_kernel << <blocksPerGrid, threadsPerBlock >> > (d_kernel, width, height, channels);
}


__global__ void reorganize_inputs_kernel(const type* inputs_device, type* outputs_device, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = rows * cols;

    if (idx < totalSize) {

        int row = idx % rows;
        int col = idx / rows; 

        int newIdx = row * cols + col;
        outputs_device[idx] = inputs_device[newIdx];
    }
}


void reorganize_inputs_cuda(const type* inputs_device, type* outputs_device, int rows, int cols)
{
    int num_elements = rows * cols;
    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;

    reorganize_inputs_kernel << <numBlocks, blockSize >> > (inputs_device, outputs_device, rows, cols);
}


__global__ void reorganize_deltas_kernel(const type* inputs_device, type* outputs_device, int rows, int cols) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = rows * cols;

    if (idx < totalSize) 
    {
        int row = idx % rows;
        int col = idx / rows;

        int originalIdx = col + row * cols;

        outputs_device[originalIdx] = inputs_device[idx];
    }
}

void reorganize_deltas_cuda(const type* inputs_device, type* outputs_device, int rows, int cols) 
{
    int num_elements = rows * cols;
    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;

    reorganize_deltas_kernel << <numBlocks, blockSize >> > (inputs_device, outputs_device, rows, cols);
}


void copy_to_vector_cuda(float* destination, const float* source, const Index& size, Index& index)
{
    if (cudaMemcpy(destination + index, source, size * sizeof(type), cudaMemcpyDeviceToDevice) != cudaSuccess)
        cout << "copy_to_vector_cuda error" << endl;

    index += size;
}


void copy_from_vector_cuda(float* destination, const float* source, const Index& size, Index& index)
{
    if (cudaMemcpy(destination, source + index, size * sizeof(type), cudaMemcpyDeviceToDevice) != cudaSuccess)
        cout << "copy_from_vector_cuda error" << endl;

    index += size;
}


type* vector_to_device(const Tensor<type, 1>& vector)
{
    type* pointer = nullptr;

    const size_t this_size = vector.size();

    if (this_size == 0) cout << "Empty vector" << endl;

    if (cudaMalloc(&pointer, this_size * sizeof(type)) != cudaSuccess) cout << "Cuda vector malloc error" << endl;

    cudaMemcpy(pointer, vector.data(), this_size * sizeof(type), cudaMemcpyHostToDevice);

    return pointer;
}


Tensor<type, 1> vector_from_device(const type* pointer, const size_t& new_size)
{
    if (new_size == 0) cout << "Empty vector" << endl;

    Tensor<type, 1> vector(new_size);

    if (cudaMemcpy(vector.data(), pointer, new_size * sizeof(type), cudaMemcpyDeviceToHost) != cudaSuccess)
        cout << "Cuda vector memcpy error" << endl;

    return vector;
}


void print_device_data(const type* pointer, const size_t size) {
    type* host_data = new type[size];
    cudaMemcpy(host_data, pointer, size * sizeof(type), cudaMemcpyDeviceToHost);

    cout << "Device Data: ";
    for (size_t i = 0; i < size; ++i) {
        cout << host_data[i] << " ";
    }
    cout << endl;

    delete[] host_data;
}


type* matrix_to_device(const Tensor<type, 2>& matrix)
{
    const size_t this_size = matrix.size();

    if (this_size == 0) cout << "Empty matrix" << endl;

    type* pointer = nullptr;

    if (cudaMalloc(&pointer, this_size * sizeof(type)) != cudaSuccess) cout << "Cuda matrix malloc error" << endl;

    cudaMemcpy(pointer, matrix.data(), this_size * sizeof(type), cudaMemcpyHostToDevice);

    return pointer;
}


Tensor<type, 2> matrix_from_device(const type* pointer, const size_t& new_rows_number, const size_t& new_raw_variables_number)
{
    Tensor<type, 2> matrix(new_rows_number, new_raw_variables_number);

    matrix.setZero();

    if (matrix.size() == 0) cout << "Empty matrix" << endl;

    if (cudaMemcpy(matrix.data(), pointer, new_rows_number * new_raw_variables_number * sizeof(type), cudaMemcpyDeviceToHost) != cudaSuccess)
        cout << "Cuda matrix memcpy error" << endl;

    return matrix;
}

Tensor<type, 3> matrix_3d_from_device(const type* pointer, const size_t& new_depth_number, const size_t& new_rows_number, const size_t& new_columns_number)
{

    Tensor<type, 3> matrix_3d(static_cast<long long>(new_depth_number), static_cast<long long>(new_rows_number), static_cast<long long>(new_columns_number));

    matrix_3d.setZero();

    if (matrix_3d.size() == 0) {
        cout << "Empty matrix_3d" << endl;
    }

    if (cudaMemcpy(matrix_3d.data(), pointer, new_rows_number * new_columns_number * new_depth_number * sizeof(type), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cout << "CUDA tensor memcpy error" << endl;
    }

    return matrix_3d;
}

Tensor<type, 4> matrix_4d_from_device(const type* pointer, const size_t& new_batch_samples_number, const size_t& new_rows_number, const size_t& new_columns_number, const size_t& new_channels_number)
{

    Tensor<type, 4> matrix_4d(static_cast<long long>(new_batch_samples_number), static_cast<long long>(new_rows_number), static_cast<long long>(new_columns_number), static_cast<long long>(new_channels_number));

    matrix_4d.setZero();

    if (matrix_4d.size() == 0) {
        cout << "Empty matrix_4d" << endl;
    }

    if (cudaMemcpy(matrix_4d.data(), pointer, new_batch_samples_number * new_rows_number * new_columns_number * new_channels_number * sizeof(type), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cout << "CUDA tensor memcpy error" << endl;
    }

    return matrix_4d;
}


// Operation kernel

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


__global__
void log_kernel(const int n, const type* x, type* y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) y[i] = log(x[i]);
}

void log(const size_t& n, const type* x, type* y)
{
    log_kernel << <(n + 255) / 256, 256 >> > (n, x, y);
}


__global__
void log_in_place_kernel(const int n, type* x)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) x[i] = log(x[i]);
}

void log_in_place(const size_t& n, type* x)
{
    log_in_place_kernel << <(n + 255) / 256, 256 >> > (n, x);
}


__global__
void division_kernel(const int n, const type* vector_1, const type* vector_2, type* result)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) result[i] = vector_1[i] / vector_2[i];
}

void division(const size_t& size, const type* vector_1, const type* vector2, type* result)
{
    division_kernel << <(size + 255) / 256, 256 >> > (size, vector_1, vector2, result);
}


__global__
void divide_subtract_kernel(int size, type* parameters, const type* numerator, const type* denominator) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        parameters[i] -= numerator[i] / denominator[i];
    }
}

void divide_subtract(const size_t& n, type* parameters, const type* numerator, const type* denominator) 
{
    divide_subtract_kernel << <(n + 255) / 256, 256 >> > (n, parameters, numerator, denominator);
}


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


__global__ void calculate_binary_cross_entropy_kernel(const int n, float* term_results, const float* targets, const float* outputs, const float epsilon)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float out = outputs[i];
        const float tgt = targets[i];

        if (out > 1.0f) out = 1.0f;
        if (out < 0.0f) out = 0.0f;

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


// Regularization

__global__ void apply_l1_gradient_kernel(const int n, float* deltas, const float* params, const float weight)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float param_val = params[i];
        const float sign = (param_val > 0.0f) ? 1.0f : ((param_val < 0.0f) ? -1.0f : 0.0f);
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
            const float range = max_val - min_val;
            if (range > 1e-9)
            {
                const float scaled_01 = (input_val - min_val) / range;
                output_val = scaled_01 * (max_range - min_range) + min_range;
            }
            break;
        }
        case CudaScalerMeanStandardDeviation:
        {
            const float mean = means_device[col];
            const float std_dev = std_devs_device[col];
            if (fabsf(std_dev) > 1e-9)
            {
                output_val = (input_val - mean) / std_dev;
            }
            break;
        }
        case CudaScalerStandardDeviation:
        {
            const float std_dev = std_devs_device[col];
            if (fabsf(std_dev) > 1e-9)
            {
                output_val = input_val / std_dev;
            }
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