#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cudnn.h>
#include "kernel.cuh"

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

Tensor<type, 3> matrix_3d_from_device(const type* pointer, const size_t& new_rows_number, const size_t& new_columns_number, const size_t& new_depth_number)
{

    Tensor<type, 3> matrix_3d(static_cast<long long>(new_rows_number), static_cast<long long>(new_columns_number), static_cast<long long>(new_depth_number));

    matrix_3d.setZero();

    if (matrix_3d.size() == 0) {
        cout << "Empty matrix_3d" << endl;
    }

    if (cudaMemcpy(matrix_3d.data(), pointer, new_rows_number * new_columns_number * new_depth_number * sizeof(type), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cout << "CUDA tensor memcpy error" << endl;
    }

    return matrix_3d;
}


auto tensor_from_device(const type* pointer, const int dims[])
{
    cout << "Hello" << endl;
    return 0;
    //    const int rows_number = dims[0];
    //    const int cols_number = dims[1];
    //    const int channels = dims[2];

    //    Tensor<type,4> matrix(new_rows_number, new_raw_variables_number);

    //    matrix.setZero();

    //    if(matrix.size() == 0) cout << "Empty matrix" << endl;

    //    if(cudaMemcpy(matrix.data(), pointer, new_rows_number*new_raw_variables_number*sizeof(type), cudaMemcpyDeviceToHost) != cudaSuccess)
    //        cout << "Cuda matrix memcpy error" << endl;

    //    return matrix;
}


__global__
void scaled_exponential_linear_kernel(int n, const type* x, type* y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    const type lambda = 1.0507;
    const type alpha = 1.67326;

    if (i < n)  x[i] < 0.0 ? y[i] = lambda * alpha * (exp(x[i]) - 1) : y[i] = lambda * x[i];
}


__global__
void soft_plus_kernel(int n, const type* x, type* y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) y[i] = log(1 + exp(x[i]));
}


__global__
void soft_sign_kernel(int n, const type* x, type* y)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) x[i] < 0.0 ? y[i] = x[i] / (1.0 - x[i]) : y[i] = x[i] / (1.0 + x[i]);
}


__global__
void hard_logistic_kernel(int n, const type* x, type* y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        if (x[i] < -2.5)
        {
            y[i] = 0;
        }
        else if (x[i] > 2.5)
        {
            y[i] = 1;
        }
        else
        {
            y[i] = 0.2 * x[i] + 0.5;
        }
    }
}


// Activation derivatives kernel


__global__
void scaled_exponential_linear_derivative_kernel(int n, const type* x, type* y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    const type lambda = 1.0507;
    const type alpha = 1.67326;

    if (i < n) x[i] < 0.0 ? y[i] = lambda * alpha * exp(x[i]) : y[i] = lambda;
}


__global__
void soft_plus_derivative_kernel(int n, const type* x, type* y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) y[i] = 1 / (1 + exp(-x[i]));
    if (i < n) y[i] = 1 / (1 + exp(-x[i]));
}


__global__
void soft_sign_derivative_kernel(int n, const type* x, type* y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) x[i] < 0.0 ? y[i] = 1 / pow((1 - x[i]), 2) : y[i] = 1 / pow((1 + x[i]), 2);
}


__global__
void hard_logistic_derivative_kernel(int n, const type* x, type* y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) x[i] < -2.5 || x[i] > 2.5 ? y[i] = 0.0 : y[i] = 0.2;
}


// Operation kernel

__global__
void multiplication_kernel(const int n, const type* vector_1, const type* vector_2, type* result)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) result[i] = vector_1[i] * vector_2[i];
}


__global__
void division_kernel(const int n, const type* vector_1, const type* vector_2, type* result)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) result[i] = vector_1[i] / vector_2[i];
}


__global__
void square_kernel(const int n, const type* x, type* y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) y[i] = x[i] * x[i];
}


__global__
void exponential_kernel(const int n, const type* x, type* y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) y[i] = exp(x[i]);
}


__global__
void log_kernel(const int n, const type* x, type* y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) y[i] = log(x[i]);
}

__global__
void log_in_place_kernel(const int n, type* x)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) x[i] = log(x[i]);
}

__global__
void pow_kernel(const int n, const type p, type* x)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) x[i] = powf(float(x[i]), p);
}


__global__
void sign_kernel(const int  n, const float* x, float* y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) x[i] < 0 ? y[i] = -1.0 : y[i] = 1;
}


__global__
void subtract_kernel(const int n, const type* x, const type* y, type* z)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) z[i] = x[i] - y[i];
}

// Operation in place kernel

__global__
void multiplication_in_place_kernel(int n, const type* vector_1, type* vector_2)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) vector_2[i] = vector_1[i] * vector_2[i];
}


__global__
void abs_kernel(const int n, type* x)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) x[i] = fabs(x[i]);
}


__global__
void square_root_kernel(int n, type* x)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) x[i] = sqrt(x[i]);
}


__global__
void divide_in_place_kernel(int n, type* numerator, const type* denominator)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) numerator[i] = numerator[i] / denominator[i];
}


__global__
void subtract_in_place_kernel(int n, type* x, type* y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) y[i] = x[i] - y[i];
}


__global__
void sum_in_place_kernel(int n, type x, type* y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) y[i] = x + y[i];
}

__global__
void sum_kernel(type* input, type* output, const size_t& size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        output[0] += input[idx];
    }
}


// Wrappers activations

void scaled_exponential_linear_cuda(const size_t& size, const type* x, type* y)
{
    scaled_exponential_linear_kernel << <(size + 255) / 256, 256 >> > (size, x, y);
}


void soft_plus_cuda(const size_t& size, const type* x, type* y)
{
    soft_plus_kernel << <(size + 255) / 256, 256 >> > (size, x, y);
}


void soft_sign_cuda(const size_t& size, const type* x, type* y)
{
    soft_sign_kernel << <(size + 255) / 256, 256 >> > (size, x, y);
}


void hard_logistic_cuda(const size_t& size, const type* x, type* y)
{
    hard_logistic_kernel << <(size + 255) / 256, 256 >> > (size, x, y);
}


// Wrappers activations derivatives

void logistic_derivatives_cuda(const size_t& size, const type* x, type* y)
{
    //logistic_derivative_kernel << <(size + 255) / 256, 256 >> > (size, x, y);
}



void hyperbolic_tangent_derivatives_cuda(const size_t& size, const type* x, type* y)
{
    //hyperbolic_tangent_derivative_kernel << <(size + 255) / 256, 256 >> > (size, x, y);
}


void linear_derivatives_cuda(const size_t& size, const type* x, type* y)
{
    //linear_derivative_kernel << <(size + 255) / 256, 256 >> > (size, x, y);
}


void rectified_linear_derivatives_cuda(const size_t& size, const type* x, type* y)
{
    //rectified_linear_derivative_kernel << <(size + 255) / 256, 256 >> > (size, x, y);
}


void scaled_exponential_linear_derivatives_cuda(const size_t& size, const type* x, type* y)
{
    scaled_exponential_linear_derivative_kernel << <(size + 255) / 256, 256 >> > (size, x, y);
}


void soft_plus_derivatives_cuda(const size_t& size, const type* x, type* y)
{
    soft_plus_derivative_kernel << <(size + 255) / 256, 256 >> > (size, x, y);
}


void soft_sign_derivatives_cuda(const size_t& size, const type* x, type* y)
{
    soft_sign_derivative_kernel << <(size + 255) / 256, 256 >> > (size, x, y);
}


void hard_logistic_derivatives_cuda(const size_t& size, const type* x, type* y)
{
    hard_logistic_derivative_kernel << <(size + 255) / 256, 256 >> > (size, x, y);
}


void exponential_linear_derivatives_cuda(const size_t& size, const type* x, type* y)
{
    //exponential_linear_derivative_kernel << <(size + 255) / 256, 256 >> > (size, x, y);
}


void multiplication(const size_t& size, const type* vector_1, const type* vector2, type* result)
{
    multiplication_kernel << <(size + 255) / 256, 256 >> > (size, vector_1, vector2, result);
}


void division(const size_t& size, const type* vector_1, const type* vector2, type* result)
{
    division_kernel << <(size + 255) / 256, 256 >> > (size, vector_1, vector2, result);
}

/*
void square(const size_t& n, const type* x, type* y)
{
    square_kernel << <(n + 255) / 256, 256 >> > (n, x, y);
}
*/

void exponential(const size_t& n, const type* x, type* y)
{
    exponential_kernel << <(n + 255) / 256, 256 >> > (n, x, y);
}


void log(const size_t& n, const type* x, type* y)
{
    log_kernel << <(n + 255) / 256, 256 >> > (n, x, y);
}

void log_in_place(const size_t& n, type* x)
{
    log_in_place_kernel << <(n + 255) / 256, 256 >> > (n, x);
}


void pow(const int& n, const type& p, type* x)
{
    pow_kernel << <(n + 255) / 256, 256 >> > (n, p, x);
}


void sign_cuda(int n, const float* numerator, float* denominator)
{
    sign_kernel << <(n + 255) / 256, 256 >> > (n, numerator, denominator);
}


void subtract(const size_t& n, const type* x, const type* y, type* z)
{
    subtract_kernel << <(n + 255) / 256, 256 >> > (n, x, y, z);
}


void multiplication_in_place(int size, const type* vector_1, type* result)
{
    multiplication_in_place_kernel << <(size + 255) / 256, 256 >> > (size, vector_1, result);
}


void abs(const size_t& n, type* x)
{
    abs_kernel << <(n + 255) / 256, 256 >> > (n, x);
}


void square_root(const size_t& n, type* x)
{
    square_root_kernel << <(n + 255) / 256, 256 >> > (n, x);
}


void division_in_place(const size_t& n, type* numerator, const type* denominator)
{
    divide_in_place_kernel << <(n + 255) / 256, 256 >> > (n, numerator, denominator);
}


void subtract_in_place(const size_t& n, type* x, type* y)
{
    subtract_in_place_kernel << <(n + 255) / 256, 256 >> > (n, x, y);
}


void sum_in_place(const size_t& n, type x, type* y)
{
    sum_in_place_kernel << <(n + 255) / 256, 256 >> > (n, x, y);
}

void sum(type* input, type* output, const size_t& size)
{
    sum_kernel << <(size + 255) / 256, 256 >> > (input, output, size);
}


__global__
void calculate_error_combinations_derivatives_kernel(const int batch_size,
    const int neurons_number,
    const float* delta,
    const float* activations,
    float* result)
{
    const int id_thread = blockIdx.x * blockDim.x + threadIdx.x;

    const int step = neurons_number * neurons_number;

    if (id_thread < batch_size)
    {
        for (int i = 0; i < neurons_number; i++)
        {
            for (int j = 0; j < neurons_number; j++)
            {
                result[id_thread + i * batch_size] += delta[id_thread + j * batch_size] * activations[id_thread * step + j + i * neurons_number];
            }
        }
    }
}


__global__
void multiply_rows_kernel(const int size, const int raw_variables_number, const float* matrix, const float* vector, float* result)
{
    const int id_thread = blockIdx.x * blockDim.x + threadIdx.x;

    const int rows_number = size / raw_variables_number;

    if (id_thread < raw_variables_number)
    {
        for (int i = 0; i < rows_number; i++)
        {
            result[id_thread * rows_number + i] = matrix[id_thread * rows_number + i] * vector[id_thread];
        }
    }
}


__global__
void sum_recurrence_kernel(const int size, const int raw_variables_number, const float* vector, float* matrix)
{
    const int id_thread = blockIdx.x * blockDim.x + threadIdx.x;

    const int rows_number = size / raw_variables_number;

    if (id_thread < raw_variables_number)
    {
        for (int i = 0; i < rows_number; i++)
        {
            matrix[id_thread * rows_number + id_thread * (raw_variables_number)+i] += vector[i];
        }
    }
}


__global__
void sum_identity_kernel(const int size, const int raw_variables_number, float* matrix)
{
    const int id_thread = blockIdx.x * blockDim.x + threadIdx.x;

    const int rows_number = size / raw_variables_number;

    if (id_thread < raw_variables_number)
    {
        for (int i = 0; i < rows_number; i++)
        {
            if (id_thread == i) matrix[id_thread + i * rows_number] += 1.0;
        }
    }
}


__global__
void chip_row_kernel(const int size, const int raw_variables_number, const int idx, const float* matrix, float* vector)
{
    const int id_thread = blockIdx.x * blockDim.x + threadIdx.x;

    const int rows_number = size / raw_variables_number;

    if (id_thread < raw_variables_number)
    {
        vector[id_thread] = matrix[id_thread * rows_number + idx];
    }
}


__global__
void append_rows_kernel(const int size, const int raw_variables_number, const int idx, float* matrix, const float* vector)
{
    const int id_thread = blockIdx.x * blockDim.x + threadIdx.x;

    const int rows_number = size / raw_variables_number;

    if (id_thread < raw_variables_number)
    {
        matrix[id_thread * rows_number + idx] = vector[id_thread];
    }
}


// this method performs C = A*B

__global__
void dot_device(int row_max, int col_max, int offset, float* x, const float* y, float* z)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < offset && col < offset)
    {
        float tmpSum = 0;

        for (int i = 0; i < offset; i++)
        {
            tmpSum += x[i * row_max + row] * y[i + col * offset];
        }

        z[row + row_max * col] = tmpSum;
    }
}

void calculate_error_combinations_derivatives(const int& batch,
    const int& neurons,
    const float* matrix,
    const float* vector,
    float* result)
{
    calculate_error_combinations_derivatives_kernel << <(batch + 255) / 256, 256 >> > (batch, neurons, matrix, vector, result);
}


void multiply_rows_device(const int& size, const int& lda, const float* matrix, const float* vector, float* result)
{
    multiply_rows_kernel << <(size + 255) / 256, 256 >> > (size, lda, matrix, vector, result);
}


void sum_recurrence(const int& size, const int& lda, const float* vector, float* matrix)
{
    sum_recurrence_kernel << <(size + 255) / 256, 256 >> > (size, lda, vector, matrix);
}


void sum_identity(const int& size, const int& lda, float* matrix)
{
    sum_identity_kernel << <(size + 255) / 256, 256 >> > (size, lda, matrix);
}


void chip_row_device(const int& size, const int& lda, const int& idx, const float* matrix, float* vector)
{
    chip_row_kernel << <(size + 255) / 256, 256 >> > (size, lda, idx, matrix, vector);
}


void append_rows_device(const int& size, const int& lda, const int& idx, float* matrix, const float* vector)
{
    append_rows_kernel << <(size + 255) / 256, 256 >> > (size, lda, idx, matrix, vector);
}


__global__
void weighted_squared_error_kernel(const int size,
    const float positives_weight,
    const float negatives_weight,
    const float* errors,
    const float* targets,
    float* results)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
    {
        if (targets[i] == 1)
        {
            results[i] = positives_weight * errors[i] * errors[i];
        }
        else
        {
            results[i] = negatives_weight * errors[i] * errors[i];
        }
    }
}


__global__
void cross_entropy_error_kernel(const int size, const float* outputs, const float* targets, float* results)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
    {
        if (targets[i] == 1)
        {
            results[i] = -1.0 * log(outputs[i]);
        }
        else
        {
            results[i] = -1.0 * log(1.0 - outputs[i]);
        }
    }
}


__global__
void cross_entropy_error_derivative_kernel(const int size, const float* outputs, const float* targets, float* results)
{
    /// @todo This should be done with cublas

    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
    {
        results[i] = -1.0 * targets[i] / outputs[i] + (1.0 - targets[i]) / (1.0 - outputs[i]);
    }
}


void calculate_weighted_squared_error(const int& size,
    const float& positives_weight,
    const float& negatives_weight,
    const float* errors,
    const float* targets,
    float* results)
{
    weighted_squared_error_kernel << <(size + 255) / 256, 256 >> > (size, positives_weight, negatives_weight, errors, targets, results);
}


void calculate_cross_entropy_error(const int& size, const float* errors, const float* targets, float* results)
{
    cross_entropy_error_kernel << <(size + 255) / 256, 256 >> > (size, errors, targets, results);
}


void calculate_cross_entropy_error_derivative(const int& size, const float* errors, const float* targets, float* results)
{
    cross_entropy_error_derivative_kernel << <(size + 255) / 256, 256 >> > (size, errors, targets, results);
}


void sum_matrices_rows_cuda(float* vector_device_1d, float* vector_device_3d, const int rows, const int collums, const int channels) {

}

void sum_matrices_collums_cuda(float* vector_device_1d, float* vector_device_3d, const int rows, const int collums, const int channels) {

}

void sum_matrices_channels_cuda(float* vector_device_1d, float* vector_device_3d, const int rows, const int collums, const int channels) {

}

void divide_subtract(const size_t& n, type* parameters, const type* numerator, const type* denominator) {
    divide_subtract_kernel << <(n + 255) / 256, 256 >> > (n, parameters, numerator, denominator);
}
__global__
void divide_subtract_kernel(int size, type* parameters, const type* numerator, const type* denominator) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        parameters[i] -= numerator[i] / denominator[i];
    }
}