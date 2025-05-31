#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "../../opennn/eigen/unsupported/Eigen/CXX11/Tensor"

#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include <curand.h>

// System includes

#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <algorithm>
#include <time.h>

using namespace std;
using namespace Eigen;

typedef float type;

// Utilities

__global__ void reorder_inputs_kernel(const float* __restrict__, float* __restrict__ , int, int, int, int);
void reorder_inputs_cuda(const float* source, float* destination, int, int, int, int);

__global__ void invert_reorder_inputs_kernel(const float* __restrict__, float* __restrict__, const int, const int, const int, const int);
void invert_reorder_inputs_cuda(const float* source, float* destination, int N, int C, int H, int W);

__global__ void reverse_kernel(type*, int, int, int);
void reverse_cuda(int, int, int, type*);

__global__ void reorganize_inputs_kernel(const type*, type*, int, int);
void reorganize_inputs_cuda(const type*, type*, int, int);

__global__ void reorganize_deltas_kernel(const type*, type*, int, int);
void reorganize_deltas_cuda(const type*, type*, int, int);

void copy_to_vector_cuda(float* destination, const float* source, const Index& size, Index& index);
void copy_from_vector_cuda(float* destination, const float* source, const Index& size, Index& index);

type* vector_to_device(const Tensor<type, 1>&);

Tensor<type, 1> vector_from_device(const type*, const size_t&);

type* matrix_to_device(const Tensor<type, 2>&);

Tensor<type, 2> matrix_from_device(const type*, const size_t&, const size_t&);

Tensor<type, 3> matrix_3d_from_device(const type*, const size_t&, const size_t&, const size_t&);

Tensor<type, 4> matrix_4d_from_device(const type*, const size_t&, const size_t&, const size_t&, const size_t&);

void print_device_data(const type*, const size_t);


// Operation kernel

__global__ void multiplication_kernel(const int, const type*, const type*, type*);

__global__ void division_kernel(const int, const type*, const type*, type*);

__global__ void square_kernel(int, const type*, type*);

__global__ void exponential_kernel(int, const type*, type*);

__global__ void log_kernel(int, const type*, type*);

__global__ void log_in_place_kernel(const int, type*);

__global__ void pow_kernel(const int, const type, type*);

__global__ void sign_kernel(int, const type*, type*);

__global__ void subtract_kernel(int, const type*, const type*, type*);

__global__ void dot_device(int, int, int, float*, const float*, float*);

__global__ void divide_subtract_kernel(int, type*, const type*, const type*);


// Operation in place kernel

__global__ void multiplication_in_place_kernel(int, const type*, type*);

__global__ void abs_kernel(int, type*); // in place

__global__ void square_root_kernel(int, type*); // in place

__global__ void divide_in_place_kernel(int, type*, const type*); // in place

__global__ void subtract_in_place_kernel(int, type*, type*); // in place

__global__ void sum_kernel(type*, type*, const size_t&); // vector sum


// Wrappers operations

void multiplication(const size_t&, const type*, const type*, type*);

void division(const size_t&, const type*, const type*, type*);

void square(const size_t&, const type*, type*);

void exponential(const size_t&, const type*, type*);

void log(const size_t&, const type*, type*);

void log_in_place(const size_t&, type*);

void sign_cuda(int, const type*, type*);

void subtract(const size_t&, const type*, const type*, type*);

void divide_subtract(const size_t&, type*, const type*, const type*);

// Wrappers operations in place

void multiplication_in_place(int, const type*, type*);

void abs(const size_t&, type*);

void pow(const int&, const type&, type*);

void square_root(const size_t&, type*);

void division_in_place(const size_t&, type*, const type*);

void subtract_in_place(const size_t&, type*, type*);

void sum_in_place(const size_t&, type, type*);

void sum(type*, type*, const size_t&);


// aux methods probabilistic

void calculate_error_combinations_derivatives(const int&, const int&, const float*, const float*, float*);
__global__ void calculate_error_combinations_derivatives_kernel(const int, const int, const float*, const float*, float*);


// aux methods RNN

void multiply_rows_device(const int&, const int&, const float*, const float*, float*);
__global__ void multiply_rows_kernel(const int, const int, const float*, const float*, float*);


void sum_recurrence(const int& size, const int&, const float*, float*);
__global__ void sum_recurrence_kernel(const int, const int, const float*, float*);


void sum_identity(const int& size, const int&, float*);
__global__ void sum_identity_kernel(const int, const int, float*);


void chip_row_device(const int& size, const int&, const int&, const float*, float*);
__global__ void chip_row_kernel(const int, const int, const int, const float*, float*);


void append_rows_device(const int&, const int&, const int&, float*, const float*);
__global__ void append_rows_kernel(const int, const int, const int, float*, const float*);


// Loss wrappers

void calculate_weighted_squared_error(const int&, const float&, const float&, const float*, const float*, float*);
__global__ void weighted_squared_error_kernel(const int, const float, const float, const float*, const float*, float*);


void calculate_cross_entropy_error(const int&, const float*, const float*, float*);
__global__ void cross_entropy_error_kernel(const int, const float*, const float*, float*);


void calculate_cross_entropy_error_derivative(const int&, const float*, const float*, float*);
__global__ void cross_entropy_error_derivative_kernel(const int, const float*, const float*, float*);

#endif // KERNEL_CUH
