#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "../eigen/unsupported/Eigen/CXX11/Tensor"

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

__global__ void addition_kernel(const int, const float*, const float*, float*);

__global__ void division_kernel(const int, const type*, const type*, type*);

__global__ void log_kernel(int, const type*, type*);

__global__ void log_in_place_kernel(const int, type*);

__global__ void divide_subtract_kernel(int, type*, const type*, const type*);


// Wrappers operations

void addition_cuda(const size_t, const float*, const float*, float*);

void division(const size_t&, const type*, const type*, type*);

void log(const size_t&, const type*, type*);

void log_in_place(const size_t&, type*);

void divide_subtract(const size_t&, type*, const type*, const type*);

// ADAM

__global__ void adam_update_kernel(const int, float*, float*, float*, const float*, const float, const float, const float, const float, const float, const float);
void adam_update_device(const size_t, float*, float*, float*, const float*, const float, const float, const float, const float, const float, const float);

// SGD

__global__ void sgd_update_kernel(const int, float*, float*, const float*, const float, const float, const bool);
void sgd_update_device(const size_t, float*, float*, const float*, const float, const float, const bool);

 // Errors

 __global__ void calculate_binary_cross_entropy_kernel(const int, type*, const type*, const type*, const type);
 void calculate_binary_cross_entropy_cuda(const size_t&, type*, const type*, const type*, const type);

 __global__ void calculate_binary_cross_entropy_delta_kernel(const int, type*, const type*, const type*, const type, const type);
 void calculate_binary_cross_entropy_delta_cuda(const size_t&, type*, const type*, const type*, const type, const type);

 __global__ void calculate_multiple_cross_entropy_kernel(const int, type*, const type*, const type*, const type);
 void calculate_multiple_cross_entropy_cuda(const size_t&, type*, const type*, const type*, const type);

 __global__ void calculate_multiple_cross_entropy_delta_kernel(const int, type*, const type*, const type*, const type);
 void calculate_multiple_cross_entropy_delta_cuda(const size_t&, type*, const type*, const type*, const type);

 // Regularization

 __global__ void apply_l1_gradient_kernel(const int, float*, const float*, const float);
 void apply_l1_gradient_cuda(const size_t, float*, const float*, const float);

 __global__ void apply_elastic_net_gradient_kernel(const int, float*, const float*, const float, const float);
 void apply_elastic_net_gradient_cuda(const size_t, float*, const float*, const float, const float);

#endif // KERNEL_CUH
