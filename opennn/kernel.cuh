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
#include <string>
#include <time.h>

using namespace std;
using namespace Eigen;

typedef float type;

#define CUDA_CHECK(call) do {                                 \
    cudaError_t err__ = (call);                               \
    if (err__ != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n",       \
                #call, __FILE__, __LINE__,                    \
                cudaGetErrorString(err__));                   \
        abort();                                              \
    }                                                         \
} while(0)

#ifndef NDEBUG
    #define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
#else
    #define CUDA_CHECK_KERNEL() ((void)0)
#endif

// ADAM

__global__ void adam_update_kernel(const int, float*, float*, float*, const float*, const float, const float, const float, const float, const float, const float);
void adam_update_cuda(const Index, float*, float*, float*, const float*, const float, const float, const float, const float, const float, const float);

// SGD

__global__ void sgd_update_kernel(const int, float*, float*, const float*, const float, const float, const bool);
void sgd_update_cuda(const Index, float*, float*, const float*, const float, const float, const bool);

 // Errors

 __global__ void binary_cross_entropy_kernel(const int, type*, const type*, const type*, const type);
 void binary_cross_entropy_cuda(const Index, type*, const type*, const type*, const type);

 __global__ void binary_cross_entropy_gradient_kernel(const int, type*, const type*, const type*, const type, const type);
 void binary_cross_entropy_gradient_cuda(const Index, type*, const type*, const type*, const type, const type);

 __global__ void multiple_cross_entropy_kernel(const int, type*, const type*, const type*, const type);
 void multiple_cross_entropy_cuda(const Index, type*, const type*, const type*, const type);

 __global__ void multiple_cross_entropy_gradient_kernel(const int, type*, const type*, const type*, const type);
 void multiple_cross_entropy_gradient_cuda(const Index, type*, const type*, const type*, const type);

 __global__ void weighted_squared_error_kernel(const int, type*, const type*, const type*, const type, const type);
 void weighted_squared_error_cuda(const Index, type*, const type*, const type*, const type, const type);

 __global__ void weighted_squared_error_gradient_kernel(const int, type*, const type*, const type*, const type, const type, const type);
 void weighted_squared_error_gradient_cuda(const Index, type*, const type*, const type*, const type, const type, const type);

 __global__ void cross_entropy_3d_multiple_forward_kernel(const int, const int, const float*, const float*, float*, float*, const float);
 void cross_entropy_3d_multiple_forward_cuda(const Index, const int, const float*, const float*, float*, float*, const float);

 __global__ void cross_entropy_3d_multiple_backward_kernel(const int, const int, const float*, const float*, float*, const float);
 void cross_entropy_3d_multiple_backward_cuda(const Index, const int, const float*, const float*, float*, const float);

 // Regularization

 __global__ void l1_gradient_kernel(const int, float*, const float*, const float);
 void l1_gradient_cuda(const Index, float*, const float*, const float);

 // Addition

 void addition_cuda(const Index, const float*, const float*, float*);
 __global__ void addition_kernel(const int, const float*, const float*, float*);

 // Embedding

 __global__ void embedding_forward_kernel(const int n, const float* inputs, const float* weights, const float* positional_encoding, float* outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding, const bool add_positional_encoding);
 void embedding_forward_cuda(const Index n, const float* inputs, const float* weights, const float* positional_encoding, float* outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding, const bool add_positional_encoding);

 __global__ void embedding_backward_kernel(const int n, const float* inputs, const float* output_gradients, float* weight_gradients, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding);
 void embedding_backward_cuda(const Index n, const float* inputs, const float* output_gradients, float* weight_gradients, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding);

 // MultiHead Attention

 __global__ void split_heads_kernel(const int n, const float* in, float* out, const int S, const int H, const int D);
 void split_heads_cuda(const Index n, const float* in, float* out, const int S, const int H, const int D);

 __global__ void merge_heads_kernel(const int n, const float* in, float* out, const int S, const int H, const int D);
 void merge_heads_cuda(const Index n, const float* in, float* out, const int S, const int H, const int D);

 void attention_masks_cuda(const int batch_size, const int heads_number, const int query_sequence_length,
                           const int source_sequence_length, const int embedding_dimension,
                           const float* source_input, float* attention_weights, float* padding_mask,
                           const bool use_causal_mask);

 // Element-wise addition
 
 void addition_cuda(const Index n, const float* input1, const float* input2, float* output);

 // Pooling 3D

 __global__ void max_pooling_3d_forward_kernel(const int n, const float* in, float* out, float* indices, const int S, const int F);
 void max_pooling_3d_forward_cuda(const Index n, const float* in, float* out, float* indices, const int S, const int F);

 __global__ void max_pooling_3d_backward_kernel(const int n, const float* delta, float* in_grad, const float* indices, const int S, const int F);
 void max_pooling_3d_backward_cuda(const Index n, const float* delta, float* in_grad, const float* indices, const int S, const int F);

 __global__ void average_pooling_3d_forward_kernel(const int n, const float* in, float* out, const int S, const int F);
 void average_pooling_3d_forward_cuda(const Index n, const float* in, float* out, const int S, const int F);

 __global__ void average_pooling_3d_backward_kernel(const int n, const float* in, const float* delta, float* in_grad, const int S, const int F);
 void average_pooling_3d_backward_cuda(const Index n, const float* in, const float* delta, float* in_grad, const int S, const int F);

 // Normalization Layer

 __global__ void layernorm_forward_kernel(const int N, const int D, const float* X, float* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps);
 void layernorm_forward_cuda(const int N, const int D, const float* X, float* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps);

 __global__ void layernorm_backward_kernel(const int N, const int D, const float* dY, const float* X, const float* means, const float* inv_vars, const float* gamma, float* dX);
 __global__ void layernorm_gamma_beta_gradient_kernel(const int N, const int D, const float* dY, const float* X, const float* means, const float* inv_vars, float* dGamma, float* dBeta);
 void layernorm_backward_cuda(const int N, const int D, const float* dY, const float* X, const float* means, const float* inv_vars, const float* gamma, float* dX, float* dGamma, float* dBeta);

#endif // KERNEL_CUH
