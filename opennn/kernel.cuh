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

 __global__ void calculate_weighted_squared_error_kernel(const int, type*, const type*, const type*, const type, const type);
 void calculate_weighted_squared_error_cuda(const size_t&, type*, const type*, const type*, const type, const type);

 __global__ void calculate_weighted_squared_error_delta_kernel(const int n, type* deltas, const type* targets, const type* outputs, const type positives_weight, const type negatives_weight, const type scaling_factor);
 void calculate_weighted_squared_error_delta_cuda(const size_t& n, type* deltas, const type* targets, const type* outputs, const type positives_weight, const type negatives_weight, const type scaling_factor);

 // Regularization

 __global__ void apply_l1_gradient_kernel(const int, float*, const float*, const float);
 void apply_l1_gradient_cuda(const size_t, float*, const float*, const float);

 __global__ void apply_elastic_net_gradient_kernel(const int, float*, const float*, const float, const float);
 void apply_elastic_net_gradient_cuda(const size_t, float*, const float*, const float, const float);

 // Addition

 void addition_cuda(const size_t, const float*, const float*, float*);
 __global__ void addition_kernel(const int, const float*, const float*, float*);

 // Scaling

 enum CudaScaler {
     CudaScalerNone = 0,
     CudaScalerMinimumMaximum,
     CudaScalerMeanStandardDeviation,
     CudaScalerStandardDeviation,
     CudaScalerLogarithm,
     CudaScalerImageMinMax
 };

 __global__ void scale_2d_kernel(const int n, const int batch_size, const int outputs_number,
                                 const float* inputs_device, float* outputs_device,
                                 const int* scalers_device,
                                 const float* minimums_device, const float* maximums_device,
                                 const float* means_device, const float* std_devs_device,
                                 const float min_range, const float max_range);

 void scale_2d_cuda(const size_t n, const int batch_size, const int outputs_number,
                    const float* inputs_device, float* outputs_device,
                    const int* scalers_device,
                    const float* minimums_device, const float* maximums_device,
                    const float* means_device, const float* std_devs_device,
                    const float min_range, const float max_range);

 // Embedding

 __global__ void embedding_forward_kernel(const int n, const float* inputs, const float* weights, const float* positional_encoding, float* outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding, const bool add_positional_encoding);
 void embedding_forward_cuda(const size_t n, const float* inputs, const float* weights, const float* positional_encoding, float* outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding, const bool add_positional_encoding);

 __global__ void embedding_backward_kernel(const int n, const float* inputs, const float* output_gradients, float* weight_gradients, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding);
 void embedding_backward_cuda(const size_t n, const float* inputs, const float* output_gradients, float* weight_gradients, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding);

 // Multihead

 // MultiHead Attention

 __global__ void mha_transpose_qkv_kernel(const int n, const float* in, float* out, const int S, const int H, const int D);
 void mha_transpose_qkv_cuda(const size_t n, const float* in, float* out, const int S, const int H, const int D);

 __global__ void mha_transpose_o_kernel(const int n, const float* in, float* out, const int S, const int H, const int D);
 void mha_transpose_o_cuda(const size_t n, const float* in, float* out, const int S, const int H, const int D);

 __global__ void mha_key_padding_mask_kernel(const int n, const float* source_input, float* attention_weights, const int H, const int Sq, const int Sk, const int E);
 void mha_key_padding_mask_cuda(const size_t n, const float* source_input, float* attention_weights, const int H, const int Sq, const int Sk, const int E);

 __global__ void mha_causal_mask_kernel(const int n, float* scores, const int seq_q, const int seq_k);
 void mha_causal_mask_cuda(const size_t n, float* scores, const int seq_q, const int seq_k);


#endif // KERNEL_CUH
