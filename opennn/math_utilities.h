//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M A T H   U T I L I T I E S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "tensor_utilities.h"
#include "random_utilities.h"

namespace opennn
{

constexpr float SELU_ALPHA = 1.6732632423543772848170429916717f;
constexpr float SELU_LAMBDA = 1.0507009873554804934193349852946f;

enum class ActivationFunction{
    Linear, Sigmoid, HyperbolicTangent, RectifiedLinear, ScaledExponentialLinear, Softmax, Logistic
};

inline ActivationFunction string_to_activation(const string& name)
{
    if (name == "Sigmoid") return ActivationFunction::Sigmoid;
    if (name == "HyperbolicTangent") return ActivationFunction::HyperbolicTangent;
    if (name == "RectifiedLinear") return ActivationFunction::RectifiedLinear;
    if (name == "ScaledExponentialLinear") return ActivationFunction::ScaledExponentialLinear;
    if (name == "Softmax") return ActivationFunction::Softmax;
    if (name == "Logistic") return ActivationFunction::Logistic;
    return ActivationFunction::Linear;
}

struct ActivationArguments
{
    ActivationFunction activation_function = ActivationFunction::Linear;
#ifdef CUDA
    cudnnActivationDescriptor_t activation_descriptor = nullptr;
#endif
};

struct ConvolutionArguments
{
    string convolution_type;
    Shape stride_shape;
    Shape padding_shape;
#ifdef CUDA
    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;
    cudnnFilterDescriptor_t kernel_descriptor = nullptr;
    cudnnConvolutionFwdAlgo_t algorithm_forward = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionBwdDataAlgo_t algorithm_data = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    cudnnConvolutionBwdFilterAlgo_t algorithm_filter = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    void* workspace = nullptr;
    size_t workspace_size = 0;
    void* backward_filter_workspace = nullptr;
    size_t backward_filter_workspace_size = 0;
#endif
};

struct PoolingArguments
{
    string pooling_method;
    Shape pool_dimensions;
    Shape stride_shape;
    Shape padding_shape;
#ifdef CUDA
    cudnnPoolingDescriptor_t pooling_descriptor = nullptr;
#endif
};

struct BatchNormalizationArguments
{
    type momentum;
#ifdef CUDA
    cudnnBatchNormMode_t batch_normalization_mode = CUDNN_BATCHNORM_PER_ACTIVATION;
    cudnnTensorDescriptor_t per_activation_descriptor = nullptr;
#endif
};

// Pooling 4D

void max_pooling(const TensorView& input, TensorView& output, TensorView& maximal_indices, const PoolingArguments& arguments, bool is_training = false);
void average_pooling(const TensorView& input, TensorView& output, const PoolingArguments& arguments);
void max_pooling_backward(const TensorView& input, const TensorView& output, const TensorView& output_gradient, const TensorView& maximal_indices, TensorView& input_gradient, const PoolingArguments& args);
void average_pooling_backward(const TensorView& input, const TensorView& output, const TensorView& output_gradient, TensorView& input_gradient, const PoolingArguments& args);

// Generic

void padding(const TensorView& input, TensorView& output);
void bounding(const TensorView& input, const TensorView& lower_bounds, const TensorView& upper_bounds, TensorView& output);
void copy(const TensorView& source, TensorView& destination);
void addition(const TensorView& input_1, const TensorView& input_2, TensorView& output);
void multiply(const TensorView& input_A, bool transpose_A, const TensorView& input_B, bool transpose_B, TensorView& output_C, type alpha = 1.0f, type beta = 0.0f);
void multiply_elementwise(const TensorView& A, const TensorView& B, TensorView& C);
void sum(const TensorView& A, TensorView& B, type alpha = 1.0f, type beta = 0.0f);
void softmax(TensorView& output);

// Dense layer

void combination(const TensorView& input, const TensorView& weights, const TensorView& biases, TensorView& output);
void activation(TensorView& output, ActivationFunction func);
void activation(TensorView& output, ActivationArguments arguments);
void activation_gradient(const TensorView& outputs, const TensorView& output_gradient, TensorView& activation_derivative, const ActivationFunction func, void* act_desc = nullptr);
void dropout(TensorView& output, type dropout_rate);
void dropout_gradient(const TensorView& output_gradient, const TensorView& mask, type dropout_rate, TensorView& input_gradient);

// Batch normalization

void batch_normalization_inference(const TensorView& input, const TensorView& gamma, const TensorView& beta, const VectorR& running_mean, const VectorR& running_variance, TensorView& output);
void batch_normalization_training(const TensorView& input, const TensorView& gamma, const TensorView& beta, VectorR& running_mean, VectorR& running_variance, TensorView& mean, TensorView& inverse_variance, TensorView& output, type momentum = type(0.9));
void batch_normalization_backward(const TensorView& input, const TensorView& output, const TensorView& output_gradient, const TensorView& mean, const TensorView& inverse_variance, const TensorView& gamma, TensorView& gamma_gradient, TensorView& beta_gradient, TensorView& input_gradient);

// Convolution

void convolution(const TensorView& input, const TensorView& kernel, const TensorView& bias, TensorView& output, const ConvolutionArguments& args = {});
void convolution_activation(const TensorView& input, const TensorView& weight, const TensorView& bias, TensorView& output, const ConvolutionArguments& conv_args = {}, const ActivationArguments& act_args = {});
void convolution_backward_weights(const TensorView& input, const TensorView& delta, TensorView& weight_grad, TensorView& bias_grad, const ConvolutionArguments& args = {});
void convolution_backward_data(const TensorView& delta, const TensorView& kernel, TensorView& input_grad, TensorView& padded_input_grad, const ConvolutionArguments& args = {});

// Pooling 3D

void max_pooling_3d_forward(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training);
void average_pooling_3d_forward(const TensorView& input, TensorView& output);
void max_pooling_3d_backward(const TensorView& maximal_indices, const TensorView& output_gradient, TensorView& input_gradient);
void average_pooling_3d_backward(const TensorView& input, const TensorView& output_gradient, TensorView& input_gradient);

// Embedding

void embedding_backward(const TensorView& input_indices, const TensorView& output_gradient, TensorView& weight_gradient, Index embedding_dimension, bool scale_embedding);

// Multi-head attention

void projection(const TensorView& input, const TensorView& weights, const TensorView& biases, TensorView& output);
void projection_gradient(const TensorView& d_head, const TensorView& input, const TensorView& weights, TensorView& d_bias, TensorView& d_weights, TensorView& d_input, Index batch_size, Index heads_number, Index sequence_length, Index embedding_dimension, Index head_dimension, bool accumulate);

void multihead_attention_forward(const TensorView& query, const TensorView& key, const TensorView& value, TensorView& attention_weights, TensorView& concatenated, TensorView& output, const TensorView& projection_weights, const TensorView& projection_biases, const TensorView& source_input, Index batch_size, Index heads_number, Index query_sequence_length, Index source_sequence_length, Index embedding_dimension, Index head_dimension, type scaling_factor, bool use_causal_mask, const MatrixR& causal_mask);

void multihead_attention_backward(const TensorView& query_input, const TensorView& source_input, TensorView& output_gradient, const TensorView& query, const TensorView& key, const TensorView& value, const TensorView& attention_weights, const TensorView& concatenated, const TensorView& projection_weights, TensorView& proj_weight_grad, TensorView& proj_bias_grad, TensorView& concat_grad, TensorView& att_weight_grad, TensorView& query_grad, TensorView& key_grad, TensorView& value_grad, TensorView& query_weight_grad, TensorView& query_bias_grad, TensorView& key_weight_grad, TensorView& key_bias_grad, TensorView& value_weight_grad, TensorView& value_bias_grad, TensorView& input_query_grad, const TensorView& query_weights, const TensorView& key_weights, const TensorView& value_weights, Index batch_size, Index heads_number, Index query_sequence_length, Index source_sequence_length, Index embedding_dimension, Index head_dimension, type scaling_factor, bool self_attention);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
