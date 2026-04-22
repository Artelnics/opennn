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

namespace opennn
{

enum class ActivationFunction{
    Linear, Sigmoid, HyperbolicTangent, RectifiedLinear, ScaledExponentialLinear, Softmax, Logistic
};

inline const EnumMap<ActivationFunction>& activation_function_map()
{
    static const vector<pair<ActivationFunction, string>> entries = {
        {ActivationFunction::Linear,                  "Linear"},
        {ActivationFunction::Sigmoid,                 "Sigmoid"},
        {ActivationFunction::HyperbolicTangent,       "HyperbolicTangent"},
        {ActivationFunction::RectifiedLinear,         "RectifiedLinear"},
        {ActivationFunction::ScaledExponentialLinear, "ScaledExponentialLinear"},
        {ActivationFunction::Softmax,                 "Softmax"},
        {ActivationFunction::Logistic,                "Logistic"}
    };
    static const EnumMap<ActivationFunction> map{entries};
    return map;
}

inline ActivationFunction string_to_activation(const string& name)
{
    return activation_function_map().from_string(name, ActivationFunction::Linear);
}

inline const string& activation_to_string(ActivationFunction function)
{
    return activation_function_map().to_string(function);
}

inline cudnnActivationMode_t to_cudnn_activation_mode(ActivationFunction function)
{
    switch(function)
    {
    case ActivationFunction::Sigmoid:                 return CUDNN_ACTIVATION_SIGMOID;
    case ActivationFunction::HyperbolicTangent:       return CUDNN_ACTIVATION_TANH;
    case ActivationFunction::RectifiedLinear:         return CUDNN_ACTIVATION_RELU;
    case ActivationFunction::ScaledExponentialLinear: return CUDNN_ACTIVATION_ELU;
    default:                                          return CUDNN_ACTIVATION_IDENTITY;
    }
}

struct ActivationArguments
{
    ActivationFunction activation_function = ActivationFunction::Linear;
    cudnnActivationDescriptor_t activation_descriptor = nullptr;
};

struct ConvolutionArguments
{
    Shape stride_shape;
    Shape padding_shape;
    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;
    cudnnFilterDescriptor_t kernel_descriptor = nullptr;
    cudnnConvolutionFwdAlgo_t algorithm_forward = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionBwdDataAlgo_t algorithm_data = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    cudnnConvolutionBwdFilterAlgo_t algorithm_filter = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    void* workspace = nullptr;
    size_t workspace_size = 0;
    void* backward_filter_workspace = nullptr;
    size_t backward_filter_workspace_size = 0;
};

struct PoolingArguments
{
    Shape pool_dimensions;
    Shape stride_shape;
    Shape padding_shape;
    cudnnPoolingDescriptor_t pooling_descriptor = nullptr;
};

struct BatchNormalizationArguments
{
    type momentum;
    cudnnBatchNormMode_t batch_normalization_mode = CUDNN_BATCHNORM_PER_ACTIVATION;
    cudnnTensorDescriptor_t per_activation_descriptor = nullptr;
};

struct DropoutArguments
{
    type rate = type(0);
    VectorR mask_cpu;
    cudnnDropoutDescriptor_t descriptor = nullptr;
    void* states = nullptr;
    size_t states_size = 0;
    void* reserve_space = nullptr;
    size_t reserve_size = 0;
};


// Generic

void padding(const TensorView& input, TensorView& output);
void bounding(const TensorView& input, const TensorView& lower_bounds, const TensorView& upper_bounds, TensorView& output);
void scale(const TensorView& input,
           const TensorView& minimums, const TensorView& maximums,
           const TensorView& means, const TensorView& standard_deviations,
           const TensorView& scalers,
           type min_range, type max_range,
           TensorView& output);
void unscale(const TensorView& input,
             const TensorView& minimums, const TensorView& maximums,
             const TensorView& means, const TensorView& standard_deviations,
             const TensorView& scalers,
             type min_range, type max_range,
             TensorView& output);
void copy(const TensorView& source, TensorView& destination);
void addition(const TensorView& input_1, const TensorView& input_2, TensorView& output);
void multiply(const TensorView& input_a, bool transpose_a, const TensorView& input_b, bool transpose_b, TensorView& output, type alpha = 1.0f, type beta = 0.0f);
void multiply_elementwise(const TensorView& input_a, const TensorView& input_b, TensorView& output);
void sum(const TensorView& input, TensorView& output, type alpha = 1.0f, type beta = 0.0f);
void softmax(TensorView& output);
void softmax_backward(const TensorView& softmax_out, TensorView& output_gradient);

// Dense layer

void combination(const TensorView& input, const TensorView& weights, const TensorView& biases, TensorView& output);
void combination_gradient(const TensorView& output_gradient, const TensorView& input, const TensorView& weights, TensorView& input_gradient, TensorView& weight_gradient, TensorView& bias_gradient, bool accumulate_input_gradient);
void activation(TensorView& output, ActivationArguments arguments);
void activation_gradient(const TensorView& outputs, const TensorView& output_gradient, TensorView& activation_derivative, const ActivationArguments& arguments);
void dropout(TensorView& output, DropoutArguments& args);
void dropout_gradient(const TensorView& output_gradient, TensorView& input_gradient, const DropoutArguments& args);

// Batch normalization

void batch_normalization_inference(const TensorView& input, const TensorView& gamma, const TensorView& beta, const TensorView& running_mean, const TensorView& running_variance, TensorView& output);
void batch_normalization_training(const TensorView& input, const TensorView& gamma, const TensorView& beta, TensorView& running_mean, TensorView& running_variance, TensorView& mean, TensorView& inverse_variance, TensorView& output, type momentum = type(0.9));
void batch_normalization_backward(const TensorView& input, const TensorView& output, const TensorView& output_gradient, const TensorView& mean, const TensorView& inverse_variance, const TensorView& gamma, TensorView& gamma_gradient, TensorView& beta_gradient, TensorView& input_gradient);

// Layer normalization (3D)

void layernorm_forward(const TensorView& input, const TensorView& gamma, const TensorView& beta,
                       TensorView& means, TensorView& standard_deviations, TensorView& normalized,
                       TensorView& output,
                       Index batch_size, Index sequence_length, Index embedding_dimension);

void layernorm_backward(const TensorView& input, const TensorView& output_gradient,
                        const TensorView& means, const TensorView& standard_deviations,
                        const TensorView& normalized, const TensorView& gamma,
                        TensorView& gamma_gradient, TensorView& beta_gradient, TensorView& input_gradient,
                        Index batch_size, Index sequence_length, Index embedding_dimension);

// Convolution

void convolution(const TensorView& input, const TensorView& kernel, const TensorView& bias, TensorView& output, const ConvolutionArguments& args = {});
void convolution_activation(const TensorView& input, const TensorView& weight, const TensorView& bias, TensorView& output, const ConvolutionArguments& conv_args = {}, const ActivationArguments& activation_arguments = {});
void convolution_backward_weights(const TensorView& input, const TensorView& output_gradient, TensorView& weight_grad, TensorView& bias_grad, const ConvolutionArguments& args = {});
void convolution_backward_data(const TensorView& output_gradient, const TensorView& kernel, TensorView& input_grad, const ConvolutionArguments& args = {});

// Pooling 4D

void max_pooling(const TensorView& input, TensorView& output, TensorView& maximal_indices, const PoolingArguments& arguments, bool is_training = false);
void average_pooling(const TensorView& input, TensorView& output, const PoolingArguments& arguments);
void max_pooling_backward(const TensorView& input, const TensorView& output, const TensorView& output_gradient, const TensorView& maximal_indices, TensorView& input_gradient, const PoolingArguments& args);
void average_pooling_backward(const TensorView& input, const TensorView& output, const TensorView& output_gradient, TensorView& input_gradient, const PoolingArguments& args);

// Pooling 3D

void max_pooling_3d_forward(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training);
void average_pooling_3d_forward(const TensorView& input, TensorView& output);
void max_pooling_3d_backward(const TensorView& maximal_indices, const TensorView& output_gradient, TensorView& input_gradient);
void average_pooling_3d_backward(const TensorView& input, const TensorView& output_gradient, TensorView& input_gradient);

// Embedding

void embedding_backward(const TensorView& input_indices, const TensorView& output_gradient, TensorView& weight_gradient, Index embedding_dimension, bool scale_embedding);

// Multi-head attention

void projection(const TensorView& input, const TensorView& weights, const TensorView& biases, TensorView& output, float* transpose_scratch);

void split_heads(const TensorView& source, TensorView& destination);
void merge_heads(const TensorView& source, TensorView& destination);

void projection_gradient(const TensorView& head_gradient,
                         const TensorView& input,
                         const TensorView& weights,
                         TensorView& bias_gradient,
                         TensorView& weight_gradient,
                         TensorView& input_gradient,
                         float* transpose_scratch,
                         bool accumulate);

void attention_masks(const TensorView& source_input, TensorView& attention_weights, const MatrixR& causal_mask, bool use_causal_mask, float* padding_mask_scratch);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
