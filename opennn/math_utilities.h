//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M A T H   U T I L I T I E S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"
#include "tensor_utilities.h"
#include "random_utilities.h"

#pragma once

namespace opennn
{

enum class ActivationFunction{
    Linear,
    Sigmoid,
    HyperbolicTangent,
    RectifiedLinear,
    ScaledExponentialLinear,
    Softmax,
    Logistic
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
    string activation_function;
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
    cudnnConvolutionBwdDataAlgo_t algorithm_data;
    cudnnConvolutionBwdFilterAlgo_t algorithm_filter;
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


inline void max_pooling(const TensorView& input,
                        TensorView& output,
                        TensorView& maximal_indices,
                        const PoolingArguments& arguments,
                        bool is_training = false)
{
#ifndef CUDA
    const TensorMap4 inputs = input.as_tensor<4>();
    TensorMap4 outputs = output.as_tensor<4>();

    const Index batch_size = inputs.dimension(0);
    const Index input_height = inputs.dimension(1);
    const Index input_width = inputs.dimension(2);
    const Index channels = inputs.dimension(3);

    const Index output_height = outputs.dimension(1);
    const Index output_width = outputs.dimension(2);

    const Index pool_height = arguments.pool_dimensions[0];
    const Index pool_width = arguments.pool_dimensions[1];
    const Index row_stride = arguments.stride_shape[0];
    const Index column_stride = arguments.stride_shape[1];
    const Index padding_height = arguments.padding_shape[0];
    const Index padding_width = arguments.padding_shape[1];

#pragma omp parallel for collapse(2)
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for(Index channel_index = 0; channel_index < channels; ++channel_index)
            for(Index output_row = 0; output_row < output_height; ++output_row)
                for(Index output_column = 0; output_column < output_width; ++output_column)
                {
                    const Index input_row_start = output_row * row_stride - padding_height;
                    const Index input_column_start = output_column * column_stride - padding_width;

                    type maximum_value = -numeric_limits<type>::infinity();
                    Index maximum_index = 0;

                    for(Index pool_row = 0; pool_row < pool_height; ++pool_row)
                        for(Index pool_column = 0; pool_column < pool_width; ++pool_column)
                        {
                            const Index input_row = input_row_start + pool_row;
                            const Index input_column = input_column_start + pool_column;

                            if(input_row >= 0 && input_row < input_height && input_column >= 0 && input_column < input_width)
                            {
                                const type current_value = inputs(batch_index, input_row, input_column, channel_index);

                                if(current_value > maximum_value)
                                {
                                    maximum_value = current_value;
                                    maximum_index = pool_row * pool_width + pool_column;
                                }
                            }
                        }

                    outputs(batch_index, output_row, output_column, channel_index) =
                        (maximum_value == -numeric_limits<type>::infinity()) ? type(0) : maximum_value;

                    if(is_training)
                    {
                        TensorMap4 maximal_indices_map = maximal_indices.as_tensor<4>();
                        maximal_indices_map(batch_index, output_row, output_column, channel_index) = maximum_index;
                    }
                }

#else
    CHECK_CUDNN(cudnnPoolingForward(get_cudnn_handle(),
                                    arguments.pooling_descriptor,
                                    &one,
                                    input.descriptor,
                                    input.device,
                                    &zero,
                                    output.descriptor,
                                    output.device));
#endif
}


inline void average_pooling(const TensorView& input, TensorView& output, const PoolingArguments& arguments)
{
#ifndef CUDA
    const TensorMap4 inputs = input.as_tensor<4>();
    const TensorMap4 outputs = output.as_tensor<4>();

    const Index batch_size = inputs.dimension(0);
    const Index input_height = inputs.dimension(1);
    const Index input_width = inputs.dimension(2);
    const Index channels = inputs.dimension(3);
    const Index output_height = outputs.dimension(1);
    const Index output_width = outputs.dimension(2);
/*
    const type inv_pool_size = type(1) / (pool_height * pool_width);

#pragma omp parallel for collapse(2)
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for(Index channel_index = 0; channel_index < channels; ++channel_index)
            for(Index output_row = 0; output_row < output_height; ++output_row)
                for(Index output_column = 0; output_column < output_width; ++output_column)
                {
                    const Index input_row_start = output_row * row_stride - padding_height;
                    const Index input_column_start = output_column * column_stride - padding_width;

                    type sum = 0;

                    for(Index pool_row = 0; pool_row < pool_height; ++pool_row)
                        for(Index pool_column = 0; pool_column < pool_width; ++pool_column)
                        {
                            const Index input_row = input_row_start + pool_row;
                            const Index input_column = input_column_start + pool_column;

                            if(input_row >= 0 && input_row < input_height && input_column >= 0 && input_column < input_width)
                                sum += inputs(batch_index, input_row, input_column, channel_index);
                        }

                    outputs(batch_index, output_row, output_column, channel_index) = sum * inv_pool_size;
                }
*/
#else

#endif
}


inline void padding(const TensorView& input, TensorView& output)
{       
#ifndef CUDA
    const TensorMap4 input_map = input.as_tensor<4>();
    TensorMap4 output_map = output.as_tensor<4>();
/*
    output_map.device(get_device()) = input_map.pad(input_map);
*/
#else

#endif
}

inline void bounding(const TensorView& input,
                     const TensorView& lower_bounds,
                     const TensorView& upper_bounds,
                     TensorView& output)
{
    const Index features = lower_bounds.size();

#ifndef CUDA
    const MatrixMap input_matrix = input.as_matrix();
    const VectorMap lower_bounds_vector = lower_bounds.as_vector();
    const VectorMap upper_bounds_vector = upper_bounds.as_vector();

    MatrixMap output_matrix = output.as_matrix();

    for(Index j = 0; j < features; ++j)
        output_matrix.col(j) = input_matrix.col(j)
                                           .cwiseMax(lower_bounds_vector(j))
                                           .cwiseMin(upper_bounds_vector(j));

#else
    const Index total_rows = input.shape[0];

    bounding_cuda(input.size(),
                  total_rows,
                  features,
                  input.data,
                  lower_bounds.device, // Assuming these were moved to device
                  upper_bounds.device,
                  output.data);
#endif
}


inline void copy(const TensorView& source, TensorView& destination)
{
    if(source.size() != destination.size())
        throw runtime_error("Math Error: Tensor sizes mismatch in copy operation.");

#ifndef CUDA
    destination.as_vector() = source.as_vector();
#else
    CHECK_CUDA(cudaMemcpy(destination.data,
                          source.data,
                          source.size() * sizeof(type),
                          cudaMemcpyDeviceToDevice));
#endif
}

inline void addition(const TensorView& input_1, const TensorView& input_2, TensorView& output)
{
    if(input_1.size() != input_2.size() || input_1.size() != output.size())
        throw runtime_error("Addition Error: Tensor dimensions do not match.");

#ifndef CUDA
    output.as_vector().array() = input_1.as_vector().array() + input_2.as_vector().array();
#else
    CHECK_CUDNN(cudnnOpTensor(get_cudnn_handle(),
                              get_operator_sum_descriptor(),
                              &one,
                              input_1.get_descriptor(),
                              input_1.data,
                              &one,
                              input_2.get_descriptor(),
                              input_2.data,
                              &zero,
                              output.get_descriptor(),
                              output.data));
#endif
}


inline void projection(const TensorView& input,
                       const TensorView& weights,
                       const TensorView& biases,
                       TensorView& output)
{
//    multiply(input, false, weights, false, output, type(1.0), type(0.0));

//    addition(biases, output);
}


inline void projection_gradient(const Tensor4& d_head,
                                const TensorMap3& input,
                                const TensorView& weights,
                                VectorMap& d_bias,
                                MatrixMap& d_weights,
                                TensorMap3& d_input,
                                Index batch_size,
                                bool accumulate)
{
/*
    const Index sequence_length = input.dimension(1);
    const Index embedding_dimension = get_embedding_dimension();
    const Index head_dimension = get_head_dimension();

    const MatrixMap W = matrix_map(weights);

#pragma omp parallel for
    for (Index b = 0; b < batch_size; ++b)
    {
        type* dx_ptr = d_input.data() + b * (sequence_length * embedding_dimension);
        MatrixMap dX_b(dx_ptr, sequence_length, embedding_dimension);

        if(!accumulate)
            dX_b.setZero();

        for (Index h = 0; h < heads_number; ++h)
        {
            const type* delta_ptr =
                d_head.data()
                + b * (heads_number * sequence_length * head_dimension)
                + h * (sequence_length * head_dimension);

            const MatrixMap Delta_bh(const_cast<type*>(delta_ptr), sequence_length, head_dimension);

            auto W_h = W.block(0, h * head_dimension, embedding_dimension, head_dimension);

            dX_b.noalias() += Delta_bh * W_h.transpose();
        }
    }

#pragma omp parallel for
    for (Index h = 0; h < heads_number; ++h)
    {
        auto dW_h = d_weights.block(0, h * head_dimension, embedding_dimension, head_dimension);
        auto db_h = d_bias.segment(h * head_dimension, head_dimension);

        dW_h.setZero();
        db_h.setZero();

        for (Index b = 0; b < batch_size; ++b)
        {
            const type* delta_ptr = d_head.data() + b * (heads_number * sequence_length * head_dimension) + h * (sequence_length * head_dimension);
            const MatrixMap Delta_bh(const_cast<type*>(delta_ptr), sequence_length, head_dimension);

            const type* in_ptr = input.data() + b * (sequence_length * embedding_dimension);
            const MatrixMap X_b(const_cast<type*>(in_ptr), sequence_length, embedding_dimension);

            dW_h.noalias() += X_b.transpose() * Delta_bh;
            db_h.noalias() += Delta_bh.colwise().sum().transpose();
        }
    }
*/
}


inline void batch_normalization(const TensorView& input, TensorView& output)
{
#ifndef CUDA
/*
    const Index batch_size = inputs.dimension(0);
    const Index features_number = inputs.dimension(1);

    const array<int, 1> reduction_axis({0});

    const array<Index, 2> reshape_dims({1, features_number});
    const array<Index, 2> broadcast_dims({batch_size, 1});

    means.device(get_device()) = inputs.mean(reduction_axis);

    standard_deviations.device(get_device()) = (inputs - means.reshape(reshape_dims).broadcast(broadcast_dims))
                                                   .square()
                                                   .mean(reduction_axis)
                                                   .sqrt();

    outputs.device(get_device()) = (inputs - means.reshape(reshape_dims).broadcast(broadcast_dims)) /
                                   (standard_deviations.reshape(reshape_dims).broadcast(broadcast_dims) + EPSILON);

    if (batch_normalization && parameters[Gammas].data != nullptr && parameters[Betas].data != nullptr)
    {
        const MatrixMap gammas_map(parameters[Gammas].data, 1, features_number);
        const MatrixMap betas_map(parameters[Betas].data, 1, features_number);

        TensorMap2 g(parameters[Gammas].data, 1, features_number);
        TensorMap2 b(parameters[Betas].data, 1, features_number);

        outputs.device(get_device()) = outputs * g.broadcast(broadcast_dims) + b.broadcast(broadcast_dims);
    }
*/
#else
    if (is_training)
        CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
            get_cudnn_handle(),
            CUDNN_BATCHNORM_PER_ACTIVATION,
            &one,
            &zero,
            outputs.get_descriptor(),
            outputs_buffer,
            outputs.get_descriptor(),
            outputs_buffer,
            gammas_device.get_descriptor(),
            gammas_device.data,
            betas_device.data,
            arguments.momentum,
            running_means_device.data,
            running_variances_device.data,
            EPSILON,
            means.data,
            inverse_variance.data));
    else
        CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
            get_cudnn_handle(),
            CUDNN_BATCHNORM_PER_ACTIVATION,
            &one, &zero,
            outputs.get_descriptor(),
            outputs_buffer,
            outputs.get_descriptor(),
            outputs_buffer,
            gammas_device.get_descriptor(),
            gammas_device.data,
            betas_device.data,
            running_means_device.data,
            running_variances_device.data,
            EPSILON));
#endif
}


inline void batch_normalization_backward(
    const TensorView& input,
    const TensorView& output,
    const TensorView& output_gradient,
    const TensorView& mean,
    const TensorView& inverse_variance, // Stored as 1/sqrt(variance + epsilon)
    const TensorView& gamma,
    TensorView& gamma_gradient,
    TensorView& beta_gradient,
    TensorView& input_gradient)
{
#ifndef CUDA
    const Index neurons_number = gamma.size();
    // Handles Dense (batch) and Convolutional (batch * height * width)
    const Index effective_batch_size = input.size() / neurons_number;

    const MatrixMap input_matrix(input.data, effective_batch_size, neurons_number);
    const MatrixMap output_gradients(output_gradient.data, effective_batch_size, neurons_number);

    const VectorMap means = mean.as_vector();
    const VectorMap inverse_variances = inverse_variance.as_vector();
    const VectorMap gammas = gamma.as_vector();

    VectorMap gamma_gradients = gamma_gradient.as_vector();
    VectorMap beta_gradients = beta_gradient.as_vector();
    MatrixMap input_gradients(input_gradient.data, effective_batch_size, neurons_number);

    // 1. Calculate x_hat (normalized input) using inverse_variance
    const MatrixR x_hat = (input_matrix.rowwise() - means.transpose()).array().rowwise() * inverse_variances.transpose().array();

    // 2. Beta gradient: sum of output gradients
    beta_gradients.noalias() = output_gradients.colwise().sum();
/*
    // 3. Gamma gradient: sum of (output gradient * x_hat)
    gamma_gradients.noalias() = (output_gradients.array() * x_hat.array()).colwise().sum();
*/
    // 4. Input gradient (input_gradient):
    // Formula: (gamma * inv_std / m) * (m * dy - sum_dy - x_hat * sum_dy_xhat)
    const type batch_size_type = static_cast<type>(effective_batch_size);

    input_gradients.array() = (gammas.array() * inverse_variances.array() / batch_size_type).transpose().replicate(effective_batch_size, 1) *
                              (batch_size_type * output_gradients.array() -
                               beta_gradients.transpose().replicate(effective_batch_size, 1).array() -
                               x_hat.array() * gamma_gradients.transpose().replicate(effective_batch_size, 1).array());

#else
    // Use SPATIAL for 4D (Conv) and PER_ACTIVATION for 2D (Dense)

    const cudnnBatchNormMode_t mode = (input.rank() == 4)
        ? CUDNN_BATCHNORM_SPATIAL
        : CUDNN_BATCHNORM_PER_ACTIVATION;

    CHECK_CUDNN(cudnnBatchNormalizationBackward(
        get_cudnn_handle(),
        mode,
        &one, &zero, // Data alpha/beta
        &one, &zero, // Param alpha/beta
        input.get_descriptor(),
        input.data,
        output_gradient.get_descriptor(),
        output_gradient.data,
        input_gradient.get_descriptor(),
        input_gradient.data,
        gamma.get_descriptor(),
        gamma.data,
        gamma_gradient.data,
        beta_gradient.data,
        EPSILON,
        mean.data,
        inverse_variance.data));
#endif
}

inline void combination(const TensorView& input,
                        const TensorView& weights,
                        const TensorView& biases,
                        TensorView& output)
{
#ifndef CUDA
    output.as_matrix().noalias()
        = (input.as_matrix() * weights.as_matrix()).rowwise() + biases.as_vector().transpose();
#else

    multiply(input, false, weights, false, output, 1.0f, 0.0f);

    CHECK_CUDNN(cudnnAddTensor(get_cudnn_handle(),
                               &one, biases.get_descriptor(), biases.data,
                               &one, output.get_descriptor(), output.data));
#endif
}

inline void batch_normalization_training(
    const TensorView& input,
    const TensorView& gamma,
    const TensorView& beta,
    VectorR& running_mean,
    VectorR& running_variance,
    TensorView& mean,             // Output: current batch mean
    TensorView& inverse_variance, // Output: current batch 1/sqrt(var + epsilon)
    TensorView& output,
    type momentum = type(0.9))
{
#ifndef CUDA
    const Index neurons_number = gamma.size();
    const Index effective_batch_size = input.size() / neurons_number;

    const MatrixMap input_matrix(input.data, effective_batch_size, neurons_number);
    MatrixMap output_matrix(output.data, effective_batch_size, neurons_number);

    const VectorMap gammas = gamma.as_vector();
    const VectorMap betas = beta.as_vector();

    VectorMap means = mean.as_vector();
    VectorMap inverse_variances = inverse_variance.as_vector();

    means = input_matrix.colwise().mean();

    const VectorR batch_variance = (input_matrix.rowwise() - means.transpose()).array().square().colwise().mean();
    inverse_variances.array() = 1.0f / (batch_variance.array() + EPSILON).sqrt();

    running_mean = running_mean * momentum + means * (type(1) - momentum);
    running_variance = running_variance * momentum + batch_variance * (type(1) - momentum);

    output_matrix.array() = (input_matrix.rowwise() - means.transpose()).array().rowwise() *
                            (inverse_variances.array() * gammas.array()).transpose();

    output_matrix.rowwise() += betas.transpose();

#else
    const cudnnBatchNormMode_t mode = (input.rank() == 4)
        ? CUDNN_BATCHNORM_SPATIAL
        : CUDNN_BATCHNORM_PER_ACTIVATION;

    CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
        get_cudnn_handle(),
        mode,
        &one, &zero,
        input.get_descriptor(), input.data,
        output.get_descriptor(), output.data,
        gamma.get_descriptor(), gamma.data,
        beta.data,
        static_cast<double>(momentum),
        running_mean.data(), running_variance.data(),
        EPSILON,
        mean.data,
        inverse_variance.data));
#endif
}


inline void activation(TensorView& output, ActivationFunction func)
{
    if (output.empty() || func == ActivationFunction::Linear)
        return;

#ifndef CUDA
    auto arr = output.as_vector().array();

    switch (func)
    {
    case ActivationFunction::Sigmoid:
    case ActivationFunction::Logistic:
    {
        arr = (1.0f + (-arr).exp()).inverse();
        return;
    }

    case ActivationFunction::HyperbolicTangent:
    {
        arr = arr.tanh();
        return;
    }

    case ActivationFunction::RectifiedLinear:
    {
        arr = arr.cwiseMax(0.0f);
        return;
    }

    case ActivationFunction::ScaledExponentialLinear:
    {
        const float alpha = 1.6732632423543772848170429916717f;
        const float lambda = 1.0507009873554804934193349852946f;

        arr = lambda * (arr > 0.0f).select(arr, alpha * (arr.exp() - 1.0f));
        return;
    }

    case ActivationFunction::Softmax:
    {
//        softmax(output);
        return;
    }

    default:
        return;
    }
#else
    func == ActivationFunction::Softmax
        ? CHECK_CUDNN(cudnnSoftmaxForward(get_cudnn_handle(),
                                          CUDNN_SOFTMAX_ACCURATE,
                                          CUDNN_SOFTMAX_MODE_CHANNEL, // Softmax per spatial location
                                          &one,
                                          output.get_descriptor(), output.data,
                                          &zero,
                                          output.get_descriptor(), output.data))
        : CHECK_CUDNN(cudnnActivationForward(get_cudnn_handle(),
                                             act_desc,
                                             &one,
                                             output.get_descriptor(), output.data,
                                             &zero,
                                             output.get_descriptor(), output.data));

#endif
}



inline void activation_gradient(const TensorView& outputs,
                                const TensorView& output_gradient,
                                TensorView& activation_derivative,
                                const ActivationFunction func)
{
    if (outputs.empty()) return;

#ifndef CUDA
    const auto y = outputs.as_vector().array();
    const auto dy = output_gradient.as_vector().array();
    auto dx = activation_derivative.as_vector().array();

    switch (func)
    {
    case ActivationFunction::Linear:
    {
        dx = dy;
        return;
    }

    case ActivationFunction::Sigmoid:
    case ActivationFunction::Logistic:
    {
        dx = dy * (y * (1.0f - y));
        return;
    }

    case ActivationFunction::HyperbolicTangent:
    {
        dx = dy * (1.0f - y.square());
        return;
    }

    case ActivationFunction::RectifiedLinear:
    {
        dx = (y > 0.0f).select(dy, 0.0f);
        return;
    }

    case ActivationFunction::ScaledExponentialLinear:
    {
        const float alpha = 1.6732632423543772848170429916717f;
        const float lambda = 1.0507009873554804934193349852946f;

        dx = (y > 0.0f).select(lambda * dy, (y + (alpha * lambda)) * dy);
        return;
    }

    case ActivationFunction::Softmax:
    {
        dx = dy;
        return;
    }

    default:
        throw runtime_error("Math Error: Unknown activation function in activation_gradient.");
    }
#else
    CHECK_CUDNN(cudnnActivationBackward(get_cudnn_handle(),
                                        activation_descriptor,
                                        &one,
                                        gradients.descriptor,
                                        outputs.data,
                                        gradients_tensor_descriptor,
                                        output_gradients_data,
                                        gradients_tensor_descriptor,
                                        (use_convolutions() && convolutions) ? convolutions : outputs_view.data,
                                        &zero,
                                        gradients_tensor_descriptor,
                                        output_gradients_data));

#endif
}


inline void dropout(TensorView& output, type dropout_rate)
{
#ifndef CUDA
    const type scale = type(1) / (type(1) - dropout_rate);

    type* data = output.data;
    const Index n = output.size();

    for (Index i = 0; i < n; ++i)
        data[i] = (random_uniform(type(0), type(1)) < dropout_rate) ? type(0) : data[i] * scale;
#else
    CHECK_CUDNN(cudnnDropoutForward(get_cudnn_handle(),
                                    dropout_descriptor,
                                    tensor.get_descriptor(),
                                    tensor.data,
                                    tensor.get_descriptor(),
                                    tensor.data,
                                    dropout_reserve_space,
                                    dropout_reserve_space_size));
#endif
}


inline void dropout_gradient(const TensorView& output_gradient,
                             const TensorView& mask, type dropout_rate,
                             TensorView& input_gradient)
{
#ifndef CUDA
    const type scale = type(1) / (type(1) - dropout_rate);

    input_gradient.as_vector().array() = output_gradient.as_vector().array() * mask.as_vector().array().cast<type>() * scale;
#else
    CHECK_CUDNN(cudnnDropoutBackward(get_cudnn_handle(),
                                     dropout_descriptor,
                                     gradients.descriptor,
                                     output_gradients.device,
                                     gradients.descriptor,
                                     output_gradients.data,
                                     dropout_reserve_space,
                                     dropout_reserve_space_size));
#endif
}

inline void convolution(const TensorView& input,
                        const TensorView& kernel,
                        const TensorView& bias,
                        TensorView& output)
{
#ifndef CUDA

    const TensorMap4 inputs = input.as_tensor<4>();
    const VectorMap biases = bias.as_vector();

    const Index batch_size = inputs.dimension(0);
/*
    const Index output_height = convolutions.dimension(1);
    const Index output_width = convolutions.dimension(2);

    const Index kernels_number = get_kernels_number();
    const Index kernel_height = get_kernel_height();
    const Index kernel_width = get_kernel_width();
    const Index kernel_channels = get_kernel_channels();

    const Index single_kernel_size = kernel_height * kernel_width * kernel_channels;

    const Eigen::array<Index, 3> conv_dims({1, 2, 3});

    const Eigen::array<Index, 3> out_slice_shape({batch_size, output_height, output_width});

    for(Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {
        type* current_kernel_ptr = parameters[Weights].data + (kernel_index * single_kernel_size);
        TensorMap3 kernel_weights(current_kernel_ptr, kernel_height, kernel_width, kernel_channels);

        convolutions.chip(kernel_index, 3).device(get_device()) =
            inputs.convolve(kernel_weights, conv_dims).reshape(out_slice_shape) + biases(kernel_index);
    }
*/
#else
    CHECK_CUDNN(cudnnConvolutionForward(get_cudnn_handle(),
                                        &one,
                                        input.descriptor,
                                        input.device,
                                        kernel_descriptor,
                                        weights_device.data,
                                        convolution_descriptor,
                                        convolution_algorithm,
                                        workspace,
                                        workspace_size,
                                        &zero,
                                        current_output_descriptor,
                                        outputs_buffer));

    // Biases

    CHECK_CUDNN(cudnnAddTensor(get_cudnn_handle(),
                               &one,
                               biases.get_descriptor(),
                               biases.data,
                               &one,
                               current_output_descriptor,
                               outputs_buffer));

#endif
}


inline void convolution_activation(const TensorView& input,
                                   const TensorView& weight,
                                   const TensorView& bias,
                                   TensorView& output,
                                   const string& activation)
{
#ifndef CUDA
    convolution(input, weight, bias, output);

//    activation(output, activation);
#else
    CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
        get_cudnn_handle(),
        &one,
        input.descriptor,
        input_data,
        kernel_descriptor,
        weights_device.data,
        convolution_descriptor,
        convolution_algorithm,
        workspace,
        workspace_size,
        &zero,
        current_output_descriptor,
        outputs.data,
        biases_device.get_descriptor(),
        biases_device.data,
        activation_descriptor,
        current_output_descriptor,
        outputs.data));
#endif
}


inline void multiply(const TensorView& input_A, bool transpose_A,
                     const TensorView& input_B, bool transpose_B,
                     TensorView& output_C,
                     type alpha = 1.0f, type beta = 0.0f)
{
    const size_t rank = input_A.rank();

#ifndef CUDA
    if (rank <= 2)
    {
        const auto matrix_A = input_A.as_matrix();
        const auto matrix_B = input_B.as_matrix();
        auto matrix_C = output_C.as_matrix();

        if (!transpose_A && !transpose_B)
            matrix_C.noalias() = alpha * (matrix_A * matrix_B) + beta * matrix_C;
        else if (transpose_A && !transpose_B)
            matrix_C.noalias() = alpha * (matrix_A.transpose() * matrix_B) + beta * matrix_C;
        else if (!transpose_A && transpose_B)
            matrix_C.noalias() = alpha * (matrix_A * matrix_B.transpose()) + beta * matrix_C;
        else
            matrix_C.noalias() = alpha * (matrix_A.transpose() * matrix_B.transpose()) + beta * matrix_C;
    }
    else
    {
        // For Rank 3 or 4, we loop over the outer dimensions (batch/heads) on CPU
        const Index outer_dimensions_count = input_A.size() / (input_A.shape[rank - 2] * input_A.shape[rank - 1]);
        const Index size_A = input_A.shape[rank - 2] * input_A.shape[rank - 1];
        const Index size_B = input_B.shape[rank - 2] * input_B.shape[rank - 1];
        const Index size_C = output_C.shape[rank - 2] * output_C.shape[rank - 1];

#pragma omp parallel for
        for (Index i = 0; i < outer_dimensions_count; ++i)
        {
            const MatrixMap mat_A(input_A.data + i * size_A, input_A.shape[rank - 2], input_A.shape[rank - 1]);
            const MatrixMap mat_B(input_B.data + i * size_B, input_B.shape[rank - 2], input_B.shape[rank - 1]);
            MatrixMap mat_C(output_C.data + i * size_C, output_C.shape[rank - 2], output_C.shape[rank - 1]);

            if (!transpose_A && !transpose_B)
                mat_C.noalias() = alpha * (mat_A * mat_B) + beta * mat_C;
            else if (transpose_A && !transpose_B)
                mat_C.noalias() = alpha * (mat_A.transpose() * mat_B) + beta * mat_C;
            else if (!transpose_A && transpose_B)
                mat_C.noalias() = alpha * (mat_A * mat_B.transpose()) + beta * mat_C;
            else
                mat_C.noalias() = alpha * (mat_A.transpose() * mat_B.transpose()) + beta * mat_C;
        }
    }
#else
    // M, N, K are derived from the last two dimensions (the matrix dimensions)
    const int m = transpose_B ? (int)input_B.shape[rank - 2] : (int)input_B.shape[rank - 1];
    const int n = transpose_A ? (int)input_A.shape[rank - 1] : (int)input_A.shape[rank - 2];
    const int k = transpose_A ? (int)input_A.shape[rank - 2] : (int)input_A.shape[rank - 1];

    const int lda = (int)input_A.shape[rank - 1];
    const int ldb = (int)input_B.shape[rank - 1];
    const int ldc = m;

    // Calculate how many matrices are in the batch (e.g., batch_size * heads_number)
    const int batch_count = (int)(input_A.size() / (input_A.shape[rank - 2] * input_A.shape[rank - 1]));

    // Strides: how far to jump in memory to get to the next matrix
    const long long stride_A = input_A.shape[rank - 2] * input_A.shape[rank - 1];
    const long long stride_B = input_B.shape[rank - 2] * input_B.shape[rank - 1];
    const long long stride_C = output_C.shape[rank - 2] * output_C.shape[rank - 1];

    CHECK_CUBLAS(cublasSgemmStridedBatched(get_cublas_handle(),
                                           transpose_B ? CUBLAS_OP_N : CUBLAS_OP_T,
                                           transpose_A ? CUBLAS_OP_N : CUBLAS_OP_T,
                                           m, n, k,
                                           &one,
                                           input_B.data, ldb, stride_B,
                                           input_A.data, lda, stride_A,
                                           &zero,
                                           output_C.data, ldc, stride_C,
                                           batch_count));
#endif
}


inline void multiply_elementwise(const TensorView& A, const TensorView& B, TensorView& C)
{
#ifndef OPENNN_CUDA
    C.as_vector().array() = A.as_vector().array() * B.as_vector().array();
#else
    CHECK_CUDNN(cudnnOpTensor(get_cudnn_handle(), get_operator_multiplication_descriptor(),
                              &one, A.get_descriptor(), A.data,
                              &one, B.get_descriptor(), B.data,
                              &zero, C.get_descriptor(), C.data));
#endif
}

inline void sum(const TensorView& A, TensorView& B, type alpha = 1.0f, type beta = 0.0f)
{
#ifndef OPENNN_CUDA
    B.as_vector().noalias() = alpha * A.as_matrix().colwise().sum() + beta * B.as_vector();
#else
    CHECK_CUDNN(cudnnAddTensor(get_cudnn_handle(),
                               &one,
                               input_tensor.descriptor,
                               input_tensor.device,
                               &beta_one,
                               output_tensor.descriptor,
                               output_tensor.data));
#endif
}


inline void softmax(TensorView& output)
{
    if (output.empty()) return;

#ifndef CUDA

    const Index columns = output.shape.back();
    const Index rows = output.size() / columns;

    MatrixMap mat(output.data, rows, columns);
    mat.colwise() -= mat.rowwise().maxCoeff();
    mat.array() = mat.array().exp();
    mat.array().colwise() /= mat.rowwise().sum().array();

#else
    CHECK_CUDNN(cudnnSoftmaxForward(get_cudnn_handle(),
                                    CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &one,
                                    output.get_descriptor(), output.data,
                                    &zero,
                                    output.get_descriptor(), output.data));
#endif
}

}
