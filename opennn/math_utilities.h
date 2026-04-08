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
    Linear,
    Sigmoid,
    HyperbolicTangent,
    RectifiedLinear,
    ScaledExponentialLinear,
    Softmax
};


inline ActivationFunction string_to_activation(const string& name)
{
    if (name == "Sigmoid" || name == "Logistic") return ActivationFunction::Sigmoid;
    if (name == "HyperbolicTangent") return ActivationFunction::HyperbolicTangent;
    if (name == "RectifiedLinear") return ActivationFunction::RectifiedLinear;
    if (name == "ScaledExponentialLinear") return ActivationFunction::ScaledExponentialLinear;
    if (name == "Softmax") return ActivationFunction::Softmax;
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
    type momentum = type(0.9);

#ifdef CUDA
    cudnnBatchNormMode_t batch_normalization_mode = CUDNN_BATCHNORM_PER_ACTIVATION;
    cudnnTensorDescriptor_t per_activation_descriptor = nullptr;
#endif
};


//  Generic operators

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
    output.as_vector() = input_1.as_vector() + input_2.as_vector();
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

inline void multiply(const TensorView& input_A, bool transpose_A,
                     const TensorView& input_B, bool transpose_B,
                     TensorView& output_C,
                     type alpha = 1.0f, type beta = 0.0f)
{
    const size_t rank = input_A.get_rank();

#ifndef CUDA
    auto gemm = [&](const MatrixMap& A, const MatrixMap& B, MatrixMap& C)
    {
        if (!transpose_A && !transpose_B)
            C.noalias() = alpha * (A * B) + beta * C;
        else if (transpose_A && !transpose_B)
            C.noalias() = alpha * (A.transpose() * B) + beta * C;
        else if (!transpose_A && transpose_B)
            C.noalias() = alpha * (A * B.transpose()) + beta * C;
        else
            C.noalias() = alpha * (A.transpose() * B.transpose()) + beta * C;
    };

    if (rank <= 2)
    {
        auto matrix_A = input_A.as_matrix();
        auto matrix_B = input_B.as_matrix();
        auto matrix_C = output_C.as_matrix();
        gemm(matrix_A, matrix_B, matrix_C);
    }
    else
    {
        const Index outer = input_A.size() / (input_A.shape[rank - 2] * input_A.shape[rank - 1]);
        const Index size_A = input_A.shape[rank - 2] * input_A.shape[rank - 1];
        const Index size_B = input_B.shape[rank - 2] * input_B.shape[rank - 1];
        const Index size_C = output_C.shape[rank - 2] * output_C.shape[rank - 1];

#pragma omp parallel for
        for (Index i = 0; i < outer; ++i)
        {
            MatrixMap mat_A(input_A.data + i * size_A, input_A.shape[rank - 2], input_A.shape[rank - 1]);
            MatrixMap mat_B(input_B.data + i * size_B, input_B.shape[rank - 2], input_B.shape[rank - 1]);
            MatrixMap mat_C(output_C.data + i * size_C, output_C.shape[rank - 2], output_C.shape[rank - 1]);
            gemm(mat_A, mat_B, mat_C);
        }
    }
#else
    const int m = transpose_B ? static_cast<int>(input_B.shape[rank - 2]) : static_cast<int>(input_B.shape[rank - 1]);
    const int n = transpose_A ? static_cast<int>(input_A.shape[rank - 1]) : static_cast<int>(input_A.shape[rank - 2]);
    const int k = transpose_A ? static_cast<int>(input_A.shape[rank - 2]) : static_cast<int>(input_A.shape[rank - 1]);

    const int lda = static_cast<int>(input_A.shape[rank - 1]);
    const int ldb = static_cast<int>(input_B.shape[rank - 1]);
    const int ldc = m;

    const int batch_count = static_cast<int>(input_A.size() / (input_A.shape[rank - 2] * input_A.shape[rank - 1]));

    const long long stride_A = input_A.shape[rank - 2] * input_A.shape[rank - 1];
    const long long stride_B = input_B.shape[rank - 2] * input_B.shape[rank - 1];
    const long long stride_C = output_C.shape[rank - 2] * output_C.shape[rank - 1];

    CHECK_CUBLAS(cublasSgemmStridedBatched(get_cublas_handle(),
                                           transpose_B ? CUBLAS_OP_N : CUBLAS_OP_T,
                                           transpose_A ? CUBLAS_OP_N : CUBLAS_OP_T,
                                           m, n, k,
                                           &alpha,
                                           input_B.data, ldb, stride_B,
                                           input_A.data, lda, stride_A,
                                           &beta,
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

inline void softmax(TensorView& output);


//  Dense layer operators

inline void combination(const TensorView& input,
                        const TensorView& weights,
                        const TensorView& biases,
                        TensorView& output)
{
#ifndef CUDA
    output.as_matrix().noalias()
        = (input.as_matrix() * weights.as_matrix()).rowwise() + biases.as_vector().transpose();
#else
    CHECK_CUDNN(cudnnAddTensor(get_cudnn_handle(),
                               &one, biases.get_descriptor(), biases.data,
                               &zero, output.get_descriptor(), output.data));

    multiply(input, false, weights, false, output, 1.0f, 1.0f);
#endif
}

inline void activation(TensorView& output, ActivationArguments arguments)
{
    if (output.empty()) return;

#ifndef CUDA

    const ActivationFunction activation_function = arguments.activation_function;

    auto arr = output.as_vector().array();

    switch (activation_function)
    {
    case ActivationFunction::Linear:
        return;

    case ActivationFunction::Sigmoid:
        arr = (1.0f + (-arr).exp()).inverse();
        return;

    case ActivationFunction::HyperbolicTangent:
        arr = arr.tanh();
        return;

    case ActivationFunction::RectifiedLinear:
        arr = arr.cwiseMax(0.0f);
        return;

    case ActivationFunction::ScaledExponentialLinear:
        arr = SELU_LAMBDA * (arr > 0.0f).select(arr, SELU_ALPHA * (arr.exp() - 1.0f));
        return;

    case ActivationFunction::Softmax:
        softmax(output);
        return;

    default:
        return;
    }
#else
    func == ActivationFunction::Softmax
        ? CHECK_CUDNN(cudnnSoftmaxForward(get_cudnn_handle(),
                                          CUDNN_SOFTMAX_ACCURATE,
                                          CUDNN_SOFTMAX_MODE_CHANNEL,
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
        dx = dy;
        return;

    case ActivationFunction::Sigmoid:
        dx = dy * (y * (1.0f - y));
        return;

    case ActivationFunction::HyperbolicTangent:
        dx = dy * (1.0f - y.square());
        return;

    case ActivationFunction::RectifiedLinear:
        dx = (y > 0.0f).select(dy, 0.0f);
        return;

    case ActivationFunction::ScaledExponentialLinear:
        dx = (y > 0.0f).select(SELU_LAMBDA * dy, (y + (SELU_ALPHA * SELU_LAMBDA)) * dy);
        return;

    case ActivationFunction::Softmax:
        dx = dy;
        return;

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


//  Batch normalization operators

inline void batch_normalization(const TensorView& input, TensorView& output)
{
#ifndef CUDA
    // @todo CPU batch normalization inference
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

inline void batch_normalization_training(
    const TensorView& input,
    const TensorView& gamma,
    const TensorView& beta,
    VectorR& running_mean,
    VectorR& running_variance,
    TensorView& mean,
    TensorView& inverse_variance,
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
    const cudnnBatchNormMode_t mode = (input.get_rank() == 4)
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

inline void batch_normalization_backward(
    const TensorView& input,
    const TensorView& output,
    const TensorView& output_gradient,
    const TensorView& mean,
    const TensorView& inverse_variance,
    const TensorView& gamma,
    TensorView& gamma_gradient,
    TensorView& beta_gradient,
    TensorView& input_gradient)
{
#ifndef CUDA
    const Index neurons_number = gamma.size();
    const Index effective_batch_size = input.size() / neurons_number;

    const MatrixMap input_matrix(input.data, effective_batch_size, neurons_number);
    const MatrixMap output_gradients(output_gradient.data, effective_batch_size, neurons_number);

    const VectorMap means = mean.as_vector();
    const VectorMap inverse_variances = inverse_variance.as_vector();
    const VectorMap gammas = gamma.as_vector();

    VectorMap gamma_gradients = gamma_gradient.as_vector();
    VectorMap beta_gradients = beta_gradient.as_vector();
    MatrixMap input_gradients(input_gradient.data, effective_batch_size, neurons_number);

    const MatrixR x_hat = (input_matrix.rowwise() - means.transpose()).array().rowwise() * inverse_variances.transpose().array();

    beta_gradients.noalias() = output_gradients.colwise().sum();

    const type batch_size_type = static_cast<type>(effective_batch_size);

    input_gradients.array() = (gammas.array() * inverse_variances.array() / batch_size_type).transpose().replicate(effective_batch_size, 1) *
                              (batch_size_type * output_gradients.array() -
                               beta_gradients.transpose().replicate(effective_batch_size, 1).array() -
                               x_hat.array() * gamma_gradients.transpose().replicate(effective_batch_size, 1).array());

#else
    const cudnnBatchNormMode_t mode = (input.get_rank() == 4)
        ? CUDNN_BATCHNORM_SPATIAL
        : CUDNN_BATCHNORM_PER_ACTIVATION;

    CHECK_CUDNN(cudnnBatchNormalizationBackward(
        get_cudnn_handle(),
        mode,
        &one, &zero,
        &one, &zero,
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


//  Scaling / Bounding layer operators

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
                  lower_bounds.device,
                  upper_bounds.device,
                  output.data);
#endif
}


//  Convolutional layer operators

inline void padding(const TensorView& input, TensorView& output)
{
#ifndef CUDA
    const TensorMap4 input_map = input.as_tensor<4>();
    TensorMap4 output_map = output.as_tensor<4>();

    const Index pad_height = (output_map.dimension(1) - input_map.dimension(1)) / 2;
    const Index pad_width = (output_map.dimension(2) - input_map.dimension(2)) / 2;

    const Eigen::array<pair<Index, Index>, 4> paddings =
        { make_pair(Index(0), Index(0)),
          make_pair(pad_height, pad_height),
          make_pair(pad_width, pad_width),
          make_pair(Index(0), Index(0)) };

    output_map.device(get_device()) = input_map.pad(paddings);
#else

#endif
}

inline void convolution(const TensorView& input,
                        const TensorView& kernel,
                        const TensorView& bias,
                        TensorView& output,
                        const ConvolutionArguments& arguments = {})
{
#ifndef CUDA
    const TensorMap4 inputs = input.as_tensor<4>();
    TensorMap4 outputs = output.as_tensor<4>();
    const VectorMap biases = bias.as_vector();

    const Index batch_size = inputs.dimension(0);
    const Index output_height = outputs.dimension(1);
    const Index output_width = outputs.dimension(2);

    const Index kernels_number = kernel.shape[0];
    const Index kernel_height = kernel.shape[1];
    const Index kernel_width = kernel.shape[2];
    const Index kernel_channels = kernel.shape[3];

    const Index single_kernel_size = kernel_height * kernel_width * kernel_channels;

    const Eigen::array<Index, 3> conv_dims({1, 2, 3});
    const Eigen::array<Index, 3> out_slice_shape({batch_size, output_height, output_width});

    for(Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {
        type* current_kernel_ptr = kernel.data + kernel_index * single_kernel_size;
        TensorMap3 kernel_weights(current_kernel_ptr, kernel_height, kernel_width, kernel_channels);

        outputs.chip(kernel_index, 3).device(get_device()) =
            inputs.convolve(kernel_weights, conv_dims).reshape(out_slice_shape) + biases(kernel_index);
    }
#else
    CHECK_CUDNN(cudnnConvolutionForward(get_cudnn_handle(),
                                        &one,
                                        input.get_descriptor(), input.data,
                                        arguments.kernel_descriptor,
                                        kernel.data,
                                        arguments.convolution_descriptor,
                                        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                        nullptr, 0,
                                        &zero,
                                        output.get_descriptor(), output.data));

    CHECK_CUDNN(cudnnAddTensor(get_cudnn_handle(),
                               &one,
                               bias.get_descriptor(), bias.data,
                               &one,
                               output.get_descriptor(), output.data));
#endif
}

inline void convolution_backward_weights(const TensorView& padded_input,
                                          const TensorView& output_gradient,
                                          TensorView& weight_gradient,
                                          TensorView& bias_gradient,
                                          const ConvolutionArguments& arguments = {})
{
#ifndef CUDA
    const TensorMap4 inputs = padded_input.as_tensor<4>();
    const TensorMap4 out_grad = output_gradient.as_tensor<4>();

    const Index kernels_number = weight_gradient.shape[0];
    const Index kernel_height = weight_gradient.shape[1];
    const Index kernel_width = weight_gradient.shape[2];
    const Index kernel_channels = weight_gradient.shape[3];
    const Index single_kernel_size = kernel_height * kernel_width * kernel_channels;

    const Eigen::array<Index, 3> conv_dims({0, 1, 2});

    #pragma omp parallel for
    for(Index ki = 0; ki < kernels_number; ki++)
    {
        const Tensor3 kernel_grads = out_grad.chip(ki, 3);

        TensorMap4 kernel_weight_grads(weight_gradient.data + (ki * single_kernel_size),
                                        1, kernel_height, kernel_width, kernel_channels);

        kernel_weight_grads.device(get_device()) = inputs.convolve(kernel_grads, conv_dims);
    }

    sum(output_gradient, bias_gradient);

#else
    CHECK_CUDNN(cudnnConvolutionBackwardFilter(get_cudnn_handle(),
                                                &one,
                                                padded_input.get_descriptor(), padded_input.data,
                                                output_gradient.get_descriptor(), output_gradient.data,
                                                arguments.convolution_descriptor,
                                                arguments.algorithm_filter,
                                                nullptr, 0,
                                                &zero,
                                                arguments.kernel_descriptor,
                                                weight_gradient.data));

    CHECK_CUDNN(cudnnConvolutionBackwardBias(get_cudnn_handle(),
                                              &one,
                                              output_gradient.get_descriptor(), output_gradient.data,
                                              &zero,
                                              bias_gradient.get_descriptor(), bias_gradient.data));
#endif
}

inline void convolution_backward_data(const TensorView& output_gradient,
                                       const TensorView& weights,
                                       TensorView& input_gradient,
                                       TensorView& rotated_weights_buffer,
                                       const ConvolutionArguments& arguments = {})
{
#ifndef CUDA
    const TensorMap4 out_grad = output_gradient.as_tensor<4>();
    TensorMap4 in_grad = input_gradient.as_tensor<4>();

    const Index batch_size = in_grad.dimension(0);
    const Index input_height = in_grad.dimension(1);
    const Index input_width = in_grad.dimension(2);
    const Index input_channels = in_grad.dimension(3);

    const Index kernels_number = weights.shape[0];
    const Index kernel_height = weights.shape[1];
    const Index kernel_width = weights.shape[2];

    in_grad.setZero();

    TensorMap4 rotated(rotated_weights_buffer.data, kernels_number, kernel_height, kernel_width, input_channels);
    const TensorMap4 weights_map(weights.data, kernels_number, kernel_height, kernel_width, input_channels);
    rotated.device(get_device()) = weights_map.reverse(Eigen::array<bool, 4>({false, true, true, false}));

    const Index out_h = out_grad.dimension(1);
    const Index out_w = out_grad.dimension(2);
    const Index pad_h = (input_height + kernel_height - 1) - out_h;
    const Index pad_w = (input_width + kernel_width - 1) - out_w;

    const Eigen::array<pair<Index, Index>, 2> paddings =
        {make_pair(pad_h / 2, pad_h - pad_h / 2), make_pair(pad_w / 2, pad_w - pad_w / 2)};

    vector<vector<Tensor2>> rotated_slices(kernels_number, vector<Tensor2>(input_channels));

    #pragma omp parallel for
    for(Index ki = 0; ki < kernels_number; ++ki)
    {
        auto kernel_rotated = rotated.chip(ki, 0);
        for(Index ci = 0; ci < input_channels; ++ci)
            rotated_slices[ki][ci] = kernel_rotated.chip(ci, 2);
    }

    const Eigen::array<Index, 2> conv_dims_2d = {0, 1};

    for(Index ki = 0; ki < kernels_number; ++ki)
    {
        auto kernel_grads = out_grad.chip(ki, 3);

        #pragma omp parallel for
        for(Index bi = 0; bi < batch_size; ++bi)
        {
            const Tensor2 padded_grads = kernel_grads.chip(bi, 0).pad(paddings);

            for(Index ci = 0; ci < input_channels; ++ci)
            {
                const Tensor2 result = padded_grads.convolve(rotated_slices[ki][ci], conv_dims_2d);

                for(Index h = 0; h < input_height; ++h)
                    for(Index w = 0; w < input_width; ++w)
                        in_grad(bi, h, w, ci) += result(h, w);
            }
        }
    }

#else
    CHECK_CUDNN(cudnnConvolutionBackwardData(get_cudnn_handle(),
                                              &one,
                                              weights.get_descriptor(), weights.data,
                                              output_gradient.get_descriptor(), output_gradient.data,
                                              arguments.convolution_descriptor,
                                              arguments.algorithm_data,
                                              nullptr, 0,
                                              &zero,
                                              input_gradient.get_descriptor(), input_gradient.data));
#endif
}


//  Pooling layer operators (4D — images)

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

    TensorMap4 maximal_indices_map = maximal_indices.as_tensor<4>();

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
                        maximal_indices_map(batch_index, output_row, output_column, channel_index) = maximum_index;
                }

#else
    CHECK_CUDNN(cudnnPoolingForward(get_cudnn_handle(),
                                    arguments.pooling_descriptor,
                                    &one,
                                    input.get_descriptor(), input.data,
                                    &zero,
                                    output.get_descriptor(), output.data));
#endif
}

inline void max_pooling_backward(const TensorView& output_gradient,
                                  const TensorView& maximal_indices,
                                  TensorView& input_gradient,
                                  const PoolingArguments& arguments)
{
#ifndef CUDA
    const TensorMap4 out_grad = output_gradient.as_tensor<4>();
    const TensorMap4 max_idx = maximal_indices.as_tensor<4>();
    TensorMap4 in_grad = input_gradient.as_tensor<4>();

    const Index batch_size = in_grad.dimension(0);
    const Index input_height = in_grad.dimension(1);
    const Index input_width = in_grad.dimension(2);
    const Index channels = in_grad.dimension(3);

    const Index output_height = out_grad.dimension(1);
    const Index output_width = out_grad.dimension(2);

    const Index pool_width = arguments.pool_dimensions[1];
    const Index row_stride = arguments.stride_shape[0];
    const Index column_stride = arguments.stride_shape[1];
    const Index padding_height = arguments.padding_shape[0];
    const Index padding_width = arguments.padding_shape[1];

    in_grad.setZero();

    #pragma omp parallel for collapse(2)
    for(Index b = 0; b < batch_size; ++b)
        for(Index c = 0; c < channels; ++c)
            for(Index oh = 0; oh < output_height; ++oh)
                for(Index ow = 0; ow < output_width; ++ow)
                {
                    const Index idx = static_cast<Index>(max_idx(b, oh, ow, c));
                    const Index ih = oh * row_stride + idx / pool_width - padding_height;
                    const Index iw = ow * column_stride + idx % pool_width - padding_width;

                    if(ih >= 0 && ih < input_height && iw >= 0 && iw < input_width)
                        in_grad(b, ih, iw, c) += out_grad(b, oh, ow, c);
                }
#else
    CHECK_CUDNN(cudnnPoolingBackward(get_cudnn_handle(),
                                      arguments.pooling_descriptor,
                                      &one,
                                      output_gradient.get_descriptor(), output_gradient.data,
                                      output_gradient.get_descriptor(), output_gradient.data,
                                      input_gradient.get_descriptor(), input_gradient.data,
                                      &zero,
                                      input_gradient.get_descriptor(), input_gradient.data));
#endif
}

inline void average_pooling(const TensorView& input, TensorView& output, const PoolingArguments& arguments)
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
#else
    CHECK_CUDNN(cudnnPoolingForward(get_cudnn_handle(),
                                    arguments.pooling_descriptor,
                                    &one,
                                    input.get_descriptor(), input.data,
                                    &zero,
                                    output.get_descriptor(), output.data));
#endif
}

inline void average_pooling_backward(const TensorView& output_gradient,
                                      TensorView& input_gradient,
                                      const PoolingArguments& arguments)
{
#ifndef CUDA
    const TensorMap4 out_grad = output_gradient.as_tensor<4>();
    TensorMap4 in_grad = input_gradient.as_tensor<4>();

    const Index batch_size = in_grad.dimension(0);
    const Index input_height = in_grad.dimension(1);
    const Index input_width = in_grad.dimension(2);
    const Index channels = in_grad.dimension(3);

    const Index output_height = out_grad.dimension(1);
    const Index output_width = out_grad.dimension(2);

    const Index pool_height = arguments.pool_dimensions[0];
    const Index pool_width = arguments.pool_dimensions[1];
    const Index row_stride = arguments.stride_shape[0];
    const Index column_stride = arguments.stride_shape[1];
    const Index padding_height = arguments.padding_shape[0];
    const Index padding_width = arguments.padding_shape[1];

    const type inv_pool_size = type(1) / (pool_height * pool_width);

    in_grad.setZero();

    #pragma omp parallel for collapse(2)
    for(Index b = 0; b < batch_size; ++b)
        for(Index c = 0; c < channels; ++c)
            for(Index oh = 0; oh < output_height; ++oh)
                for(Index ow = 0; ow < output_width; ++ow)
                {
                    const type avg_grad = out_grad(b, oh, ow, c) * inv_pool_size;
                    const Index ih_start = oh * row_stride - padding_height;
                    const Index iw_start = ow * column_stride - padding_width;

                    for(Index ph = 0; ph < pool_height; ++ph)
                        for(Index pw = 0; pw < pool_width; ++pw)
                        {
                            const Index ih = ih_start + ph;
                            const Index iw = iw_start + pw;

                            if(ih >= 0 && ih < input_height && iw >= 0 && iw < input_width)
                                in_grad(b, ih, iw, c) += avg_grad;
                        }
                }
#else
    CHECK_CUDNN(cudnnPoolingBackward(get_cudnn_handle(),
                                      arguments.pooling_descriptor,
                                      &one,
                                      output_gradient.get_descriptor(), output_gradient.data,
                                      output_gradient.get_descriptor(), output_gradient.data,
                                      input_gradient.get_descriptor(), input_gradient.data,
                                      &zero,
                                      input_gradient.get_descriptor(), input_gradient.data));
#endif
}


//  Pooling3d layer operators (3D — sequences)

inline void max_pooling_3d_forward(const TensorView& input,
                                    TensorView& output,
                                    TensorView& maximal_indices,
                                    bool is_training)
{
#ifndef CUDA
    const TensorMap3 inputs = input.as_tensor<3>();
    MatrixMap outputs(output.data, output.shape[0], output.shape[1]);

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index features = inputs.dimension(2);

    #pragma omp parallel for
    for(Index b = 0; b < batch_size; ++b)
    {
        outputs.row(b).setConstant(-numeric_limits<type>::infinity());

        for(Index s = 0; s < sequence_length; ++s)
            for(Index f = 0; f < features; ++f)
            {
                const type value = inputs(b, s, f);
                if(value > outputs(b, f))
                {
                    outputs(b, f) = value;
                    if(is_training)
                        maximal_indices.data[b * features + f] = static_cast<type>(s);
                }
            }
    }
#else
#endif
}

inline void max_pooling_3d_backward(const TensorView& maximal_indices,
                                     const TensorView& output_gradient,
                                     TensorView& input_gradient)
{
#ifndef CUDA
    TensorMap3 in_grad = input_gradient.as_tensor<3>();
    const MatrixMap delta(output_gradient.data, output_gradient.shape[0], output_gradient.shape[1]);

    in_grad.setZero();

    const Index batch_size = in_grad.dimension(0);
    const Index features = in_grad.dimension(2);

    #pragma omp parallel for
    for(Index b = 0; b < batch_size; ++b)
        for(Index f = 0; f < features; ++f)
        {
            const Index max_idx = static_cast<Index>(maximal_indices.data[b * features + f]);
            in_grad(b, max_idx, f) = delta(b, f);
        }
#else
#endif
}

inline void average_pooling_3d_forward(const TensorView& input, TensorView& output)
{
#ifndef CUDA
    const TensorMap3 inputs = input.as_tensor<3>();
    MatrixMap outputs(output.data, output.shape[0], output.shape[1]);

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index features = inputs.dimension(2);

    #pragma omp parallel for
    for(Index b = 0; b < batch_size; ++b)
    {
        outputs.row(b).setZero();
        Index valid_count = 0;

        for(Index s = 0; s < sequence_length; ++s)
        {
            bool is_padding = true;
            for(Index f = 0; f < features; ++f)
                if(inputs(b, s, f) != type(0)) { is_padding = false; break; }

            if(!is_padding)
            {
                for(Index f = 0; f < features; ++f)
                    outputs(b, f) += inputs(b, s, f);
                ++valid_count;
            }
        }

        if(valid_count > 0)
            outputs.row(b) /= static_cast<type>(valid_count);
    }
#else
#endif
}

inline void average_pooling_3d_backward(const TensorView& input,
                                         const TensorView& output_gradient,
                                         TensorView& input_gradient)
{
#ifndef CUDA
    const TensorMap3 inputs = input.as_tensor<3>();
    const MatrixMap delta(output_gradient.data, output_gradient.shape[0], output_gradient.shape[1]);
    TensorMap3 in_grad = input_gradient.as_tensor<3>();

    in_grad.setZero();

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index features = inputs.dimension(2);

    #pragma omp parallel for
    for(Index b = 0; b < batch_size; ++b)
    {
        Index valid_count = 0;
        for(Index s = 0; s < sequence_length; ++s)
        {
            bool is_padding = true;
            for(Index f = 0; f < features; ++f)
                if(inputs(b, s, f) != type(0)) { is_padding = false; break; }
            if(!is_padding) ++valid_count;
        }

        if(valid_count == 0) continue;

        const type inv = type(1) / static_cast<type>(valid_count);

        for(Index s = 0; s < sequence_length; ++s)
        {
            bool is_padding = true;
            for(Index f = 0; f < features; ++f)
                if(inputs(b, s, f) != type(0)) { is_padding = false; break; }

            if(!is_padding)
                for(Index f = 0; f < features; ++f)
                    in_grad(b, s, f) = delta(b, f) * inv;
        }
    }
#else
#endif
}


//  Embedding layer operators

inline void embedding_backward(const TensorView& input_indices,
                                const TensorView& output_gradient,
                                TensorView& weight_gradient,
                                Index embedding_dimension,
                                bool scale_embedding)
{
#ifndef CUDA
    const Index total_tokens = output_gradient.size() / embedding_dimension;

    MatrixMap grad_map(output_gradient.data, total_tokens, embedding_dimension);
    MatrixMap weight_grad(weight_gradient.data, weight_gradient.shape[0], weight_gradient.shape[1]);

    if(scale_embedding)
        grad_map *= sqrt(static_cast<type>(embedding_dimension));

    weight_grad.setZero();

    for(Index i = 0; i < total_tokens; i++)
    {
        const Index vocab_idx = static_cast<Index>(input_indices.data[i]);

        if(vocab_idx < 0 || vocab_idx >= weight_grad.rows()) continue;

        weight_grad.row(vocab_idx).noalias() += grad_map.row(i);
    }

    weight_grad.row(0).setZero();
#else
#endif
}


//  MultiHeadAttention layer operators

inline void projection(const TensorView& input,
                       const TensorView& weights,
                       const TensorView& biases,
                       TensorView& output)
{
#ifndef CUDA
    const Index batch_size = input.shape[0];
    const Index sequence_length = input.shape[1];
    const Index embedding_dimension = weights.shape[0];
    const Index heads_number = output.shape[1];
    const Index head_dimension = output.shape[3];

    const MatrixMap W(weights.data, embedding_dimension, embedding_dimension);
    const VectorMap b(biases.data, embedding_dimension);

    #pragma omp parallel for collapse(2)
    for(Index batch = 0; batch < batch_size; ++batch)
    {
        for(Index h = 0; h < heads_number; ++h)
        {
            const type* in_ptr = input.data + batch * (sequence_length * embedding_dimension);
            const MatrixMap X(const_cast<type*>(in_ptr), sequence_length, embedding_dimension);

            type* out_ptr = output.data + batch * (heads_number * sequence_length * head_dimension)
                            + h * (sequence_length * head_dimension);
            MatrixMap Out(out_ptr, sequence_length, head_dimension);

            auto W_h = W.block(0, h * head_dimension, embedding_dimension, head_dimension);
            auto b_h = b.segment(h * head_dimension, head_dimension);

            Out.noalias() = (X * W_h).rowwise() + b_h.transpose();
        }
    }
#else
#endif
}

inline void multihead_attention_forward(
    const TensorView& query,
    const TensorView& key,
    const TensorView& value,
    TensorView& attention_weights,
    TensorView& concatenated_outputs,
    TensorView& output,
    const TensorView& projection_weights,
    const TensorView& projection_biases,
    const TensorView& source_input,
    Index batch_size,
    Index heads_number,
    Index query_sequence_length,
    Index source_sequence_length,
    Index embedding_dimension,
    Index head_dimension,
    type scaling_factor,
    bool use_causal_mask,
    const MatrixR& causal_mask)
{
#ifndef CUDA
    const Index total_heads = batch_size * heads_number;

    type* query_data = query.data;
    type* key_data = key.data;
    type* att_weights_data = attention_weights.data;

    #pragma omp parallel for
    for(Index i = 0; i < total_heads; ++i)
    {
        const Index offset_q = i * query_sequence_length * head_dimension;
        const Index offset_k = i * source_sequence_length * head_dimension;
        const Index offset_w = i * query_sequence_length * source_sequence_length;

        const MatrixMap q_mat(query_data + offset_q, query_sequence_length, head_dimension);
        const MatrixMap k_mat(key_data + offset_k, source_sequence_length, head_dimension);
        MatrixMap w_mat(att_weights_data + offset_w, query_sequence_length, source_sequence_length);

        w_mat.noalias() = (q_mat * k_mat.transpose()) * scaling_factor;
    }

    // Key padding mask
    {
        const TensorMap3 src_map(source_input.data,
                                  batch_size,
                                  source_sequence_length,
                                  source_input.size() / (batch_size * source_sequence_length));

        #pragma omp parallel for
        for(Index b = 0; b < batch_size; ++b)
            for(Index s = 0; s < source_sequence_length; ++s)
            {
                bool is_pad = true;
                for(Index d = 0; d < src_map.dimension(2); ++d)
                    if(abs(src_map(b, s, d)) > 1e-7f) { is_pad = false; break; }

                if(is_pad)
                    for(Index h = 0; h < heads_number; ++h)
                    {
                        const Index offset = (b * heads_number + h) * query_sequence_length * source_sequence_length;
                        MatrixMap w(att_weights_data + offset, query_sequence_length, source_sequence_length);
                        w.col(s).setConstant(-1e9f);
                    }
            }
    }

    if(use_causal_mask)
    {
        const Index matrix_size = query_sequence_length * source_sequence_length;
        const VectorMap causal_vec(const_cast<type*>(causal_mask.data()), matrix_size);
        MatrixMap scores(att_weights_data, total_heads, matrix_size);
        scores.rowwise() += causal_vec.transpose();
    }

    // Softmax
    const Index total_rows_softmax = total_heads * query_sequence_length;
    MatrixMap att_map(att_weights_data, total_rows_softmax, source_sequence_length);
    att_map.colwise() -= att_map.rowwise().maxCoeff();
    att_map.array() = att_map.array().exp();
    att_map.array().colwise() /= att_map.rowwise().sum().array();

    // Attention * V
    type* value_data = value.data;
    type* concat_data = concatenated_outputs.data;

    const int original_eigen_threads = Eigen::nbThreads();
    Eigen::setNbThreads(1);

    #pragma omp parallel for collapse(2)
    for(Index b = 0; b < batch_size; ++b)
        for(Index h = 0; h < heads_number; ++h)
        {
            const Index offset_v = b * (heads_number * source_sequence_length * head_dimension) + h * (source_sequence_length * head_dimension);
            const Index offset_w = b * (heads_number * query_sequence_length * source_sequence_length) + h * (query_sequence_length * source_sequence_length);

            const MatrixMap w_mat(att_weights_data + offset_w, query_sequence_length, source_sequence_length);
            const MatrixMap v_mat(value_data + offset_v, source_sequence_length, head_dimension);

            type* out_ptr = concat_data + b * (query_sequence_length * embedding_dimension) + h * head_dimension;
            using StrideType = Eigen::OuterStride<Eigen::Dynamic>;
            Eigen::Map<MatrixR, 0, StrideType> o_mat(out_ptr, query_sequence_length, head_dimension, StrideType(embedding_dimension));

            o_mat.noalias() = w_mat * v_mat;
        }

    Eigen::setNbThreads(original_eigen_threads);

    // Output projection
    const MatrixMap proj_W(projection_weights.data, embedding_dimension, embedding_dimension);
    const VectorMap proj_b(projection_biases.data, embedding_dimension);

    const Index total_rows = batch_size * query_sequence_length;
    const MatrixMap concat_map(concat_data, total_rows, embedding_dimension);
    MatrixMap out_map(output.data, total_rows, embedding_dimension);

    out_map.noalias() = (concat_map * proj_W).rowwise() + proj_b.transpose();
#else
#endif
}

inline void multihead_attention_backward(
    const TensorView& query_input,
    const TensorView& source_input,
    const TensorView& output_gradient,
    const TensorView& query,
    const TensorView& key,
    const TensorView& value,
    const TensorView& attention_weights,
    const TensorView& concatenated_outputs,
    const TensorView& projection_weights,
    TensorView& projection_weight_gradient,
    TensorView& projection_bias_gradient,
    TensorView& concatenated_output_gradient,
    TensorView& attention_weight_gradient,
    TensorView& query_gradient,
    TensorView& key_gradient,
    TensorView& value_gradient,
    TensorView& query_weight_gradient,
    TensorView& query_bias_gradient,
    TensorView& key_weight_gradient,
    TensorView& key_bias_gradient,
    TensorView& value_weight_gradient,
    TensorView& value_bias_gradient,
    TensorView& input_query_gradient,
    const TensorView& query_weights_param,
    const TensorView& key_weights_param,
    const TensorView& value_weights_param,
    Index batch_size,
    Index heads_number,
    Index query_sequence_length,
    Index source_sequence_length,
    Index embedding_dimension,
    Index head_dimension,
    type scaling_factor,
    bool self_attention)
{
#ifndef CUDA
    const Index total_rows = batch_size * query_sequence_length;
    const Index total_heads = batch_size * heads_number;

    const MatrixMap concat_map(concatenated_outputs.data, total_rows, embedding_dimension);
    const MatrixMap delta_Y(output_gradient.data, total_rows, embedding_dimension);

    MatrixMap proj_wg(projection_weight_gradient.data, embedding_dimension, embedding_dimension);
    VectorMap proj_bg(projection_bias_gradient.data, embedding_dimension);

    proj_wg.noalias() = concat_map.transpose() * delta_Y;
    proj_bg.noalias() = delta_Y.colwise().sum();

    MatrixMap concat_grad(concatenated_output_gradient.data, total_rows, embedding_dimension);
    const MatrixMap proj_W(projection_weights.data, embedding_dimension, embedding_dimension);
    concat_grad.noalias() = delta_Y * proj_W.transpose();

    type* att_w_data = attention_weights.data;
    type* v_data = value.data;
    type* v_grad_data = value_gradient.data;
    type* att_wg_data = attention_weight_gradient.data;
    type* concat_grad_data = concatenated_output_gradient.data;

    const int original_eigen_threads = Eigen::nbThreads();
    Eigen::setNbThreads(1);

    #pragma omp parallel for collapse(2)
    for(Index b = 0; b < batch_size; ++b)
        for(Index h = 0; h < heads_number; ++h)
        {
            const Index offset_w = b * (heads_number * query_sequence_length * source_sequence_length) + h * (query_sequence_length * source_sequence_length);
            const Index offset_v = b * (heads_number * source_sequence_length * head_dimension) + h * (source_sequence_length * head_dimension);

            const MatrixMap W(att_w_data + offset_w, query_sequence_length, source_sequence_length);
            const MatrixMap V(v_data + offset_v, source_sequence_length, head_dimension);

            MatrixMap dV(v_grad_data + offset_v, source_sequence_length, head_dimension);
            MatrixMap dW(att_wg_data + offset_w, query_sequence_length, source_sequence_length);

            type* dO_ptr = concat_grad_data + b * (query_sequence_length * embedding_dimension) + h * head_dimension;
            using StrideType = Eigen::OuterStride<Eigen::Dynamic>;
            Eigen::Map<const MatrixR, 0, StrideType> dO(dO_ptr, query_sequence_length, head_dimension, StrideType(embedding_dimension));

            dV.noalias() = W.transpose() * dO;
            dW.noalias() = dO * V.transpose();
        }

    Eigen::setNbThreads(original_eigen_threads);

    // Softmax gradient
    #pragma omp parallel for
    for(Index i = 0; i < total_heads; ++i)
    {
        const Index offset_w = i * query_sequence_length * source_sequence_length;
        const MatrixMap W(att_w_data + offset_w, query_sequence_length, source_sequence_length);
        MatrixMap dW(att_wg_data + offset_w, query_sequence_length, source_sequence_length);

        VectorR dot_product = (W.array() * dW.array()).rowwise().sum();
        dW.array() = W.array() * (dW.colwise() - dot_product).array();
    }

    // Q and K gradients
    type* q_data = query.data;
    type* k_data = key.data;
    type* q_grad_data = query_gradient.data;
    type* k_grad_data = key_gradient.data;

    #pragma omp parallel for
    for(Index i = 0; i < total_heads; ++i)
    {
        const Index offset_w = i * query_sequence_length * source_sequence_length;
        const Index offset_q = i * query_sequence_length * head_dimension;
        const Index offset_k = i * source_sequence_length * head_dimension;

        const MatrixMap dW(att_wg_data + offset_w, query_sequence_length, source_sequence_length);
        const MatrixMap Q(q_data + offset_q, query_sequence_length, head_dimension);
        const MatrixMap K(k_data + offset_k, source_sequence_length, head_dimension);

        MatrixMap dQ(q_grad_data + offset_q, query_sequence_length, head_dimension);
        MatrixMap dK(k_grad_data + offset_k, source_sequence_length, head_dimension);

        dQ.noalias() = (dW * K) * scaling_factor;
        dK.noalias() = (dW.transpose() * Q) * scaling_factor;
    }

    // Projection gradients for Q, K, V → input gradients
    auto proj_grad = [&](type* d_head_data, const type* input_data, Index seq_len,
                          const TensorView& weights, TensorView& d_weights, TensorView& d_biases,
                          type* d_input_data, bool accumulate)
    {
        const MatrixMap W(weights.data, embedding_dimension, embedding_dimension);
        MatrixMap dW(d_weights.data, embedding_dimension, embedding_dimension);
        VectorMap db(d_biases.data, embedding_dimension);

        #pragma omp parallel for
        for(Index b = 0; b < batch_size; ++b)
        {
            MatrixMap dX(d_input_data + b * seq_len * embedding_dimension, seq_len, embedding_dimension);
            if(!accumulate) dX.setZero();

            for(Index h = 0; h < heads_number; ++h)
            {
                const type* delta_ptr = d_head_data + b * (heads_number * seq_len * head_dimension) + h * (seq_len * head_dimension);
                const MatrixMap Delta(const_cast<type*>(delta_ptr), seq_len, head_dimension);
                auto W_h = W.block(0, h * head_dimension, embedding_dimension, head_dimension);
                dX.noalias() += Delta * W_h.transpose();
            }
        }

        #pragma omp parallel for
        for(Index h = 0; h < heads_number; ++h)
        {
            auto dW_h = dW.block(0, h * head_dimension, embedding_dimension, head_dimension);
            auto db_h = db.segment(h * head_dimension, head_dimension);
            dW_h.setZero();
            db_h.setZero();

            for(Index b = 0; b < batch_size; ++b)
            {
                const type* delta_ptr = d_head_data + b * (heads_number * seq_len * head_dimension) + h * (seq_len * head_dimension);
                const MatrixMap Delta(const_cast<type*>(delta_ptr), seq_len, head_dimension);
                const MatrixMap X(const_cast<type*>(input_data) + b * seq_len * embedding_dimension, seq_len, embedding_dimension);

                dW_h.noalias() += X.transpose() * Delta;
                db_h.noalias() += Delta.colwise().sum().transpose();
            }
        }
    };

    proj_grad(q_grad_data, query_input.data, query_sequence_length,
              query_weights_param, query_weight_gradient, query_bias_gradient,
              input_query_gradient.data, false);

    if(self_attention)
    {
        proj_grad(k_grad_data, source_input.data, source_sequence_length,
                  key_weights_param, key_weight_gradient, key_bias_gradient,
                  input_query_gradient.data, true);

        proj_grad(v_grad_data, source_input.data, source_sequence_length,
                  value_weights_param, value_weight_gradient, value_bias_gradient,
                  input_query_gradient.data, true);
    }
#else
#endif
}

}
