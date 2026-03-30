//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M A T H   U T I L I T I E S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"
#include "tensor_utilities.h"

#pragma once

namespace opennn
{

// In math_utilities.h

/**
 * Clips the values of the input tensor between lower and upper bounds.
 * Assuming bounds are per-feature (columns).
 */
inline void bounding(const TensorView& input,
                     const VectorR& lower_bounds,
                     const VectorR& upper_bounds,
                     TensorView& output)
{
    const Index features = lower_bounds.size();

#ifndef OPENNN_CUDA
    const MatrixMap input_matrix = input.as_matrix();
    MatrixMap output_matrix = output.as_matrix();

    for(Index j = 0; j < features; ++j)
        output_matrix.col(j) = input_matrix.col(j).cwiseMax(lower_bounds(j)).cwiseMin(upper_bounds(j));

#else
    const Index total_rows = input.shape[0];

    bounding_cuda(input.size(),
                  total_rows,
                  features,
                  input.data,
                  lower_bounds_device, // Assuming these were moved to device
                  upper_bounds_device,
                  output.data);
#endif
}


inline void copy(const TensorView& source, TensorView& destination)
{
    if(source.size() != destination.size())
        throw std::runtime_error("Math Error: Tensor sizes mismatch in copy operation.");

#ifndef OPENNN_CUDA
    destination.as_vector() = source.as_vector();
#else
    if (source.data != destination.data)
        CHECK_CUDA(cudaMemcpy(destination.data,
                              source.data,
                              source.size() * sizeof(type),
                              cudaMemcpyDeviceToDevice));
#endif
}

void addition(const TensorView& input_1, const TensorView& input_2, TensorView& output)
{
    if(input_1.size() != input_2.size() || input_1.size() != output.size())
        throw runtime_error("Addition Error: Tensor dimensions do not match.");

#ifndef OPENNN_CUDA
    output.as_vector().array() = input_1.as_vector().array() + input_2.as_vector().array();
#else
    const float alpha1 = 1.0f;
    const float alpha2 = 1.0f;
    const float beta   = 0.0f;

    CHECK_CUDNN(cudnnOpTensor(get_cudnn_handle(),
                              get_operator_sum_descriptor(),
                              &alpha1,
                              input_1.get_descriptor(),
                              input_1.data,
                              &alpha2,
                              input_2.get_descriptor(),
                              input_2.data,
                              &beta,
                              output.get_descriptor(),
                              output.data));
#endif
}


void projection(const TensorMap3 &inputs,
                const TensorView &weights,
                const TensorView &biases,
                Index sequence_length,
                Index batch_size,
                Tensor4 &output)
{
#ifndef OPENNN_CUDA
/*
    const Index embedding_dimension = get_embedding_dimension();
    const Index head_dimension = get_head_dimension();

    const MatrixMap weights_map = matrix_map(weights);
    const VectorMap biases_map = vector_map(biases);

#pragma omp parallel for collapse(2)
    for(Index b = 0; b < batch_size; ++b)
    {
        for(Index h = 0; h < heads_number; ++h)
        {
            const type* in_ptr = inputs.data() + b * (sequence_length * embedding_dimension);
            const MatrixMap X_b(const_cast<type*>(in_ptr), sequence_length, embedding_dimension);

            type* out_ptr = output.data() + b * (heads_number * sequence_length * head_dimension)
                            + h * (sequence_length * head_dimension);

            MatrixMap Out_bh(out_ptr, sequence_length, head_dimension);

            auto W_h = weights_map.block(0, h * head_dimension, embedding_dimension, head_dimension);

            auto b_h = biases_map.segment(h * head_dimension, head_dimension);

            Out_bh.noalias() = (X_b * W_h).rowwise() + b_h.transpose();
        }
    }
*/
#else
    CHECK_CUBLAS(cublasSgemm(get_cublas_handle(),
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             output_dim, batch_seq_len, input_dim,
                             &alpha_one,
                             weights, output_dim,
                             input, input_dim,
                             &beta_zero,
                             output, output_dim));

    CHECK_CUDNN(cudnnAddTensor(get_cudnn_handle(),
                               &alpha_one,
                               biases_desc, biases,
                               &alpha_one,
                               output_desc, output));
#endif
}


void projection_gradient(const Tensor4& d_head,
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

void normalization(Tensor1& means,
                   Tensor1& standard_deviations,
                   const Tensor2& inputs,
                   Tensor2& outputs)
{
#ifndef OPENNN_CUDA
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
/*
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
    if (batch_normalization && is_training)
        CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
            get_cudnn_handle(),
            CUDNN_BATCHNORM_PER_ACTIVATION,
            &alpha,
            &beta_add,
            outputs.get_descriptor(),
            outputs_buffer,
            outputs.get_descriptor(),
            outputs_buffer,
            gammas_device.get_descriptor(),
            gammas_device.data,
            betas_device.data,
            momentum,
            running_means_device.data,
            running_variances_device.data,
            EPSILON,
            dense_forward_propagation->means.data,
            dense_forward_propagation->inverse_variance.data));
    else if (batch_normalization && !is_training)
        CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
            get_cudnn_handle(),
            CUDNN_BATCHNORM_PER_ACTIVATION,
            &alpha, &beta_add,
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

inline void combination(const TensorView& input,
                        const TensorView& weights,
                        const TensorView& biases,
                        TensorView& output)
{
#ifndef OPENNN_CUDA
    output.as_matrix().noalias()
        = (input.as_matrix() * weights.as_matrix()).rowwise() + biases.as_vector().transpose();
#else
    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int m = static_cast<int>(output.size() / biases.size()); // Total Rows
    const int n = static_cast<int>(biases.size());                // Outputs Number
    const int k = static_cast<int>(input.size() / m);              // Inputs Number

    CHECK_CUBLAS(cublasSgemm(get_cublas_handle(),
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k,
                             &alpha,
                             weights.data, n,
                             input.data, k,
                             &beta,
                             output.data, n));

    CHECK_CUDNN(cudnnAddTensor(get_cudnn_handle(),
                               &alpha, biases.get_descriptor(), biases.data,
                               &alpha, output.get_descriptor(), output.data));
#endif
}

void batch_normalization_training()
{
#ifndef OPENNN_CUDA

#else
    CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
        get_cudnn_handle(),
        CUDNN_BATCHNORM_PER_ACTIVATION,
        &alpha,
        &beta_add,
        outputs.get_descriptor(),
        outputs_buffer,
        outputs.get_descriptor(),
        outputs_buffer,
        gammas_device.get_descriptor(),
        gammas_device.data,
        betas_device.data,
        momentum,
        running_means_device.data,
        running_variances_device.data,
        EPSILON,
        dense_forward_propagation->means.data,
        dense_forward_propagation->inverse_variance.data));

#endif
}


void batch_normalization_inference()
{
#ifndef OPENNN_CUDA

#else
    CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
        get_cudnn_handle(),
        CUDNN_BATCHNORM_PER_ACTIVATION,
        &alpha, &beta_add,
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


void activation()
{
#ifndef OPENNN_CUDA


#else
    if (activation_function == "Linear")
        cudaMemcpy(outputs.data, combinations, total_rows * outputs_number * sizeof(type), cudaMemcpyDeviceToDevice);
    else if (activation_function == "Softmax")
        CHECK_CUDNN(cudnnSoftmaxForward(get_cudnn_handle(),
                                        CUDNN_SOFTMAX_ACCURATE,
                                        CUDNN_SOFTMAX_MODE_CHANNEL,
                                        &alpha,
                                        outputs.get_descriptor(),
                                        outputs_buffer,
                                        &beta,
                                        outputs.get_descriptor(),
                                        outputs.data));
    else
        CHECK_CUDNN(cudnnActivationForward(get_cudnn_handle(),
                                           activation_descriptor,
                                           &alpha,
                                           outputs.get_descriptor(),
                                           outputs_buffer,
                                           &beta,
                                           outputs.get_descriptor(),
                                           outputs.data));
#endif
}

void dropout()
{
#ifndef OPENNN_CUDA

#else
    CHECK_CUDNN(cudnnDropoutForward(get_cudnn_handle(),
                                    dropout_descriptor,
                                    outputs.get_descriptor(),
                                    outputs.data,
                                    outputs.get_descriptor(),
                                    outputs.data,
                                    dropout_reserve_space,
                                    dropout_reserve_space_size));
#endif

}


void convolutions(const TensorView& inputs, const TensorView& kernel, TensorView& outputs)
{
#ifndef OPENNN_CUDA
/*
    const Index batch_size = inputs.dimension(0);
    const Index output_height = convolutions.dimension(1);
    const Index output_width = convolutions.dimension(2);

    const Index kernels_number = get_kernels_number();
    const Index kernel_height = get_kernel_height();
    const Index kernel_width = get_kernel_width();
    const Index kernel_channels = get_kernel_channels();

    const Index single_kernel_size = kernel_height * kernel_width * kernel_channels;

    const VectorMap biases = vector_map(parameters[Biases]);

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
                                        &alpha,
                                        input_tensor_descriptor,
                                        input_data,
                                        kernel_descriptor,
                                        weights_device.data,
                                        convolution_descriptor,
                                        convolution_algorithm,
                                        workspace,
                                        workspace_size,
                                        &beta,
                                        current_output_descriptor,
                                        outputs_buffer));

    // Biases

    CHECK_CUDNN(cudnnAddTensor(get_cudnn_handle(),
                               &alpha,
                               biases.get_descriptor(),
                               biases.data,
                               &alpha,
                               current_output_descriptor,
                               outputs_buffer));

#endif
}

}
