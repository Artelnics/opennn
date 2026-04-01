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

inline void max_pooling(const TensorView& input, TensorView& output)
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
                        pooling_forward_propagation->maximal_indices(batch_index, output_row, output_column, channel_index) = maximum_index;
                }
*/
#else
    CHECK_CUDNN(cudnnPoolingForward(get_cudnn_handle(),
                                    pooling_descriptor,
                                    &alpha,
                                    input.descriptor,
                                    input.device,
                                    &beta,
                                    output.descriptor,
                                    output.device));

#endif

}


inline void average_pooling(const TensorView& input, TensorView& output)
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
                     const VectorR& lower_bounds,
                     const VectorR& upper_bounds,
                     TensorView& output)
{
    const Index features = lower_bounds.size();

#ifndef CUDA
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
        throw runtime_error("Math Error: Tensor sizes mismatch in copy operation.");

#ifndef CUDA
    destination.as_vector() = source.as_vector();
#else
    if (source.data != destination.data)
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


inline void projection(const TensorView& input,
                       const TensorView& weights,
                       const TensorView& biases,
                       TensorView& output)
{
#ifndef CUDA
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
            means.data,
            inverse_variance.data));
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
#ifndef CUDA
    output.as_matrix().noalias()
        = (input.as_matrix() * weights.as_matrix()).rowwise() + biases.as_vector().transpose();
#else

    multiply(input, false, weights, false, output, 1.0f, 0.0f);

    CHECK_CUDNN(cudnnAddTensor(get_cudnn_handle(),
                               &alpha, biases.get_descriptor(), biases.data,
                               &alpha, output.get_descriptor(), output.data));
#endif
}

inline void batch_normalization_training(
    TensorView& output,
    const TensorView& gammas,
    const TensorView& betas,
    VectorR& running_means,
    VectorR& running_standard_deviations,
    type momentum = type(0.9))
{
#ifndef CUDA
    const Index neurons = running_means.size();
    const Index total_rows = output.size() / neurons;

    MatrixMap outputs_mat(output.data, total_rows, neurons);

    VectorR means = outputs_mat.colwise().mean();
    MatrixR norm_mat = outputs_mat.rowwise() - means.transpose();
    VectorR standard_deviations = (norm_mat.array().square().colwise().mean() + EPSILON).sqrt();

    norm_mat.array().rowwise() /= standard_deviations.transpose().array();

    running_means = running_means * momentum + means * (type(1) - momentum);
    running_standard_deviations = running_standard_deviations * momentum + standard_deviations * (type(1) - momentum);

    const VectorMap gammas_vec(gammas.data, neurons);
    const VectorMap betas_vec(betas.data, neurons);

    outputs_mat.array() = (norm_mat.array().rowwise() * gammas_vec.transpose().array()).rowwise() +
                          betas_vec.transpose().array();
#else
    CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
        get_cudnn_handle(),
        CUDNN_BATCHNORM_PER_ACTIVATION,
        &alpha_one,
        &beta_zero,
        output.get_descriptor(),
        output.data,
        output.get_descriptor(),
        output.data,
        gammas.get_descriptor(),
        gammas.data,
        betas.data,
        momentum,
        running_means.data(),
        running_standard_deviations.data(),
        EPSILON,
        nullptr,
        nullptr));
#endif
}


inline void batch_normalization_inference(
    TensorView& output,
    const TensorView& gammas,
    const TensorView& betas,
    const VectorR& running_means,
    const VectorR& running_standard_deviations)
{
#ifndef CUDA
    const Index neurons = running_means.size();
    const Index total_rows = output.size() / neurons;

    MatrixMap outputs_mat(output.data, total_rows, neurons);

    const VectorMap gammas_vec(gammas.data, neurons);
    const VectorMap betas_vec(betas.data, neurons);

    const VectorR scale = gammas_vec.array() / (running_standard_deviations.array() + EPSILON);
    const VectorR shift = betas_vec.array() - running_means.array() * scale.array();

    outputs_mat.array() = (outputs_mat.array().rowwise() * scale.transpose().array()).rowwise() +
                          shift.transpose().array();
#else
    CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
        get_cudnn_handle(),
        CUDNN_BATCHNORM_PER_ACTIVATION,
        &alpha_one,
        &beta_zero,
        output.get_descriptor(),
        output.data,
        output.get_descriptor(),
        output.data,
        gammas.get_descriptor(),
        gammas.data,
        betas.data,
        running_means.data(),
        running_standard_deviations.data(),
        EPSILON));
#endif
}


inline void activation(TensorView& output, const string& function)
{
    if (function == "Linear" || output.empty()) return;

#ifndef CUDA
/*
    auto output_array = output.as_vector().array();

    if (function == "Sigmoid" || function == "Logistic")
    {
        output_array = (1.0f + (-output_array).exp()).inverse();
    }
    else if (function == "HyperbolicTangent")
    {
        output_array = output_array.tanh();
    }
    else if (function == "RectifiedLinear")
    {
        output_array = output_array.cwiseMax(0.0f);
    }
    else if (function == "ScaledExponentialLinear")
    {
        const type alpha = 1.67326f;
        const type lambda = 1.05070f;
        output_array = lambda * (output_array > 0.0f).select(output_array, alpha * (output_array.exp() - 1.0f));
    }
    else if (function == "Softmax")
    {
        MatrixMap output_matrix = output.as_matrix();
        VectorR row_max = output_matrix.rowwise().maxCoeff();
        output_matrix.colwise() -= row_max;
        output_matrix.array() = output_matrix.array().exp();
        VectorR row_sum = output_matrix.rowwise().sum();
        output_matrix.array().colwise() /= row_sum.array();
    }
    else
    {
        throw runtime_error("Activation Error: Unknown function '" + function + "'");
    }
*/
#else
    const float alpha = 1.0f;
    const float beta = 0.0f;

    if (function == "Softmax")
    {
        CHECK_CUDNN(cudnnSoftmaxForward(get_cudnn_handle(),
                                        CUDNN_SOFTMAX_ACCURATE,
                                        CUDNN_SOFTMAX_MODE_CHANNEL, // Softmax per spatial location
                                        &alpha,
                                        output.get_descriptor(), output.data,
                                        &beta,
                                        output.get_descriptor(), output.data));
    }
    else
    {
        CHECK_CUDNN(cudnnActivationForward(get_cudnn_handle(),
                                           act_desc,
                                           &alpha,
                                           output.get_descriptor(), output.data,
                                           &beta,
                                           output.get_descriptor(), output.data));
    }
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


inline void convolution(const TensorView& input,
                 const TensorView& kernel,
                 const TensorView& biases,
                 TensorView& output)
{
#ifndef CUDA
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
                                        input.descriptor,
                                        input.device,
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


inline void convolution_activation(const TensorView& input,
                                   const TensorView& weight,
                                   const TensorView& bias,
                                   TensorView& output,
                                   const string& activation)
{
#ifndef CUDA
    convolution(input, weight, bias, output);

//    activation(input, output, activation);
#else
    CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
        get_cudnn_handle(),
        &alpha,
        input.descriptor,
        input_data,
        kernel_descriptor,
        weights_device.data,
        convolution_descriptor,
        convolution_algorithm,
        workspace,
        workspace_size,
        &beta,
        current_output_descriptor,
        outputs.data,
        biases_device.get_descriptor(),
        biases_device.data,
        activation_descriptor,
        current_output_descriptor,
        outputs.data));
#endif
}


inline void multiply(const TensorView& A, bool transA,
                     const TensorView& B, bool transB,
                     TensorView& C,
                     type alpha = 1.0f, type beta = 0.0f)
{
#ifndef OPENNN_CUDA
    auto matA = A.as_matrix();
    auto matB = B.as_matrix();
    auto matC = C.as_matrix();

    if (!transA && !transB)      matC.noalias() = alpha * (matA * matB) + beta * matC;
    else if (transA && !transB)  matC.noalias() = alpha * (matA.transpose() * matB) + beta * matC;
    else if (!transA && transB)  matC.noalias() = alpha * (matA * matB.transpose()) + beta * matC;
    else                         matC.noalias() = alpha * (matA.transpose() * matB.transpose()) + beta * matC;
#else
    const int m = transA ? (int)A.shape[1] : (int)A.shape[0];
    const int n = transB ? (int)B.shape[0] : (int)B.shape[1];
    const int k = transA ? (int)A.shape[0] : (int)A.shape[1];

    const int lda = (int)A.shape[1];
    const int ldb = (int)B.shape[1];
    const int ldc = n;

    // C^T = B^T * A^T (RowMajor)
    CHECK_CUBLAS(cublasSgemm(get_cublas_handle(),
                             transB ? CUBLAS_OP_N : CUBLAS_OP_T,
                             transA ? CUBLAS_OP_N : CUBLAS_OP_T,
                             n, m, k,
                             &alpha, B.data, ldb, A.data, lda,
                             &beta, C.data, ldc));
#endif
}

inline void multiply_elementwise(const TensorView& A, const TensorView& B, TensorView& C)
{
#ifndef OPENNN_CUDA
    C.as_vector().array() = A.as_vector().array() * B.as_vector().array();
#else
    const float one = 1.0f;
    const float zero = 0.0f;
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
    // @todo
#endif
}


inline void softmax(const TensorView& input, TensorView& output)
{
#ifndef OPENNN_CUDA

#else

#endif
}


inline void activation_gradient(const TensorView& output, const TensorView& dY, TensorView& delta, const string& function)
{
#ifndef OPENNN_CUDA
    const auto out = output.as_vector().array();
    const auto dy  = dY.as_vector().array();
    auto d         = delta.as_vector().array();

    if      (function == "Linear")           d = dy;
    else if (function == "Logistic")         d = dy * out * (type(1) - out);
    else if (function == "HyperbolicTangent")d = dy * (type(1) - out.square());
    else if (function == "RectifiedLinear")  d = dy * (out > type(0)).cast<type>();
    else if (function == "ExponentialLinear")d = dy * (out > type(0)).select(out.Constant(out.rows(), out.cols(), type(1)), out + type(1));
    else                                     d = dy;
#else
    // @todo
#endif
}
}
