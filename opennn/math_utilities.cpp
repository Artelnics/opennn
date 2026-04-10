//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M A T H   U T I L I T I E S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "math_utilities.h"
#ifdef CUDA
#include "kernel.cuh"
#endif

namespace opennn
{

void max_pooling(const TensorView& input,
                        TensorView& output,
                        TensorView& maximal_indices,
                        const PoolingArguments& arguments,
                        bool is_training)
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
    (void)maximal_indices; (void)is_training;

    CHECK_CUDNN(cudnnPoolingForward(Device::get_cudnn_handle(),
        arguments.pooling_descriptor,
        &one,
        input.get_descriptor(), input.data,
        &zero,
        output.get_descriptor(), output.data));
#endif
}


void average_pooling(const TensorView& input, TensorView& output, const PoolingArguments& arguments)
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
    CHECK_CUDNN(cudnnPoolingForward(Device::get_cudnn_handle(),
        arguments.pooling_descriptor,
        &one,
        input.get_descriptor(), input.data,
        &zero,
        output.get_descriptor(), output.data));
#endif
}


void padding(const TensorView& input, TensorView& output)
{
#ifndef CUDA
    const TensorMap4 input_map = input.as_tensor<4>();
    TensorMap4 output_map = output.as_tensor<4>();

    const Index pad_h = (output.shape[1] - input.shape[1]) / 2;
    const Index pad_w = (output.shape[2] - input.shape[2]) / 2;

    const Eigen::array<pair<Index,Index>, 4> paddings = {
        make_pair(Index(0), Index(0)),
        make_pair(pad_h, pad_h),
        make_pair(pad_w, pad_w),
        make_pair(Index(0), Index(0))
    };

    output_map.device(get_device()) = input_map.pad(paddings);
#else
    // CUDA: cuDNN handles padding via convolution_descriptor
    (void)input; (void)output;
#endif
}

void bounding(const TensorView& input,
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
    // @todo CUDA bounding
    (void)input; (void)lower_bounds; (void)upper_bounds; (void)output;
#endif
}


void copy(const TensorView& source, TensorView& destination)
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

void addition(const TensorView& input_1, const TensorView& input_2, TensorView& output)
{
    if(input_1.size() != input_2.size() || input_1.size() != output.size())
        throw runtime_error("Addition Error: Tensor dimensions do not match.");

#ifndef CUDA
    output.as_vector().array() = input_1.as_vector().array() + input_2.as_vector().array();
#else
    CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(),
                              Device::get_operator_sum_descriptor(),
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


void projection(const TensorView& input,
                       const TensorView& weights,
                       const TensorView& biases,
                       TensorView& output)
{
    // Projection: output = input * weights + biases
    // Input is 3D (batch, seq, emb), output is 4D (batch, heads, seq, head_dim)
    // Reshape to 2D for matmul: (batch*seq, emb) * (emb, heads*head_dim) + biases
    const Index total_rows = input.size() / input.shape[input.get_rank() - 1];
    const Index in_cols = input.shape[input.get_rank() - 1];
    const Index out_cols = weights.shape[weights.get_rank() - 1];

    TensorView in_2d(input.data, {total_rows, in_cols});
    TensorView out_2d(output.data, {total_rows, out_cols});
#ifdef CUDA
    in_2d.set_descriptor(in_2d.shape);
    out_2d.set_descriptor(out_2d.shape);
#endif
    combination(in_2d, weights, biases, out_2d);
}


void projection_gradient(const TensorView& d_head,
                                const TensorView& input,
                                const TensorView& weights,
                                TensorView& d_bias,
                                TensorView& d_weights,
                                TensorView& d_input,
                                Index batch_size,
                                Index heads_number,
                                Index sequence_length,
                                Index embedding_dimension,
                                Index head_dimension,
                                bool accumulate)
{
#ifdef CUDA
    // CUDA fallback: copy to CPU, compute, copy back
    const Index dh_size = d_head.size();
    const Index in_size = input.size();
    const Index w_size = weights.size();
    const Index di_size = d_input.size();
    const Index dw_size = d_weights.size();
    const Index db_size = d_bias.size();

    vector<float> dh_h(dh_size), in_h(in_size), w_h(w_size), di_h(di_size), dw_h(dw_size), db_h(db_size);

    cudaMemcpy(dh_h.data(), d_head.data, dh_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(in_h.data(), input.data, in_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(w_h.data(), weights.data, w_size * sizeof(float), cudaMemcpyDeviceToHost);
    if(accumulate)
        cudaMemcpy(di_h.data(), d_input.data, di_size * sizeof(float), cudaMemcpyDeviceToHost);

    TensorView dh_v(dh_h.data(), d_head.shape);
    TensorView in_v(in_h.data(), input.shape);
    TensorView w_v(w_h.data(), weights.shape);
    TensorView di_v(di_h.data(), d_input.shape);
    TensorView dw_v(dw_h.data(), d_weights.shape);
    TensorView db_v(db_h.data(), d_bias.shape);

    // Fall through to CPU code below with host views
    #define PG_WEIGHTS_PTR w_v.data
    #define PG_DINPUT_PTR di_v.data
    #define PG_DHEAD_PTR dh_v.data
    #define PG_INPUT_PTR in_v.data
    #define PG_DWEIGHTS_PTR dw_v.data
    #define PG_DBIAS_PTR db_v.data
#else
    #define PG_WEIGHTS_PTR weights.data
    #define PG_DINPUT_PTR d_input.data
    #define PG_DHEAD_PTR d_head.data
    #define PG_INPUT_PTR input.data
    #define PG_DWEIGHTS_PTR d_weights.data
    #define PG_DBIAS_PTR d_bias.data
#endif

    const MatrixMap W(PG_WEIGHTS_PTR, embedding_dimension, heads_number * head_dimension);

    #pragma omp parallel for
    for(Index b = 0; b < batch_size; ++b)
    {
        MatrixMap dX_b(PG_DINPUT_PTR + b * sequence_length * embedding_dimension, sequence_length, embedding_dimension);

        if(!accumulate) dX_b.setZero();

        for(Index h = 0; h < heads_number; ++h)
        {
            const MatrixMap Delta(PG_DHEAD_PTR + b * (heads_number * sequence_length * head_dimension) + h * (sequence_length * head_dimension),
                                  sequence_length, head_dimension);
            auto W_h = W.block(0, h * head_dimension, embedding_dimension, head_dimension);
            dX_b.noalias() += Delta * W_h.transpose();
        }
    }

    MatrixMap dW(PG_DWEIGHTS_PTR, embedding_dimension, heads_number * head_dimension);
    VectorMap db(PG_DBIAS_PTR, heads_number * head_dimension);

    #pragma omp parallel for
    for(Index h = 0; h < heads_number; ++h)
    {
        auto dW_h = dW.block(0, h * head_dimension, embedding_dimension, head_dimension);
        auto db_h = db.segment(h * head_dimension, head_dimension);
        dW_h.setZero();
        db_h.setZero();

        for(Index b = 0; b < batch_size; ++b)
        {
            const MatrixMap Delta(PG_DHEAD_PTR + b * (heads_number * sequence_length * head_dimension) + h * (sequence_length * head_dimension),
                                  sequence_length, head_dimension);
            const MatrixMap X_b(PG_INPUT_PTR + b * sequence_length * embedding_dimension, sequence_length, embedding_dimension);

            dW_h.noalias() += X_b.transpose() * Delta;
            db_h.noalias() += Delta.colwise().sum().transpose();
        }
    }

#ifdef CUDA
    cudaMemcpy(d_input.data, di_h.data(), di_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights.data, dw_h.data(), dw_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias.data, db_h.data(), db_size * sizeof(float), cudaMemcpyHostToDevice);
#endif

    #undef PG_WEIGHTS_PTR
    #undef PG_DINPUT_PTR
    #undef PG_DHEAD_PTR
    #undef PG_INPUT_PTR
    #undef PG_DWEIGHTS_PTR
    #undef PG_DBIAS_PTR
}


void batch_normalization_inference(
    const TensorView& input,
    const TensorView& gamma,
    const TensorView& beta,
    const VectorR& running_mean,
    const VectorR& running_variance,
    TensorView& output)
{
#ifndef CUDA
    const Index neurons_number = gamma.size();
    const Index effective_batch_size = input.size() / neurons_number;

    const MatrixMap input_matrix(input.data, effective_batch_size, neurons_number);
    MatrixMap output_matrix(output.data, effective_batch_size, neurons_number);

    const VectorMap gammas = gamma.as_vector();
    const VectorMap betas = beta.as_vector();

    // output = gamma * (input - running_mean) / sqrt(running_variance + eps) + beta
    output_matrix = ((input_matrix.rowwise() - running_mean.transpose()).array()
                     .rowwise() / (running_variance.array() + EPSILON).sqrt().transpose())
                    .rowwise() * gammas.transpose().array()
                    + betas.transpose().replicate(effective_batch_size, 1).array();
#else
    const cudnnBatchNormMode_t mode = (input.get_rank() == 4)
        ? CUDNN_BATCHNORM_SPATIAL
        : CUDNN_BATCHNORM_PER_ACTIVATION;

    CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
        Device::get_cudnn_handle(),
        mode,
        &one, &zero,
        input.get_descriptor(), input.data,
        output.get_descriptor(), output.data,
        gamma.get_descriptor(),
        gamma.data, beta.data,
        running_mean.data(), running_variance.data(),
        EPSILON));
#endif
}


void batch_normalization_backward(
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

    // 3. Gamma gradient: sum of (output gradient * x_hat)
    gamma_gradients = (output_gradients.array() * x_hat.array()).matrix().colwise().sum();
    // 4. Input gradient (input_gradient):
    // Formula: (gamma * inv_std / m) * (m * dy - sum_dy - x_hat * sum_dy_xhat)
    const type batch_size_type = static_cast<type>(effective_batch_size);

    const Eigen::Array<type, 1, Eigen::Dynamic> scale = (gammas.array() * inverse_variances.array() / batch_size_type).transpose();

    input_gradients.array() = ((batch_size_type * output_gradients.array()).rowwise() - beta_gradients.transpose().array()
                               - x_hat.array().rowwise() * gamma_gradients.transpose().array())
                              .rowwise() * scale;

#else
    // Use SPATIAL for 4D (Conv) and PER_ACTIVATION for 2D (Dense)

    const cudnnBatchNormMode_t mode = (input.get_rank() == 4)
        ? CUDNN_BATCHNORM_SPATIAL
        : CUDNN_BATCHNORM_PER_ACTIVATION;

    CHECK_CUDNN(cudnnBatchNormalizationBackward(
        Device::get_cudnn_handle(),
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

void combination(const TensorView& input,
                        const TensorView& weights,
                        const TensorView& biases,
                        TensorView& output)
{
#ifndef CUDA
    output.as_matrix().noalias()
        = (input.as_matrix() * weights.as_matrix()).rowwise() + biases.as_vector().transpose();
#else
    CHECK_CUDNN(cudnnAddTensor(Device::get_cudnn_handle(),
                               &one, biases.get_descriptor(), biases.data,
                               &zero, output.get_descriptor(), output.data));

    multiply(input, false, weights, false, output, 1.0f, 1.0f);
#endif
}

void batch_normalization_training(
    const TensorView& input,
    const TensorView& gamma,
    const TensorView& beta,
    VectorR& running_mean,
    VectorR& running_variance,
    TensorView& mean,             // Output: current batch mean
    TensorView& inverse_variance, // Output: current batch 1/sqrt(var + epsilon)
    TensorView& output,
    type momentum)
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
        Device::get_cudnn_handle(),
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


void activation(TensorView& output, ActivationFunction func)
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
        softmax(output);
        return;
    }

    default:
        return;
    }
#else
    // @todo CUDA activation without descriptor — not used by Dense (Dense passes ActivationArguments)
    (void)output; (void)func;
#endif
}

void activation(TensorView& output, ActivationArguments arguments)
{
    if (output.empty() || arguments.activation_function == ActivationFunction::Linear)
        return;

#ifndef CUDA
    activation(output, arguments.activation_function);
#else
    const ActivationFunction func = arguments.activation_function;

    if(func == ActivationFunction::Softmax)
    {
        CHECK_CUDNN(cudnnSoftmaxForward(Device::get_cudnn_handle(),
                                        CUDNN_SOFTMAX_ACCURATE,
                                        CUDNN_SOFTMAX_MODE_CHANNEL,
                                        &one,
                                        output.get_descriptor(), output.data,
                                        &zero,
                                        output.get_descriptor(), output.data));
    }
    else
    {
        CHECK_CUDNN(cudnnActivationForward(Device::get_cudnn_handle(),
                                           arguments.activation_descriptor,
                                           &one,
                                           output.get_descriptor(), output.data,
                                           &zero,
                                           output.get_descriptor(), output.data));
    }
#endif
}



void activation_gradient(const TensorView& outputs,
                                const TensorView& output_gradient,
                                TensorView& activation_derivative,
                                const ActivationFunction func,
                                void* act_desc)
{
    if (outputs.empty()) return;

#ifndef CUDA
    (void)act_desc;

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
    if(func == ActivationFunction::Linear || func == ActivationFunction::Softmax)
    {
        if(activation_derivative.data != output_gradient.data)
            CHECK_CUDA(cudaMemcpy(activation_derivative.data, output_gradient.data,
                                  output_gradient.size() * sizeof(float), cudaMemcpyDeviceToDevice));
        return;
    }

    cudnnActivationDescriptor_t descriptor = static_cast<cudnnActivationDescriptor_t>(act_desc);

    CHECK_CUDNN(cudnnActivationBackward(Device::get_cudnn_handle(),
                                        descriptor,
                                        &one,
                                        outputs.get_descriptor(),
                                        outputs.data,
                                        output_gradient.get_descriptor(),
                                        output_gradient.data,
                                        outputs.get_descriptor(),
                                        outputs.data,
                                        &zero,
                                        activation_derivative.get_descriptor(),
                                        activation_derivative.data));
#endif
}


void dropout(TensorView& output, type dropout_rate)
{
#ifndef CUDA
    const type scale = type(1) / (type(1) - dropout_rate);

    type* data = output.data;
    const Index n = output.size();

    for (Index i = 0; i < n; ++i)
        data[i] = (random_uniform(type(0), type(1)) < dropout_rate) ? type(0) : data[i] * scale;
#else
    // @todo CUDA dropout: needs dropout descriptor and reserve space from layer
    (void)output; (void)dropout_rate;
#endif
}


void dropout_gradient(const TensorView& output_gradient,
                             const TensorView& mask, type dropout_rate,
                             TensorView& input_gradient)
{
#ifndef CUDA
    const type scale = type(1) / (type(1) - dropout_rate);

    input_gradient.as_vector().array() = output_gradient.as_vector().array() * mask.as_vector().array().cast<type>() * scale;
#else
    // @todo CUDA dropout backward
    (void)output_gradient; (void)mask; (void)dropout_rate; (void)input_gradient;
#endif
}

void convolution(const TensorView& input,
                        const TensorView& kernel,
                        const TensorView& bias,
                        TensorView& output,
                        const ConvolutionArguments& args)
{
#ifndef CUDA
    (void)args;

    const TensorMap4 inputs = input.as_tensor<4>();
    const VectorMap biases = bias.as_vector();

    const Index batch_size = inputs.dimension(0);
    const Index output_height = output.shape[1];
    const Index output_width = output.shape[2];
    const Index kernels_number = kernel.shape[0];
    const Index kernel_height = kernel.shape[1];
    const Index kernel_width = kernel.shape[2];
    const Index kernel_channels = kernel.shape[3];
    const Index single_kernel_size = kernel_height * kernel_width * kernel_channels;

    const Eigen::array<Index, 3> conv_dims({1, 2, 3});
    const Eigen::array<Index, 3> out_slice_shape({batch_size, output_height, output_width});

    TensorMap4 outputs = output.as_tensor<4>();

    for(Index ki = 0; ki < kernels_number; ki++)
    {
        TensorMap3 kw(kernel.data + ki * single_kernel_size, kernel_height, kernel_width, kernel_channels);
        outputs.chip(ki, 3).device(get_device()) =
            inputs.convolve(kw, conv_dims).reshape(out_slice_shape) + biases(ki);
    }
#else
    CHECK_CUDNN(cudnnConvolutionForward(Device::get_cudnn_handle(),
        &one,
        input.get_descriptor(), input.data,
        args.kernel_descriptor, kernel.data,
        args.convolution_descriptor,
        args.algorithm_forward,
        args.workspace, args.workspace_size,
        &zero,
        output.get_descriptor(), output.data));

    CHECK_CUDNN(cudnnAddTensor(Device::get_cudnn_handle(),
        &one, bias.get_descriptor(), bias.data,
        &one, output.get_descriptor(), output.data));
#endif
}


void convolution_activation(const TensorView& input,
                            const TensorView& weight,
                            const TensorView& bias,
                            TensorView& output,
                            const ConvolutionArguments& conv_args,
                            const ActivationArguments& act_args)
{
#ifndef CUDA
    convolution(input, weight, bias, output, conv_args);
    opennn::activation(output, act_args);
#else
    CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
        Device::get_cudnn_handle(),
        &one,
        input.get_descriptor(), input.data,
        conv_args.kernel_descriptor, weight.data,
        conv_args.convolution_descriptor,
        conv_args.algorithm_forward,
        conv_args.workspace, conv_args.workspace_size,
        &zero,
        output.get_descriptor(), output.data,
        bias.get_descriptor(), bias.data,
        act_args.activation_descriptor,
        output.get_descriptor(), output.data));
#endif
}


void multiply(const TensorView& input_A, bool transpose_A,
                     const TensorView& input_B, bool transpose_B,
                     TensorView& output_C,
                     type alpha, type beta)
{
    const size_t rank = input_A.get_rank();

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
    // Row-major to cuBLAS col-major: C = A * B (row-major) ≡ C^T = B^T * A^T (col-major)
    // Row-major matrix (r,c) in memory == col-major (c,r) with ld=c
    // User "no transpose" → cuBLAS op=N, User "transpose" → cuBLAS op=T

    const int rows_A = static_cast<int>(input_A.shape[rank - 2]);
    const int cols_A = static_cast<int>(input_A.shape[rank - 1]);
    const int rows_B = static_cast<int>(input_B.shape[rank - 2]);
    const int cols_B = static_cast<int>(input_B.shape[rank - 1]);

    const int m = transpose_B ? rows_B : cols_B;
    const int n = transpose_A ? cols_A : rows_A;
    const int k = transpose_A ? rows_A : cols_A;

    const cublasOperation_t op_B_cublas = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t op_A_cublas = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int ld_B = cols_B;
    const int ld_A = cols_A;
    const int ld_C = m;

    const int batch_count = static_cast<int>(input_A.size() / (rows_A * cols_A));
    const long long stride_A = rows_A * cols_A;
    const long long stride_B = rows_B * cols_B;
    const long long stride_C = output_C.shape[rank - 2] * output_C.shape[rank - 1];

    if(batch_count == 1)
    {
        CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(),
                                 op_B_cublas, op_A_cublas,
                                 m, n, k,
                                 &alpha,
                                 input_B.data, ld_B,
                                 input_A.data, ld_A,
                                 &beta,
                                 output_C.data, ld_C));
    }
    else
    {
        CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
                                               op_B_cublas, op_A_cublas,
                                               m, n, k,
                                               &alpha,
                                               input_B.data, ld_B, stride_B,
                                               input_A.data, ld_A, stride_A,
                                               &beta,
                                               output_C.data, ld_C, stride_C,
                                               batch_count));
    }
#endif
}


void multiply_elementwise(const TensorView& A, const TensorView& B, TensorView& C)
{
#ifndef CUDA
    C.as_vector().array() = A.as_vector().array() * B.as_vector().array();
#else
    CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(), Device::get_operator_multiplication_descriptor(),
                              &one, A.get_descriptor(), A.data,
                              &one, B.get_descriptor(), B.data,
                              &zero, C.get_descriptor(), C.data));
#endif
}

void sum(const TensorView& A, TensorView& B, type alpha, type beta)
{
#ifndef CUDA
    B.as_vector().noalias() = alpha * A.as_matrix().colwise().sum() + beta * B.as_vector();
#else
    const int rows = static_cast<int>(A.shape[0]);
    const int cols = static_cast<int>(A.shape.size() / A.shape[0]);

    static float* ones_device = nullptr;
    static int ones_size = 0;
    if(rows > ones_size)
    {
        if(ones_device) cudaFree(ones_device);
        cudaMalloc(&ones_device, rows * sizeof(float));
        vector<float> ones_host(rows, 1.0f);
        cudaMemcpy(ones_device, ones_host.data(), rows * sizeof(float), cudaMemcpyHostToDevice);
        ones_size = rows;
    }

    CHECK_CUBLAS(cublasSgemv(Device::get_cublas_handle(),
                             CUBLAS_OP_N,
                             cols, rows,
                             &alpha,
                             A.data, cols,
                             ones_device, 1,
                             &beta,
                             B.data, 1));
#endif
}


void softmax(TensorView& output)
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
    CHECK_CUDNN(cudnnSoftmaxForward(Device::get_cudnn_handle(),
                                    CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &one,
                                    output.get_descriptor(), output.data,
                                    &zero,
                                    output.get_descriptor(), output.data));
#endif
}


// Convolution backward stubs

void convolution_backward_weights(const TensorView& input,
                                         const TensorView& delta,
                                         TensorView& weight_grad,
                                         TensorView& bias_grad,
                                         const ConvolutionArguments& args)
{
#ifndef CUDA
    (void)args;

    const TensorMap4 inputs = input.as_tensor<4>();
    const TensorMap4 deltas = delta.as_tensor<4>();

    const Index batch_size = inputs.dimension(0);
    const Index kernels_number = weight_grad.shape[0];
    const Index kernel_height = weight_grad.shape[1];
    const Index kernel_width = weight_grad.shape[2];
    const Index kernel_channels = weight_grad.shape[3];
    const Index single_kernel_size = kernel_height * kernel_width * kernel_channels;
    const Index output_height = deltas.dimension(1);
    const Index output_width = deltas.dimension(2);
    const Index input_height = inputs.dimension(1);
    const Index input_width = inputs.dimension(2);

    VectorMap bias_gradients = bias_grad.as_vector();
    bias_gradients.setZero();

    for(Index ki = 0; ki < kernels_number; ki++)
    {
        type* wg = weight_grad.data + ki * single_kernel_size;
        memset(wg, 0, single_kernel_size * sizeof(type));
        TensorMap3 wg_map(wg, kernel_height, kernel_width, kernel_channels);

        for(Index b = 0; b < batch_size; b++)
        {
            for(Index oh = 0; oh < output_height; oh++)
                for(Index ow = 0; ow < output_width; ow++)
                {
                    const type d = deltas(b, oh, ow, ki);
                    bias_gradients(ki) += d;

                    for(Index kh = 0; kh < kernel_height; kh++)
                        for(Index kw = 0; kw < kernel_width; kw++)
                        {
                            const Index ih = oh + kh;
                            const Index iw = ow + kw;

                            if(ih < input_height && iw < input_width)
                                for(Index c = 0; c < kernel_channels; c++)
                                    wg_map(kh, kw, c) += d * inputs(b, ih, iw, c);
                        }
                }
        }
    }
#else
    CHECK_CUDNN(cudnnConvolutionBackwardFilter(Device::get_cudnn_handle(),
        &one,
        input.get_descriptor(), input.data,
        delta.get_descriptor(), delta.data,
        args.convolution_descriptor,
        args.algorithm_filter,
        args.backward_filter_workspace, args.backward_filter_workspace_size,
        &zero,
        args.kernel_descriptor, weight_grad.data));

    CHECK_CUDNN(cudnnConvolutionBackwardBias(Device::get_cudnn_handle(),
        &one,
        delta.get_descriptor(), delta.data,
        &zero,
        bias_grad.get_descriptor(), bias_grad.data));
#endif
}

void convolution_backward_data(const TensorView& delta,
                                      const TensorView& kernel,
                                      TensorView& input_grad,
                                      TensorView& /*padded_input_grad*/,
                                      const ConvolutionArguments& args)
{
#ifndef CUDA
    (void)args;

    const TensorMap4 deltas = delta.as_tensor<4>();
    TensorMap4 in_grad = input_grad.as_tensor<4>();
    in_grad.setZero();

    const Index batch_size = deltas.dimension(0);
    const Index output_height = deltas.dimension(1);
    const Index output_width = deltas.dimension(2);
    const Index kernels_number = kernel.shape[0];
    const Index kernel_height = kernel.shape[1];
    const Index kernel_width = kernel.shape[2];
    const Index kernel_channels = kernel.shape[3];
    const Index single_kernel_size = kernel_height * kernel_width * kernel_channels;
    const Index input_height = in_grad.dimension(1);
    const Index input_width = in_grad.dimension(2);

    for(Index ki = 0; ki < kernels_number; ki++)
    {
        const type* kw = kernel.data + ki * single_kernel_size;

        for(Index b = 0; b < batch_size; b++)
            for(Index oh = 0; oh < output_height; oh++)
                for(Index ow = 0; ow < output_width; ow++)
                {
                    const type d = deltas(b, oh, ow, ki);

                    for(Index kh = 0; kh < kernel_height; kh++)
                        for(Index kwi = 0; kwi < kernel_width; kwi++)
                        {
                            const Index ih = oh + kh;
                            const Index iw = ow + kwi;

                            if(ih < input_height && iw < input_width)
                                for(Index ci = 0; ci < kernel_channels; ci++)
                                    in_grad(b, ih, iw, ci) += d * kw[kh * kernel_width * kernel_channels + kwi * kernel_channels + ci];
                        }
                }
    }
#else
    CHECK_CUDNN(cudnnConvolutionBackwardData(Device::get_cudnn_handle(),
        &one,
        args.kernel_descriptor, kernel.data,
        delta.get_descriptor(), delta.data,
        args.convolution_descriptor,
        args.algorithm_data,
        args.workspace, args.workspace_size,
        &zero,
        input_grad.get_descriptor(), input_grad.data));
#endif
}

// Pooling backward

void max_pooling_backward(const TensorView& input,
                                 const TensorView& output,
                                 const TensorView& output_gradient,
                                 const TensorView& maximal_indices,
                                 TensorView& input_gradient,
                                 const PoolingArguments& args)
{
#ifndef CUDA
    (void)output; (void)args;

    const TensorMap4 out_grads = output_gradient.as_tensor<4>();
    const TensorMap4 max_indices = maximal_indices.as_tensor<4>();
    TensorMap4 in_grads = input_gradient.as_tensor<4>();
    in_grads.setZero();

    const Index batch_size = out_grads.dimension(0);
    const Index output_height = out_grads.dimension(1);
    const Index output_width = out_grads.dimension(2);
    const Index channels = out_grads.dimension(3);

    const Index pool_height = args.pool_dimensions[0];
    const Index pool_width = args.pool_dimensions[1];
    const Index row_stride = args.stride_shape[0];
    const Index column_stride = args.stride_shape[1];
    const Index padding_height = args.padding_shape[0];
    const Index padding_width = args.padding_shape[1];

    #pragma omp parallel for collapse(2)
    for(Index b = 0; b < batch_size; b++)
        for(Index c = 0; c < channels; c++)
            for(Index oh = 0; oh < output_height; oh++)
                for(Index ow = 0; ow < output_width; ow++)
                {
                    const Index max_idx = static_cast<Index>(max_indices(b, oh, ow, c));
                    const Index kh = max_idx / pool_width;
                    const Index kw = max_idx % pool_width;
                    const Index ih = oh * row_stride - padding_height + kh;
                    const Index iw = ow * column_stride - padding_width + kw;

                    if(ih >= 0 && ih < in_grads.dimension(1) && iw >= 0 && iw < in_grads.dimension(2))
                        in_grads(b, ih, iw, c) += out_grads(b, oh, ow, c);
                }
#else
    CHECK_CUDNN(cudnnPoolingBackward(Device::get_cudnn_handle(),
        args.pooling_descriptor,
        &one,
        output.get_descriptor(), output.data,
        output_gradient.get_descriptor(), output_gradient.data,
        input.get_descriptor(), input.data,
        &zero,
        input_gradient.get_descriptor(), input_gradient.data));
#endif
}

void average_pooling_backward(const TensorView& input,
                                     const TensorView& output,
                                     const TensorView& output_gradient,
                                     TensorView& input_gradient,
                                     const PoolingArguments& args)
{
#ifndef CUDA
    (void)output;

    const TensorMap4 inputs = input.as_tensor<4>();
    const TensorMap4 out_grads = output_gradient.as_tensor<4>();
    TensorMap4 in_grads = input_gradient.as_tensor<4>();
    in_grads.setZero();

    const Index batch_size = inputs.dimension(0);
    const Index input_height = inputs.dimension(1);
    const Index input_width = inputs.dimension(2);
    const Index channels = inputs.dimension(3);
    const Index output_height = out_grads.dimension(1);
    const Index output_width = out_grads.dimension(2);

    const Index pool_height = args.pool_dimensions[0];
    const Index pool_width = args.pool_dimensions[1];
    const Index row_stride = args.stride_shape[0];
    const Index column_stride = args.stride_shape[1];
    const Index padding_height = args.padding_shape[0];
    const Index padding_width = args.padding_shape[1];

    const type inv_pool_size = type(1) / (pool_height * pool_width);

    #pragma omp parallel for collapse(2)
    for(Index b = 0; b < batch_size; b++)
        for(Index c = 0; c < channels; c++)
            for(Index oh = 0; oh < output_height; oh++)
                for(Index ow = 0; ow < output_width; ow++)
                {
                    const type avg_grad = out_grads(b, oh, ow, c) * inv_pool_size;

                    const Index ih_start = oh * row_stride - padding_height;
                    const Index iw_start = ow * column_stride - padding_width;

                    for(Index ph = 0; ph < pool_height; ph++)
                        for(Index pw = 0; pw < pool_width; pw++)
                        {
                            const Index ih = ih_start + ph;
                            const Index iw = iw_start + pw;

                            if(ih >= 0 && ih < input_height && iw >= 0 && iw < input_width)
                                in_grads(b, ih, iw, c) += avg_grad;
                        }
                }
#else
    CHECK_CUDNN(cudnnPoolingBackward(Device::get_cudnn_handle(),
        args.pooling_descriptor,
        &one,
        output.get_descriptor(), output.data,
        output_gradient.get_descriptor(), output_gradient.data,
        input.get_descriptor(), input.data,
        &zero,
        input_gradient.get_descriptor(), input_gradient.data));
#endif
}

// Pooling 3D stubs

void max_pooling_3d_forward(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training)
{
#ifndef CUDA
    const TensorMap3 inputs = input.as_tensor<3>();
    MatrixMap outputs = output.as_matrix();

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index features = inputs.dimension(2);

    MatrixMap max_idx = maximal_indices.as_matrix();

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
                    if(is_training) max_idx(b, f) = static_cast<type>(s);
                }
            }
    }
#else
    (void)is_training;
    const int B = static_cast<int>(input.shape[0]);
    const int S = static_cast<int>(input.shape[1]);
    const int F = static_cast<int>(input.shape[2]);
    pooling3d_max_forward_cuda(B * F, input.data, output.data, maximal_indices.data, B, S, F);
#endif
}

void average_pooling_3d_forward(const TensorView& input, TensorView& output)
{
#ifndef CUDA
    const TensorMap3 inputs = input.as_tensor<3>();
    MatrixMap outputs = output.as_matrix();

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
    const int B = static_cast<int>(input.shape[0]);
    const int S = static_cast<int>(input.shape[1]);
    const int F = static_cast<int>(input.shape[2]);
    pooling3d_avg_forward_cuda(B * F, input.data, output.data, B, S, F);
#endif
}

void max_pooling_3d_backward(const TensorView& maximal_indices, const TensorView& output_gradient, TensorView& input_gradient)
{
#ifndef CUDA
    const MatrixMap max_idx = maximal_indices.as_matrix();
    const MatrixMap delta = output_gradient.as_matrix();
    TensorMap3 in_grad = input_gradient.as_tensor<3>();
    in_grad.setZero();

    const Index batch_size = delta.rows();
    const Index features = delta.cols();

    #pragma omp parallel for
    for(Index b = 0; b < batch_size; ++b)
        for(Index f = 0; f < features; ++f)
        {
            const Index s = static_cast<Index>(max_idx(b, f));
            in_grad(b, s, f) = delta(b, f);
        }
#else
    const int B = static_cast<int>(output_gradient.shape[0]);
    const int F = static_cast<int>(output_gradient.shape[1]);
    const int S = static_cast<int>(input_gradient.shape[1]);
    CHECK_CUDA(cudaMemset(input_gradient.data, 0, input_gradient.size() * sizeof(float)));
    pooling3d_max_backward_cuda(B * F, output_gradient.data, input_gradient.data, maximal_indices.data, B, S, F);
#endif
}

void average_pooling_3d_backward(const TensorView& input, const TensorView& output_gradient, TensorView& input_gradient)
{
#ifndef CUDA
    const TensorMap3 inputs = input.as_tensor<3>();
    const MatrixMap delta = output_gradient.as_matrix();
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
    const int B = static_cast<int>(input.shape[0]);
    const int S = static_cast<int>(input.shape[1]);
    const int F = static_cast<int>(input.shape[2]);
    CHECK_CUDA(cudaMemset(input_gradient.data, 0, input_gradient.size() * sizeof(float)));
    pooling3d_avg_backward_cuda(B * F, input.data, output_gradient.data, input_gradient.data, B, S, F);
#endif
}

// Embedding backward stub

void embedding_backward(const TensorView& input_indices,
                               const TensorView& output_gradient,
                               TensorView& weight_gradient,
                               Index embedding_dimension,
                               bool scale_embedding)
{
    const Index total_elements = input_indices.size();

    MatrixMap gradients_map(output_gradient.data, total_elements, embedding_dimension);

    if(scale_embedding)
        gradients_map *= sqrt(static_cast<type>(embedding_dimension));

    MatrixMap weight_gradients = weight_gradient.as_matrix();
    weight_gradients.setZero();

    for(Index i = 0; i < total_elements; i++)
    {
        const Index vocab_idx = static_cast<Index>(input_indices.data[i]);

        if(vocab_idx < 0 || vocab_idx >= weight_gradients.rows())
            continue;

        weight_gradients.row(vocab_idx).noalias() += gradients_map.row(i);
    }

    weight_gradients.row(0).setZero();
}

// Multi-head attention stubs

void multihead_attention_forward(
    const TensorView& query, const TensorView& key, const TensorView& value,
    TensorView& attention_weights, TensorView& concatenated, TensorView& output,
    const TensorView& projection_weights, const TensorView& projection_biases,
    const TensorView& source_input,
    Index batch_size, Index heads_number,
    Index query_sequence_length, Index source_sequence_length,
    Index embedding_dimension, Index head_dimension,
    type scaling_factor, bool use_causal_mask, const MatrixR& causal_mask)
{
#ifndef CUDA
    const Index total_heads = batch_size * heads_number;

    // Attention scores: Q * K^T * scaling_factor
    #pragma omp parallel for
    for(Index i = 0; i < total_heads; ++i)
    {
        const MatrixMap q(query.data + i * query_sequence_length * head_dimension, query_sequence_length, head_dimension);
        const MatrixMap k(key.data + i * source_sequence_length * head_dimension, source_sequence_length, head_dimension);
        MatrixMap w(attention_weights.data + i * query_sequence_length * source_sequence_length, query_sequence_length, source_sequence_length);
        w.noalias() = (q * k.transpose()) * scaling_factor;
    }

    if(causal_mask.size() > 0)
        for(Index i = 0; i < total_heads; ++i)
        {
            MatrixMap w(attention_weights.data + i * query_sequence_length * source_sequence_length, query_sequence_length, source_sequence_length);
            w += causal_mask;
        }

    // Softmax
    const Index total_rows = total_heads * query_sequence_length;
    MatrixMap att_map(attention_weights.data, total_rows, source_sequence_length);
    for(Index r = 0; r < total_rows; ++r)
    {
        const type mx = att_map.row(r).maxCoeff();
        att_map.row(r).array() = (att_map.row(r).array() - mx).exp();
        att_map.row(r) /= att_map.row(r).sum();
    }

    // Attention output: P * V → concatenated (with head interleaving)
    #pragma omp parallel for collapse(2)
    for(Index b = 0; b < batch_size; ++b)
        for(Index h = 0; h < heads_number; ++h)
        {
            const Index off_w = (b * heads_number + h) * query_sequence_length * source_sequence_length;
            const Index off_v = (b * heads_number + h) * source_sequence_length * head_dimension;
            const MatrixMap w(attention_weights.data + off_w, query_sequence_length, source_sequence_length);
            const MatrixMap v(value.data + off_v, source_sequence_length, head_dimension);
            type* out_ptr = concatenated.data + b * query_sequence_length * embedding_dimension + h * head_dimension;
            using Stride = Eigen::OuterStride<Eigen::Dynamic>;
            Eigen::Map<MatrixR, 0, Stride> o(out_ptr, query_sequence_length, head_dimension, Stride(embedding_dimension));
            o.noalias() = w * v;
        }

    // Output projection
    const Index total = batch_size * query_sequence_length;
    const MatrixMap concat_map(concatenated.data, total, embedding_dimension);
    MatrixMap out_map(output.data, total, embedding_dimension);
    const MatrixMap proj_w(projection_weights.data, embedding_dimension, embedding_dimension);
    const VectorMap proj_b(projection_biases.data, embedding_dimension);
    out_map.noalias() = (concat_map * proj_w).rowwise() + proj_b.transpose();

#else
    (void)causal_mask;

    const int BH = static_cast<int>(batch_size * heads_number);
    const int Sq = static_cast<int>(query_sequence_length);
    const int Sk = static_cast<int>(source_sequence_length);
    const int E  = static_cast<int>(embedding_dimension);
    const int D  = static_cast<int>(head_dimension);
    const int H  = static_cast<int>(heads_number);
    const int B  = static_cast<int>(batch_size);
    const float sf = static_cast<float>(scaling_factor);

    // Temp buffers (static for reuse)
    static float* q_t = nullptr; static float* k_t = nullptr; static float* v_t = nullptr;
    static float* att_out_t = nullptr; static float* att_probs = nullptr;
    static float* pad_mask = nullptr;
    static size_t alloc_qkv = 0, alloc_att = 0, alloc_out = 0, alloc_probs = 0, alloc_mask = 0;

    const size_t qkv_size = batch_size * heads_number * max(query_sequence_length, source_sequence_length) * head_dimension;
    const size_t att_size = batch_size * heads_number * query_sequence_length * source_sequence_length;
    const size_t out_size = batch_size * heads_number * query_sequence_length * head_dimension;
    const size_t mask_size = batch_size * source_sequence_length;

    if(qkv_size > alloc_qkv) { if(q_t) cudaFree(q_t); if(k_t) cudaFree(k_t); if(v_t) cudaFree(v_t);
        cudaMalloc(&q_t, qkv_size*sizeof(float)); cudaMalloc(&k_t, qkv_size*sizeof(float)); cudaMalloc(&v_t, qkv_size*sizeof(float)); alloc_qkv = qkv_size; }
    if(att_size > alloc_probs) { if(att_probs) cudaFree(att_probs); cudaMalloc(&att_probs, att_size*sizeof(float)); alloc_probs = att_size; }
    if(out_size > alloc_out) { if(att_out_t) cudaFree(att_out_t); cudaMalloc(&att_out_t, out_size*sizeof(float)); alloc_out = out_size; }
    if(mask_size > alloc_mask) { if(pad_mask) cudaFree(pad_mask); cudaMalloc(&pad_mask, mask_size*sizeof(float)); alloc_mask = mask_size; }

    // Transpose Q,K,V: [B,S,H,D] → [B,H,S,D]
    mha_transpose_qkv_cuda(B * Sq * E, query.data, q_t, Sq, H, D);
    mha_transpose_qkv_cuda(B * Sk * E, key.data, k_t, Sk, H, D);
    mha_transpose_qkv_cuda(B * Sk * E, value.data, v_t, Sk, H, D);

    // Q * K^T → attention_weights [BH, Sq, Sk] (col-major cublas)
    CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
        CUBLAS_OP_T, CUBLAS_OP_N, Sk, Sq, D,
        &sf,
        k_t, D, Sk * D,
        q_t, D, Sq * D,
        &zero,
        attention_weights.data, Sk, Sq * Sk,
        BH));

    // Apply masks (padding + causal)
    {
        const size_t att_n = static_cast<size_t>(BH) * Sq * Sk;
        mha_key_padding_mask_cuda(att_n, source_input.data, attention_weights.data, H, Sq, Sk, E);
        if(use_causal_mask)
            mha_causal_mask_cuda(att_n, attention_weights.data, Sq, Sk);
    }

    // Softmax: need descriptor for [BH*Sq, Sk] as [BH*Sq, 1, 1, Sk] NCHW
    TensorView att_view(attention_weights.data, {(Index)(BH * Sq), (Index)Sk});
    att_view.set_descriptor(att_view.shape);

    TensorView probs_view(att_probs, {(Index)(BH * Sq), (Index)Sk});
    probs_view.set_descriptor(probs_view.shape);

    CHECK_CUDNN(cudnnSoftmaxForward(Device::get_cudnn_handle(),
        CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
        &one, att_view.get_descriptor(), attention_weights.data,
        &zero, probs_view.get_descriptor(), att_probs));

    // Copy probs back to attention_weights for backward
    CHECK_CUDA(cudaMemcpy(attention_weights.data, att_probs, att_size * sizeof(float), cudaMemcpyDeviceToDevice));

    // P * V → att_out_transposed [BH, Sq, D]
    CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
        CUBLAS_OP_N, CUBLAS_OP_N, D, Sq, Sk,
        &one,
        v_t, D, Sk * D,
        att_probs, Sk, Sq * Sk,
        &zero,
        att_out_t, D, Sq * D,
        BH));

    // Transpose back: [B,H,Sq,D] → [B,Sq,H,D] = [B,Sq,E]
    mha_transpose_o_cuda(B * Sq * E, att_out_t, concatenated.data, Sq, H, D);

    // Output projection: output = concat * W_proj + b_proj
    TensorView concat_2d(concatenated.data, {(Index)(B * Sq), (Index)E});
    TensorView output_2d(output.data, {(Index)(B * Sq), (Index)E});
    concat_2d.set_descriptor(concat_2d.shape);
    output_2d.set_descriptor(output_2d.shape);
    combination(concat_2d, projection_weights, projection_biases, output_2d);
#endif
}

void multihead_attention_backward(
    const TensorView& query_input, const TensorView& source_input, TensorView& output_gradient,
    const TensorView& query, const TensorView& key, const TensorView& value,
    const TensorView& attention_weights, const TensorView& concatenated,
    const TensorView& projection_weights,
    TensorView& proj_weight_grad, TensorView& proj_bias_grad,
    TensorView& concat_grad, TensorView& att_weight_grad,
    TensorView& query_grad, TensorView& key_grad, TensorView& value_grad,
    TensorView& query_weight_grad, TensorView& query_bias_grad,
    TensorView& key_weight_grad, TensorView& key_bias_grad,
    TensorView& value_weight_grad, TensorView& value_bias_grad,
    TensorView& input_query_grad,
    const TensorView& query_weights, const TensorView& key_weights, const TensorView& value_weights,
    Index batch_size, Index heads_number,
    Index query_sequence_length, Index source_sequence_length,
    Index embedding_dimension, Index head_dimension,
    type scaling_factor, bool self_attention)
{
    const Index total_rows = batch_size * query_sequence_length;
    const Index total_heads = batch_size * heads_number;

    // 1. Output projection gradients: dW_proj = concat^T * dY, db_proj = sum(dY)
    TensorView concat_2d(concatenated.data, {total_rows, embedding_dimension});
    TensorView dY_2d(output_gradient.data, {total_rows, embedding_dimension});

    multiply(concat_2d, true, dY_2d, false, proj_weight_grad);
    sum(dY_2d, proj_bias_grad);

    // 2. Backprop through output projection: d_concat = dY * W_proj^T
    TensorView concat_grad_2d(concat_grad.data, {total_rows, embedding_dimension});
    multiply(dY_2d, false, projection_weights, true, concat_grad_2d);

#ifndef CUDA
    // 3-6: CPU path uses per-head strided access for concat↔heads rearrangement

    // 3. dV = P^T * dO, dP = dO * V^T (per head with strided concat_grad access)
    #pragma omp parallel for collapse(2)
    for(Index b = 0; b < batch_size; ++b)
        for(Index h = 0; h < heads_number; ++h)
        {
            const Index off_w = b * (heads_number * query_sequence_length * source_sequence_length) + h * (query_sequence_length * source_sequence_length);
            const Index off_v = b * (heads_number * source_sequence_length * head_dimension) + h * (source_sequence_length * head_dimension);

            const MatrixMap P(attention_weights.data + off_w, query_sequence_length, source_sequence_length);
            const MatrixMap V(value.data + off_v, source_sequence_length, head_dimension);
            MatrixMap dV(value_grad.data + off_v, source_sequence_length, head_dimension);
            MatrixMap dP(att_weight_grad.data + off_w, query_sequence_length, source_sequence_length);

            type* dO_ptr = concat_grad.data + b * (query_sequence_length * embedding_dimension) + h * head_dimension;
            using Stride = Eigen::OuterStride<Eigen::Dynamic>;
            Eigen::Map<const MatrixR, 0, Stride> dO(dO_ptr, query_sequence_length, head_dimension, Stride(embedding_dimension));

            dV.noalias() = P.transpose() * dO;
            dP.noalias() = dO * V.transpose();
        }

    // 4. Softmax gradient: dP = P * (dP_raw - rowwise_sum(P * dP_raw))
    #pragma omp parallel for
    for(Index i = 0; i < total_heads; ++i)
    {
        const Index off = i * query_sequence_length * source_sequence_length;
        const MatrixMap P(attention_weights.data + off, query_sequence_length, source_sequence_length);
        MatrixMap dP(att_weight_grad.data + off, query_sequence_length, source_sequence_length);
        VectorR dot = (P.array() * dP.array()).rowwise().sum();
        dP.array() = P.array() * (dP.colwise() - dot).array();
    }

    // 5. dQ = dP * K * scale, dK = dP^T * Q * scale
    #pragma omp parallel for
    for(Index i = 0; i < total_heads; ++i)
    {
        const Index off_w = i * query_sequence_length * source_sequence_length;
        const Index off_q = i * query_sequence_length * head_dimension;
        const Index off_k = i * source_sequence_length * head_dimension;

        const MatrixMap dP(att_weight_grad.data + off_w, query_sequence_length, source_sequence_length);
        const MatrixMap Q(query.data + off_q, query_sequence_length, head_dimension);
        const MatrixMap K(key.data + off_k, source_sequence_length, head_dimension);
        MatrixMap dQ(query_grad.data + off_q, query_sequence_length, head_dimension);
        MatrixMap dK(key_grad.data + off_k, source_sequence_length, head_dimension);

        dQ.noalias() = (dP * K) * scaling_factor;
        dK.noalias() = (dP.transpose() * Q) * scaling_factor;
    }

    // 6. Projection gradients for Q, K, V
    projection_gradient(query_grad, query_input, query_weights, query_bias_grad, query_weight_grad, input_query_grad, batch_size, heads_number, query_sequence_length, embedding_dimension, head_dimension, false);

    if(self_attention)
    {
        projection_gradient(key_grad, source_input, key_weights, key_bias_grad, key_weight_grad, input_query_grad, batch_size, heads_number, source_sequence_length, embedding_dimension, head_dimension, true);
        projection_gradient(value_grad, source_input, value_weights, value_bias_grad, value_weight_grad, input_query_grad, batch_size, heads_number, source_sequence_length, embedding_dimension, head_dimension, true);
    }

#else
    const int BH = static_cast<int>(total_heads);
    const int Sq = static_cast<int>(query_sequence_length);
    const int Sk = static_cast<int>(source_sequence_length);
    const int E  = static_cast<int>(embedding_dimension);
    const int D  = static_cast<int>(head_dimension);
    const int H  = static_cast<int>(heads_number);
    const int B  = static_cast<int>(batch_size);
    const float sf = static_cast<float>(scaling_factor);

    // Temp buffers
    static float* dO_t = nullptr; static float* dV_t = nullptr; static float* dQ_t = nullptr;
    static float* dK_t = nullptr; static float* softmax_grad = nullptr;
    static float* q_grad_flat = nullptr; static float* k_grad_flat = nullptr; static float* v_grad_flat = nullptr;
    static float* q_input_grad = nullptr; static float* src_input_grad = nullptr; static float* ones_buf = nullptr;
    static size_t ba_qkv = 0, ba_att = 0, ba_io = 0, ba_ones = 0;

    const size_t qkv_sz = total_heads * max(query_sequence_length, source_sequence_length) * head_dimension;
    const size_t att_sz = total_heads * query_sequence_length * source_sequence_length;
    const size_t io_sz = max(batch_size * query_sequence_length, batch_size * source_sequence_length) * embedding_dimension;
    const size_t ones_sz = max(batch_size * query_sequence_length, batch_size * source_sequence_length);

    if(qkv_sz > ba_qkv) { if(dO_t) cudaFree(dO_t); if(dV_t) cudaFree(dV_t); if(dQ_t) cudaFree(dQ_t); if(dK_t) cudaFree(dK_t);
        cudaMalloc(&dO_t, qkv_sz*sizeof(float)); cudaMalloc(&dV_t, qkv_sz*sizeof(float));
        cudaMalloc(&dQ_t, qkv_sz*sizeof(float)); cudaMalloc(&dK_t, qkv_sz*sizeof(float)); ba_qkv = qkv_sz; }
    if(att_sz > ba_att) { if(softmax_grad) cudaFree(softmax_grad); cudaMalloc(&softmax_grad, att_sz*sizeof(float)); ba_att = att_sz; }
    if(io_sz > ba_io) { if(q_grad_flat) cudaFree(q_grad_flat); if(k_grad_flat) cudaFree(k_grad_flat); if(v_grad_flat) cudaFree(v_grad_flat);
        if(q_input_grad) cudaFree(q_input_grad); if(src_input_grad) cudaFree(src_input_grad);
        cudaMalloc(&q_grad_flat, io_sz*sizeof(float)); cudaMalloc(&k_grad_flat, io_sz*sizeof(float)); cudaMalloc(&v_grad_flat, io_sz*sizeof(float));
        cudaMalloc(&q_input_grad, io_sz*sizeof(float)); cudaMalloc(&src_input_grad, io_sz*sizeof(float)); ba_io = io_sz; }
    if(ones_sz > ba_ones) { if(ones_buf) cudaFree(ones_buf); cudaMalloc(&ones_buf, ones_sz*sizeof(float));
        vector<float> oh(ones_sz, 1.0f); cudaMemcpy(ones_buf, oh.data(), ones_sz*sizeof(float), cudaMemcpyHostToDevice); ba_ones = ones_sz; }

    // Need transposed Q,K,V from forward (recompute from query/key/value which are in [B,S,H,D] layout)
    static float* q_t = nullptr; static float* k_t = nullptr; static float* v_t = nullptr;
    static size_t ba_fwd = 0;
    if(qkv_sz > ba_fwd) { if(q_t) cudaFree(q_t); if(k_t) cudaFree(k_t); if(v_t) cudaFree(v_t);
        cudaMalloc(&q_t, qkv_sz*sizeof(float)); cudaMalloc(&k_t, qkv_sz*sizeof(float)); cudaMalloc(&v_t, qkv_sz*sizeof(float)); ba_fwd = qkv_sz; }

    mha_transpose_qkv_cuda(B*Sq*E, query.data, q_t, Sq, H, D);
    mha_transpose_qkv_cuda(B*Sk*E, key.data, k_t, Sk, H, D);
    mha_transpose_qkv_cuda(B*Sk*E, value.data, v_t, Sk, H, D);

    // 3. Transpose concat_grad [B,Sq,H,D] → dO_t [B,H,Sq,D]
    mha_transpose_qkv_cuda(B*Sq*E, concat_grad.data, dO_t, Sq, H, D);

    // dV_t = P^T * dO_t (batched)
    CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
        CUBLAS_OP_N, CUBLAS_OP_T, D, Sk, Sq,
        &one, dO_t, D, Sq*D, attention_weights.data, Sk, Sq*Sk,
        &zero, dV_t, D, Sk*D, BH));

    // dP = V_t^T * dO_t^T → att_weight_grad
    CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
        CUBLAS_OP_T, CUBLAS_OP_N, Sk, Sq, D,
        &one, v_t, D, Sk*D, dO_t, D, Sq*D,
        &zero, att_weight_grad.data, Sk, Sq*Sk, BH));

    // 4. Softmax backward
    TensorView att_view(attention_weights.data, {(Index)(BH*Sq), (Index)Sk});
    att_view.set_descriptor(att_view.shape);
    TensorView datt_view(att_weight_grad.data, {(Index)(BH*Sq), (Index)Sk});
    datt_view.set_descriptor(datt_view.shape);
    TensorView sgrad_view(softmax_grad, {(Index)(BH*Sq), (Index)Sk});
    sgrad_view.set_descriptor(sgrad_view.shape);

    CHECK_CUDNN(cudnnSoftmaxBackward(Device::get_cudnn_handle(),
        CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
        &one, att_view.get_descriptor(), attention_weights.data,
        datt_view.get_descriptor(), att_weight_grad.data,
        &zero, sgrad_view.get_descriptor(), softmax_grad));

    // 5. dQ_t = K_t * softmax_grad^T * scale, dK_t = Q_t * softmax_grad * scale
    CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
        CUBLAS_OP_N, CUBLAS_OP_N, D, Sq, Sk,
        &sf, k_t, D, Sk*D, softmax_grad, Sk, Sq*Sk,
        &zero, dQ_t, D, Sq*D, BH));

    CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
        CUBLAS_OP_N, CUBLAS_OP_T, D, Sk, Sq,
        &sf, q_t, D, Sq*D, softmax_grad, Sk, Sq*Sk,
        &zero, dK_t, D, Sk*D, BH));

    // Transpose back: dQ [B,H,Sq,D]→[B,Sq,E], dK/dV [B,H,Sk,D]→[B,Sk,E]
    mha_transpose_o_cuda(B*Sq*E, dQ_t, q_grad_flat, Sq, H, D);
    mha_transpose_o_cuda(B*Sk*E, dK_t, k_grad_flat, Sk, H, D);
    mha_transpose_o_cuda(B*Sk*E, dV_t, v_grad_flat, Sk, H, D);

    // 6. Projection weight/bias/input gradients using cublasSgemm (col-major)
    // Query
    CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
        E, E, B*Sq, &one, q_grad_flat, E, query_input.data, E, &zero, query_weight_grad.data, E));
    CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        E, 1, B*Sq, &one, q_grad_flat, E, ones_buf, B*Sq, &zero, query_bias_grad.data, E));
    CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
        E, B*Sq, E, &one, query_weights.data, E, q_grad_flat, E, &zero, q_input_grad, E));

    // Key
    CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
        E, E, B*Sk, &one, k_grad_flat, E, source_input.data, E, &zero, key_weight_grad.data, E));
    CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        E, 1, B*Sk, &one, k_grad_flat, E, ones_buf, B*Sk, &zero, key_bias_grad.data, E));
    CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
        E, B*Sk, E, &one, key_weights.data, E, k_grad_flat, E, &zero, src_input_grad, E));

    // Value (accumulate source input grad)
    CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
        E, E, B*Sk, &one, v_grad_flat, E, source_input.data, E, &zero, value_weight_grad.data, E));
    CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        E, 1, B*Sk, &one, v_grad_flat, E, ones_buf, B*Sk, &zero, value_bias_grad.data, E));
    CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
        E, B*Sk, E, &one, value_weights.data, E, v_grad_flat, E, &one, src_input_grad, E));

    // Self-attention: input_query_grad = q_input_grad + src_input_grad
    if(self_attention)
        addition_cuda(B * Sq * E, q_input_grad, src_input_grad, input_query_grad.data);
    else
        CHECK_CUDA(cudaMemcpy(input_query_grad.data, q_input_grad, B*Sq*E*sizeof(float), cudaMemcpyDeviceToDevice));
#endif
}

}
