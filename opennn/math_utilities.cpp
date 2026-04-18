//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M A T H   U T I L I T I E S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "math_utilities.h"
#include "random_utilities.h"

#ifdef OPENNN_WITH_CUDA
#include "kernel.cuh"
#endif

namespace opennn
{

void padding(const TensorView& input, TensorView& output)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu())
        throw runtime_error("padding: GPU implementation not available.");
#endif

    const TensorMap4 input_map = input.as_tensor<4>();
    TensorMap4 output_map = output.as_tensor<4>();

    const Index padding_height = (output.shape[1] - input.shape[1]) / 2;
    const Index padding_width = (output.shape[2] - input.shape[2]) / 2;

    const Eigen::array<pair<Index,Index>, 4> paddings = {
        make_pair(Index(0), Index(0)),
        make_pair(padding_height, padding_height),
        make_pair(padding_width, padding_width),
        make_pair(Index(0), Index(0))
    };

    output_map.device(get_device()) = input_map.pad(paddings);
}

void bounding(const TensorView& input,
              const TensorView& lower_bounds,
              const TensorView& upper_bounds,
              TensorView& output)
{
    const Index features = lower_bounds.size();

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu())
    {
        throw runtime_error("bounding: GPU implementation not available.");
    }
#endif
    const MatrixMap input_matrix = input.as_matrix();
    const VectorMap lower_bounds_vector = lower_bounds.as_vector();
    const VectorMap upper_bounds_vector = upper_bounds.as_vector();

    MatrixMap output_matrix = output.as_matrix();

    for(Index feature_index = 0; feature_index < features; ++feature_index)
        output_matrix.col(feature_index) = input_matrix.col(feature_index)
                                                        .cwiseMax(lower_bounds_vector(feature_index))
                                                        .cwiseMin(upper_bounds_vector(feature_index));
}

void copy(const TensorView& source, TensorView& destination)
{
    if(source.size() != destination.size())
        throw runtime_error("Math Error: Tensor sizes mismatch in copy operation.");

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDA(cudaMemcpy(destination.data,
                              source.data,
                              source.size() * sizeof(type),
                              cudaMemcpyDeviceToDevice));
        return;
    }
#endif
    memcpy(destination.data, source.data, source.size() * sizeof(type));
}

void addition(const TensorView& input_1, 
              const TensorView& input_2, 
              TensorView& output)
{
    if(input_1.size() != input_2.size() || input_1.size() != output.size())
        throw runtime_error("Addition Error: Tensor dimensions do not match.");

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu())
    {
        const int total_size = to_int(input_1.size());

        CHECK_CUDA(cudaMemcpy(output.data, input_1.data, total_size * sizeof(float), cudaMemcpyDeviceToDevice));

        CHECK_CUBLAS(cublasSaxpy(Device::get_cublas_handle(), total_size, &one, input_2.data, 1, output.data, 1));
        
        return;
    }
#endif

    output.as_vector().noalias() = input_1.as_vector() + input_2.as_vector();
}

void multiply(const TensorView& input_a, bool transpose_a,
              const TensorView& input_b, bool transpose_b,
              TensorView& output,
              type alpha, type beta)
{
    const size_t rank = input_a.get_rank();

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) 
    {
        const int rows_a = to_int(input_a.shape[rank - 2]);
        const int cols_a = to_int(input_a.shape[rank - 1]);
        const int rows_b = to_int(input_b.shape[rank - 2]);
        const int cols_b = to_int(input_b.shape[rank - 1]);

        const int output_columns = transpose_b ? rows_b : cols_b;
        const int output_rows = transpose_a ? cols_a : rows_a;
        const int inner_dimension = transpose_a ? rows_a : cols_a;

        const cublasOperation_t operation_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasOperation_t operation_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
        const int leading_dimension_b = cols_b;
        const int leading_dimension_a = cols_a;
        const int leading_dimension_output = output_columns;

        const int batch_count = to_int(input_a.size() / (rows_a * cols_a));
        const long long stride_a = rows_a * cols_a;
        const long long stride_b = rows_b * cols_b;
        const long long stride_output = output.shape[rank - 2] * output.shape[rank - 1];

        if(batch_count == 1)
            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(),
                                     operation_b, operation_a,
                                     output_columns, output_rows, inner_dimension,
                                     &alpha,
                                     input_b.data, leading_dimension_b,
                                     input_a.data, leading_dimension_a,
                                     &beta,
                                     output.data, leading_dimension_output));
        else
            CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
                                                   operation_b, operation_a,
                                                   output_columns, output_rows, inner_dimension,
                                                   &alpha,
                                                   input_b.data, leading_dimension_b, stride_b,
                                                   input_a.data, leading_dimension_a, stride_a,
                                                   &beta,
                                                   output.data, leading_dimension_output, stride_output,
                                                   batch_count));
        return;
    }
#endif
    const bool simple = (alpha == 1.0f && beta == 0.0f);

    const Index batch_count = input_a.size() / (input_a.shape[rank - 2] * input_a.shape[rank - 1]);

    #pragma omp parallel for
    for(Index batch_index = 0; batch_index < batch_count; ++batch_index)
    {
        const MatrixMap matrix_a = input_a.as_matrix(batch_index);
        const MatrixMap matrix_b = input_b.as_matrix(batch_index);
        MatrixMap matrix_output = output.as_matrix(batch_index);

        if(simple)
        {
            if (!transpose_a && !transpose_b)
                matrix_output.noalias() = matrix_a * matrix_b;
            else if (transpose_a && !transpose_b)
                matrix_output.noalias() = matrix_a.transpose() * matrix_b;
            else if (!transpose_a && transpose_b)
                matrix_output.noalias() = matrix_a * matrix_b.transpose();
            else
                matrix_output.noalias() = matrix_a.transpose() * matrix_b.transpose();
        }
        else
        {
            if (!transpose_a && !transpose_b)
                matrix_output.noalias() = alpha * (matrix_a * matrix_b) + beta * matrix_output;
            else if (transpose_a && !transpose_b)
                matrix_output.noalias() = alpha * (matrix_a.transpose() * matrix_b) + beta * matrix_output;
            else if (!transpose_a && transpose_b)
                matrix_output.noalias() = alpha * (matrix_a * matrix_b.transpose()) + beta * matrix_output;
            else
                matrix_output.noalias() = alpha * (matrix_a.transpose() * matrix_b.transpose()) + beta * matrix_output;
        }
    }
}

void multiply_elementwise(const TensorView& input_a, const TensorView& input_b, TensorView& output)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(), Device::get_operator_multiplication_descriptor(),
                                  &one, input_a.get_descriptor(), input_a.data,
                                  &one, input_b.get_descriptor(), input_b.data,
                                  &zero, output.get_descriptor(), output.data));
        return;
    }
#endif
    output.as_vector().noalias() = input_a.as_vector().cwiseProduct(input_b.as_vector());
}

void sum(const TensorView& input, TensorView& output, type alpha, type beta)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const int total_rows = to_int(input.shape[0]);
        const int total_columns = to_int(input.shape.size() / input.shape[0]);

        CHECK_CUBLAS(cublasSgemv(Device::get_cublas_handle(),
                                 CUBLAS_OP_N,
                                 total_columns, total_rows,
                                 &alpha,
                                 input.data, total_columns,
                                 Device::get_ones(total_rows), 1,
                                 &beta,
                                 output.data, 1));
        return;
    }
#endif
    output.as_vector().noalias() = alpha * input.as_matrix().colwise().sum() + beta * output.as_vector();
}

void softmax(TensorView& output)
{
    if (output.empty()) return;

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDNN(cudnnSoftmaxForward(Device::get_cudnn_handle(),
                                        CUDNN_SOFTMAX_ACCURATE,
                                        CUDNN_SOFTMAX_MODE_CHANNEL,
                                        &one,
                                        output.get_descriptor(), output.data,
                                        &zero,
                                        output.get_descriptor(), output.data));
        return;
    }
#endif
    const Index output_columns = output.shape.back();
    const Index total_rows = output.size() / output_columns;

    MatrixMap output_matrix(output.data, total_rows, output_columns);
    output_matrix.colwise() -= output_matrix.rowwise().maxCoeff();
    output_matrix.array() = output_matrix.array().exp();
    output_matrix.array().colwise() /= output_matrix.rowwise().sum().array();
}

void combination(const TensorView& input,
                 const TensorView& weights,
                 const TensorView& biases,
                 TensorView& output)
{
    const Index input_columns = input.shape.back();
    const Index total_rows = input.size() / input_columns;
    const Index output_columns = weights.shape.back();

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu())
    {
        CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(),
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 to_int(output_columns),
                                 to_int(total_rows),
                                 to_int(input_columns),
                                 &one,
                                 weights.data, to_int(output_columns),
                                 input.data, to_int(input_columns),
                                 &zero,
                                 output.data, to_int(output_columns)));

        CHECK_CUDNN(cudnnAddTensor(Device::get_cudnn_handle(),
                                   &one,
                                   biases.get_descriptor(), biases.data,
                                   &one,
                                   output.get_descriptor(), output.data));
        return;
    }
#endif

    const MatrixMap input_matrix(input.data, total_rows, input_columns);
    MatrixMap output_matrix(output.data, total_rows, output_columns);
    output_matrix.noalias() = (input_matrix * weights.as_matrix()).rowwise() + biases.as_vector().transpose();
}

void activation(TensorView& output, ActivationArguments arguments)
{
    if (output.empty() || arguments.activation_function == ActivationFunction::Linear)
        return;

    const ActivationFunction activation_function = arguments.activation_function;

    if(activation_function == ActivationFunction::Softmax)
    {
        softmax(output);
        return;
    }

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDNN(cudnnActivationForward(Device::get_cudnn_handle(),
                                           arguments.activation_descriptor,
                                           &one,
                                           output.get_descriptor(), output.data,
                                           &zero,
                                           output.get_descriptor(), output.data));
        return;
    }
#endif
    auto arr = output.as_vector().array();

    switch (activation_function)
    {
    case ActivationFunction::Sigmoid:
    case ActivationFunction::Logistic:
        arr = (1.0f + (-arr).exp()).inverse();
        return;

    case ActivationFunction::HyperbolicTangent:
        arr = arr.tanh();
        return;

    case ActivationFunction::RectifiedLinear:
        arr = arr.cwiseMax(0.0f);
        return;

    case ActivationFunction::ScaledExponentialLinear:
    {
        const float alpha = 1.6732632423543772848170429916717f;
        const float lambda = 1.0507009873554804934193349852946f;
        arr = lambda * (arr > 0.0f).select(arr, alpha * (arr.exp() - 1.0f));
        return;
    }

    default:
        return;
    }
}

void activation_gradient(const TensorView& outputs,
                                const TensorView& output_gradient,
                                TensorView& activation_derivative,
                                const ActivationArguments& arguments)
{
    if (outputs.empty()) return;

    const ActivationFunction activation_function = arguments.activation_function;

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        if(activation_function == ActivationFunction::Linear || func == ActivationFunction::Softmax)
        {
            if(activation_derivative.data != output_gradient.data)
                CHECK_CUDA(cudaMemcpy(activation_derivative.data, output_gradient.data,
                                      output_gradient.size() * sizeof(float), cudaMemcpyDeviceToDevice));
            return;
        }

        CHECK_CUDNN(cudnnActivationBackward(Device::get_cudnn_handle(),
                                            arguments.activation_descriptor,
                                            &one,
                                            outputs.get_descriptor(), outputs.data,
                                            output_gradient.get_descriptor(), output_gradient.data,
                                            outputs.get_descriptor(), outputs.data,
                                            &zero,
                                            activation_derivative.get_descriptor(), activation_derivative.data));
        return;
    }
#endif

    const auto outputs_array = outputs.as_vector().array();
    const auto output_gradient_array = output_gradient.as_vector().array();
    auto derivative_array = activation_derivative.as_vector().array();

    switch (activation_function)
    {
    case ActivationFunction::Linear:
    {
        derivative_array = output_gradient_array;
        return;
    }

    case ActivationFunction::Sigmoid:
    case ActivationFunction::Logistic:
    {
        derivative_array = output_gradient_array * (outputs_array * (1.0f - outputs_array));
        return;
    }

    case ActivationFunction::HyperbolicTangent:
    {
        derivative_array = output_gradient_array * (1.0f - outputs_array.square());
        return;
    }

    case ActivationFunction::RectifiedLinear:
    {
        derivative_array = (outputs_array > 0.0f).select(output_gradient_array, 0.0f);
        return;
    }

    case ActivationFunction::ScaledExponentialLinear:
    {
        const float alpha = 1.6732632423543772848170429916717f;
        const float lambda = 1.0507009873554804934193349852946f;

        derivative_array = (outputs_array > 0.0f).select(lambda * output_gradient_array, (outputs_array + (alpha * lambda)) * output_gradient_array);
        return;
    }

    case ActivationFunction::Softmax:
    {
        derivative_array = output_gradient_array;
        return;
    }

    default:
        throw runtime_error("Math Error: Unknown activation function in activation_gradient.");
    }
}

void dropout(TensorView& output, DropoutArguments& args)
{
    if (args.rate <= type(0)) return;

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDNN(cudnnDropoutForward(Device::get_cudnn_handle(),
                                        args.descriptor,
                                        output.get_descriptor(), output.data,
                                        output.get_descriptor(), output.data,
                                        args.reserve_space, args.reserve_size));
        return;
    }
#endif
    const Index total_size = output.size();
    
    if (args.mask_cpu.size() != total_size) 
        args.mask_cpu.resize(total_size);

    const type scale = type(1) / (type(1) - args.rate);
    type* __restrict data = output.data;
    type* __restrict mask = args.mask_cpu.data();

    for (Index i = 0; i < total_size; ++i)
    {
        const bool dropped = random_uniform(type(0), type(1)) < args.rate;
        mask[i] = dropped ? type(0) : scale;
        data[i] *= mask[i];
    }
}

void dropout_gradient(const TensorView& output_gradient,
                      TensorView& input_gradient,
                      const DropoutArguments& args)
{
    if (args.rate <= type(0))
    {
        copy(output_gradient, input_gradient);
        return;
    }

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDNN(cudnnDropoutBackward(Device::get_cudnn_handle(),
                                         args.descriptor,
                                         output_gradient.get_descriptor(), output_gradient.data,
                                         input_gradient.get_descriptor(), input_gradient.data,
                                         const_cast<void*>(args.reserve_space), args.reserve_size));
        return;
    }
#endif
    const Index total_size = output_gradient.size();
    const VectorMap mask_vector(const_cast<type*>(args.mask_cpu.data()), total_size);
    input_gradient.as_vector().noalias() = output_gradient.as_vector().cwiseProduct(mask_vector);
}

void batch_normalization_inference(
    const TensorView& input,
    const TensorView& gamma,
    const TensorView& beta,
    const TensorView& running_mean,
    const TensorView& running_variance,
    TensorView& output)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;

        CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
            Device::get_cudnn_handle(),
            mode,
            &one, &zero,
            input.get_descriptor(), input.data,
            output.get_descriptor(), output.data,
            gamma.get_descriptor(),
            gamma.data, beta.data,
            running_mean.data, running_variance.data,
            EPSILON));
        return;
    }
#endif
    const Index neurons_number = gamma.size();
    const Index effective_batch_size = input.size() / neurons_number;

    const MatrixMap input_matrix(input.data, effective_batch_size, neurons_number);
    MatrixMap output_matrix(output.data, effective_batch_size, neurons_number);

    const VectorMap gammas = gamma.as_vector();
    const VectorMap betas = beta.as_vector();
    const VectorMap running_means = running_mean.as_vector();
    const VectorMap running_variances = running_variance.as_vector();

    output_matrix.array() = ((input_matrix.rowwise() - running_means.transpose()).array()
                              .rowwise() / (running_variances.array() + EPSILON).sqrt().transpose())
                             .rowwise() * gammas.transpose().array();

    output_matrix.rowwise() += betas.transpose();
}

void batch_normalization_training(
    const TensorView& input,
    const TensorView& gamma,
    const TensorView& beta,
    TensorView& running_mean,
    TensorView& running_variance,
    TensorView& mean,
    TensorView& inverse_variance,
    TensorView& output,
    type momentum)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;

        CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
            Device::get_cudnn_handle(),
            mode,
            &one, &zero,
            input.get_descriptor(), input.data,
            output.get_descriptor(), output.data,
            gamma.get_descriptor(), gamma.data,
            beta.data,
            static_cast<double>(type(1) - momentum),
            running_mean.data, running_variance.data,
            EPSILON,
            mean.data,
            inverse_variance.data));
        return;
    }
#endif
    const Index neurons_number = gamma.size();
    const Index effective_batch_size = input.size() / neurons_number;

    const MatrixMap input_matrix(input.data, effective_batch_size, neurons_number);
    MatrixMap output_matrix(output.data, effective_batch_size, neurons_number);

    const VectorMap gammas = gamma.as_vector();
    const VectorMap betas = beta.as_vector();

    VectorMap means = mean.as_vector();
    VectorMap inverse_variances = inverse_variance.as_vector();
    VectorMap running_means = running_mean.as_vector();
    VectorMap running_variances = running_variance.as_vector();

    means.noalias() = input_matrix.colwise().mean();

    output_matrix.noalias() = input_matrix.rowwise() - means.transpose();

    inverse_variances.noalias() = output_matrix.array().square().colwise().mean();

    running_means = running_means * momentum + means * (type(1) - momentum);
    running_variances = running_variances * momentum + inverse_variances * (type(1) - momentum);

    inverse_variances.array() = 1.0f / (inverse_variances.array() + EPSILON).sqrt();

    output_matrix.array().rowwise() *= (inverse_variances.array() * gammas.array()).transpose();

    output_matrix.rowwise() += betas.transpose();
}

void batch_normalization_backward(
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
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;

        CHECK_CUDNN(cudnnBatchNormalizationBackward(
            Device::get_cudnn_handle(),
            mode,
            &one, &zero,                        // alpha/beta for dx
            &one, &zero,                        // alpha/beta for dgamma/dbeta
            input.get_descriptor(), input.data,
            output_gradient.get_descriptor(), output_gradient.data,
            input_gradient.get_descriptor(), input_gradient.data,
            gamma.get_descriptor(), gamma.data,
            gamma_gradient.data, beta_gradient.data,
            EPSILON,
            mean.data, inverse_variance.data));
        return;
    }
#endif
    (void)output;// to avoid unused parameter warning
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

    beta_gradients.noalias() = output_gradients.colwise().sum();

    const MatrixR normalized = (input_matrix.rowwise() - means.transpose()).array().rowwise()
                               * inverse_variances.transpose().array();

    gamma_gradients.noalias() = (output_gradients.array() * normalized.array()).matrix().colwise().sum();

    const Eigen::Array<type, 1, Eigen::Dynamic> scale =
        (gammas.array() * inverse_variances.array() / to_type(effective_batch_size)).transpose();

    input_gradients.array() = ((to_type(effective_batch_size) * output_gradients.array()).rowwise() - beta_gradients.transpose().array()
                               - normalized.array().rowwise() * gamma_gradients.transpose().array())
                              .rowwise() * scale;
}

void layernorm_forward(const TensorView& input, const TensorView& gamma, const TensorView& beta,
                       TensorView& means, TensorView& standard_deviations, TensorView& normalized,
                       TensorView& output,
                       Index batch_size, Index sequence_length, Index embedding_dimension)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        layernorm_forward_cuda(to_int(batch_size * sequence_length), to_int(embedding_dimension),
            input.data, output.data,
            means.data, standard_deviations.data,
            gamma.data, beta.data, EPSILON);
        return;
    }
#endif
    const TensorMap3 input_map = input.as_tensor<3>();
    TensorMap2 means_map = means.as_tensor<2>();
    TensorMap2 standard_deviations_map = standard_deviations.as_tensor<2>();
    TensorMap3 normalized_map = normalized.as_tensor<3>();
    TensorMap3 output_map = output.as_tensor<3>();

    const array<Index, 3> reshape_dims({batch_size, sequence_length, 1});
    const array<Index, 3> broadcast_dims({1, 1, embedding_dimension});

    means_map = input_map.mean(array<Index, 1>({2}));

    const auto centered = input_map - means_map.reshape(reshape_dims).broadcast(broadcast_dims);
    const auto variance = centered.square().mean(array<Index, 1>({2}));
    standard_deviations_map = (variance + EPSILON).sqrt();

    normalized_map = centered / standard_deviations_map.reshape(reshape_dims).broadcast(broadcast_dims);

    const TensorMap1 gamma_map = gamma.as_tensor<1>();
    const TensorMap1 beta_map = beta.as_tensor<1>();

    output_map = normalized_map * gamma_map.reshape(array<Index, 3>({1, 1, embedding_dimension})).broadcast(array<Index, 3>({batch_size, sequence_length, 1}))
               + beta_map.reshape(array<Index, 3>({1, 1, embedding_dimension})).broadcast(array<Index, 3>({batch_size, sequence_length, 1}));
}


void layernorm_backward(const TensorView& input, const TensorView& output_gradient,
                        const TensorView& means, const TensorView& standard_deviations,
                        const TensorView& normalized, const TensorView& gamma,
                        TensorView& gamma_gradient, TensorView& beta_gradient, TensorView& input_gradient,
                        Index batch_size, Index sequence_length, Index embedding_dimension)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        layernorm_backward_cuda(to_int(batch_size * sequence_length), to_int(embedding_dimension),
            output_gradient.data, input.data,
            means.data, standard_deviations.data,
            gamma.data,
            input_gradient.data,
            gamma_gradient.data, beta_gradient.data);
        return;
    }
#endif
    const TensorMap2 standard_deviations_map = standard_deviations.as_tensor<2>();
    const TensorMap3 normalized_map = normalized.as_tensor<3>();
    const TensorMap3 output_gradient_map = output_gradient.as_tensor<3>();

    gamma_gradient.as_tensor<1>() = (output_gradient_map * normalized_map).sum(array<Index, 2>({0, 1}));
    beta_gradient.as_tensor<1>() = output_gradient_map.sum(array<Index, 2>({0, 1}));

    const array<Index, 3> reshape_3d({batch_size, sequence_length, 1});
    const array<Index, 3> broadcast_3d({1, 1, embedding_dimension});

    const Tensor3 scaled_gradient = output_gradient_map * gamma.as_tensor<1>().reshape(array<Index, 3>({1, 1, embedding_dimension})).broadcast(array<Index, 3>({batch_size, sequence_length, 1}));
    const Tensor2 sum_scaled_gradient = scaled_gradient.sum(array<Index, 1>({2}));
    const Tensor2 sum_scaled_gradient_normalized = (scaled_gradient * normalized_map).sum(array<Index, 1>({2}));

    const type inverse_embedding_dimension = type(1.0) / to_type(embedding_dimension);

    input_gradient.as_tensor<3>() = (scaled_gradient
                                     - sum_scaled_gradient.reshape(reshape_3d).broadcast(broadcast_3d) * inverse_embedding_dimension
                                     - normalized_map * sum_scaled_gradient_normalized.reshape(reshape_3d).broadcast(broadcast_3d) * inverse_embedding_dimension)
                                    / standard_deviations_map.reshape(reshape_3d).broadcast(broadcast_3d);
}


void convolution(const TensorView& input,
                        const TensorView& kernel,
                        const TensorView& bias,
                        TensorView& output,
                        const ConvolutionArguments& args)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
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
                                   &one,
                                   bias.get_descriptor(), bias.data,
                                   &one,
                                   output.get_descriptor(), output.data));
        return;
    }
#endif
    (void)args;

    const TensorMap4 inputs = input.as_tensor<4>();
    const VectorMap biases = bias.as_vector();

    const Index batch_size = inputs.dimension(0);
    const Index kernels_number = kernel.shape[0];

    const Eigen::array<Index, 3> conv_dims({1, 2, 3});
    const Eigen::array<Index, 3> out_slice_shape({batch_size, output.shape[1], output.shape[2]});

    TensorMap4 outputs = output.as_tensor<4>();

    for(Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        const TensorMap3 kernel_map = kernel.as_tensor<3>(kernel_index);
        
        outputs.chip(kernel_index, 3).device(get_device()) =
            inputs.convolve(kernel_map, conv_dims).reshape(out_slice_shape) + biases(kernel_index);
    }
}

void convolution_activation(const TensorView& input,
                            const TensorView& weight,
                            const TensorView& bias,
                            TensorView& output,
                            const ConvolutionArguments& conv_args,
                            const ActivationArguments& activation_arguments)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
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
            activation_arguments.activation_descriptor,
            output.get_descriptor(), output.data));
        return;
    }
#endif
    convolution(input, weight, bias, output, conv_args);
    activation(output, activation_arguments);
}

void convolution_backward_weights(const TensorView& input,
                                  const TensorView& delta,
                                  TensorView& weight_grad,
                                  TensorView& bias_grad,
                                  const ConvolutionArguments& args)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
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
        return;
    }
#endif
    (void)args;

    const TensorMap4 inputs = input.as_tensor<4>();
    const TensorMap4 deltas = delta.as_tensor<4>();

    const Index batch_size = inputs.dimension(0);
    const Index kernels_number = weight_grad.shape[0];
    const Index kernel_height = weight_grad.shape[1];
    const Index kernel_width = weight_grad.shape[2];
    const Index kernel_channels = weight_grad.shape[3];

    const Index output_height = deltas.dimension(1);
    const Index output_width = deltas.dimension(2);
    const Index input_height = inputs.dimension(1);
    const Index input_width = inputs.dimension(2);

    VectorMap bias_gradients = bias_grad.as_vector().setZero();

    weight_grad.as_vector().setZero();

    #pragma omp parallel for
    for(Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        TensorMap3 weight_gradient_map = weight_grad.as_tensor<3>(kernel_index);

        for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
        {
            for(Index output_row = 0; output_row < output_height; ++output_row)
            {
                const Index row_limit = min(kernel_height, input_height - output_row);

                for(Index output_column = 0; output_column < output_width; ++output_column)
                {
                    const type delta = deltas(batch_index, output_row, output_column, kernel_index);
                    bias_gradients(kernel_index) += delta;

                    const Index col_limit = min(kernel_width, input_width - output_column);

                    for(Index kernel_row = 0; kernel_row < row_limit; ++kernel_row)
                    {
                        const Index input_row = output_row + kernel_row;

                        for(Index kernel_column = 0; kernel_column < col_limit; ++kernel_column)
                        {
                            const Index input_column = output_column + kernel_column;

                            for(Index channel_index = 0; channel_index < kernel_channels; ++channel_index)
                                weight_gradient_map(kernel_row, kernel_column, channel_index) += delta * inputs(batch_index, input_row, input_column, channel_index);
                        }
                    }
                }
            }
        }
    }
}

void convolution_backward_data(const TensorView& delta,
                               const TensorView& kernel,
                               TensorView& input_grad,
                               TensorView& /*padded_input_grad*/,
                               const ConvolutionArguments& args)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDNN(cudnnConvolutionBackwardData(Device::get_cudnn_handle(),
            &one,
            args.kernel_descriptor, kernel.data,
            delta.get_descriptor(), delta.data,
            args.convolution_descriptor,
            args.algorithm_data,
            args.workspace, args.workspace_size,
            &zero,
            input_grad.get_descriptor(), input_grad.data));
        return;
    }
#endif
    (void)args;

    const TensorMap4 deltas = delta.as_tensor<4>();
    TensorMap4 in_grad = input_grad.as_tensor<4>().setZero();

    const Index batch_size = deltas.dimension(0);
    const Index output_height = deltas.dimension(1);
    const Index output_width = deltas.dimension(2);
    const Index kernels_number = kernel.shape[0];
    const Index input_height = in_grad.dimension(1);
    const Index input_width = in_grad.dimension(2);

    #pragma omp parallel for
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for(Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
        {
            const TensorMap3 kernel_map = kernel.as_tensor<3>(kernel_index);

            for(Index output_row = 0; output_row < output_height; ++output_row)
            {
                const Index row_limit = min(kernel_height, input_height - output_row);

                for(Index output_column = 0; output_column < output_width; ++output_column)
                {
                    const type delta_value = deltas(batch_index, output_row, output_column, kernel_index);

                    const Index col_limit = min(kernel_width, input_width - output_column);

                    for(Index kernel_row = 0; kernel_row < row_limit; ++kernel_row)
                    {
                        const Index input_row = output_row + kernel_row;

                        for(Index kernel_column = 0; kernel_column < col_limit; ++kernel_column)
                        {
                            const Index input_column = output_column + kernel_column;

                            for(Index channel_index = 0; channel_index < kernel_channels; ++channel_index)
                                in_grad(batch_index, input_row, input_column, channel_index) += delta_value * kernel_map(kernel_row, kernel_column, channel_index);
                        }
                    }
                }
            }
        }
}

template <bool IsTraining>
static void max_pooling_cpu(const TensorView& input,
                            TensorView& output,
                            TensorView& maximal_indices,
                            const PoolingArguments& arguments)
{
    const TensorMap4 inputs = input.as_tensor<4>();
    TensorMap4 outputs = output.as_tensor<4>();

    TensorMap4 maximal_indices_map = [&]() -> TensorMap4 {
        if constexpr (IsTraining)
            return maximal_indices.as_tensor<4>();
        else
            return TensorMap4(nullptr, 0, 0, 0, 0);
    }();

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
            {
                const Index input_row_start = output_row * row_stride - padding_height;
                const Index pool_row_start = max(Index(0), -input_row_start);
                const Index pool_row_end   = min(pool_height, input_height - input_row_start);

                for(Index output_column = 0; output_column < output_width; ++output_column)
                {
                    const Index input_column_start = output_column * column_stride - padding_width;

                    const Index pool_col_start = max(Index(0), -input_column_start);
                    const Index pool_col_end   = min(pool_width, input_width - input_column_start);

                    type maximum_value = NEG_INFINITY;
                    [[maybe_unused]] Index maximal_index = 0;

                    for(Index pool_row = pool_row_start; pool_row < pool_row_end; ++pool_row)
                    {
                        const Index input_row = input_row_start + pool_row;

                        for(Index pool_column = pool_col_start; pool_column < pool_col_end; ++pool_column)
                        {
                            const Index input_column = input_column_start + pool_column;

                            const type current_value = inputs(batch_index, input_row, input_column, channel_index);

                            if(current_value > maximum_value)
                            {
                                maximum_value = current_value;
                                if constexpr (IsTraining)
                                    maximal_index = pool_row * pool_width + pool_column;
                            }
                        }
                    }

                    outputs(batch_index, output_row, output_column, channel_index) = maximum_value;

                    if constexpr (IsTraining)
                        maximal_indices_map(batch_index, output_row, output_column, channel_index) = maximal_index;
                }
            }
}

void max_pooling(const TensorView& input,
                 TensorView& output,
                 TensorView& maximal_indices,
                 const PoolingArguments& arguments,
                 bool is_training)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        (void)maximal_indices; (void)is_training;

        CHECK_CUDNN(cudnnPoolingForward(Device::get_cudnn_handle(),
            arguments.pooling_descriptor,
            &one,
            input.get_descriptor(), input.data,
            &zero,
            output.get_descriptor(), output.data));
        return;
    }
#endif
    if(is_training)
        max_pooling_cpu<true>(input, output, maximal_indices, arguments);
    else
        max_pooling_cpu<false>(input, output, maximal_indices, arguments);
}

void average_pooling(const TensorView& input,
                     TensorView& output,
                     const PoolingArguments& arguments)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDNN(cudnnPoolingForward(Device::get_cudnn_handle(),
            arguments.pooling_descriptor,
            &one,
            input.get_descriptor(), input.data,
            &zero,
            output.get_descriptor(), output.data));
        return;
    }
#endif
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
            {
                const Index input_row_start = output_row * row_stride - padding_height;

                const Index pool_row_start = max(Index(0), -input_row_start);
                const Index pool_row_end   = min(pool_height, input_height - input_row_start);

                for(Index output_column = 0; output_column < output_width; ++output_column)
                {
                    const Index input_column_start = output_column * column_stride - padding_width;

                    const Index pool_col_start = max(Index(0), -input_column_start);
                    const Index pool_col_end   = min(pool_width, input_width - input_column_start);

                    type sum = 0;

                    for(Index pool_row = pool_row_start; pool_row < pool_row_end; ++pool_row)
                    {
                        const Index input_row = input_row_start + pool_row;

                        for(Index pool_column = pool_col_start; pool_column < pool_col_end; ++pool_column)
                        {
                            const Index input_column = input_column_start + pool_column;

                            sum += inputs(batch_index, input_row, input_column, channel_index);
                        }
                    }
                    outputs(batch_index, output_row, output_column, channel_index) = sum * inv_pool_size;
                }
            }    
}

void max_pooling_backward(const TensorView& input,
                          const TensorView& output,
                          const TensorView& output_gradient,
                          const TensorView& maximal_indices,
                          TensorView& input_gradient,
                          const PoolingArguments& args)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDNN(cudnnPoolingBackward(Device::get_cudnn_handle(),
            args.pooling_descriptor,
            &one,
            output.get_descriptor(), output.data,
            output_gradient.get_descriptor(), output_gradient.data,
            input.get_descriptor(), input.data,
            &zero,
            input_gradient.get_descriptor(), input_gradient.data));
        return;
    }
#endif
    (void)output; (void)args;

    const TensorMap4 out_grads = output_gradient.as_tensor<4>();
    const TensorMap4 max_indices = maximal_indices.as_tensor<4>();
    TensorMap4 in_grads = input_gradient.as_tensor<4>().setZero();

    const Index batch_size = out_grads.dimension(0);
    const Index output_height = out_grads.dimension(1);
    const Index output_width = out_grads.dimension(2);
    const Index channels = out_grads.dimension(3);

    const Index pool_width = args.pool_dimensions[1];
    const Index row_stride = args.stride_shape[0];
    const Index column_stride = args.stride_shape[1];
    const Index padding_height = args.padding_shape[0];
    const Index padding_width = args.padding_shape[1];

    #pragma omp parallel for collapse(2)
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for(Index channel_index = 0; channel_index < channels; ++channel_index)
            for(Index output_row = 0; output_row < output_height; ++output_row)
            {
                const Index input_row_start = output_row * row_stride - padding_height;

                for(Index output_column = 0; output_column < output_width; ++output_column)
                {
                    const Index input_column_start = output_column * column_stride - padding_width;

                    const Index maximal_index = static_cast<Index>(max_indices(batch_index, output_row, output_column, channel_index));
                    const Index pool_row = maximal_index / pool_width;
                    const Index pool_column = maximal_index % pool_width;

                    const Index input_row    = input_row_start    + pool_row;
                    const Index input_column = input_column_start + pool_column;

                    in_grads(batch_index, input_row, input_column, channel_index)
                        += out_grads(batch_index, output_row, output_column, channel_index);
                }
            }
}

void average_pooling_backward(const TensorView& input,
                                     const TensorView& output,
                                     const TensorView& output_gradient,
                                     TensorView& input_gradient,
                                     const PoolingArguments& args)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDNN(cudnnPoolingBackward(Device::get_cudnn_handle(),
            args.pooling_descriptor,
            &one,
            output.get_descriptor(), output.data,
            output_gradient.get_descriptor(), output_gradient.data,
            input.get_descriptor(), input.data,
            &zero,
            input_gradient.get_descriptor(), input_gradient.data));
        return;
    }
#endif
    (void)input; (void)output;

    const TensorMap4 out_grads = output_gradient.as_tensor<4>();
    TensorMap4 in_grads = input_gradient.as_tensor<4>().setZero();

    const Index batch_size = in_grads.dimension(0);
    const Index input_height = in_grads.dimension(1);
    const Index input_width = in_grads.dimension(2);
    const Index channels = in_grads.dimension(3);
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
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for(Index channel_index = 0; channel_index < channels; ++channel_index)
            for(Index output_row = 0; output_row < output_height; ++output_row)
            {
                const Index input_row_start = output_row * row_stride - padding_height;
                const Index pool_row_start = max(Index(0), -input_row_start);
                const Index pool_row_end   = min(pool_height, input_height - input_row_start);

                for(Index output_column = 0; output_column < output_width; ++output_column)
                {
                    const type average_gradient = out_grads(batch_index, output_row, output_column, channel_index) * inv_pool_size;

                    const Index input_column_start = output_column * column_stride - padding_width;
                    const Index pool_col_start = max(Index(0), -input_column_start);
                    const Index pool_col_end   = min(pool_width, input_width - input_column_start);

                    for(Index pool_row = pool_row_start; pool_row < pool_row_end; ++pool_row)
                    {
                        const Index input_row = input_row_start + pool_row;

                        for(Index pool_column = pool_col_start; pool_column < pool_col_end; ++pool_column)
                        {
                            const Index input_column = input_column_start + pool_column;

                            in_grads(batch_index, input_row, input_column, channel_index) += average_gradient;
                        }
                    }
                }
            }
}

void max_pooling_3d_forward(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        (void)is_training;
        pooling3d_max_forward_cuda(to_int(input.shape[0]) * to_int(input.shape[2]),
            input.data, output.data, maximal_indices.data,
            to_int(input.shape[0]), to_int(input.shape[1]), to_int(input.shape[2]));
        return;
    }
#endif
    const TensorMap3 inputs = input.as_tensor<3>();
    MatrixMap outputs = output.as_matrix();

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index features = inputs.dimension(2);

    MatrixMap max_indices = maximal_indices.as_matrix();

    #pragma omp parallel for
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        outputs.row(batch_index).setConstant(NEG_INFINITY);

        for(Index step = 0; step < sequence_length; ++step)
            for(Index feature_index = 0; feature_index < features; ++feature_index)
            {
                const type value = inputs(batch_index, step, feature_index);
                if(value > outputs(batch_index, feature_index))
                {
                    outputs(batch_index, feature_index) = value;
                    if(is_training) max_indices(batch_index, feature_index) = to_type(step);
                }
            }
    }
}

void average_pooling_3d_forward(const TensorView& input, TensorView& output)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        pooling3d_avg_forward_cuda(to_int(input.shape[0]) * to_int(input.shape[2]),
                                          input.data, 
                                          output.data,
                                          to_int(input.shape[0]), 
                                          to_int(input.shape[1]), 
                                          to_int(input.shape[2]));
        return;
    }
#endif
    const TensorMap3 inputs = input.as_tensor<3>();
    MatrixMap outputs = output.as_matrix();

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index features = inputs.dimension(2);

    #pragma omp parallel for
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        const Map<const MatrixR> seq_matrix(&inputs(batch_index, 0, 0), sequence_length, features);

        const Index valid_count = ((seq_matrix.array() != type(0)).rowwise().any()).count();

        if(valid_count > 0)
            outputs.row(batch_index) = seq_matrix.colwise().sum() / to_type(valid_count);
        else
            outputs.row(batch_index).setZero();
    }
}

void max_pooling_3d_backward(const TensorView& maximal_indices, const TensorView& output_gradient, TensorView& input_gradient)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDA(cudaMemset(input_gradient.data, 0, input_gradient.size() * sizeof(float)));

        pooling3d_max_backward_cuda(to_int(output_gradient.shape[0]) * to_int(output_gradient.shape[1]),
                                    output_gradient.data, 
                                    input_gradient.data, 
                                    maximal_indices.data,
                                    to_int(output_gradient.shape[0]), 
                                    to_int(input_gradient.shape[1]), 
                                    to_int(output_gradient.shape[1]));
        
        return;
    }
#endif
    const MatrixMap max_indices = maximal_indices.as_matrix();
    const MatrixMap output_gradient_matrix = output_gradient.as_matrix();
    TensorMap3 input_gradient_map = input_gradient.as_tensor<3>().setZero();

    const Index batch_size = output_gradient_matrix.rows();
    const Index features = output_gradient_matrix.cols();

    #pragma omp parallel for
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for(Index feature_index = 0; feature_index < features; ++feature_index)
        {
            const Index step = static_cast<Index>(max_indices(batch_index, feature_index));
            input_gradient_map(batch_index, step, feature_index) = output_gradient_matrix(batch_index, feature_index);
        }
}

void average_pooling_3d_backward(const TensorView& input, const TensorView& output_gradient, TensorView& input_gradient)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDA(cudaMemset(input_gradient.data, 0, input_gradient.size() * sizeof(float)));

        pooling3d_avg_backward_cuda(to_int(input.shape[0]) * to_int(input.shape[2]),
            input.data, output_gradient.data, input_gradient.data,
            to_int(input.shape[0]), to_int(input.shape[1]), to_int(input.shape[2]));
        return;
    }
#endif
    const TensorMap3 inputs = input.as_tensor<3>();
    const MatrixMap output_gradient_matrix = output_gradient.as_matrix();
    TensorMap3 input_gradient_map = input_gradient.as_tensor<3>().setZero();

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index features = inputs.dimension(2);

    #pragma omp parallel for
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        const Map<const MatrixR> seq_matrix(&inputs(batch_index, 0, 0), sequence_length, features);
        const auto non_padding = (seq_matrix.array() != type(0)).rowwise().any().eval();
        const Index valid_count = non_padding.count();

        if(valid_count == 0) continue;

        const type inverse_valid_count = type(1) / to_type(valid_count);
        Map<MatrixR> grad_matrix(&input_gradient_map(batch_index, 0, 0), sequence_length, features);
        const auto output_row = output_gradient_matrix.row(batch_index);

        for(Index step = 0; step < sequence_length; ++step)
            if(non_padding(step))
                grad_matrix.row(step) = output_row * inverse_valid_count;
    }
}

void embedding_backward(const TensorView& input_indices,
                        const TensorView& output_gradient,
                        TensorView& weight_gradient,
                        Index embedding_dimension,
                        bool scale_embedding)
{
    const Index total_elements = input_indices.size();

    MatrixMap gradients_map(output_gradient.data, total_elements, embedding_dimension);

    if(scale_embedding)
        gradients_map *= sqrt(to_type(embedding_dimension));

    MatrixMap weight_gradients = weight_gradient.as_matrix().setZero();

    for(Index token_index = 0; token_index < total_elements; ++token_index)
    {
        const Index vocabulary_index = static_cast<Index>(input_indices.data[token_index]);

        if(vocabulary_index < 0 || vocabulary_index >= weight_gradients.rows())
            continue;

        weight_gradients.row(vocabulary_index).noalias() += gradients_map.row(token_index);
    }

    weight_gradients.row(0).setZero();
}

void projection(const TensorView& input,
                const TensorView& weights,
                const TensorView& biases,
                TensorView& output,
                const MultiheadAttentionArguments& args)
{
    const Index batch_size = args.batch_size;
    const Index heads_number = args.heads_number;
    const Index embedding_dimension = args.embedding_dimension;
    const Index head_dimension = args.head_dimension;
    const Index sequence_length = output.size() / (batch_size * heads_number * head_dimension);

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) 
    {
        const Index total_rows = input.size() / embedding_dimension;
        TensorView in_2d(input.data, {total_rows, embedding_dimension});
        TensorView out_2d(output.data, {total_rows, embedding_dimension});
        in_2d.set_descriptor(in_2d.shape);

        out_2d.set_descriptor(out_2d.shape);
        
        combination(in_2d, weights, biases, out_2d);
        return;
    }
#endif
    const MatrixMap weight_matrix = weights.as_matrix();
    const VectorMap bias_vector = biases.as_vector();

    #pragma omp parallel for collapse(2)
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        for(Index head_index = 0; head_index < heads_number; ++head_index)
        {
            const MatrixMap input_batch(input.data + batch_index * sequence_length * embedding_dimension, sequence_length, embedding_dimension);

            MatrixMap output_batch_head(output.data + batch_index * (heads_number * sequence_length * head_dimension)
                             + head_index * (sequence_length * head_dimension),
                             sequence_length, head_dimension);

            const auto weight_head = weight_matrix.block(0, head_index * head_dimension, embedding_dimension, head_dimension);
            const auto bias_head = bias_vector.segment(head_index * head_dimension, head_dimension);

            output_batch_head.noalias() = (input_batch * weight_head).rowwise() + bias_head.transpose();
        }
    }
}

void projection_gradient(const TensorView& d_head,
                         const TensorView& input,
                         const TensorView& weights,
                         TensorView& d_bias,
                         TensorView& d_weights,
                         TensorView& d_input,
                         const MultiheadAttentionArguments& args,
                         Index sequence_length,
                         bool accumulate)
{
    const Index batch_size = args.batch_size;
    const Index heads_number = args.heads_number;
    const Index embedding_dimension = args.embedding_dimension;
    const Index head_dimension = args.head_dimension;

    const MatrixMap weight_matrix = weights.as_matrix();

    #pragma omp parallel for
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        MatrixMap input_gradient_batch(d_input.data + batch_index * sequence_length * embedding_dimension, sequence_length, embedding_dimension);

        if(!accumulate) input_gradient_batch.setZero();

        for(Index head_index = 0; head_index < heads_number; ++head_index)
        {
            const MatrixMap delta_head(d_head.data + batch_index * (heads_number * sequence_length * head_dimension) + head_index * (sequence_length * head_dimension),
                                       sequence_length, head_dimension);
            const auto weight_head = weight_matrix.block(0, head_index * head_dimension, embedding_dimension, head_dimension);
            input_gradient_batch.noalias() += delta_head * weight_head.transpose();
        }
    }

    MatrixMap weight_gradient_matrix = d_weights.as_matrix();
    VectorMap bias_gradient_vector = d_bias.as_vector();

    #pragma omp parallel for
    for(Index head_index = 0; head_index < heads_number; ++head_index)
    {
        auto weight_gradient_head = weight_gradient_matrix.block(0, head_index * head_dimension, embedding_dimension, head_dimension).setZero();
        auto bias_gradient_head = bias_gradient_vector.segment(head_index * head_dimension, head_dimension).setZero();

        for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
        {
            const MatrixMap delta_head(d_head.data + batch_index * (heads_number * sequence_length * head_dimension) + head_index * (sequence_length * head_dimension),
                                       sequence_length, head_dimension);

            const MatrixMap input_batch(input.data + batch_index * sequence_length * embedding_dimension, sequence_length, embedding_dimension);

            weight_gradient_head.noalias() += input_batch.transpose() * delta_head;
            bias_gradient_head.noalias() += delta_head.colwise().sum().transpose();
        }
    }
}

void multihead_attention_forward(
    const TensorView& query, const TensorView& key, const TensorView& value,
    TensorView& attention_weights, TensorView& concatenated, TensorView& output,
    const TensorView& projection_weights, const TensorView& projection_biases,
    const TensorView& source_input,
    const MultiheadAttentionArguments& args)
{
    const Index batch_size = args.batch_size;
    const Index heads_number = args.heads_number;
    const Index query_sequence_length = args.query_sequence_length;
    const Index source_sequence_length = args.source_sequence_length;
    const Index embedding_dimension = args.embedding_dimension;
    const Index head_dimension = args.head_dimension;
    const type scaling_factor = args.scaling_factor;
    const bool use_causal_mask = args.use_causal_mask;

    const Index total_heads = batch_size * heads_number;
    const Index total_rows = batch_size * query_sequence_length;

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu())
    {
        const int total_heads_int = to_int(total_heads);
        const int query_length_int = to_int(query_sequence_length);
        const int source_length_int = to_int(source_sequence_length);
        const int embedding_int = to_int(embedding_dimension);
        const int head_dim_int = to_int(head_dimension);
        const int heads_int = to_int(heads_number);
        const int batch_int = to_int(batch_size);
        const float scaling_factor_float = static_cast<float>(scaling_factor);

        // Transpose Q, K, V from [B, S, H, D] to [B, H, S, D]

        float* scratch = args.transpose_scratch;

        mha_transpose_qkv_cuda(batch_int * query_length_int * embedding_int, query.data, scratch, query_length_int, heads_int, head_dim_int);
        cudaMemcpy(query.data, scratch, batch_int * query_length_int * embedding_int * sizeof(float), cudaMemcpyDeviceToDevice);

        mha_transpose_qkv_cuda(batch_int * source_length_int * embedding_int, key.data, scratch, source_length_int, heads_int, head_dim_int);
        cudaMemcpy(key.data, scratch, batch_int * source_length_int * embedding_int * sizeof(float), cudaMemcpyDeviceToDevice);

        mha_transpose_qkv_cuda(batch_int * source_length_int * embedding_int, value.data, scratch, source_length_int, heads_int, head_dim_int);
        cudaMemcpy(value.data, scratch, batch_int * source_length_int * embedding_int * sizeof(float), cudaMemcpyDeviceToDevice);

        // Q * K^T — attention scores

        CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            source_length_int, query_length_int, head_dim_int,
            &scaling_factor_float,
            key.data, head_dim_int, source_length_int * head_dim_int,
            query.data, head_dim_int, query_length_int * head_dim_int,
            &zero,
            attention_weights.data, source_length_int, query_length_int * source_length_int,
            total_heads_int));

        // Fused masks: padding + causal

        mha_fused_masks_cuda(batch_int, heads_int, query_length_int, source_length_int, embedding_int, source_input.data, attention_weights.data,
                             args.padding_mask, use_causal_mask);

        // Softmax

        TensorView att_view(attention_weights.data, {(Index)(total_heads_int * query_length_int), (Index)source_length_int});
        att_view.set_descriptor(att_view.shape);
        softmax(att_view);

        // Attention * V

        float* att_out = args.attention_output_transposed;

        CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            head_dim_int, query_length_int, source_length_int,
            &one,
            value.data, head_dim_int, source_length_int * head_dim_int,
            attention_weights.data, source_length_int, query_length_int * source_length_int,
            &zero,
            att_out, head_dim_int, query_length_int * head_dim_int,
            total_heads_int));

        // Transpose back to [B*Sq, E]

        mha_transpose_o_cuda(batch_int * query_length_int * embedding_int, att_out, concatenated.data, query_length_int, heads_int, head_dim_int);

        // Output projection

        TensorView concat_2d(concatenated.data, {(Index)(batch_int * query_length_int), (Index)embedding_int});
        TensorView output_2d(output.data, {(Index)(batch_int * query_length_int), (Index)embedding_int});
        concat_2d.set_descriptor(concat_2d.shape);
        output_2d.set_descriptor(output_2d.shape);
        combination(concat_2d, projection_weights, projection_biases, output_2d);
        return;
    }
#endif
    // Q*K^T — attention scores

    #pragma omp parallel for
    for(Index head_index = 0; head_index < total_heads; ++head_index)
    {
        const MatrixMap query_map = query.as_matrix(head_index);
        const MatrixMap key_map = key.as_matrix(head_index);
        MatrixMap weights_map = attention_weights.as_matrix(head_index);
        weights_map.noalias() = (query_map * key_map.transpose()) * scaling_factor;
    }

    // Key padding mask

    #pragma omp parallel for
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        for(Index source_index = 0; source_index < source_sequence_length; ++source_index)
        {
            const type* row_ptr = source_input.data + batch_index * source_sequence_length * embedding_dimension + source_index * embedding_dimension;
            const bool is_pad = Eigen::Map<const VectorR>(row_ptr, embedding_dimension)
                                    .cwiseAbs().maxCoeff() <= type(1e-7f);

            if(is_pad)
            {
                const Index slice_size = heads_number * query_sequence_length;
                MatrixMap att_map(attention_weights.data + batch_index * slice_size * source_sequence_length,
                                  slice_size, source_sequence_length);
                att_map.col(source_index).setConstant(type(-1e9f));
            }
        }
    }

    // Causal mask

    if(use_causal_mask)
    {
        const Index matrix_size = query_sequence_length * source_sequence_length;
        MatrixMap scores(attention_weights.data, total_heads, matrix_size);
        const VectorMap causal_mask_map(const_cast<type*>(args.causal_mask->data()), matrix_size);
        scores.rowwise() += causal_mask_map.transpose();
    }

    // Softmax

    softmax(TensorView(attention_weights.data, {total_heads * query_sequence_length, source_sequence_length}));

    // W*V — attention output, scattered into concatenated

    #pragma omp parallel for collapse(2)
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for(Index head_index = 0; head_index < heads_number; ++head_index)
        {
            const Index linear_head_index = batch_index * heads_number + head_index;
            const MatrixMap weights_map = attention_weights.as_matrix(linear_head_index);
            const MatrixMap value_map = value.as_matrix(linear_head_index);
            type* output_ptr = concatenated.data + batch_index * query_sequence_length * embedding_dimension + head_index * head_dimension;
            using StrideType = Eigen::OuterStride<Eigen::Dynamic>;
            Eigen::Map<MatrixR, 0, StrideType> output_map(output_ptr, query_sequence_length, head_dimension, StrideType(embedding_dimension));
            output_map.noalias() = weights_map * value_map;
        }

    // Output projection

    const MatrixMap concatenated_matrix(concatenated.data, total_rows, embedding_dimension);
    MatrixMap output_matrix(output.data, total_rows, embedding_dimension);
    const MatrixMap projection_weight_matrix = projection_weights.as_matrix();
    const VectorMap projection_bias_vector = projection_biases.as_vector();

    output_matrix.noalias() = (concatenated_matrix * projection_weight_matrix).rowwise() + projection_bias_vector.transpose();
}


void multihead_attention_backward(
    const TensorView& query_input, const TensorView& source_input,
    TensorView& output_gradient,
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
    TensorView& input_source_grad,
    const TensorView& query_weights, const TensorView& key_weights, const TensorView& value_weights,
    const MultiheadAttentionArguments& args,
    bool self_attention)
{
    const Index batch_size = args.batch_size;
    const Index heads_number = args.heads_number;
    const Index query_sequence_length = args.query_sequence_length;
    const Index source_sequence_length = args.source_sequence_length;
    const Index embedding_dimension = args.embedding_dimension;
    const Index head_dimension = args.head_dimension;
    const type scaling_factor = args.scaling_factor;

    const Index total_rows = batch_size * query_sequence_length;
    const Index total_heads = batch_size * heads_number;

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const int total_heads_int = to_int(total_heads);
        const int query_length_int = to_int(query_sequence_length);
        const int source_length_int = to_int(source_sequence_length);
        const int embedding_int = to_int(embedding_dimension);
        const int head_dim_int = to_int(head_dimension);
        const int heads_int = to_int(heads_number);
        const int batch_int = to_int(batch_size);
        const float scaling_factor_float = static_cast<float>(scaling_factor);

        float* scratch = args.transpose_scratch;

        // Projection weight gradients: dW_proj = concat^T * dY

        CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_T,
            embedding_int, embedding_int, batch_int * query_length_int,
            &one,
            output_gradient.data, embedding_int,
            concatenated.data, embedding_int,
            &zero,
            proj_weight_grad.data, embedding_int));

        // Projection bias gradients: db_proj = sum(dY)

        CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            embedding_int, 1, batch_int * query_length_int,
            &one,
            output_gradient.data, embedding_int,
            Device::get_ones(batch_int * query_length_int), batch_int * query_length_int,
            &zero,
            proj_bias_grad.data, embedding_int));

        // Concatenated output gradients: d_concat = dY * W_proj^T

        CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            embedding_int, batch_int * query_length_int, embedding_int,
            &one,
            projection_weights.data, embedding_int,
            output_gradient.data, embedding_int,
            &zero,
            concat_grad.data, embedding_int));

        // Transpose d_concat from [B, Sq, H, D] to [B, H, Sq, D]

        mha_transpose_qkv_cuda(batch_int * query_length_int * embedding_int, concat_grad.data, scratch, query_length_int, heads_int, head_dim_int);

        // dV = P^T * dO (transposed)

        CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_T,
            head_dim_int, source_length_int, query_length_int,
            &one,
            scratch, head_dim_int, query_length_int * head_dim_int,
            attention_weights.data, source_length_int, query_length_int * source_length_int,
            &zero,
            value_grad.data, head_dim_int, source_length_int * head_dim_int,
            total_heads_int));

        // dP = dO * V^T

        CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            source_length_int, query_length_int, head_dim_int,
            &one,
            value.data, head_dim_int, source_length_int * head_dim_int,
            scratch, head_dim_int, query_length_int * head_dim_int,
            &zero,
            att_weight_grad.data, source_length_int, query_length_int * source_length_int,
            total_heads_int));

        // Softmax backward

        TensorView att_view(attention_weights.data, {(Index)(total_heads_int * query_length_int), (Index)source_length_int});
        att_view.set_descriptor(att_view.shape);
        TensorView datt_view(att_weight_grad.data, {(Index)(total_heads_int * query_length_int), (Index)source_length_int});
        datt_view.set_descriptor(datt_view.shape);
        TensorView sgrad_view(args.softmax_gradient, {(Index)(total_heads_int * query_length_int), (Index)source_length_int});
        sgrad_view.set_descriptor(sgrad_view.shape);

        CHECK_CUDNN(cudnnSoftmaxBackward(Device::get_cudnn_handle(),
            CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
            &one, att_view.get_descriptor(), attention_weights.data,
            datt_view.get_descriptor(), att_weight_grad.data,
            &zero, sgrad_view.get_descriptor(), args.softmax_gradient));

        // dQ = softmax_grad * K^T * scaling_factor

        CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            head_dim_int, query_length_int, source_length_int,
            &scaling_factor_float,
            key.data, head_dim_int, source_length_int * head_dim_int,
            args.softmax_gradient, source_length_int, query_length_int * source_length_int,
            &zero,
            query_grad.data, head_dim_int, query_length_int * head_dim_int,
            total_heads_int));

        // dK = softmax_grad^T * Q * scaling_factor

        CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_T,
            head_dim_int, source_length_int, query_length_int,
            &scaling_factor_float,
            query.data, head_dim_int, query_length_int * head_dim_int,
            args.softmax_gradient, source_length_int, query_length_int * source_length_int,
            &zero,
            key_grad.data, head_dim_int, source_length_int * head_dim_int,
            total_heads_int));

        // Transpose dQ, dK, dV from [B, H, S, D] to [B, S, H, D]

        float* q_grad_flat = args.query_input_gradient_scratch;
        float* src_grad_flat = args.source_input_gradient_scratch;

        mha_transpose_o_cuda(batch_int * query_length_int * embedding_int, query_grad.data, q_grad_flat, query_length_int, heads_int, head_dim_int);
        mha_transpose_o_cuda(batch_int * source_length_int * embedding_int, key_grad.data, scratch, source_length_int, heads_int, head_dim_int);
        float* k_grad_flat = scratch; // reuse scratch for K grad flat

        // Query weight/bias/input gradients

        CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
            embedding_int, embedding_int, batch_int * query_length_int, &one, q_grad_flat, embedding_int, query_input.data, embedding_int, &zero, query_weight_grad.data, embedding_int));

        CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
            embedding_int, 1, batch_int * query_length_int, &one, q_grad_flat, embedding_int, Device::get_ones(batch_int * query_length_int), batch_int * query_length_int, &zero, query_bias_grad.data, embedding_int));

        CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
            embedding_int, batch_int * query_length_int, embedding_int, &one, query_weights.data, embedding_int, q_grad_flat, embedding_int, &zero, input_query_grad.data, embedding_int));

        if(self_attention)
        {
            // Key weight/bias/source gradients

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                embedding_int, embedding_int, batch_int * source_length_int, &one, k_grad_flat, embedding_int, source_input.data, embedding_int, &zero, key_weight_grad.data, embedding_int));

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                embedding_int, 1, batch_int * source_length_int, &one, k_grad_flat, embedding_int, Device::get_ones(batch_int * source_length_int), batch_int * source_length_int, &zero, key_bias_grad.data, embedding_int));

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                embedding_int, batch_int * source_length_int, embedding_int, &one, key_weights.data, embedding_int, k_grad_flat, embedding_int, &zero, src_grad_flat, embedding_int));

            // Value weight/bias/source gradients (accumulate on src_grad_flat)

            mha_transpose_o_cuda(batch_int * source_length_int * embedding_int, value_grad.data, k_grad_flat, source_length_int, heads_int, head_dim_int);

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                embedding_int, embedding_int, batch_int * source_length_int, &one, k_grad_flat, embedding_int, source_input.data, embedding_int, &zero, value_weight_grad.data, embedding_int));

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                embedding_int, 1, batch_int * source_length_int, &one, k_grad_flat, embedding_int, Device::get_ones(batch_int * source_length_int), batch_int * source_length_int, &zero, value_bias_grad.data, embedding_int));

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                embedding_int, batch_int * source_length_int, embedding_int, &one, value_weights.data, embedding_int, k_grad_flat, embedding_int, &one, src_grad_flat, embedding_int));

            // input_query_grad = q_input_grad + src_grad

            addition_cuda(batch_int * query_length_int * embedding_int, input_query_grad.data, src_grad_flat, input_query_grad.data);
        }
        else
        {
            // Cross-attention: K/V gradients go to input_source_grad

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                embedding_int, embedding_int, batch_int * source_length_int, &one, k_grad_flat, embedding_int, source_input.data, embedding_int, &zero, key_weight_grad.data, embedding_int));

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                embedding_int, 1, batch_int * source_length_int, &one, k_grad_flat, embedding_int, Device::get_ones(batch_int * source_length_int), batch_int * source_length_int, &zero, key_bias_grad.data, embedding_int));

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                embedding_int, batch_int * source_length_int, embedding_int, &one, key_weights.data, embedding_int, k_grad_flat, embedding_int, &zero, input_source_grad.data, embedding_int));

            mha_transpose_o_cuda(batch_int * source_length_int * embedding_int, value_grad.data, k_grad_flat, source_length_int, heads_int, head_dim_int);

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                embedding_int, embedding_int, batch_int * source_length_int, &one, k_grad_flat, embedding_int, source_input.data, embedding_int, &zero, value_weight_grad.data, embedding_int));

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                embedding_int, 1, batch_int * source_length_int, &one, k_grad_flat, embedding_int, Device::get_ones(batch_int * source_length_int), batch_int * source_length_int, &zero, value_bias_grad.data, embedding_int));

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                embedding_int, batch_int * source_length_int, embedding_int, &one, value_weights.data, embedding_int, k_grad_flat, embedding_int, &one, input_source_grad.data, embedding_int));
        }
        return;
    }
#endif
    // Projection gradients

    const MatrixMap concatenated_matrix(concatenated.data, total_rows, embedding_dimension);
    const MatrixMap output_gradient_matrix(output_gradient.data, total_rows, embedding_dimension);

    MatrixMap(proj_weight_grad.data, embedding_dimension, embedding_dimension).noalias() = concatenated_matrix.transpose() * output_gradient_matrix;
    VectorMap(proj_bias_grad.data, embedding_dimension).noalias() = output_gradient_matrix.colwise().sum();

    MatrixMap concat_gradient_matrix(concat_grad.data, total_rows, embedding_dimension);
    const MatrixMap projection_weight_matrix = projection_weights.as_matrix();
    concat_gradient_matrix.noalias() = output_gradient_matrix * projection_weight_matrix.transpose();

    // dV and dP from concat_grad

    #pragma omp parallel for collapse(2)
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for(Index head_index = 0; head_index < heads_number; ++head_index)
        {
            const Index linear_head_index = batch_index * heads_number + head_index;

            const MatrixMap attention_map = attention_weights.as_matrix(linear_head_index);
            const MatrixMap value_map = value.as_matrix(linear_head_index);
            MatrixMap value_gradient_map = value_grad.as_matrix(linear_head_index);
            MatrixMap attention_gradient_map = att_weight_grad.as_matrix(linear_head_index);

            type* output_gradient_ptr = concat_grad.data + batch_index * (query_sequence_length * embedding_dimension) + head_index * head_dimension;
            using StrideType = Eigen::OuterStride<Eigen::Dynamic>;
            Eigen::Map<const MatrixR, 0, StrideType> concat_output_gradient(output_gradient_ptr, query_sequence_length, head_dimension, StrideType(embedding_dimension));

            value_gradient_map.noalias() = attention_map.transpose() * concat_output_gradient;
            attention_gradient_map.noalias() = concat_output_gradient * value_map.transpose();
        }

    // Softmax gradient + dQ/dK

    #pragma omp parallel for
    for(Index head_index = 0; head_index < total_heads; ++head_index)
    {
        const MatrixMap attention_map = attention_weights.as_matrix(head_index);
        MatrixMap attention_gradient_map = att_weight_grad.as_matrix(head_index);

        const VectorR dot_product = (attention_map.array() * attention_gradient_map.array()).rowwise().sum();
        attention_gradient_map.array() = attention_map.array() * (attention_gradient_map.colwise() - dot_product).array();

        const MatrixMap query_map = query.as_matrix(head_index);
        const MatrixMap key_map = key.as_matrix(head_index);
        MatrixMap query_gradient_map = query_grad.as_matrix(head_index);
        MatrixMap key_gradient_map = key_grad.as_matrix(head_index);

        query_gradient_map.noalias() = (attention_gradient_map * key_map) * scaling_factor;
        key_gradient_map.noalias() = (attention_gradient_map.transpose() * query_map) * scaling_factor;
    }

    // Projection gradients for Q, K, V

    projection_gradient(query_grad, query_input, query_weights, query_bias_grad, query_weight_grad, input_query_grad,
                        args, query_sequence_length, false);

    if(self_attention)
    {
        projection_gradient(key_grad, source_input, key_weights, key_bias_grad, key_weight_grad, input_query_grad,
                            args, source_sequence_length, true);
        projection_gradient(value_grad, source_input, value_weights, value_bias_grad, value_weight_grad, input_query_grad,
                            args, source_sequence_length, true);
    }
    else
    {
        projection_gradient(key_grad, source_input, key_weights, key_bias_grad, key_weight_grad, input_source_grad,
                            args, source_sequence_length, false);
        projection_gradient(value_grad, source_input, value_weights, value_bias_grad, value_weight_grad, input_source_grad,
                            args, source_sequence_length, true);
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
