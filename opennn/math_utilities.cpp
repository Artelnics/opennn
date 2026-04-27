//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M A T H   U T I L I T I E S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "math_utilities.h"
#include "random_utilities.h"
#include "cuda_dispatch.h"

namespace opennn
{

static constexpr float SELU_ALPHA  = 1.6732632423543772848170429916717f;
static constexpr float SELU_LAMBDA = 1.0507009873554804934193349852946f;

void padding(const TensorView& input, TensorView& output)
{
    if (Device::instance().is_gpu())
        throw runtime_error("padding: GPU implementation not available.");

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

    if (TRY_GPU_DISPATCH(output, [&](auto tag) {
        using T = decltype(tag);
        bounding_cuda<T>(output.size(), to_int(features), input.as<T>(),
                         lower_bounds.as_float(),
                         upper_bounds.as_float(),
                         output.as<T>());
    })) return;

    const MatrixMap input_matrix = input.as_matrix();
    const VectorMap lower_bounds_vector = lower_bounds.as_vector();
    const VectorMap upper_bounds_vector = upper_bounds.as_vector();

    MatrixMap output_matrix = output.as_matrix();

    for(Index feature_index = 0; feature_index < features; ++feature_index)
        output_matrix.col(feature_index) = input_matrix.col(feature_index)
                                                        .cwiseMax(lower_bounds_vector(feature_index))
                                                        .cwiseMin(upper_bounds_vector(feature_index));
}

void scale(const TensorView& input,
           const TensorView& minimums, const TensorView& maximums,
           const TensorView& means, const TensorView& standard_deviations,
           const TensorView& scalers,
           type min_range, type max_range,
           TensorView& output)
{
    const Index features = scalers.size();

    if (TRY_GPU_DISPATCH(output, [&](auto tag) {
        using T = decltype(tag);
        scale_cuda<T>(output.size(), to_int(features),
                      input.as<T>(),
                      minimums.as_float(),
                      maximums.as_float(),
                      means.as_float(),
                      standard_deviations.as_float(),
                      scalers.as_float(),
                      min_range, max_range,
                      output.as<T>());
    })) return;

    const MatrixMap input_matrix = input.as_matrix();
    const VectorMap mins = minimums.as_vector();
    const VectorMap maxs = maximums.as_vector();
    const VectorMap mus  = means.as_vector();
    const VectorMap sds  = standard_deviations.as_vector();
    const VectorMap sc   = scalers.as_vector();

    MatrixMap output_matrix = output.as_matrix();

    output_matrix.noalias() = input_matrix;

    for(Index f = 0; f < features; ++f)
    {
        const int code = static_cast<int>(sc(f));
        auto col = output_matrix.col(f).array();

        switch(code)
        {
        case 1: // MinimumMaximum
            col = (col - mins(f)) / ((maxs(f) - mins(f)) + EPSILON)
                   * (max_range - min_range) + min_range;
            break;
        case 2: // MeanStandardDeviation
            col = (col - mus(f)) / (sds(f) + EPSILON);
            break;
        case 3: // StandardDeviation
            col /= (sds(f) + EPSILON);
            break;
        case 4: // Logarithm
            col = col.log();
            break;
        case 5: // ImageMinMax
            col /= type(255);
            break;
        default: // None
            break;
        }
    }
}

void unscale(const TensorView& input,
             const TensorView& minimums, const TensorView& maximums,
             const TensorView& means, const TensorView& standard_deviations,
             const TensorView& scalers,
             type min_range, type max_range,
             TensorView& output)
{
    const Index features = scalers.size();

    if (TRY_GPU_DISPATCH(output, [&](auto tag) {
        using T = decltype(tag);
        unscale_cuda<T>(output.size(), to_int(features),
                        input.as<T>(),
                        minimums.as_float(),
                        maximums.as_float(),
                        means.as_float(),
                        standard_deviations.as_float(),
                        scalers.as_float(),
                        min_range, max_range,
                        output.as<T>());
    })) return;

    const MatrixMap input_matrix = input.as_matrix();
    const VectorMap mins = minimums.as_vector();
    const VectorMap maxs = maximums.as_vector();
    const VectorMap mus  = means.as_vector();
    const VectorMap sds  = standard_deviations.as_vector();
    const VectorMap sc   = scalers.as_vector();

    MatrixMap output_matrix = output.as_matrix();

    output_matrix.noalias() = input_matrix;

    for(Index f = 0; f < features; ++f)
    {
        const int code = static_cast<int>(sc(f));
        auto col = output_matrix.col(f).array();

        switch(code)
        {
        case 1: // MinimumMaximum
            col = (col - min_range) / (max_range - min_range)
                   * (maxs(f) - mins(f)) + mins(f);
            break;
        case 2: // MeanStandardDeviation
            col = mus(f) + col * sds(f);
            break;
        case 3: // StandardDeviation
            col *= sds(f);
            break;
        case 4: // Logarithm
            col = col.exp();
            break;
        case 5: // ImageMinMax
            col *= type(255);
            break;
        default: // None
            break;
        }
    }
}

void copy(const TensorView& source, TensorView& destination)
{
    if(source.size() != destination.size())
        throw runtime_error("Math Error: Tensor sizes mismatch in copy operation.");

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDA(cudaMemcpy(destination.data, source.data, source.byte_size(), cudaMemcpyDeviceToDevice));
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
    if (Device::instance().is_gpu()) {
        // operator_sum_descriptor is configured with CUDNN_OP_TENSOR_ADD; reused here.
        CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(),
                                  Device::get_operator_sum_descriptor(),
                                  &one, input_1.get_descriptor(), input_1.data,
                                  &one, input_2.get_descriptor(), input_2.data,
                                  &zero, output.get_descriptor(), output.data));
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

        const int batch_count = to_int(input_a.size() / (rows_a * cols_a));
        const long long stride_a = rows_a * cols_a;
        const long long stride_b = rows_b * cols_b;
        const long long stride_output = output.shape[rank - 2] * output.shape[rank - 1];

        if(batch_count == 1)
            gemm_cuda(operation_b, operation_a,
                      output_columns, output_rows, inner_dimension,
                      input_b.data, input_b.cuda_dtype(), cols_b,
                      input_a.data, input_a.cuda_dtype(), cols_a,
                      output.data,  output.cuda_dtype(),  output_columns,
                      alpha, beta);
        else
            gemm_strided_batched_cuda(operation_b, operation_a,
                                      output_columns, output_rows, inner_dimension,
                                      input_b.data, cols_b, stride_b,
                                      input_a.data, cols_a, stride_a,
                                      output.data, output_columns, stride_output,
                                      batch_count,
                                      alpha, beta);
        return;
    }
#endif
    const Index batch_count = input_a.size() / (input_a.shape[rank - 2] * input_a.shape[rank - 1]);

    #pragma omp parallel for
    for(Index batch_index = 0; batch_index < batch_count; ++batch_index)
    {
        const MatrixMap matrix_a = input_a.as_matrix(batch_index);
        const MatrixMap matrix_b = input_b.as_matrix(batch_index);
        MatrixMap matrix_output = output.as_matrix(batch_index);

        auto gemm_like = [&](auto A, auto B)
        {
            if (beta == 0.0f)
                matrix_output.noalias() = alpha * (A * B);
            else
                matrix_output.noalias() = alpha * (A * B) + beta * matrix_output;
        };

        if (!transpose_a && !transpose_b)       gemm_like(matrix_a,             matrix_b);
        else if (transpose_a && !transpose_b)   gemm_like(matrix_a.transpose(), matrix_b);
        else if (!transpose_a && transpose_b)   gemm_like(matrix_a,             matrix_b.transpose());
        else                                    gemm_like(matrix_a.transpose(), matrix_b.transpose());
    }
}

void multiply_elementwise(const TensorView& input_a, const TensorView& input_b, TensorView& output)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(), 
                                  Device::get_operator_multiplication_descriptor(),
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

        gemv_cuda(CUBLAS_OP_N,
                  total_columns, total_rows,
                  input.data, input.cuda_dtype(), total_columns,
                  Device::get_ones(total_rows), CUDA_ACTIVATION_DTYPE,
                  output.data, output.cuda_dtype(),
                  alpha, beta);
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

    MatrixMap output_matrix = output.as_flat_matrix();
    const Index rows = output_matrix.rows();

    #pragma omp parallel for
    for (Index i = 0; i < rows; ++i)
    {
        const type max_val = output_matrix.row(i).maxCoeff();
        output_matrix.row(i).array() = (output_matrix.row(i).array() - max_val).exp();
        output_matrix.row(i) /= output_matrix.row(i).sum();
    }
}

void softmax_backward(const TensorView& softmax_out, TensorView& output_delta)
{
    if (output_delta.empty()) return;

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDNN(cudnnSoftmaxBackward(Device::get_cudnn_handle(),
                                         CUDNN_SOFTMAX_ACCURATE,
                                         CUDNN_SOFTMAX_MODE_CHANNEL,
                                         &one,
                                         softmax_out.get_descriptor(),     softmax_out.data,
                                         output_delta.get_descriptor(), output_delta.data,
                                         &zero,
                                         output_delta.get_descriptor(), output_delta.data));
        return;
    }
#endif
    const MatrixMap y = softmax_out.as_flat_matrix();
    MatrixMap dY = output_delta.as_flat_matrix();

    const VectorR dot = (y.array() * dY.array()).rowwise().sum();
    dY.array() = y.array() * (dY.colwise() - dot).array();
}

void combination(const TensorView& input,
                 const TensorView& weights,
                 const TensorView& biases,
                 TensorView& output)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu())
    {
        const int input_columns  = to_int(input.shape.back());
        const int output_columns = to_int(weights.shape.back());
        const int total_rows     = to_int(input.size() / input.shape.back());

        // Fused GEMM + bias-add via cuBLASLt epilogue (one launch instead of two).
        gemm_bias_cuda(CUBLAS_OP_N, CUBLAS_OP_N,
                       output_columns, total_rows, input_columns,
                       weights.data,
                       input.data,
                       output.data,
                       biases.as<float>());
        return;
    }
#endif

    output.as_flat_matrix().noalias() = (input.as_flat_matrix() * weights.as_matrix()).rowwise()
                                      + biases.as_vector().transpose();
}

void combination_activation(const TensorView& input,
                            const TensorView& weights,
                            const TensorView& biases,
                            const ActivationArguments& activation_arguments,
                            TensorView& output)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()
        && activation_arguments.activation_function == ActivationFunction::RectifiedLinear)
    {
        const int input_columns  = to_int(input.shape.back());
        const int output_columns = to_int(weights.shape.back());
        const int total_rows     = to_int(input.size() / input.shape.back());

        // Fused GEMM + bias + ReLU in one cuBLASLt launch.
        gemm_bias_cuda(CUBLAS_OP_N, CUBLAS_OP_N,
                       output_columns, total_rows, input_columns,
                       weights.data,
                       input.data,
                       output.data,
                       biases.as<float>(),
                       CUBLASLT_EPILOGUE_RELU_BIAS);
        return;
    }
#endif

    // Fallback: unfused combination + activation.
    combination(input, weights, biases, output);
    activation(output, activation_arguments);
}

void combination_gradient(const TensorView& output_delta,
                          const TensorView& input,
                          const TensorView& weights,
                          TensorView& input_delta,
                          TensorView& weight_gradient,
                          TensorView& bias_gradient,
                          bool accumulate_input_delta)
{
    multiply(input, true, output_delta, false, weight_gradient);
    sum(output_delta, bias_gradient);

    if(input_delta.data && input_delta.size() > 0)
    {
        const type beta = accumulate_input_delta ? type(1) : type(0);
        multiply(output_delta, false, weights, true, input_delta, type(1), beta);
    }
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
        arr = SELU_LAMBDA * (arr > 0.0f).select(arr, SELU_ALPHA * (arr.exp() - 1.0f));
        return;

    default:
        return;
    }
}

void activation_delta(const TensorView& outputs,
                      const TensorView& output_delta,
                      TensorView& input_delta,
                      const ActivationArguments& arguments)
{
    if (outputs.empty()) return;

    const ActivationFunction activation_function = arguments.activation_function;

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        if(activation_function == ActivationFunction::Linear
        || activation_function == ActivationFunction::Softmax)
        {
            if(input_delta.data != output_delta.data)
                CHECK_CUDA(cudaMemcpy(input_delta.data, output_delta.data,
                                      output_delta.byte_size(), cudaMemcpyDeviceToDevice));
            return;
        }

        CHECK_CUDNN(cudnnActivationBackward(Device::get_cudnn_handle(),
                                            arguments.activation_descriptor,
                                            &one,
                                            outputs.get_descriptor(), outputs.data,
                                            output_delta.get_descriptor(), output_delta.data,
                                            outputs.get_descriptor(), outputs.data,
                                            &zero,
                                            input_delta.get_descriptor(), input_delta.data));
        return;
    }
#endif

    const auto outputs_array = outputs.as_vector().array();
    const auto output_delta_array = output_delta.as_vector().array();
    auto input_delta_array = input_delta.as_vector().array();

    switch (activation_function)
    {
    case ActivationFunction::Linear:
    case ActivationFunction::Softmax:
        input_delta_array = output_delta_array;
        return;

    case ActivationFunction::Sigmoid:
    case ActivationFunction::Logistic:
    {
        input_delta_array = output_delta_array * (outputs_array * (1.0f - outputs_array));
        return;
    }

    case ActivationFunction::HyperbolicTangent:
    {
        input_delta_array = output_delta_array * (1.0f - outputs_array.square());
        return;
    }

    case ActivationFunction::RectifiedLinear:
    {
        input_delta_array = (outputs_array > 0.0f).select(output_delta_array, 0.0f);
        return;
    }

    case ActivationFunction::ScaledExponentialLinear:
        input_delta_array = (outputs_array > 0.0f).select(SELU_LAMBDA * output_delta_array,
                                                          (outputs_array + (SELU_ALPHA * SELU_LAMBDA)) * output_delta_array);
        return;

    default:
        throw runtime_error("Math Error: Unknown activation function in activation_delta.");
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

    for (Index i = 0; i < total_size; ++i)
    {
        const type mask_value = random_uniform(type(0), type(1)) < args.rate ? type(0) : scale;
        args.mask_cpu[i] = mask_value;
        output.data[i] *= mask_value;
    }
}

void dropout_delta(const TensorView& output_delta,
                   TensorView& input_delta,
                   const DropoutArguments& args)
{
    if (args.rate <= type(0))
    {
        copy(output_delta, input_delta);
        return;
    }

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDNN(cudnnDropoutBackward(Device::get_cudnn_handle(),
                                         args.descriptor,
                                         output_delta.get_descriptor(), output_delta.data,
                                         input_delta.get_descriptor(), input_delta.data,
                                         const_cast<void*>(args.reserve_space), args.reserve_size));
        return;
    }
#endif
    input_delta.as_vector().noalias() = output_delta.as_vector().cwiseProduct(args.mask_cpu);
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
    const VectorR scale = gamma.as_vector().array() / (running_variance.as_vector().array() + EPSILON).sqrt();
    const VectorR shift = beta.as_vector().array() - scale.array() * running_mean.as_vector().array();

    const MatrixMap input_matrix = input.as_flat_matrix();
    MatrixMap output_matrix = output.as_flat_matrix();

    #pragma omp parallel for
    for (Index i = 0; i < input_matrix.rows(); ++i)
        output_matrix.row(i).array() = input_matrix.row(i).array() * scale.transpose().array() + shift.transpose().array();
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
    const MatrixMap input_matrix = input.as_flat_matrix();
    MatrixMap output_matrix = output.as_flat_matrix();

    VectorMap means = mean.as_vector();
    VectorMap inverse_variances = inverse_variance.as_vector();
    VectorMap running_means = running_mean.as_vector();
    VectorMap running_variances = running_variance.as_vector();

    means.noalias() = input_matrix.colwise().mean();
    output_matrix.noalias() = input_matrix.rowwise() - means.transpose();

    inverse_variances.noalias() = output_matrix.array().square().colwise().mean().matrix();

    running_means = running_means * momentum + means * (type(1) - momentum);
    running_variances = running_variances * momentum + inverse_variances * (type(1) - momentum);

    inverse_variances.array() = type(1) / (inverse_variances.array() + EPSILON).sqrt();
    const VectorR scale = inverse_variances.array() * gamma.as_vector().array();
    const VectorMap betas = beta.as_vector();

    #pragma omp parallel for
    for (Index i = 0; i < output_matrix.rows(); ++i)
        output_matrix.row(i).array() = output_matrix.row(i).array() * scale.transpose().array() + betas.transpose().array();
}

void batch_normalization_backward(
    const TensorView& input,
    const TensorView& output,
    const TensorView& output_delta,
    const TensorView& mean,
    const TensorView& inverse_variance,
    const TensorView& gamma,
    TensorView& gamma_gradient,
    TensorView& beta_gradient,
    TensorView& input_delta)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;

        CHECK_CUDNN(cudnnBatchNormalizationBackward(
            Device::get_cudnn_handle(),
            mode,
            &one, &zero,
            &one, &zero,
            input.get_descriptor(), input.data,
            output_delta.get_descriptor(), output_delta.data,
            input_delta.get_descriptor(), input_delta.data,
            gamma.get_descriptor(), gamma.data,
            gamma_gradient.data, beta_gradient.data,
            EPSILON,
            mean.data, inverse_variance.data));
        return;
    }
#endif
    (void)output;
    const Index effective_batch_size = input.size() / gamma.size();
    const type inv_N = type(1) / to_type(effective_batch_size);
    const type N = to_type(effective_batch_size);

    const MatrixMap input_matrix = input.as_flat_matrix();
    const MatrixMap output_deltas = output_delta.as_flat_matrix();

    const VectorMap means = mean.as_vector();
    const VectorMap inverse_variances = inverse_variance.as_vector();
    const VectorMap gammas = gamma.as_vector();

    VectorMap gamma_gradients = gamma_gradient.as_vector();
    VectorMap beta_gradients = beta_gradient.as_vector();
    MatrixMap input_deltas = input_delta.as_flat_matrix();

    beta_gradients.noalias() = output_deltas.colwise().sum();

    gamma_gradients.noalias() = (output_deltas.array()
                                 * ((input_matrix.rowwise() - means.transpose()).array().rowwise()
                                    * inverse_variances.transpose().array())
                                ).matrix().colwise().sum();

    const VectorR scale = (gammas.array() * inverse_variances.array() * inv_N).matrix();
    const Index cols = gammas.size();

    #pragma omp parallel for
    for (Index i = 0; i < effective_batch_size; ++i)
        for (Index c = 0; c < cols; ++c)
        {
            const type x_hat = (input_matrix(i, c) - means(c)) * inverse_variances(c);
            input_deltas(i, c) =
                scale(c) * (N * output_deltas(i, c) - beta_gradients(c) - x_hat * gamma_gradients(c));
        }
}

void layernorm_forward(const TensorView& input, const TensorView& gamma, const TensorView& beta,
                       TensorView& means, TensorView& standard_deviations, TensorView& normalized,
                       TensorView& output,
                       Index batch_size, Index sequence_length, Index embedding_dimension)
{
    if (TRY_GPU_DISPATCH(output, [&](auto tag) {
        using T = decltype(tag);
        layernorm_forward_cuda<T>(to_int(batch_size * sequence_length),
                                  to_int(embedding_dimension),
                                  input.as<T>(), output.as<T>(),
                                  means.as<float>(), standard_deviations.as<float>(),
                                  gamma.as<float>(), beta.as<float>(), EPSILON);
    })) return;

    const type* input_data = input.data;
    type* means_data = means.data;
    type* stds_data = standard_deviations.data;
    type* normalized_data = normalized.data;
    type* output_data = output.data;
    const type* gamma_data = gamma.data;
    const type* beta_data = beta.data;

    const Index total_rows = batch_size * sequence_length;
    const type inv_D = type(1) / to_type(embedding_dimension);

    #pragma omp parallel for
    for (Index row = 0; row < total_rows; ++row)
    {
        const type* x = input_data + row * embedding_dimension;
        type* norm_row = normalized_data + row * embedding_dimension;
        type* out_row = output_data + row * embedding_dimension;

        type sum = 0;
        type sum_sq = 0;
        for (Index d = 0; d < embedding_dimension; ++d)
        {
            const type v = x[d];
            sum += v;
            sum_sq += v * v;
        }

        const type mean = sum * inv_D;
        const type variance = sum_sq * inv_D - mean * mean;
        const type std_val = std::sqrt(variance + EPSILON);
        const type inv_std = type(1) / std_val;

        means_data[row] = mean;
        stds_data[row] = std_val;

        for (Index d = 0; d < embedding_dimension; ++d)
        {
            const type x_hat = (x[d] - mean) * inv_std;
            norm_row[d] = x_hat;
            out_row[d] = gamma_data[d] * x_hat + beta_data[d];
        }
    }
}


void layernorm_backward(const TensorView& input, const TensorView& output_delta,
                        const TensorView& means, const TensorView& standard_deviations,
                        const TensorView& normalized, const TensorView& gamma,
                        TensorView& gamma_gradient, TensorView& beta_gradient, TensorView& input_delta,
                        Index batch_size, Index sequence_length, Index embedding_dimension)
{
    if (TRY_GPU_DISPATCH(input, [&](auto tag) {
        using T = decltype(tag);
        layernorm_backward_cuda<T>(to_int(batch_size * sequence_length),
                                   to_int(embedding_dimension),
                                   output_delta.as<T>(), input.as<T>(),
                                   means.as<float>(), standard_deviations.as<float>(),
                                   gamma.as<float>(),
                                   input_delta.as<T>(),
                                   gamma_gradient.as<float>(), beta_gradient.as<float>());
    })) return;
    const MatrixMap dy_flat = output_delta.as_flat_matrix();
    const MatrixMap norm_flat = normalized.as_flat_matrix();

    beta_gradient.as_vector().noalias() = dy_flat.colwise().sum();
    gamma_gradient.as_vector().noalias() = (dy_flat.array() * norm_flat.array()).matrix().colwise().sum();

    const type* dy_data = output_delta.data;
    const type* norm_data = normalized.data;
    const type* std_data = standard_deviations.data;
    const type* gamma_data = gamma.data;
    type* dx_data = input_delta.data;

    const Index total_rows = batch_size * sequence_length;
    const type inv_D = type(1) / to_type(embedding_dimension);

    #pragma omp parallel for
    for (Index row = 0; row < total_rows; ++row)
    {
        const type* dy = dy_data + row * embedding_dimension;
        const type* norm = norm_data + row * embedding_dimension;
        type* dx = dx_data + row * embedding_dimension;
        const type inv_std = type(1) / std_data[row];

        type sum_sg = 0;
        type sum_sg_norm = 0;
        for (Index d = 0; d < embedding_dimension; ++d)
        {
            const type sg = gamma_data[d] * dy[d];
            sum_sg += sg;
            sum_sg_norm += sg * norm[d];
        }
        sum_sg *= inv_D;
        sum_sg_norm *= inv_D;

        for (Index d = 0; d < embedding_dimension; ++d)
        {
            const type sg = gamma_data[d] * dy[d];
            dx[d] = (sg - sum_sg - norm[d] * sum_sg_norm) * inv_std;
        }
    }
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
                                  const TensorView& output_delta,
                                  TensorView& weight_grad,
                                  TensorView& bias_grad,
                                  const ConvolutionArguments& args)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDNN(cudnnConvolutionBackwardFilter(Device::get_cudnn_handle(),
            &one,
            input.get_descriptor(), input.data,
            output_delta.get_descriptor(), output_delta.data,
            args.convolution_descriptor,
            args.algorithm_filter,
            args.backward_filter_workspace, args.backward_filter_workspace_size,
            &zero,
            args.kernel_descriptor, weight_grad.data));

        CHECK_CUDNN(cudnnConvolutionBackwardBias(Device::get_cudnn_handle(),
            &one,
            output_delta.get_descriptor(), output_delta.data,
            &zero,
            bias_grad.get_descriptor(), bias_grad.data));
        return;
    }
#endif
    (void)args;

    const TensorMap4 inputs = input.as_tensor<4>();
    const TensorMap4 output_deltas = output_delta.as_tensor<4>();

    const Index kernels_number = weight_grad.shape[0];
    const Index kernel_height = weight_grad.shape[1];
    const Index kernel_width = weight_grad.shape[2];
    const Index kernel_channels = weight_grad.shape[3];
    const Index kernel_size = kernel_height * kernel_width * kernel_channels;

    // Bias gradients: sum output_deltas over (batch, height, width), leaving [kernels].
    const Index grads_per_kernel = output_deltas.dimension(0)
                                 * output_deltas.dimension(1)
                                 * output_deltas.dimension(2);

    MatrixMap output_grads_mat(output_delta.data, grads_per_kernel, kernels_number);
    VectorMap(bias_grad.data, kernels_number).noalias() = output_grads_mat.colwise().sum();

    // Weight gradients: for each kernel, convolve padded input with output-gradient slice
    // along (batch, H, W). Eigen's Tensor::convolve is SIMD-vectorized — much faster than
    // the naive 6-nested loop it replaced.
    type* weight_data = weight_grad.data;

    #pragma omp parallel for
    for(Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        const auto kernel_convolution_gradients = output_deltas.chip(kernel_index, 3);

        TensorMap4 kernel_weight_gradients(weight_data + kernel_index * kernel_size,
                                           1, kernel_height, kernel_width, kernel_channels);

        kernel_weight_gradients.device(get_device()) =
            inputs.convolve(kernel_convolution_gradients, array<Index, 3>({0, 1, 2}));
    }
}

void convolution_backward_data(const TensorView& output_delta,
                               const TensorView& kernel,
                               TensorView& input_grad,
                               const ConvolutionArguments& args)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDNN(cudnnConvolutionBackwardData(Device::get_cudnn_handle(),
            &one,
            args.kernel_descriptor, kernel.data,
            output_delta.get_descriptor(), output_delta.data,
            args.convolution_descriptor,
            args.algorithm_data,
            args.workspace, args.workspace_size,
            &zero,
            input_grad.get_descriptor(), input_grad.data));
        return;
    }
#endif
    (void)args;

    const TensorMap4 output_deltas = output_delta.as_tensor<4>();
    TensorMap4 in_grad = input_grad.as_tensor<4>().setZero();

    const Index batch_size = output_deltas.dimension(0);
    const Index kernels_number = kernel.shape[0];
    const Index kernel_height = kernel.shape[1];
    const Index kernel_width = kernel.shape[2];
    const Index kernel_channels = kernel.shape[3];
    const Index input_height = in_grad.dimension(1);
    const Index input_width = in_grad.dimension(2);
    const Index output_height = output_deltas.dimension(1);
    const Index output_width = output_deltas.dimension(2);

    // Pad output_delta so full-convolution with rotated kernel produces input-sized output.
    const Index pad_height = (input_height + kernel_height - 1) - output_height;
    const Index pad_width  = (input_width  + kernel_width  - 1) - output_width;
    const Index pad_top = pad_height / 2;
    const Index pad_bottom = pad_height - pad_top;
    const Index pad_left = pad_width / 2;
    const Index pad_right = pad_width - pad_left;
    const array<pair<Index, Index>, 2> paddings = {
        make_pair(pad_top, pad_bottom),
        make_pair(pad_left, pad_right)
    };

    const TensorMap4 kernels_4d = kernel.as_tensor<4>();
    const Tensor4 rotated_weights = kernels_4d.reverse(array<bool, 4>({false, true, true, false}));

    vector<vector<Tensor2>> precomputed_rotated_slices(kernels_number, vector<Tensor2>(kernel_channels));

    #pragma omp parallel for
    for(Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        auto kernel_rotated_weights = rotated_weights.chip(kernel_index, 0);

        for(Index channel_index = 0; channel_index < kernel_channels; ++channel_index)
            precomputed_rotated_slices[kernel_index][channel_index] = kernel_rotated_weights.chip(channel_index, 2);
    }

    const array<Index, 2> convolution_dimensions_2d = {0, 1};

    for(Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        auto kernel_convolution_gradients = output_deltas.chip(kernel_index, 3);

        #pragma omp parallel for
        for(Index image_index = 0; image_index < batch_size; ++image_index)
        {
            const Tensor2 image_kernel_grads_padded = kernel_convolution_gradients.chip(image_index, 0).pad(paddings);

            for(Index channel_index = 0; channel_index < kernel_channels; ++channel_index)
            {
                const Tensor2 convolution_result = image_kernel_grads_padded
                    .convolve(precomputed_rotated_slices[kernel_index][channel_index], convolution_dimensions_2d);

                for(Index h = 0; h < input_height; ++h)
                    for(Index w = 0; w < input_width; ++w)
                        in_grad(image_index, h, w, channel_index) += convolution_result(h, w);
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
                          const TensorView& output_delta,
                          const TensorView& maximal_indices,
                          TensorView& input_delta,
                          const PoolingArguments& args)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDNN(cudnnPoolingBackward(Device::get_cudnn_handle(),
            args.pooling_descriptor,
            &one,
            output.get_descriptor(), output.data,
            output_delta.get_descriptor(), output_delta.data,
            input.get_descriptor(), input.data,
            &zero,
            input_delta.get_descriptor(), input_delta.data));
        return;
    }
#endif
    (void)output;

    const TensorMap4 out_grads = output_delta.as_tensor<4>();
    const TensorMap4 max_indices = maximal_indices.as_tensor<4>();
    TensorMap4 in_grads = input_delta.as_tensor<4>().setZero();

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
                                     const TensorView& output_delta,
                                     TensorView& input_delta,
                                     const PoolingArguments& args)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDNN(cudnnPoolingBackward(Device::get_cudnn_handle(),
            args.pooling_descriptor,
            &one,
            output.get_descriptor(), output.data,
            output_delta.get_descriptor(), output_delta.data,
            input.get_descriptor(), input.data,
            &zero,
            input_delta.get_descriptor(), input_delta.data));
        return;
    }
#endif
    (void)input; (void)output;

    const TensorMap4 out_grads = output_delta.as_tensor<4>();
    TensorMap4 in_grads = input_delta.as_tensor<4>().setZero();

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
    if (TRY_GPU_DISPATCH(output, [&](auto tag) {
        using T = decltype(tag);
        max_pooling_3d_forward_cuda<T>(to_int(input.shape[0]) * to_int(input.shape[2]),
                                       input.as<T>(), output.as<T>(),
                                       maximal_indices.as<float>(),
                                       to_int(input.shape[1]),
                                       to_int(input.shape[2]));
    })) return;
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
    if (TRY_GPU_DISPATCH(output, [&](auto tag) {
        using T = decltype(tag);
        average_pooling_3d_forward_cuda<T>(to_int(input.shape[0]) * to_int(input.shape[2]),
                                           input.as<T>(), output.as<T>(),
                                           to_int(input.shape[1]),
                                           to_int(input.shape[2]));
    })) return;
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

void max_pooling_3d_backward(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta)
{
    if (TRY_GPU_DISPATCH(input_delta, [&](auto tag) {
        using T = decltype(tag);
        CHECK_CUDA(cudaMemset(input_delta.data, 0, input_delta.byte_size()));
        max_pooling_3d_backward_cuda<T>(to_int(output_delta.shape[0]) * to_int(output_delta.shape[1]),
                                        output_delta.as<T>(), input_delta.as<T>(),
                                        maximal_indices.as<float>(),
                                        to_int(input_delta.shape[1]),
                                        to_int(output_delta.shape[1]));
    })) return;
    const MatrixMap max_indices = maximal_indices.as_matrix();
    const MatrixMap output_delta_matrix = output_delta.as_matrix();
    TensorMap3 input_delta_map = input_delta.as_tensor<3>().setZero();

    const Index batch_size = output_delta_matrix.rows();
    const Index features = output_delta_matrix.cols();

    #pragma omp parallel for
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for(Index feature_index = 0; feature_index < features; ++feature_index)
        {
            const Index step = static_cast<Index>(max_indices(batch_index, feature_index));
            input_delta_map(batch_index, step, feature_index) = output_delta_matrix(batch_index, feature_index);
        }
}

void average_pooling_3d_backward(const TensorView& input, 
                                 const TensorView& output_delta, 
                                 TensorView& input_delta)
{
    if (TRY_GPU_DISPATCH(input_delta, [&](auto tag) {
        using T = decltype(tag);
        CHECK_CUDA(cudaMemset(input_delta.data, 0, input_delta.byte_size()));
        average_pooling_3d_backward_cuda<T>(to_int(input.shape[0]) * to_int(input.shape[2]),
                                            input.as<T>(), output_delta.as<T>(),
                                            input_delta.as<T>(),
                                            to_int(input.shape[1]),
                                            to_int(input.shape[2]));
    })) return;
    const TensorMap3 inputs = input.as_tensor<3>();
    const MatrixMap output_delta_matrix = output_delta.as_matrix();
    TensorMap3 input_delta_map = input_delta.as_tensor<3>().setZero();

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
        Map<MatrixR> grad_matrix(&input_delta_map(batch_index, 0, 0), sequence_length, features);
        const auto output_row = output_delta_matrix.row(batch_index);

        for(Index step = 0; step < sequence_length; ++step)
            if(non_padding(step))
                grad_matrix.row(step) = output_row * inverse_valid_count;
    }
}

void embedding_backward(const TensorView& input_indices,
                        const TensorView& output_delta,
                        TensorView& weight_gradient,
                        Index embedding_dimension,
                        bool scale_embedding)
{
    const Index total_elements = input_indices.size();

    MatrixMap gradients_map = output_delta.as_flat_matrix();

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

static void transpose_middle_axes(const type* src, type* dst,
                                  Index batch_size, Index src_m1, Index src_m2, Index D)
{
    #pragma omp parallel for collapse(3)
    for (Index b = 0; b < batch_size; ++b)
        for (Index i = 0; i < src_m2; ++i)
            for (Index j = 0; j < src_m1; ++j)
                memcpy(dst + ((b * src_m2 + i) * src_m1 + j) * D,
                       src + ((b * src_m1 + j) * src_m2 + i) * D,
                       D * sizeof(type));
}

void split_heads(const TensorView& source, TensorView& destination)
{
    const Index batch_size = source.shape[0];
    const Index sequence_length = source.shape[1];
    const Index heads_number = source.shape[2];
    const Index head_dimension = source.shape[3];

    if (TRY_GPU_DISPATCH(destination, [&](auto tag) {
        using T = decltype(tag);
        split_heads_cuda<T>(source.size(), source.as<T>(), destination.as<T>(),
                            to_int(sequence_length),
                            to_int(heads_number),
                            to_int(head_dimension));
    })) return;

    transpose_middle_axes(source.data, destination.data,
                          batch_size, sequence_length, heads_number, head_dimension);
}

void merge_heads(const TensorView& source, TensorView& destination)
{
    const Index batch_size = source.shape[0];
    const Index heads_number = source.shape[1];
    const Index sequence_length = source.shape[2];
    const Index head_dimension = source.shape[3];

    if (TRY_GPU_DISPATCH(destination, [&](auto tag) {
        using T = decltype(tag);
        merge_heads_cuda<T>(source.size(), source.as<T>(), destination.as<T>(),
                            to_int(sequence_length),
                            to_int(heads_number),
                            to_int(head_dimension));
    })) return;

    transpose_middle_axes(source.data, destination.data,
                          batch_size, heads_number, sequence_length, head_dimension);
}

void projection(const TensorView& input,
                const TensorView& weights,
                const TensorView& biases,
                TensorView& output,
                float* transpose_scratch)
{
    const Index batch_size       = output.shape[0];
    const Index heads_number     = output.shape[1];
    const Index sequence_length  = output.shape[2];
    const Index head_dimension   = output.shape[3];
    const Index embedding_dim    = heads_number * head_dimension;
    const Index total_rows       = batch_size * sequence_length;

    TensorView input_2d = input.reshape({total_rows, embedding_dim});
    TensorView scratch_2d(transpose_scratch, {total_rows, embedding_dim});

    combination(input_2d, weights, biases, scratch_2d);

    TensorView scratch_4d(transpose_scratch,
                          {batch_size, sequence_length, heads_number, head_dimension});
    split_heads(scratch_4d, output);
}

void projection_gradient(const TensorView& head_gradient,
                         const TensorView& input,
                         const TensorView& weights,
                         TensorView& bias_gradient,
                         TensorView& weight_gradient,
                         TensorView& input_delta,
                         float* transpose_scratch,
                         bool accumulate)
{
    const Index batch_size       = head_gradient.shape[0];
    const Index heads_number     = head_gradient.shape[1];
    const Index sequence_length  = head_gradient.shape[2];
    const Index head_dimension   = head_gradient.shape[3];
    const Index embedding_dim    = heads_number * head_dimension;
    const Index total_rows       = batch_size * sequence_length;

    TensorView scratch_4d(transpose_scratch,
                          {batch_size, sequence_length, heads_number, head_dimension});
    merge_heads(head_gradient, scratch_4d);

    TensorView head_gradient_flat(transpose_scratch, {total_rows, embedding_dim});
    TensorView input_flat          = input.reshape({total_rows, embedding_dim});
    TensorView input_delta_flat = input_delta.reshape({total_rows, embedding_dim});

    combination_gradient(head_gradient_flat, input_flat, weights,
                         input_delta_flat, weight_gradient, bias_gradient, accumulate);
}

void attention_masks(const TensorView& source_input,
                           TensorView& attention_weights,
                           const MatrixR& causal_mask,
                           bool use_causal_mask,
                           float* padding_mask_scratch)
{
    const Index batch_size = source_input.shape[0];
    const Index source_sequence_length = source_input.shape[1];
    const Index embedding_dimension = source_input.shape[2];
    const Index heads_number = attention_weights.shape[1];
    const Index query_sequence_length = attention_weights.shape[2];

    if (TRY_GPU_DISPATCH(attention_weights, [&](auto tag) {
        using T = decltype(tag);
        attention_masks_cuda<T>(to_int(batch_size),
                                to_int(heads_number),
                                to_int(query_sequence_length),
                                to_int(source_sequence_length),
                                to_int(embedding_dimension),
                                source_input.as<T>(),
                                attention_weights.as<T>(),
                                reinterpret_cast<T*>(padding_mask_scratch),
                                use_causal_mask);
    })) return;

    const Index att_rows_per_batch = heads_number * query_sequence_length;

    #pragma omp parallel for
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        const type* src = source_input.data + batch_index * source_sequence_length * embedding_dimension;
        type*       att = attention_weights.data + batch_index * att_rows_per_batch * source_sequence_length;

        for(Index s = 0; s < source_sequence_length; ++s)
        {
            const type* src_row = src + s * embedding_dimension;
            type max_abs = type(0);
            for(Index k = 0; k < embedding_dimension; ++k)
            {
                const type a = std::abs(src_row[k]);
                if(a > max_abs) max_abs = a;
            }
            if(max_abs > EPSILON) continue;

            for(Index r = 0; r < att_rows_per_batch; ++r)
                att[r * source_sequence_length + s] = SOFTMAX_MASK_VALUE;
        }
    }

    if(!use_causal_mask) return;

    const Index bh = batch_size * heads_number;
    MatrixMap attention_flat(attention_weights.data, bh * query_sequence_length, source_sequence_length);
    attention_flat += causal_mask.replicate(bh, 1);
}


}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
