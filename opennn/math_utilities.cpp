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

void padding(const TensorView& input,
             TensorView& output)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        (void)input; (void)output;
        return;
    }
#endif
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
        // @todo CUDA bounding
        (void)input;
        (void)lower_bounds;
        (void)upper_bounds;
        (void)output;
        return;
    }
#endif
    const MatrixMap input_matrix = input.as_matrix();
    const VectorMap lower_bounds_vector = lower_bounds.as_vector();
    const VectorMap upper_bounds_vector = upper_bounds.as_vector();

    MatrixMap output_matrix = output.as_matrix();

    for(Index j = 0; j < features; ++j)
        output_matrix.col(j) = input_matrix.col(j)
                                           .cwiseMax(lower_bounds_vector(j))
                                           .cwiseMin(upper_bounds_vector(j));
}

void copy(const TensorView& source, 
          TensorView& destination)
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
        const int n = static_cast<int>(input_1.size());

        // output = input_1
        CHECK_CUDA(cudaMemcpy(output.data, input_1.data, n * sizeof(float), cudaMemcpyDeviceToDevice));
        // output += input_2
        CHECK_CUBLAS(cublasSaxpy(Device::get_cublas_handle(), n, &one, input_2.data, 1, output.data, 1));
        return;
    }
#endif
    output.as_vector().array() = input_1.as_vector().array() + input_2.as_vector().array();
}

void multiply(const TensorView& input_A, bool transpose_A,
              const TensorView& input_B, bool transpose_B,
              TensorView& output_C,
              type alpha, type beta)
{
    const size_t rank = input_A.get_rank();

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
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

        batch_count == 1
            ? CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(),
                                       op_B_cublas, op_A_cublas,
                                       m, n, k,
                                       &alpha,
                                       input_B.data, ld_B,
                                       input_A.data, ld_A,
                                       &beta,
                                       output_C.data, ld_C))
            : CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
                                                   op_B_cublas, op_A_cublas,
                                                   m, n, k,
                                                   &alpha,
                                                   input_B.data, ld_B, stride_B,
                                                   input_A.data, ld_A, stride_A,
                                                   &beta,
                                                   output_C.data, ld_C, stride_C,
                                                   batch_count));
        
        return;
    }
#endif
    const bool simple = (alpha == 1.0f && beta == 0.0f);

    if (rank <= 2)
    {
        const auto matrix_A = input_A.as_matrix();
        const auto matrix_B = input_B.as_matrix();
        auto matrix_C = output_C.as_matrix();

        if(simple)
        {
            if (!transpose_A && !transpose_B)
                matrix_C.noalias() = matrix_A * matrix_B;
            else if (transpose_A && !transpose_B)
                matrix_C.noalias() = matrix_A.transpose() * matrix_B;
            else if (!transpose_A && transpose_B)
                matrix_C.noalias() = matrix_A * matrix_B.transpose();
            else
                matrix_C.noalias() = matrix_A.transpose() * matrix_B.transpose();
        }
        else
        {
            if (!transpose_A && !transpose_B)
                matrix_C.noalias() = alpha * (matrix_A * matrix_B) + beta * matrix_C;
            else if (transpose_A && !transpose_B)
                matrix_C.noalias() = alpha * (matrix_A.transpose() * matrix_B) + beta * matrix_C;
            else if (!transpose_A && transpose_B)
                matrix_C.noalias() = alpha * (matrix_A * matrix_B.transpose()) + beta * matrix_C;
            else
                matrix_C.noalias() = alpha * (matrix_A.transpose() * matrix_B.transpose()) + beta * matrix_C;
        }
    }
    else
    {
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

            if(simple)
            {
                if (!transpose_A && !transpose_B)
                    mat_C.noalias() = mat_A * mat_B;
                else if (transpose_A && !transpose_B)
                    mat_C.noalias() = mat_A.transpose() * mat_B;
                else if (!transpose_A && transpose_B)
                    mat_C.noalias() = mat_A * mat_B.transpose();
                else
                    mat_C.noalias() = mat_A.transpose() * mat_B.transpose();
            }
            else
            {
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
    }
}

void multiply_elementwise(const TensorView& A, const TensorView& B, TensorView& C)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(), Device::get_operator_multiplication_descriptor(),
                                  &one, A.get_descriptor(), A.data,
                                  &one, B.get_descriptor(), B.data,
                                  &zero, C.get_descriptor(), C.data));
        return;
    }
#endif
    C.as_vector().array() = A.as_vector().array() * B.as_vector().array();
}

void sum(const TensorView& A, TensorView& B, type alpha, type beta)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const int rows = static_cast<int>(A.shape[0]);
        const int cols = static_cast<int>(A.shape.size() / A.shape[0]);

        CHECK_CUBLAS(cublasSgemv(Device::get_cublas_handle(),
                                 CUBLAS_OP_N,
                                 cols, rows,
                                 &alpha,
                                 A.data, cols,
                                 Device::get_ones(rows), 1,
                                 &beta,
                                 B.data, 1));
        return;
    }
#endif
    B.as_vector().noalias() = alpha * A.as_matrix().colwise().sum() + beta * B.as_vector();
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
    const Index columns = output.shape.back();
    const Index rows = output.size() / columns;

    MatrixMap mat(output.data, rows, columns);
    mat.colwise() -= mat.rowwise().maxCoeff();
    mat.array() = mat.array().exp();
    mat.array().colwise() /= mat.rowwise().sum().array();
}

void combination(const TensorView& input,
                 const TensorView& weights,
                 const TensorView& biases,
                 TensorView& output)
{
    const Index in_cols = input.shape.back();
    const Index rows = input.size() / in_cols;
    const Index out_cols = weights.shape.back();

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu())
    {
        const int m = static_cast<int>(out_cols);
        const int n = static_cast<int>(rows);
        const int k = static_cast<int>(in_cols);

        CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(),
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 m, n, k,
                                 &one,
                                 weights.data, m,
                                 input.data, k,
                                 &zero,
                                 output.data, m));

        CHECK_CUDNN(cudnnAddTensor(Device::get_cudnn_handle(),
                                   &one,
                                   biases.get_descriptor(), biases.data,
                                   &one,
                                   output.get_descriptor(), output.data));
        return;
    }
#endif

    const MatrixMap input_2d(input.data, rows, in_cols);
    MatrixMap output_2d(output.data, rows, out_cols);
    output_2d.noalias() = (input_2d * weights.as_matrix()).rowwise() + biases.as_vector().transpose();
}

void activation(TensorView& output, ActivationArguments arguments)
{
    if (output.empty() || arguments.activation_function == ActivationFunction::Linear)
        return;

    const ActivationFunction func = arguments.activation_function;

    if(func == ActivationFunction::Softmax)
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

    switch (func)
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

    const ActivationFunction func = arguments.activation_function;

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        if(func == ActivationFunction::Linear || func == ActivationFunction::Softmax)
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
    const Index n = output.size();
    if (args.mask_cpu.size() != n) args.mask_cpu.resize(n);

    const type scale = type(1) / (type(1) - args.rate);
    type* __restrict data = output.data;
    type* __restrict mask = args.mask_cpu.data();

    for (Index i = 0; i < n; ++i)
    {
        const bool dropped = random_uniform(type(0), type(1)) < args.rate;
        mask[i] = dropped ? type(0) : scale;
        data[i] *= mask[i];
    }
}

void dropout_gradient(const TensorView& output_gradient,
                      const DropoutArguments& args,
                      TensorView& input_gradient)
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
    const Index n = output_gradient.size();
    Eigen::Map<const VectorR> mask(args.mask_cpu.data(), n);
    input_gradient.as_vector().array() = output_gradient.as_vector().array() * mask.array();
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

    output_matrix = ((input_matrix.rowwise() - running_means.transpose()).array()
                     .rowwise() / (running_variances.array() + EPSILON).sqrt().transpose())
                    .rowwise() * gammas.transpose().array()
                    + betas.transpose().replicate(effective_batch_size, 1).array();
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
    VectorMap rmean = running_mean.as_vector();
    VectorMap rvar = running_variance.as_vector();

    means = input_matrix.colwise().mean();

    inverse_variances = (input_matrix.rowwise() - means.transpose()).array().square().colwise().mean();

    rmean = rmean * momentum + means * (type(1) - momentum);
    rvar = rvar * momentum + inverse_variances * (type(1) - momentum);

    inverse_variances.array() = 1.0f / (inverse_variances.array() + EPSILON).sqrt();

    output_matrix.array() = (input_matrix.rowwise() - means.transpose()).array().rowwise() *
                            (inverse_variances.array() * gammas.array()).transpose();

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
    (void)output;
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

    // Compute beta_gradient FIRST (only uses output_gradients).
    beta_gradients.noalias() = output_gradients.colwise().sum();

    // Allocate x_hat as a temporary so we can support in-place dx == dy
    // (caller may pass the same buffer for input_gradient and output_gradient).
    const MatrixR x_hat = (input_matrix.rowwise() - means.transpose()).array().rowwise()
                          * inverse_variances.transpose().array();

    gamma_gradients = (output_gradients.array() * x_hat.array()).matrix().colwise().sum();

    const type batch_size_type = static_cast<type>(effective_batch_size);

    const Eigen::Array<type, 1, Eigen::Dynamic> scale =
        (gammas.array() * inverse_variances.array() / batch_size_type).transpose();

    // Safe to write input_gradients now (we're done reading output_gradients).
    input_gradients.array() = ((batch_size_type * output_gradients.array()).rowwise() - beta_gradients.transpose().array()
                               - x_hat.array().rowwise() * gamma_gradients.transpose().array())
                              .rowwise() * scale;
}

void layernorm_forward(const TensorView& input, const TensorView& gamma, const TensorView& beta,
                       TensorView& means, TensorView& standard_deviations, TensorView& normalized,
                       TensorView& output,
                       Index batch_size, Index sequence_length, Index embedding_dimension)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const int N = static_cast<int>(batch_size * sequence_length);
        const int D = static_cast<int>(embedding_dimension);

        layernorm_forward_cuda(N, D,
            input.data, output.data,
            means.data, standard_deviations.data,
            gamma.data, beta.data, EPSILON);
        return;
    }
#endif
    const Index E = embedding_dimension;

    const TensorMap3 X(input.data, batch_size, sequence_length, E);
    TensorMap2 mu(means.data, batch_size, sequence_length);
    TensorMap2 sigma(standard_deviations.data, batch_size, sequence_length);
    TensorMap3 X_hat(normalized.data, batch_size, sequence_length, E);
    TensorMap3 Y(output.data, batch_size, sequence_length, E);

    const array<Index, 3> reshape_dims({batch_size, sequence_length, 1});
    const array<Index, 3> broadcast_dims({1, 1, E});

    mu = X.mean(array<Index, 1>({2}));

    auto centered = X - mu.reshape(reshape_dims).broadcast(broadcast_dims);
    auto variance = centered.square().mean(array<Index, 1>({2}));
    sigma = (variance + EPSILON).sqrt();

    X_hat = centered / sigma.reshape(reshape_dims).broadcast(broadcast_dims);

    TensorMap1 g(gamma.data, E);
    TensorMap1 b(beta.data, E);

    Y = X_hat * g.reshape(array<Index, 3>({1, 1, E})).broadcast(array<Index, 3>({batch_size, sequence_length, 1}))
      + b.reshape(array<Index, 3>({1, 1, E})).broadcast(array<Index, 3>({batch_size, sequence_length, 1}));
}


void layernorm_backward(const TensorView& input, const TensorView& output_gradient,
                        const TensorView& means, const TensorView& standard_deviations,
                        const TensorView& normalized, const TensorView& gamma,
                        TensorView& gamma_gradient, TensorView& beta_gradient, TensorView& input_gradient,
                        Index batch_size, Index sequence_length, Index embedding_dimension)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const int N = static_cast<int>(batch_size * sequence_length);
        const int D = static_cast<int>(embedding_dimension);

        layernorm_backward_cuda(N, D,
            output_gradient.data, input.data,
            means.data, standard_deviations.data,
            gamma.data,
            input_gradient.data,
            gamma_gradient.data, beta_gradient.data);
        return;
    }
#endif
    const Index E = embedding_dimension;

    const TensorMap2 sigma(standard_deviations.data, batch_size, sequence_length);
    const TensorMap3 X_hat(normalized.data, batch_size, sequence_length, E);
    const TensorMap3 dY(output_gradient.data, batch_size, sequence_length, E);

    TensorMap1 dGamma(gamma_gradient.data, E);
    TensorMap1 dBeta(beta_gradient.data, E);

    dGamma = (dY * X_hat).sum(array<Index, 2>({0, 1}));
    dBeta = dY.sum(array<Index, 2>({0, 1}));

    TensorMap3 dX(input_gradient.data, batch_size, sequence_length, E);
    TensorMap1 gamma_map(gamma.data, E);

    auto gamma_bcast = gamma_map.reshape(array<Index, 3>({1, 1, E}))
                           .broadcast(array<Index, 3>({batch_size, sequence_length, 1}));

    Tensor3 D = dY * gamma_bcast;
    Tensor2 sum_D = D.sum(array<Index, 1>({2}));
    Tensor2 sum_D_xhat = (D * X_hat).sum(array<Index, 1>({2}));

    auto sum_D_bcast = sum_D.reshape(array<Index, 3>({batch_size, sequence_length, 1}))
                           .broadcast(array<Index, 3>({1, 1, E}));
    auto sum_D_xhat_bcast = sum_D_xhat.reshape(array<Index, 3>({batch_size, sequence_length, 1}))
                                .broadcast(array<Index, 3>({1, 1, E}));
    auto std_dev_bcast = sigma.reshape(array<Index, 3>({batch_size, sequence_length, 1}))
                             .broadcast(array<Index, 3>({1, 1, E}));

    const type inv_E = type(1.0) / static_cast<type>(E);
    dX = (D - sum_D_bcast * inv_E - X_hat * sum_D_xhat_bcast * inv_E) / std_dev_bcast;
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
    opennn::activation(output, activation_arguments);
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
    const Index single_kernel_size = kernel_height * kernel_width * kernel_channels;
    const Index output_height = deltas.dimension(1);
    const Index output_width = deltas.dimension(2);
    const Index input_height = inputs.dimension(1);
    const Index input_width = inputs.dimension(2);

    VectorMap bias_gradients = bias_grad.as_vector();
    bias_gradients.setZero();

    memset(weight_grad.data, 0, kernels_number * single_kernel_size * sizeof(type));

    for(Index ki = 0; ki < kernels_number; ki++)
    {
        type* __restrict wg = weight_grad.data + ki * single_kernel_size;
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
        const type* __restrict kw = kernel.data + ki * single_kernel_size;

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
}

void max_pooling_3d_forward(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        (void)is_training;
        const int B = static_cast<int>(input.shape[0]);
        const int S = static_cast<int>(input.shape[1]);
        const int F = static_cast<int>(input.shape[2]);
        pooling3d_max_forward_cuda(B * F, input.data, output.data, maximal_indices.data, B, S, F);
        return;
    }
#endif
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
}

void average_pooling_3d_forward(const TensorView& input, TensorView& output)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const int B = static_cast<int>(input.shape[0]);
        const int S = static_cast<int>(input.shape[1]);
        const int F = static_cast<int>(input.shape[2]);
        pooling3d_avg_forward_cuda(B * F, input.data, output.data, B, S, F);
        return;
    }
#endif
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
}

void max_pooling_3d_backward(const TensorView& maximal_indices, const TensorView& output_gradient, TensorView& input_gradient)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const int B = static_cast<int>(output_gradient.shape[0]);
        const int F = static_cast<int>(output_gradient.shape[1]);
        const int S = static_cast<int>(input_gradient.shape[1]);
        CHECK_CUDA(cudaMemset(input_gradient.data, 0, input_gradient.size() * sizeof(float)));
        pooling3d_max_backward_cuda(B * F, output_gradient.data, input_gradient.data, maximal_indices.data, B, S, F);
        return;
    }
#endif
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
}

void average_pooling_3d_backward(const TensorView& input, const TensorView& output_gradient, TensorView& input_gradient)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const int B = static_cast<int>(input.shape[0]);
        const int S = static_cast<int>(input.shape[1]);
        const int F = static_cast<int>(input.shape[2]);
        CHECK_CUDA(cudaMemset(input_gradient.data, 0, input_gradient.size() * sizeof(float)));
        pooling3d_avg_backward_cuda(B * F, input.data, output_gradient.data, input_gradient.data, B, S, F);
        return;
    }
#endif
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
    const MatrixMap W(weights.data, embedding_dimension, heads_number * head_dimension);
    const VectorMap b(biases.data, heads_number * head_dimension);

    #pragma omp parallel for collapse(2)
    for(Index bb = 0; bb < batch_size; ++bb)
    {
        for(Index h = 0; h < heads_number; ++h)
        {
            const MatrixMap X_b(input.data + bb * sequence_length * embedding_dimension, sequence_length, embedding_dimension);

            MatrixMap Out_bh(output.data + bb * (heads_number * sequence_length * head_dimension)
                             + h * (sequence_length * head_dimension),
                             sequence_length, head_dimension);

            auto W_h = W.block(0, h * head_dimension, embedding_dimension, head_dimension);
            auto b_h = b.segment(h * head_dimension, head_dimension);

            Out_bh.noalias() = (X_b * W_h).rowwise() + b_h.transpose();
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

    const MatrixMap W(weights.data, embedding_dimension, heads_number * head_dimension);

    #pragma omp parallel for
    for(Index b = 0; b < batch_size; ++b)
    {
        MatrixMap dX_b(d_input.data + b * sequence_length * embedding_dimension, sequence_length, embedding_dimension);

        if(!accumulate) dX_b.setZero();

        for(Index h = 0; h < heads_number; ++h)
        {
            const MatrixMap Delta(d_head.data + b * (heads_number * sequence_length * head_dimension) + h * (sequence_length * head_dimension),
                                  sequence_length, head_dimension);
            auto W_h = W.block(0, h * head_dimension, embedding_dimension, head_dimension);
            dX_b.noalias() += Delta * W_h.transpose();
        }
    }

    MatrixMap dW(d_weights.data, embedding_dimension, heads_number * head_dimension);
    VectorMap db(d_bias.data, heads_number * head_dimension);

    #pragma omp parallel for
    for(Index h = 0; h < heads_number; ++h)
    {
        auto dW_h = dW.block(0, h * head_dimension, embedding_dimension, head_dimension);
        auto db_h = db.segment(h * head_dimension, head_dimension);
        dW_h.setZero();
        db_h.setZero();

        for(Index b = 0; b < batch_size; ++b)
        {
            const MatrixMap Delta(d_head.data + b * (heads_number * sequence_length * head_dimension) + h * (sequence_length * head_dimension),
                                  sequence_length, head_dimension);
            const MatrixMap X_b(input.data + b * sequence_length * embedding_dimension, sequence_length, embedding_dimension);

            dW_h.noalias() += X_b.transpose() * Delta;
            db_h.noalias() += Delta.colwise().sum().transpose();
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
        const int BH = static_cast<int>(total_heads);
        const int Sq = static_cast<int>(query_sequence_length);
        const int Sk = static_cast<int>(source_sequence_length);
        const int E  = static_cast<int>(embedding_dimension);
        const int D  = static_cast<int>(head_dimension);
        const int H  = static_cast<int>(heads_number);
        const int B  = static_cast<int>(batch_size);
        const float sf = static_cast<float>(scaling_factor);

        // Transpose Q, K, V from [B, S, H, D] to [B, H, S, D]

        float* scratch = args.transpose_scratch;

        mha_transpose_qkv_cuda(B * Sq * E, query.data, scratch, Sq, H, D);
        cudaMemcpy(query.data, scratch, B * Sq * E * sizeof(float), cudaMemcpyDeviceToDevice);

        mha_transpose_qkv_cuda(B * Sk * E, key.data, scratch, Sk, H, D);
        cudaMemcpy(key.data, scratch, B * Sk * E * sizeof(float), cudaMemcpyDeviceToDevice);

        mha_transpose_qkv_cuda(B * Sk * E, value.data, scratch, Sk, H, D);
        cudaMemcpy(value.data, scratch, B * Sk * E * sizeof(float), cudaMemcpyDeviceToDevice);

        // Q * K^T — attention scores

        CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            Sk, Sq, D,
            &sf,
            key.data, D, Sk * D,
            query.data, D, Sq * D,
            &zero,
            attention_weights.data, Sk, Sq * Sk,
            BH));

        // Fused masks: padding + causal

        mha_fused_masks_cuda(B, H, Sq, Sk, E, source_input.data, attention_weights.data,
                             args.padding_mask, use_causal_mask);

        // Softmax

        TensorView att_view(attention_weights.data, {(Index)(BH * Sq), (Index)Sk});
        att_view.set_descriptor(att_view.shape);
        softmax(att_view);

        // Attention * V

        float* att_out = args.attention_output_transposed;

        CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, Sq, Sk,
            &one,
            value.data, D, Sk * D,
            attention_weights.data, Sk, Sq * Sk,
            &zero,
            att_out, D, Sq * D,
            BH));

        // Transpose back to [B*Sq, E]

        mha_transpose_o_cuda(B * Sq * E, att_out, concatenated.data, Sq, H, D);

        // Output projection

        TensorView concat_2d(concatenated.data, {(Index)(B * Sq), (Index)E});
        TensorView output_2d(output.data, {(Index)(B * Sq), (Index)E});
        concat_2d.set_descriptor(concat_2d.shape);
        output_2d.set_descriptor(output_2d.shape);
        combination(concat_2d, projection_weights, projection_biases, output_2d);
        return;
    }
#endif
    // Q*K^T — attention scores

    #pragma omp parallel for
    for(Index i = 0; i < total_heads; ++i)
    {
        const MatrixMap q(query.data + i * query_sequence_length * head_dimension, query_sequence_length, head_dimension);
        const MatrixMap k(key.data + i * source_sequence_length * head_dimension, source_sequence_length, head_dimension);
        MatrixMap w(attention_weights.data + i * query_sequence_length * source_sequence_length, query_sequence_length, source_sequence_length);
        w.noalias() = (q * k.transpose()) * scaling_factor;
    }

    // Key padding mask

    #pragma omp parallel for
    for(Index b = 0; b < batch_size; ++b)
    {
        for(Index s = 0; s < source_sequence_length; ++s)
        {
            const type* row_ptr = source_input.data + b * source_sequence_length * embedding_dimension + s * embedding_dimension;
            const bool is_pad = Eigen::Map<const VectorR>(row_ptr, embedding_dimension)
                                    .cwiseAbs().maxCoeff() <= type(1e-7f);

            if(is_pad)
            {
                const Index slice_size = heads_number * query_sequence_length;
                MatrixMap att_map(attention_weights.data + b * slice_size * source_sequence_length,
                                  slice_size, source_sequence_length);
                att_map.col(s).setConstant(type(-1e9f));
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

    TensorView att_view(attention_weights.data, {total_heads * query_sequence_length, source_sequence_length});
    softmax(att_view);

    // W*V — attention output, scattered into concatenated

    #pragma omp parallel for collapse(2)
    for(Index b = 0; b < batch_size; ++b)
        for(Index h = 0; h < heads_number; ++h)
        {
            const Index off_w = (b * heads_number + h) * query_sequence_length * source_sequence_length;
            const Index off_v = (b * heads_number + h) * source_sequence_length * head_dimension;
            const MatrixMap w(attention_weights.data + off_w, query_sequence_length, source_sequence_length);
            const MatrixMap v(value.data + off_v, source_sequence_length, head_dimension);
            type* out_ptr = concatenated.data + b * query_sequence_length * embedding_dimension + h * head_dimension;
            using StrideType = Eigen::OuterStride<Eigen::Dynamic>;
            Eigen::Map<MatrixR, 0, StrideType> o(out_ptr, query_sequence_length, head_dimension, StrideType(embedding_dimension));
            o.noalias() = w * v;
        }

    // Output projection

    const MatrixMap concat_map(concatenated.data, total_rows, embedding_dimension);
    MatrixMap out_map(output.data, total_rows, embedding_dimension);
    const MatrixMap proj_w(projection_weights.data, embedding_dimension, embedding_dimension);
    const VectorMap proj_b(projection_biases.data, embedding_dimension);
    out_map.noalias() = (concat_map * proj_w).rowwise() + proj_b.transpose();
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
        const int BH = static_cast<int>(total_heads);
        const int Sq = static_cast<int>(query_sequence_length);
        const int Sk = static_cast<int>(source_sequence_length);
        const int E  = static_cast<int>(embedding_dimension);
        const int D  = static_cast<int>(head_dimension);
        const int H  = static_cast<int>(heads_number);
        const int B  = static_cast<int>(batch_size);
        const float sf = static_cast<float>(scaling_factor);

        float* scratch = args.transpose_scratch;

        // Projection weight gradients: dW_proj = concat^T * dY

        CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_T,
            E, E, B * Sq,
            &one,
            output_gradient.data, E,
            concatenated.data, E,
            &zero,
            proj_weight_grad.data, E));

        // Projection bias gradients: db_proj = sum(dY)

        CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            E, 1, B * Sq,
            &one,
            output_gradient.data, E,
            Device::get_ones(B * Sq), B * Sq,
            &zero,
            proj_bias_grad.data, E));

        // Concatenated output gradients: d_concat = dY * W_proj^T

        CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            E, B * Sq, E,
            &one,
            projection_weights.data, E,
            output_gradient.data, E,
            &zero,
            concat_grad.data, E));

        // Transpose d_concat from [B, Sq, H, D] to [B, H, Sq, D]

        mha_transpose_qkv_cuda(B * Sq * E, concat_grad.data, scratch, Sq, H, D);

        // dV = P^T * dO (transposed)

        CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, Sk, Sq,
            &one,
            scratch, D, Sq * D,
            attention_weights.data, Sk, Sq * Sk,
            &zero,
            value_grad.data, D, Sk * D,
            BH));

        // dP = dO * V^T

        CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            Sk, Sq, D,
            &one,
            value.data, D, Sk * D,
            scratch, D, Sq * D,
            &zero,
            att_weight_grad.data, Sk, Sq * Sk,
            BH));

        // Softmax backward

        TensorView att_view(attention_weights.data, {(Index)(BH * Sq), (Index)Sk});
        att_view.set_descriptor(att_view.shape);
        TensorView datt_view(att_weight_grad.data, {(Index)(BH * Sq), (Index)Sk});
        datt_view.set_descriptor(datt_view.shape);
        TensorView sgrad_view(args.softmax_gradient, {(Index)(BH * Sq), (Index)Sk});
        sgrad_view.set_descriptor(sgrad_view.shape);

        CHECK_CUDNN(cudnnSoftmaxBackward(Device::get_cudnn_handle(),
            CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
            &one, att_view.get_descriptor(), attention_weights.data,
            datt_view.get_descriptor(), att_weight_grad.data,
            &zero, sgrad_view.get_descriptor(), args.softmax_gradient));

        // dQ = softmax_grad * K^T * scaling_factor

        CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, Sq, Sk,
            &sf,
            key.data, D, Sk * D,
            args.softmax_gradient, Sk, Sq * Sk,
            &zero,
            query_grad.data, D, Sq * D,
            BH));

        // dK = softmax_grad^T * Q * scaling_factor

        CHECK_CUBLAS(cublasSgemmStridedBatched(Device::get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, Sk, Sq,
            &sf,
            query.data, D, Sq * D,
            args.softmax_gradient, Sk, Sq * Sk,
            &zero,
            key_grad.data, D, Sk * D,
            BH));

        // Transpose dQ, dK, dV from [B, H, S, D] to [B, S, H, D]

        float* q_grad_flat = args.query_input_gradient_scratch;
        float* src_grad_flat = args.source_input_gradient_scratch;

        mha_transpose_o_cuda(B * Sq * E, query_grad.data, q_grad_flat, Sq, H, D);
        mha_transpose_o_cuda(B * Sk * E, key_grad.data, scratch, Sk, H, D);
        float* k_grad_flat = scratch; // reuse scratch for K grad flat

        // Query weight/bias/input gradients

        CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
            E, E, B * Sq, &one, q_grad_flat, E, query_input.data, E, &zero, query_weight_grad.data, E));

        CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
            E, 1, B * Sq, &one, q_grad_flat, E, Device::get_ones(B * Sq), B * Sq, &zero, query_bias_grad.data, E));

        CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
            E, B * Sq, E, &one, query_weights.data, E, q_grad_flat, E, &zero, input_query_grad.data, E));

        if(self_attention)
        {
            // Key weight/bias/source gradients

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                E, E, B * Sk, &one, k_grad_flat, E, source_input.data, E, &zero, key_weight_grad.data, E));

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                E, 1, B * Sk, &one, k_grad_flat, E, Device::get_ones(B * Sk), B * Sk, &zero, key_bias_grad.data, E));

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                E, B * Sk, E, &one, key_weights.data, E, k_grad_flat, E, &zero, src_grad_flat, E));

            // Value weight/bias/source gradients (accumulate on src_grad_flat)

            mha_transpose_o_cuda(B * Sk * E, value_grad.data, k_grad_flat, Sk, H, D); // reuse scratch again

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                E, E, B * Sk, &one, k_grad_flat, E, source_input.data, E, &zero, value_weight_grad.data, E));

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                E, 1, B * Sk, &one, k_grad_flat, E, Device::get_ones(B * Sk), B * Sk, &zero, value_bias_grad.data, E));

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                E, B * Sk, E, &one, value_weights.data, E, k_grad_flat, E, &one, src_grad_flat, E));

            // input_query_grad = q_input_grad + src_grad

            addition_cuda(B * Sq * E, input_query_grad.data, src_grad_flat, input_query_grad.data);
        }
        else
        {
            // Cross-attention: K/V gradients go to input_source_grad

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                E, E, B * Sk, &one, k_grad_flat, E, source_input.data, E, &zero, key_weight_grad.data, E));

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                E, 1, B * Sk, &one, k_grad_flat, E, Device::get_ones(B * Sk), B * Sk, &zero, key_bias_grad.data, E));

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                E, B * Sk, E, &one, key_weights.data, E, k_grad_flat, E, &zero, input_source_grad.data, E));

            mha_transpose_o_cuda(B * Sk * E, value_grad.data, k_grad_flat, Sk, H, D);

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                E, E, B * Sk, &one, k_grad_flat, E, source_input.data, E, &zero, value_weight_grad.data, E));

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                E, 1, B * Sk, &one, k_grad_flat, E, Device::get_ones(B * Sk), B * Sk, &zero, value_bias_grad.data, E));

            CHECK_CUBLAS(cublasSgemm(Device::get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                E, B * Sk, E, &one, value_weights.data, E, k_grad_flat, E, &one, input_source_grad.data, E));
        }
        return;
    }
#endif
    // Projection gradients

    const MatrixMap concat_map(concatenated.data, total_rows, embedding_dimension);
    const MatrixMap dY_map(output_gradient.data, total_rows, embedding_dimension);

    MatrixMap(proj_weight_grad.data, embedding_dimension, embedding_dimension).noalias() = concat_map.transpose() * dY_map;
    VectorMap(proj_bias_grad.data, embedding_dimension).noalias() = dY_map.colwise().sum();

    MatrixMap concat_grad_map(concat_grad.data, total_rows, embedding_dimension);
    const MatrixMap proj_w(projection_weights.data, embedding_dimension, embedding_dimension);
    concat_grad_map.noalias() = dY_map * proj_w.transpose();

    // dV and dP from concat_grad

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
            using StrideType = Eigen::OuterStride<Eigen::Dynamic>;
            Eigen::Map<const MatrixR, 0, StrideType> dO(dO_ptr, query_sequence_length, head_dimension, StrideType(embedding_dimension));

            dV.noalias() = P.transpose() * dO;
            dP.noalias() = dO * V.transpose();
        }

    // Softmax gradient + dQ/dK

    #pragma omp parallel for
    for(Index i = 0; i < total_heads; ++i)
    {
        const Index off_w = i * query_sequence_length * source_sequence_length;
        const Index off_q = i * query_sequence_length * head_dimension;
        const Index off_k = i * source_sequence_length * head_dimension;

        const MatrixMap P(attention_weights.data + off_w, query_sequence_length, source_sequence_length);
        MatrixMap dP(att_weight_grad.data + off_w, query_sequence_length, source_sequence_length);

        VectorR dot = (P.array() * dP.array()).rowwise().sum();
        dP.array() = P.array() * (dP.colwise() - dot).array();

        const MatrixMap Q(query.data + off_q, query_sequence_length, head_dimension);
        const MatrixMap K(key.data + off_k, source_sequence_length, head_dimension);
        MatrixMap dQ(query_grad.data + off_q, query_sequence_length, head_dimension);
        MatrixMap dK(key_grad.data + off_k, source_sequence_length, head_dimension);

        dQ.noalias() = (dP * K) * scaling_factor;
        dK.noalias() = (dP.transpose() * Q) * scaling_factor;
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
