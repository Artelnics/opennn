//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E R R O R   U T I L I T I E S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "error_utilities.h"
#include "cuda_dispatch.h"

namespace opennn
{

#ifdef OPENNN_WITH_CUDA

static float sum_squared_diff_cuda(const TensorView& input, const TensorView& target, float* workspace)
{
    const int total_size = to_int(input.size());
    CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(), Device::get_operator_sum_descriptor(),
                              &one, input.get_descriptor(), input.data,
                              &minus_one, target.get_descriptor(), target.data,
                              &zero, input.get_descriptor(), workspace));
    float sum_squared = 0.0f;
    CHECK_CUBLAS(cublasDotEx(Device::get_cublas_handle(), total_size,
                             workspace, CUDA_ACTIVATION_DTYPE, 1,
                             workspace, CUDA_ACTIVATION_DTYPE, 1,
                             &sum_squared, CUDA_REDUCTION_DTYPE,
                             CUDA_REDUCTION_DTYPE));
    return sum_squared;
}

static void scaled_diff_cuda(const TensorView& input, const TensorView& target, float scale, TensorView& output)
{
    const int total_size = to_int(input.size());
    CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(), Device::get_operator_sum_descriptor(),
                              &one, input.get_descriptor(), input.data,
                              &minus_one, target.get_descriptor(), target.data,
                              &zero, output.get_descriptor(), output.data));
    CHECK_CUBLAS(cublasScalEx(Device::get_cublas_handle(), total_size,
                              &scale, CUDA_REDUCTION_DTYPE,
                              output.data, CUDA_ACTIVATION_DTYPE, 1,
                              CUDA_REDUCTION_DTYPE));
}

static float sum_abs_cuda(const float* data, Index n)
{
    float sum = 0.0f;
    CHECK_CUBLAS(cublasSasum(Device::get_cublas_handle(), to_int(n), data, 1, &sum));
    return sum;
}

static float squared_norm_cuda(const float* data, Index n)
{
    float dot = 0.0f;
    CHECK_CUBLAS(cublasDotEx(Device::get_cublas_handle(), to_int(n),
                             data, CUDA_ACTIVATION_DTYPE, 1,
                             data, CUDA_ACTIVATION_DTYPE, 1,
                             &dot, CUDA_REDUCTION_DTYPE,
                             CUDA_REDUCTION_DTYPE));
    return dot;
}

#endif

void mean_squared_error(const TensorView& input, const TensorView& target, type& error, float* workspace_device)
{
    const Index batch_size = input.shape[0];
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        error = sum_squared_diff_cuda(input, target, workspace_device) / to_int(2 * batch_size);
        return;
    }
#endif
    error = (input.as_vector() - target.as_vector()).squaredNorm() / to_type(2 * batch_size);
}

void mean_squared_error_gradient(const TensorView& input, const TensorView& target, TensorView& input_delta)
{
    const Index batch_size = input.shape[0];
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        scaled_diff_cuda(input, target, 1.0f / to_int(batch_size), input_delta);
        return;
    }
#endif
    input_delta.as_vector().noalias() = (input.as_vector() - target.as_vector()) / to_type(batch_size);
}

void normalized_squared_error(const TensorView& input, const TensorView& target, type coefficient, type& error, float* workspace_device)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        error = sum_squared_diff_cuda(input, target, workspace_device) / (2.0f * (coefficient + EPSILON));
        return;
    }
#endif
    error = (input.as_vector() - target.as_vector()).squaredNorm() / (2.0f * (coefficient + EPSILON));
}

void normalized_squared_error_gradient(const TensorView& input, const TensorView& target, type coefficient, TensorView& input_delta)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        scaled_diff_cuda(input, target, 1.0f / (static_cast<float>(coefficient) + EPSILON), input_delta);
        return;
    }
#endif
    input_delta.as_vector().noalias() = (input.as_vector() - target.as_vector()) / (coefficient + EPSILON);
}

void weighted_squared_error(const TensorView& input, const TensorView& target, type pos_w, type neg_w, type& error, float* workspace_device)
{
<<<<<<< HEAD
    if (TRY_GPU_DISPATCH(input, [&](auto tag) {
        using T = decltype(tag);
        weighted_squared_error_cuda<T>(input.size(),
            workspace_device, target.as<T>(), input.as<T>(), pos_w, neg_w);
        error = sum_abs_cuda(workspace_device, input.size());
    })) return;
=======
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        input.dispatch([&](auto tag) {
            using T = decltype(tag);
            weighted_squared_error_cuda<T>(input.size(),
                workspace_device, target.as<T>(), input.as<T>(), pos_w, neg_w);
        });
        error = 0.5f * sum_abs_cuda(workspace_device, input.size());
        return;
    }
#endif
>>>>>>> 034d3de168c576a8914d0e3750f2850c75013e28
    const auto inputs = input.as_vector().array();
    const auto targets = target.as_vector().array();
    const auto squared_error = (inputs - targets).square();
    error = 0.5f * (targets == 1.0f).select(squared_error * pos_w, squared_error * neg_w).sum();
}

void weighted_squared_error_gradient(const TensorView& input, const TensorView& target, type pos_w, type neg_w, type coefficient, TensorView& input_delta)
{
    if (TRY_GPU_DISPATCH(input, [&](auto tag) {
        using T = decltype(tag);
        weighted_squared_error_gradient_cuda<T>(input.size(),
            input_delta.as<T>(), target.as<T>(), input.as<T>(), pos_w, neg_w, coefficient);
    })) return;
    const auto inputs = input.as_vector().array();
    const auto targets = target.as_vector().array();
    input_delta.as_vector().array()
        = (targets == 1.0f).select(pos_w * (inputs - targets), neg_w * (inputs - targets)) * coefficient;
}

void binary_cross_entropy(const TensorView& input, const TensorView& target, type& error, float* workspace_device)
{
    if (TRY_GPU_DISPATCH(input, [&](auto tag) {
        using T = decltype(tag);
        binary_cross_entropy_cuda<T>(input.size(),
            workspace_device, target.as<T>(), input.as<T>(), EPSILON);
        error = sum_abs_cuda(workspace_device, input.size()) / input.shape[0];
    })) return;
    const Index samples_number = input.shape[0];

    const MatrixMap outputs = input.as_matrix();
    const MatrixMap targets = target.as_matrix();

    const auto clamped_outputs = outputs.array().cwiseMax(EPSILON).cwiseMin(1.0f - EPSILON);

    error = -(targets.array() * clamped_outputs.log() + (1.0f - targets.array()) * (1.0f - clamped_outputs).log()).sum()
            / to_type(samples_number);

    if(isnan(error) || isinf(error)) error = 10.0f;
}

void categorical_cross_entropy(const TensorView& input, const TensorView& target, type& error, float* workspace_device)
{
    if (TRY_GPU_DISPATCH(input, [&](auto tag) {
        using T = decltype(tag);
        multiple_cross_entropy_cuda<T>(input.size(),
            workspace_device, target.as<T>(), input.as<T>(), EPSILON);
        error = sum_abs_cuda(workspace_device, input.size()) / input.shape[0];
    })) return;
    const Index samples_number = input.shape[0];

    const MatrixMap outputs = input.as_matrix();
    const MatrixMap targets = target.as_matrix();

    error = (targets.array() * (outputs.array() + EPSILON).log()).sum() / to_type(-samples_number);

    if(isnan(error) || isinf(error)) error = type(10);
}

void cross_entropy_gradient(const TensorView& input, const TensorView& target, TensorView& input_delta)
{
    if (TRY_GPU_DISPATCH(input, [&](auto tag) {
        using T = decltype(tag);
        const Index num_classes = input.shape.back();
        const float scale = 1.0f / static_cast<float>(input.shape[0]);
        if (num_classes == 1)
            binary_cross_entropy_gradient_cuda<T>(input.size(),
                input_delta.as<T>(), target.as<T>(), input.as<T>(), EPSILON, scale);
        else
            multiple_cross_entropy_gradient_cuda<T>(input.size(),
                input_delta.as<T>(), target.as<T>(), input.as<T>(), scale);
    })) return;
    const Index samples_number = input.shape[0];
    const Index num_classes = input.shape.back();

    const MatrixMap outputs = input.as_matrix();
    const MatrixMap targets = target.as_matrix();
    MatrixMap gradients = input_delta.as_matrix();

    if(num_classes == 1)
        gradients.array() = (-targets.array() / (outputs.array() + EPSILON)
                             + (1.0f - targets.array()) / (1.0f - outputs.array() + EPSILON))
                            / to_type(samples_number);
    else
        gradients = (outputs - targets) / to_type(samples_number);
}

void minkowski_error(const TensorView& input, const TensorView& target, type p, type& error, float* workspace_device)
{
    if (Device::instance().is_gpu())
        throw runtime_error("minkowski_error: GPU implementation not available.");

    (void)workspace_device;
    const Index batch_size = input.shape[0];
    error = (input.as_vector() - target.as_vector()).array().abs().pow(p).sum() / to_type(p * batch_size);
}

void minkowski_error_gradient(const TensorView& input, const TensorView& target, type p, TensorView& input_delta)
{
    if (Device::instance().is_gpu())
        throw runtime_error("minkowski_error_gradient: GPU implementation not available.");

    const Index batch_size = input.shape[0];
    const auto difference = (input.as_vector() - target.as_vector()).array();
    input_delta.as_vector().array() = (1.0f / to_type(batch_size)) * difference.sign() * (difference.abs() + EPSILON).pow(p - 1.0f);
}

void cross_entropy_3d(const TensorView& input, const TensorView& target, type& error,
                      Index& active_tokens_out, Index& correct_tokens_out, float* errors_device)
{
    const Index vocabulary_size = input.shape.back();
    const Index sequence_length = input.shape[input.get_rank() - 2];
    const Index batch_size = input.size() / (sequence_length * vocabulary_size);

    if (TRY_GPU_DISPATCH(input, [&](auto tag) {
        using T = decltype(tag);
        const size_t token_count = batch_size * sequence_length;

        float* valid_mask_device   = errors_device + token_count;
        float* correct_mask_device = errors_device + 2 * token_count;

        cross_entropy_3d_multiple_forward_cuda<T>(token_count, to_int(vocabulary_size),
            input.as<T>(), target.as<float>(),
            errors_device, valid_mask_device, correct_mask_device, EPSILON);

        const float sum_loss     = sum_abs_cuda(errors_device,       token_count);
        const float active_count = sum_abs_cuda(valid_mask_device,   token_count);
        const float correct_count = sum_abs_cuda(correct_mask_device, token_count);

        active_tokens_out = static_cast<Index>(active_count);
        correct_tokens_out = static_cast<Index>(correct_count);
        error = active_count > 0 ? sum_loss / active_count : type(0);
    })) return;

    (void)errors_device;

    const Index token_count = batch_size * sequence_length;
    const MatrixMap outputs_flat = input.as_flat_matrix();
    const VectorMap targets_flat = target.as_vector();

    type total_log_loss = 0;
    Index active_tokens = 0;
    Index correct_tokens = 0;

    #pragma omp parallel for reduction(+:total_log_loss, active_tokens, correct_tokens)
    for(Index t = 0; t < token_count; ++t)
    {
        const Index target_index = static_cast<Index>(targets_flat(t));
        if(target_index > 0 && target_index < vocabulary_size)
        {
            total_log_loss -= log(outputs_flat(t, target_index) + EPSILON);
            ++active_tokens;

            Index best_index = 0;
            type best_value = outputs_flat(t, 0);
            for(Index k = 1; k < vocabulary_size; ++k)
            {
                if(outputs_flat(t, k) > best_value)
                {
                    best_value = outputs_flat(t, k);
                    best_index = k;
                }
            }
            if(best_index == target_index) ++correct_tokens;
        }
    }

    active_tokens_out = active_tokens;
    correct_tokens_out = correct_tokens;
    error = active_tokens > 0 ? total_log_loss / to_type(active_tokens) : type(0);
}

void cross_entropy_3d_gradient(const TensorView& input, const TensorView& target, TensorView& input_delta,
                               Index active_tokens_count)
{
    const Index vocabulary_size = input.shape.back();
    const Index sequence_length = input.shape[input.get_rank() - 2];
    const Index batch_size = input.size() / (sequence_length * vocabulary_size);

    if (TRY_GPU_DISPATCH(input, [&](auto tag) {
        using T = decltype(tag);
        const float scale = active_tokens_count > 0 ? 1.0f / static_cast<float>(active_tokens_count) : 0.0f;
        const size_t total = static_cast<size_t>(batch_size) * sequence_length * vocabulary_size;
        cross_entropy_3d_multiple_backward_cuda<T>(total, to_int(vocabulary_size),
            input.as<T>(), target.as<float>(), input_delta.as<T>(), scale);
    })) return;
    const Index token_count = batch_size * sequence_length;
    MatrixMap gradients_flat = input_delta.as_flat_matrix();
    const MatrixMap outputs_flat = input.as_flat_matrix();
    const VectorMap targets_flat = target.as_vector();

    const type scale = active_tokens_count > 0 ? type(1) / to_type(active_tokens_count) : type(0);

    #pragma omp parallel for
    for(Index t = 0; t < token_count; ++t)
    {
        const Index target_index = static_cast<Index>(targets_flat(t));
        if(target_index > 0 && target_index < vocabulary_size)
        {
            gradients_flat.row(t).noalias() = scale * outputs_flat.row(t);
            gradients_flat(t, target_index) -= scale;
        }
        else
        {
            gradients_flat.row(t).setZero();
        }
    }
}

void l1_regularization(const TensorView& parameters, type lambda, type& penalty)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        penalty = lambda * sum_abs_cuda(parameters.data, parameters.size());
        return;
    }
#endif
    penalty = lambda * parameters.as_vector().lpNorm<1>();
}

void l1_regularization_gradient(const TensorView& parameters, type lambda, TensorView& gradient)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        l1_gradient_cuda<float>(parameters.size(), gradient.as<float>(), parameters.as<float>(), lambda);
        return;
    }
#endif
    gradient.as_vector().array() += lambda * parameters.as_vector().array().sign();
}

void l2_regularization(const TensorView& parameters, type lambda, type& penalty)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        penalty = 0.5f * lambda * squared_norm_cuda(parameters.data, parameters.size());
        return;
    }
#endif
    penalty = 0.5f * lambda * parameters.as_vector().squaredNorm();
}

void l2_regularization_gradient(const TensorView& parameters, type lambda, TensorView& gradient)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const int total_size = to_int(parameters.size());
        CHECK_CUBLAS(cublasAxpyEx(Device::get_cublas_handle(), total_size,
                                  &lambda, CUDA_REDUCTION_DTYPE,
                                  parameters.data, CUDA_ACTIVATION_DTYPE, 1,
                                  gradient.data,   CUDA_ACTIVATION_DTYPE, 1,
                                  CUDA_REDUCTION_DTYPE));
        return;
    }
#endif
    gradient.as_vector().noalias() += lambda * parameters.as_vector();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.