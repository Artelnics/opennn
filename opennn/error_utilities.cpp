//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E R R O R   U T I L I T I E S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "error_utilities.h"
#ifdef OPENNN_WITH_CUDA
#include "kernel.cuh"
#endif

namespace opennn
{

#ifdef OPENNN_WITH_CUDA

// (input - target), sum of squares of the result, returned as a scalar.
// Stores the diff in `workspace`.
static float sum_squared_diff_cuda(const TensorView& input, const TensorView& target, float* workspace)
{
    const int total_size = to_int(input.size());
    CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(), Device::get_operator_sum_descriptor(),
                              &one, input.get_descriptor(), input.data,
                              &minus_one, target.get_descriptor(), target.data,
                              &zero, input.get_descriptor(), workspace));
    float sum_squared = 0.0f;
    CHECK_CUBLAS(cublasSdot(Device::get_cublas_handle(), total_size, workspace, 1, workspace, 1, &sum_squared));
    return sum_squared;
}

// output = scale * (input - target)
static void scaled_diff_cuda(const TensorView& input, const TensorView& target, float scale, TensorView& output)
{
    const int total_size = to_int(input.size());
    CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(), Device::get_operator_sum_descriptor(),
                              &one, input.get_descriptor(), input.data,
                              &minus_one, target.get_descriptor(), target.data,
                              &zero, output.get_descriptor(), output.data));
    CHECK_CUBLAS(cublasSscal(Device::get_cublas_handle(), total_size, &scale, output.data, 1));
}

// Sum of absolute values. For inputs known to be non-negative (per-element losses,
// validity masks, L1 norms) this equals the plain sum.
static float sum_abs_cuda(const float* data, Index n)
{
    float sum = 0.0f;
    CHECK_CUBLAS(cublasSasum(Device::get_cublas_handle(), to_int(n), data, 1, &sum));
    return sum;
}

// Sum of squares (equivalently, squared L2-norm).
static float squared_norm_cuda(const float* data, Index n)
{
    float dot = 0.0f;
    CHECK_CUBLAS(cublasSdot(Device::get_cublas_handle(), to_int(n), data, 1, data, 1, &dot));
    return dot;
}

#endif

void mean_squared_error(const TensorView& input, const TensorView& target, type& error, float* workspace_device)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        error = sum_squared_diff_cuda(input, target, workspace_device) / to_int(input.size());
        return;
    }
#endif
    error = (input.as_vector() - target.as_vector()).squaredNorm() / to_type(input.size());
}

void mean_squared_error_gradient(const TensorView& input, const TensorView& target, TensorView& input_gradient)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        scaled_diff_cuda(input, target, 2.0f / to_int(input.size()), input_gradient);
        return;
    }
#endif
    input_gradient.as_vector().noalias() = (input.as_vector() - target.as_vector()) * (2.0f / to_type(input.size()));
}

void normalized_squared_error(const TensorView& input, const TensorView& target, type coefficient, type& error, float* workspace_device)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        error = sum_squared_diff_cuda(input, target, workspace_device) / (coefficient + EPSILON);
        return;
    }
#endif
    error = (input.as_vector() - target.as_vector()).squaredNorm() / (coefficient + EPSILON);
}

void normalized_squared_error_gradient(const TensorView& input, const TensorView& target, type coefficient, TensorView& input_gradient)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        scaled_diff_cuda(input, target, 2.0f / (static_cast<float>(coefficient) + EPSILON), input_gradient);
        return;
    }
#endif
    input_gradient.as_vector().noalias() = (input.as_vector() - target.as_vector()) * (2.0f / (coefficient + EPSILON));
}

void weighted_squared_error(const TensorView& input, const TensorView& target, type pos_w, type neg_w, type& error, float* workspace_device)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        weighted_squared_error_cuda(input.size(), workspace_device, target.data, input.data, pos_w, neg_w);
        error = sum_abs_cuda(workspace_device, input.size());
        return;
    }
#endif
    const auto inputs = input.as_vector().array();
    const auto targets = target.as_vector().array();
    const auto squared_error = (inputs - targets).square();
    error = (targets == 1.0f).select(squared_error * pos_w, squared_error * neg_w).sum();
}

void weighted_squared_error_gradient(const TensorView& input, const TensorView& target, type pos_w, type neg_w, type coefficient, TensorView& input_gradient)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        weighted_squared_error_gradient_cuda(input.size(), input_gradient.data, target.data, input.data, pos_w, neg_w, coefficient);
        return;
    }
#endif
    const auto inputs = input.as_vector().array();
    const auto targets = target.as_vector().array();
    input_gradient.as_vector().array()
        = (targets == 1.0f).select(2.0f * pos_w * (inputs - targets), 2.0f * neg_w * (inputs - targets)) * coefficient;
}

void binary_cross_entropy(const TensorView& input, const TensorView& target, type& error, float* workspace_device)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        binary_cross_entropy_cuda(input.size(), workspace_device, target.data, input.data, EPSILON);
        error = sum_abs_cuda(workspace_device, input.size()) / input.shape[0];
        return;
    }
#endif
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
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        multiple_cross_entropy_cuda(input.size(), workspace_device, target.data, input.data, EPSILON);
        error = sum_abs_cuda(workspace_device, input.size()) / input.shape[0];
        return;
    }
#endif
    const Index samples_number = input.shape[0];

    const MatrixMap outputs = input.as_matrix();
    const MatrixMap targets = target.as_matrix();

    error = (targets.array() * (outputs.array() + EPSILON).log()).sum() / to_type(-samples_number);

    if(isnan(error)) throw runtime_error("Error is NAN.");
}

void cross_entropy_gradient(const TensorView& input, const TensorView& target, TensorView& input_gradient)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const Index num_classes = input.shape.back();
        const float scale = 1.0f / static_cast<float>(input.shape[0]);

        if(num_classes == 1)
            binary_cross_entropy_gradient_cuda(input.size(), input_gradient.data, target.data, input.data, EPSILON, scale);
        else
            multiple_cross_entropy_gradient_cuda(input.size(), input_gradient.data, target.data, input.data, scale);
        return;
    }
#endif
    const Index samples_number = input.shape[0];
    const Index num_classes = input.shape.back();

    const MatrixMap outputs = input.as_matrix();
    const MatrixMap targets = target.as_matrix();
    MatrixMap gradients = input_gradient.as_matrix();

    if(num_classes == 1)
        gradients.array() = (-targets.array() / (outputs.array() + EPSILON)
                             + (1.0f - targets.array()) / (1.0f - outputs.array() + EPSILON))
                            / to_type(samples_number);
    else
        gradients = (outputs - targets) / to_type(samples_number);
}

void minkowski_error(const TensorView& input, const TensorView& target, type p, type& error, float* workspace_device)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu())
        throw runtime_error("minkowski_error: GPU implementation not available.");
#endif
    (void)workspace_device;
    const Index total_size = input.size();
    error = (input.as_vector() - target.as_vector()).array().abs().pow(p).sum() / to_type(total_size);
}

void minkowski_error_gradient(const TensorView& input, const TensorView& target, type p, TensorView& input_gradient)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu())
        throw runtime_error("minkowski_error_gradient: GPU implementation not available.");
#endif
    const Index total_size = input.size();
    const auto difference = (input.as_vector() - target.as_vector()).array();
    input_gradient.as_vector().array() = (p / to_type(total_size)) * difference.sign() * (difference.abs() + EPSILON).pow(p - 1.0f);
}

void cross_entropy_3d(const TensorView& input, const TensorView& target, type& error,
                      Index& active_tokens_out, float* errors_device)
{
    const Index vocabulary_size = input.shape.back();
    const Index sequence_length = input.shape[input.get_rank() - 2];
    const Index batch_size = input.size() / (sequence_length * vocabulary_size);

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const size_t token_count = batch_size * sequence_length;

        // Reuse the tail of errors_device as the valid-token mask buffer.
        // errors_device is sized for batch_size * sequence_length * vocabulary_size floats;
        // we only need 2 * token_count here (errors + mask), and vocabulary_size >= 2 in any real use.
        float* valid_mask_device = errors_device + token_count;

        cross_entropy_3d_multiple_forward_cuda(token_count, to_int(vocabulary_size),
                                               input.data, target.data,
                                               errors_device, valid_mask_device, EPSILON);

        const float sum_loss    = sum_abs_cuda(errors_device,     token_count);
        const float active_count = sum_abs_cuda(valid_mask_device, token_count);

        active_tokens_out = static_cast<Index>(active_count);
        error = active_count > 0 ? sum_loss / active_count : type(0);
        return;
    }
#endif
    (void)errors_device;

    const Index token_count = batch_size * sequence_length;
    const MatrixMap outputs_flat = input.as_flat_matrix();
    const VectorMap targets_flat = target.as_vector();

    type total_log_loss = 0;
    Index active_tokens = 0;

    #pragma omp parallel for reduction(+:total_log_loss, active_tokens)
    for(Index t = 0; t < token_count; ++t)
    {
        const Index target_index = static_cast<Index>(targets_flat(t));
        if(target_index > 0 && target_index < vocabulary_size)
        {
            total_log_loss -= log(outputs_flat(t, target_index) + EPSILON);
            ++active_tokens;
        }
    }

    active_tokens_out = active_tokens;
    error = active_tokens > 0 ? total_log_loss / to_type(active_tokens) : type(0);
}

void cross_entropy_3d_gradient(const TensorView& input, const TensorView& target, TensorView& input_gradient,
                               Index active_tokens_count)
{
    const Index vocabulary_size = input.shape.back();
    const Index sequence_length = input.shape[input.get_rank() - 2];
    const Index batch_size = input.size() / (sequence_length * vocabulary_size);

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const float scale = active_tokens_count > 0 ? 1.0f / static_cast<float>(active_tokens_count) : 0.0f;

        const size_t total = static_cast<size_t>(batch_size) * sequence_length * vocabulary_size;
        cross_entropy_3d_multiple_backward_cuda(total, to_int(vocabulary_size), input.data, target.data, input_gradient.data, scale);
        return;
    }
#endif
    const Index token_count = batch_size * sequence_length;
    MatrixMap gradients_flat = input_gradient.as_flat_matrix();
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
        l1_gradient_cuda(parameters.size(), gradient.data, parameters.data, lambda);
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
        CHECK_CUBLAS(cublasSaxpy(Device::get_cublas_handle(), total_size, &lambda, parameters.data, 1, gradient.data, 1));
        return;
    }
#endif
    gradient.as_vector().noalias() += lambda * parameters.as_vector();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.