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

#ifdef OPENNN_HAS_CUDA

static float sum_squared_diff_cuda(const TensorView& input, const TensorView& target, float* workspace)
{
    const int total_size = to_int(input.size());

    input.dispatch([&](auto tag) {
        using TIn = decltype(tag);
        diff_to_fp32_cuda<TIn>(input.size(), input.as<TIn>(), target.as_float(), workspace);
    });

    float sum_squared = 0.0f;
    CHECK_CUBLAS(cublasSdot(Backend::get_cublas_handle(), total_size,
                            workspace, 1, workspace, 1, &sum_squared));
    return sum_squared;
}

static void scaled_diff_cuda(const TensorView& input, const TensorView& target, float scale, TensorView& output)
{
    visit_type_pair<Type::FP32, Type::BF16>(input.type, output.type, [&](auto in, auto out) {
        using TIn  = typename decltype(in)::type;
        using TOut = typename decltype(out)::type;
        scaled_diff_cuda_typed<TIn, TOut>(input.size(),
                                          input.as<TIn>(),
                                          target.as_float(),
                                          scale,
                                          output.as<TOut>());
    });
}

static float sum_abs_cuda(const float* data, Index size)
{
    float sum = 0.0f;
    CHECK_CUBLAS(cublasSasum(Backend::get_cublas_handle(), to_int(size), data, 1, &sum));
    return sum;
}

static float squared_norm_cuda(const float* data, Index size)
{
    float dot = 0.0f;
    CHECK_CUBLAS(cublasSdot(Backend::get_cublas_handle(), to_int(size),
                            data, 1, data, 1, &dot));
    return dot;
}

#endif

void mean_squared_error(const TensorView& input, const TensorView& target, float& error, float* workspace_device)
{
    const Index batch_size = input.shape[0];
    IF_GPU({
        error = sum_squared_diff_cuda(input, target, workspace_device) / to_int(2 * batch_size);
        return;
    });
    error = (input.as_vector() - target.as_vector()).squaredNorm() / to_type(2 * batch_size);
}

void mean_squared_error_gradient(const TensorView& input, const TensorView& target, TensorView& input_delta)
{
    const Index batch_size = input.shape[0];
    IF_GPU({
        scaled_diff_cuda(input, target, 1.0f / to_int(batch_size), input_delta);
        return;
    });
    input_delta.as_vector().noalias() = (input.as_vector() - target.as_vector()) / to_type(batch_size);
}

void normalized_squared_error(const TensorView& input, const TensorView& target, float coefficient, float& error, float* workspace_device)
{
    IF_GPU({
        error = sum_squared_diff_cuda(input, target, workspace_device) / (2.0f * (coefficient + EPSILON));
        return;
    });
    error = (input.as_vector() - target.as_vector()).squaredNorm() / (2.0f * (coefficient + EPSILON));
}

void normalized_squared_error_gradient(const TensorView& input, const TensorView& target, float coefficient, TensorView& input_delta)
{
    IF_GPU({
        scaled_diff_cuda(input, target, 1.0f / (static_cast<float>(coefficient) + EPSILON), input_delta);
        return;
    });
    input_delta.as_vector().noalias() = (input.as_vector() - target.as_vector()) / (coefficient + EPSILON);
}

void weighted_squared_error(const TensorView& input, const TensorView& target, float pos_w, float neg_w, float& error, float* workspace_device)
{
    if (TRY_GPU_DISPATCH(input, [&](auto tag) {
        using T = decltype(tag);
        weighted_squared_error_cuda<T>(input.size(),
                                       workspace_device,
                                       target.as<float>(),
                                       input.as<T>(),
                                       pos_w,
                                       neg_w);

        error = 0.5f * sum_abs_cuda(workspace_device, input.size());
    })) return;

    const auto inputs = input.as_vector().array();
    const auto targets = target.as_vector().array();
    const auto squared_error = (inputs - targets).square();
    error = 0.5f * (targets == 1.0f).select(squared_error * pos_w, squared_error * neg_w).sum();
}

void weighted_squared_error_gradient(const TensorView& input, const TensorView& target, float pos_w, float neg_w, float coefficient, TensorView& input_delta)
{
    if (TRY_GPU_DISPATCH(input, [&](auto tag) {
        using T = decltype(tag);
        weighted_squared_error_gradient_cuda<T>(input.size(),
            input_delta.as<T>(), target.as<float>(), input.as<T>(), pos_w, neg_w, coefficient);
    })) return;

    const auto inputs = input.as_vector().array();
    const auto targets = target.as_vector().array();
    const auto difference = inputs - targets;
    input_delta.as_vector().array()
        = (targets == 1.0f).select(pos_w * difference, neg_w * difference) * coefficient;
}

void binary_cross_entropy(const TensorView& input, const TensorView& target, float& error, float* workspace_device)
{
    if (TRY_GPU_DISPATCH(input, [&](auto tag) {
        using T = decltype(tag);
        binary_cross_entropy_cuda<T>(input.size(),
            workspace_device, target.as<float>(), input.as<T>(), EPSILON);
        error = sum_abs_cuda(workspace_device, input.size()) / input.shape[0];
    })) return;
    const Index samples_number = input.shape[0];

    const MatrixMap outputs = input.as_matrix();
    const MatrixMap targets = target.as_matrix();

    const auto clamped_outputs = outputs.array().cwiseMax(EPSILON).cwiseMin(1.0f - EPSILON);

    error = -(targets.array() * clamped_outputs.log() + (1.0f - targets.array()) * (1.0f - clamped_outputs).log()).sum()
            / to_type(samples_number);

    if (isnan(error) || isinf(error)) error = 10.0f;
}

void categorical_cross_entropy(const TensorView& input, const TensorView& target, float& error, float* workspace_device)
{
    if (TRY_GPU_DISPATCH(input, [&](auto tag) {
        using T = decltype(tag);
        multiple_cross_entropy_cuda<T>(input.size(),
            workspace_device, target.as<float>(), input.as<T>(), EPSILON);
        error = sum_abs_cuda(workspace_device, input.size()) / input.shape[0];
    })) return;
    const Index samples_number = input.shape[0];

    const MatrixMap outputs = input.as_matrix();
    const MatrixMap targets = target.as_matrix();

    error = (targets.array() * (outputs.array() + EPSILON).log()).sum() / to_type(-samples_number);

    if (isnan(error) || isinf(error)) error = 10.0f;
}

void cross_entropy_gradient(const TensorView& input, const TensorView& target, TensorView& input_delta)
{
    if (TRY_GPU_DISPATCH(input, [&](auto tag) {
        using T = decltype(tag);
        const Index num_classes = input.shape.back();
        const float scale = 1.0f / static_cast<float>(input.shape[0]);
        if (num_classes == 1)
            binary_cross_entropy_gradient_cuda<T>(input.size(),
                input_delta.as<T>(), target.as<float>(), input.as<T>(), EPSILON, scale);
        else
            multiple_cross_entropy_gradient_cuda<T>(input.size(),
                input_delta.as<T>(), target.as<float>(), input.as<T>(), scale);
    })) return;
    const Index samples_number = input.shape[0];
    const Index num_classes = input.shape.back();

    const MatrixMap outputs = input.as_matrix();
    const MatrixMap targets = target.as_matrix();
    MatrixMap gradients = input_delta.as_matrix();

    if (num_classes == 1)
        gradients.array() = (-targets.array() / (outputs.array() + EPSILON)
                             + (1.0f - targets.array()) / (1.0f - outputs.array() + EPSILON))
                            / to_type(samples_number);
    else
        gradients = (outputs - targets) / to_type(samples_number);
}

void minkowski_error(const TensorView& input, const TensorView& target, float power, float& error, float* workspace_device)
{
    if (is_gpu())
        throw runtime_error("minkowski_error: GPU implementation not available.");

    (void)workspace_device;
    const Index batch_size = input.shape[0];
    error = (input.as_vector() - target.as_vector()).array().abs().pow(power).sum() / to_type(power * batch_size);
}

void minkowski_error_gradient(const TensorView& input, const TensorView& target, float power, TensorView& input_delta)
{
    if (is_gpu())
        throw runtime_error("minkowski_error_gradient: GPU implementation not available.");

    const Index batch_size = input.shape[0];
    const auto difference = (input.as_vector() - target.as_vector()).array();
    input_delta.as_vector().array() = (1.0f / to_type(batch_size)) * difference.sign() * (difference.abs() + EPSILON).pow(power - 1.0f);
}

void cross_entropy_3d(const TensorView& input, const TensorView& target, float& error,
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

        static Buffer device_results(Device::CUDA);
        device_results.grow_to(Index(3 * sizeof(float)));
        float* device_results_ptr = device_results.as<float>();

        cublasHandle_t handle = Backend::get_cublas_handle();
        CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
        CHECK_CUBLAS(cublasSasum(handle, to_int(token_count), errors_device,       1, device_results_ptr + 0));
        CHECK_CUBLAS(cublasSasum(handle, to_int(token_count), valid_mask_device,   1, device_results_ptr + 1));
        CHECK_CUBLAS(cublasSasum(handle, to_int(token_count), correct_mask_device, 1, device_results_ptr + 2));
        CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

        float host_results[3];
        CHECK_CUDA(cudaMemcpyAsync(host_results, device_results_ptr, 3 * sizeof(float),
                                   cudaMemcpyDeviceToHost, Backend::get_compute_stream()));
        CHECK_CUDA(cudaStreamSynchronize(Backend::get_compute_stream()));

        const float sum_loss      = host_results[0];
        const float active_count  = host_results[1];
        const float correct_count = host_results[2];

        active_tokens_out = static_cast<Index>(active_count);
        correct_tokens_out = static_cast<Index>(correct_count);
        error = active_count > 0 ? sum_loss / active_count : 0.0f;
    })) return;

    (void)errors_device;

    const Index token_count = batch_size * sequence_length;
    const MatrixMap outputs_flat = input.as_flat_matrix();
    const VectorMap targets_flat = target.as_vector();

    float total_log_loss = 0;
    Index active_tokens = 0;
    Index correct_tokens = 0;

    #pragma omp parallel for reduction(+:total_log_loss, active_tokens, correct_tokens)
    for (Index token_index = 0; token_index < token_count; ++token_index)
    {
        const Index target_index = static_cast<Index>(targets_flat(token_index));
        if (target_index > 0 && target_index < vocabulary_size)
        {
            total_log_loss -= log(outputs_flat(token_index, target_index) + EPSILON);
            ++active_tokens;

            Index best_index;
            outputs_flat.row(token_index).maxCoeff(&best_index);
            if (best_index == target_index) ++correct_tokens;
        }
    }

    active_tokens_out = active_tokens;
    correct_tokens_out = correct_tokens;
    error = active_tokens > 0 ? total_log_loss / to_type(active_tokens) : 0.0f;
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

    const float scale = active_tokens_count > 0 ? 1.0f / to_type(active_tokens_count) : 0.0f;

    #pragma omp parallel for
    for (Index token_index = 0; token_index < token_count; ++token_index)
    {
        const Index target_index = static_cast<Index>(targets_flat(token_index));
        if (target_index > 0 && target_index < vocabulary_size)
        {
            gradients_flat.row(token_index).noalias() = scale * outputs_flat.row(token_index);
            gradients_flat(token_index, target_index) -= scale;
        }
        else
        {
            gradients_flat.row(token_index).setZero();
        }
    }
}

void l1_regularization(const TensorView& parameters, float lambda, float& penalty)
{
    IF_GPU({
        penalty = lambda * sum_abs_cuda(parameters.as<float>(), parameters.size());
        return;
    });
    penalty = lambda * parameters.as_vector().lpNorm<1>();
}

void l1_regularization_gradient(const TensorView& parameters, float lambda, TensorView& gradient)
{
    IF_GPU({
        l1_gradient_cuda<float>(parameters.size(), gradient.as<float>(), parameters.as<float>(), lambda);
        return;
    });
    gradient.as_vector().array() += lambda * parameters.as_vector().array().sign();
}

void l2_regularization(const TensorView& parameters, float lambda, float& penalty)
{
    IF_GPU({
        penalty = 0.5f * lambda * squared_norm_cuda(parameters.as<float>(), parameters.size());
        return;
    });
    penalty = 0.5f * lambda * parameters.as_vector().squaredNorm();
}

void l2_regularization_gradient(const TensorView& parameters, float lambda, TensorView& gradient)
{
    IF_GPU({
        const int total_size = to_int(parameters.size());
        CHECK_CUBLAS(cublasAxpyEx(Backend::get_cublas_handle(), total_size,
                                  &lambda,         CUDA_R_32F,
                                  parameters.data, CUDA_R_32F, 1,
                                  gradient.data,   CUDA_R_32F, 1,
                                  CUDA_R_32F));
        return;
    });
    gradient.as_vector().noalias() += lambda * parameters.as_vector();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.