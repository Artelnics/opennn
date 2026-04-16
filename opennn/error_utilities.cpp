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

void mean_squared_error(const TensorView& input, const TensorView& target, type& error, float* workspace_device)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const int n = static_cast<int>(input.size());
        CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(), Device::get_operator_sum_descriptor(),
                                  &one, input.get_descriptor(), input.data,
                                  &minus_one, target.get_descriptor(), target.data,
                                  &zero, input.get_descriptor(), workspace_device));
        float sse = 0.0f;
        CHECK_CUBLAS(cublasSdot(Device::get_cublas_handle(), n, workspace_device, 1, workspace_device, 1, &sse));
        error = sse / n;
        return;
    }
#endif
    const Index size = input.size();
    error = (input.as_vector() - target.as_vector()).squaredNorm() / static_cast<type>(size);
}

void mean_squared_error_gradient(const TensorView& input, const TensorView& target, TensorView& input_gradient)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const int n = static_cast<int>(input.size());
        const float scale = 2.0f / n;
        CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(), Device::get_operator_sum_descriptor(),
                                  &one, input.get_descriptor(), input.data,
                                  &minus_one, target.get_descriptor(), target.data,
                                  &zero, input_gradient.get_descriptor(), input_gradient.data));
        CHECK_CUBLAS(cublasSscal(Device::get_cublas_handle(), n, &scale, input_gradient.data, 1));
        return;
    }
#endif
    const Index size = input.size();
    input_gradient.as_vector().array() = (input.as_vector().array() - target.as_vector().array()) * (2.0f / static_cast<type>(size));
}

void normalized_squared_error(const TensorView& input, const TensorView& target, type coefficient, type& error, float* workspace_device)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const int n = static_cast<int>(input.size());
        CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(), Device::get_operator_sum_descriptor(),
                                  &one, input.get_descriptor(), input.data,
                                  &minus_one, target.get_descriptor(), target.data,
                                  &zero, input.get_descriptor(), workspace_device));
        float sse = 0.0f;
        CHECK_CUBLAS(cublasSdot(Device::get_cublas_handle(), n, workspace_device, 1, workspace_device, 1, &sse));
        error = sse / (coefficient + EPSILON);
        return;
    }
#endif
    error = (input.as_vector() - target.as_vector()).squaredNorm() / (coefficient + EPSILON);
}

void normalized_squared_error_gradient(const TensorView& input, const TensorView& target, type coefficient, TensorView& input_gradient)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const int n = static_cast<int>(input.size());
        const float scale = 2.0f / (static_cast<float>(coefficient) + EPSILON);
        CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(), Device::get_operator_sum_descriptor(),
                                  &one, input.get_descriptor(), input.data,
                                  &minus_one, target.get_descriptor(), target.data,
                                  &zero, input_gradient.get_descriptor(), input_gradient.data));
        CHECK_CUBLAS(cublasSscal(Device::get_cublas_handle(), n, &scale, input_gradient.data, 1));
        return;
    }
#endif
    input_gradient.as_vector().array() = (input.as_vector().array() - target.as_vector().array()) * (2.0f / (coefficient + EPSILON));
}

void weighted_squared_error(const TensorView& input, const TensorView& target, type pos_w, type neg_w, type& error, float* workspace_device)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        calculate_weighted_squared_error_cuda(input.size(), workspace_device, target.data, input.data, pos_w, neg_w);
        float result = 0.0f;
        CHECK_CUBLAS(cublasSasum(Device::get_cublas_handle(), static_cast<int>(input.size()), workspace_device, 1, &result));
        error = result;
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
        calculate_weighted_squared_error_delta_cuda(input.size(), input_gradient.data, target.data, input.data, pos_w, neg_w, coefficient);
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
        calculate_binary_cross_entropy_cuda(input.size(), workspace_device, target.data, input.data, EPSILON);
        float sum_ce = 0.0f;
        CHECK_CUBLAS(cublasSasum(Device::get_cublas_handle(), static_cast<int>(input.size()), workspace_device, 1, &sum_ce));
        error = sum_ce / input.shape[0];
        return;
    }
#endif
    const Index samples_number = input.shape[0];

    const MatrixMap outputs(input.data, samples_number, input.size() / samples_number);
    const MatrixMap targets(target.data, samples_number, target.size() / samples_number);

    auto y = outputs.array().cwiseMax(EPSILON).cwiseMin(1.0f - EPSILON);

    MatrixR ce = targets.array() * y.log() + (1.0f - targets.array()) * (1.0f - y).log();

    error = ce.sum() / static_cast<type>(-samples_number);

    if(isnan(error) || isinf(error)) error = 10.0f;
}

void categorical_cross_entropy(const TensorView& input, const TensorView& target, type& error, float* workspace_device)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        calculate_multiple_cross_entropy_cuda(input.size(), workspace_device, target.data, input.data, EPSILON);
        float sum_ce = 0.0f;
        CHECK_CUBLAS(cublasSasum(Device::get_cublas_handle(), static_cast<int>(input.size()), workspace_device, 1, &sum_ce));
        error = sum_ce / input.shape[0];
        return;
    }
#endif
    const Index samples_number = input.shape[0];

    const MatrixMap outputs(input.data, samples_number, input.size() / samples_number);
    const MatrixMap targets(target.data, samples_number, target.size() / samples_number);

    error = (targets.array() * (outputs.array() + EPSILON).log()).sum() / static_cast<type>(-samples_number);

    if(isnan(error)) throw runtime_error("Error is NAN.");
}

void cross_entropy_gradient(const TensorView& input, const TensorView& target, TensorView& input_gradient)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const Index num_classes = input.shape.back();
        const float scale = 1.0f / static_cast<float>(input.shape[0]);

        if(num_classes == 1)
            calculate_binary_cross_entropy_delta_cuda(input.size(), input_gradient.data, target.data, input.data, EPSILON, scale);
        else
            calculate_multiple_cross_entropy_delta_cuda(input.size(), input_gradient.data, target.data, input.data, scale);
        return;
    }
#endif
    const Index samples_number = input.shape[0];
    const Index num_classes = input.shape.back();

    const MatrixMap outputs(input.data, samples_number, num_classes);
    const MatrixMap targets(target.data, samples_number, num_classes);
    MatrixMap gradients(input_gradient.data, samples_number, num_classes);

    if(num_classes == 1)
        gradients.array() = (-targets.array() / (outputs.array() + EPSILON)
                             + (1.0f - targets.array()) / (1.0f - outputs.array() + EPSILON))
                            / static_cast<type>(samples_number);
    else
        gradients = (outputs - targets) / static_cast<type>(samples_number);
}

void minkowski_error(const TensorView& input, const TensorView& target, type p, type& error, float* workspace_device)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu())
        throw runtime_error("minkowski_error: GPU implementation not available.");
#endif
    (void)workspace_device;
    const Index size = input.size();
    error = (input.as_vector() - target.as_vector()).array().abs().pow(p).sum() / static_cast<type>(size);
}

void minkowski_error_gradient(const TensorView& input, const TensorView& target, type p, TensorView& input_gradient)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu())
        throw runtime_error("minkowski_error_gradient: GPU implementation not available.");
#endif
    const Index size = input.size();
    const auto diff = input.as_vector().array() - target.as_vector().array();
    input_gradient.as_vector().array() = (p / static_cast<type>(size)) * diff.sign() * (diff.abs() + EPSILON).pow(p - 1.0f);
}

void cross_entropy_3d(const TensorView& input, const TensorView& target, type& error,
                      Index& active_tokens_out, float* errors_device)
{
    const Index vocabulary_size = input.shape.back();
    const Index sequence_length = input.shape[input.get_rank() - 2];
    const Index batch_size = input.size() / (sequence_length * vocabulary_size);

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const int B = static_cast<int>(batch_size);
        const int S = static_cast<int>(sequence_length);
        const int V = static_cast<int>(vocabulary_size);
        const size_t token_count = batch_size * sequence_length;

        // Per-token losses into errors_device (pre-allocated in BackPropagation)
        cross_entropy_3d_multiple_forward_cuda(token_count, B, S, V, input.data, target.data, errors_device, EPSILON);

        // Sum losses with cublasSasum
        float sum_loss = 0;
        CHECK_CUBLAS(cublasSasum(Device::get_cublas_handle(), static_cast<int>(token_count), errors_device, 1, &sum_loss));

        // Count active tokens on host (small copy: only token_count floats)
        vector<float> targets_host(token_count);
        CHECK_CUDA(cudaMemcpy(targets_host.data(), target.data, token_count * sizeof(float), cudaMemcpyDeviceToHost));

        Index active = 0;
        for(size_t i = 0; i < token_count; i++)
            if(static_cast<Index>(targets_host[i]) > 0 && static_cast<Index>(targets_host[i]) < V)
                active++;

        active_tokens_out = active;
        error = active > 0 ? sum_loss / static_cast<float>(active) : type(0);
        return;
    }
#endif
    (void)errors_device;

    const TensorMap3 outputs(input.data, batch_size, sequence_length, vocabulary_size);
    const MatrixMap targets(target.data, batch_size, sequence_length);

    type total_log_loss = 0;
    Index active_tokens = 0;

    #pragma omp parallel for reduction(+:total_log_loss, active_tokens)
    for(Index i = 0; i < batch_size; ++i)
        for(Index j = 0; j < sequence_length; ++j)
        {
            const Index idx = static_cast<Index>(targets(i, j));
            if(idx > 0 && idx < vocabulary_size)
            {
                total_log_loss -= log(outputs(i, j, idx) + EPSILON);
                active_tokens++;
            }
        }

    active_tokens_out = active_tokens;
    error = active_tokens > 0 ? total_log_loss / static_cast<type>(active_tokens) : type(0);
}

void cross_entropy_3d_gradient(const TensorView& input, const TensorView& target, TensorView& input_gradient,
                               Index active_tokens_count)
{
    const Index vocabulary_size = input.shape.back();
    const Index sequence_length = input.shape[input.get_rank() - 2];
    const Index batch_size = input.size() / (sequence_length * vocabulary_size);

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const int B = static_cast<int>(batch_size);
        const int S = static_cast<int>(sequence_length);
        const int V = static_cast<int>(vocabulary_size);

        const float scale = active_tokens_count > 0 ? 1.0f / static_cast<float>(active_tokens_count) : 0.0f;

        const size_t total = static_cast<size_t>(B) * S * V;
        cross_entropy_3d_multiple_backward_cuda(total, B, S, V, input.data, target.data, input_gradient.data, scale);
        return;
    }
#endif
    const TensorMap3 outputs(input.data, batch_size, sequence_length, vocabulary_size);
    const MatrixMap targets(target.data, batch_size, sequence_length);
    TensorMap3 gradients(input_gradient.data, batch_size, sequence_length, vocabulary_size);

    const type scale = active_tokens_count > 0 ? type(1) / static_cast<type>(active_tokens_count) : type(0);

    #pragma omp parallel for
    for(Index i = 0; i < batch_size; ++i)
        for(Index j = 0; j < sequence_length; ++j)
        {
            const Index idx = static_cast<Index>(targets(i, j));

            if(idx > 0 && idx < vocabulary_size)
            {
                for(Index k = 0; k < vocabulary_size; ++k)
                    gradients(i, j, k) = (k == idx)
                        ? (outputs(i, j, k) - type(1)) * scale
                        : outputs(i, j, k) * scale;
            }
            else
            {
                for(Index k = 0; k < vocabulary_size; ++k)
                    gradients(i, j, k) = type(0);
            }
        }
}

void l1_regularization(const TensorView& parameters, type lambda, type& penalty)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        float sum_abs = 0.0f;
        CHECK_CUBLAS(cublasSasum(Device::get_cublas_handle(), static_cast<int>(parameters.size()), parameters.data, 1, &sum_abs));
        penalty = lambda * sum_abs;
        return;
    }
#endif
    penalty = lambda * parameters.as_vector().lpNorm<1>();
}

void l1_regularization_gradient(const TensorView& parameters, type lambda, TensorView& gradient)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        apply_l1_gradient_cuda(parameters.size(), gradient.data, parameters.data, lambda);
        return;
    }
#endif
    gradient.as_vector().array() += lambda * parameters.as_vector().array().sign();
}

void l2_regularization(const TensorView& parameters, type lambda, type& penalty)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        float dot_product = 0.0f;
        const int n = static_cast<int>(parameters.size());
        CHECK_CUBLAS(cublasSdot(Device::get_cublas_handle(), n, parameters.data, 1, parameters.data, 1, &dot_product));
        penalty = 0.5f * lambda * dot_product;
        return;
    }
#endif
    penalty = 0.5f * lambda * parameters.as_vector().squaredNorm();
}

void l2_regularization_gradient(const TensorView& parameters, type lambda, TensorView& gradient)
{
#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const int n = static_cast<int>(parameters.size());
        CHECK_CUBLAS(cublasSaxpy(Device::get_cublas_handle(), n, &lambda, parameters.data, 1, gradient.data, 1));
        return;
    }
#endif
    gradient.as_vector().noalias() += lambda * parameters.as_vector();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.