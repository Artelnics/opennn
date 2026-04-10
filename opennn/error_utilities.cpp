//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E R R O R   U T I L I T I E S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "error_utilities.h"
#ifdef CUDA
#include "kernel.cuh"
#endif

namespace opennn
{

void mean_squared_error(const TensorView& input, const TensorView& target, type& error, float* workspace_device)
{
#ifndef CUDA
    const Index size = input.size();
    error = (input.as_vector() - target.as_vector()).squaredNorm() / static_cast<type>(size);
#else
    const int n = static_cast<int>(input.size());
    CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(), Device::get_operator_sum_descriptor(),
                              &one, input.get_descriptor(), input.data,
                              &minus_one, target.get_descriptor(), target.data,
                              &zero, input.get_descriptor(), workspace_device));
    float sse = 0.0f;
    CHECK_CUBLAS(cublasSdot(Device::get_cublas_handle(), n, workspace_device, 1, workspace_device, 1, &sse));
    error = sse / n;
#endif
}

void mean_squared_error_gradient(const TensorView& input, const TensorView& target, TensorView& input_gradient)
{
#ifndef CUDA
    const Index size = input.size();
    input_gradient.as_vector().array() = (input.as_vector().array() - target.as_vector().array()) * (2.0f / static_cast<type>(size));
#else
    const int n = static_cast<int>(input.size());
    const float scale = 2.0f / n;
    CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(), Device::get_operator_sum_descriptor(),
                              &one, input.get_descriptor(), input.data,
                              &minus_one, target.get_descriptor(), target.data,
                              &zero, input_gradient.get_descriptor(), input_gradient.data));
    CHECK_CUBLAS(cublasSscal(Device::get_cublas_handle(), n, &scale, input_gradient.data, 1));
#endif
}

void normalized_squared_error(const TensorView& input, const TensorView& target, type coefficient, type& error, float* workspace_device)
{
#ifndef CUDA
    error = (input.as_vector() - target.as_vector()).squaredNorm() / (coefficient + EPSILON);
#else
    const int n = static_cast<int>(input.size());
    CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(), Device::get_operator_sum_descriptor(),
                              &one, input.get_descriptor(), input.data,
                              &minus_one, target.get_descriptor(), target.data,
                              &zero, input.get_descriptor(), workspace_device));
    float sse = 0.0f;
    CHECK_CUBLAS(cublasSdot(Device::get_cublas_handle(), n, workspace_device, 1, workspace_device, 1, &sse));
    error = sse / (coefficient + EPSILON);
#endif
}

void normalized_squared_error_gradient(const TensorView& input, const TensorView& target, type coefficient, TensorView& input_gradient)
{
#ifndef CUDA
    input_gradient.as_vector().array() = (input.as_vector().array() - target.as_vector().array()) * (2.0f / (coefficient + EPSILON));
#else
    const int n = static_cast<int>(input.size());
    const float scale = 2.0f / (static_cast<float>(coefficient) + EPSILON);
    CHECK_CUDNN(cudnnOpTensor(Device::get_cudnn_handle(), Device::get_operator_sum_descriptor(),
                              &one, input.get_descriptor(), input.data,
                              &minus_one, target.get_descriptor(), target.data,
                              &zero, input_gradient.get_descriptor(), input_gradient.data));
    CHECK_CUBLAS(cublasSscal(Device::get_cublas_handle(), n, &scale, input_gradient.data, 1));
#endif
}

void weighted_squared_error(const TensorView& input, const TensorView& target, type pos_w, type neg_w, type& error, float* workspace_device)
{
#ifndef CUDA
    const auto inputs = input.as_vector().array();
    const auto targets = target.as_vector().array();
    const auto squared_error = (inputs - targets).square();
    error = (targets == 1.0f).select(squared_error * pos_w, squared_error * neg_w).sum();
#else
    calculate_weighted_squared_error_cuda(input.size(), workspace_device, target.data, input.data, pos_w, neg_w);
    float result = 0.0f;
    CHECK_CUBLAS(cublasSasum(Device::get_cublas_handle(), static_cast<int>(input.size()), workspace_device, 1, &result));
    error = result;
#endif
}

void weighted_squared_error_gradient(const TensorView& input, const TensorView& target, type pos_w, type neg_w, type coefficient, TensorView& input_gradient)
{
#ifndef CUDA
    const auto inputs = input.as_vector().array();
    const auto targets = target.as_vector().array();
    input_gradient.as_vector().array()
        = (targets == 1.0f).select(2.0f * pos_w * (inputs - targets), 2.0f * neg_w * (inputs - targets)) * coefficient;
#else
    calculate_weighted_squared_error_delta_cuda(input.size(), input_gradient.data, target.data, input.data, pos_w, neg_w, coefficient);
#endif
}

void binary_cross_entropy(const TensorView& input, const TensorView& target, type& error, float* workspace_device)
{
#ifndef CUDA
    const auto y = input.as_vector().array().cwiseMax(EPSILON).cwiseMin(1.0f - EPSILON);
    const auto t = target.as_vector().array();
    error = -(t * y.log() + (1.0f - t) * (1.0f - y).log()).sum() / static_cast<type>(input.shape[0]);
#else
    calculate_binary_cross_entropy_cuda(input.size(), workspace_device, target.data, input.data, EPSILON);
    float sum_ce = 0.0f;
    CHECK_CUBLAS(cublasSasum(Device::get_cublas_handle(), static_cast<int>(input.size()), workspace_device, 1, &sum_ce));
    error = sum_ce / input.shape[0];
#endif
}

void categorical_cross_entropy(const TensorView& input, const TensorView& target, type& error, float* workspace_device)
{
#ifndef CUDA
    const auto y = input.as_vector().array().cwiseMax(EPSILON);
    const auto t = target.as_vector().array();
    error = -(t * y.log()).sum() / static_cast<type>(input.shape[0]);
#else
    calculate_multiple_cross_entropy_cuda(input.size(), workspace_device, target.data, input.data, EPSILON);
    float sum_ce = 0.0f;
    CHECK_CUBLAS(cublasSasum(Device::get_cublas_handle(), static_cast<int>(input.size()), workspace_device, 1, &sum_ce));
    error = sum_ce / input.shape[0];
#endif
}

void cross_entropy_gradient(const TensorView& input, const TensorView& target, TensorView& input_gradient)
{
#ifndef CUDA
    const Index n = input.shape[0];
    const Index num_classes = input.shape.back();
    const auto y = input.as_vector().array().cwiseMax(EPSILON).cwiseMin(1.0f - EPSILON);
    const auto t = target.as_vector().array();

    if(num_classes == 1)
        input_gradient.as_vector().array() = (-t / (y + EPSILON) + (type(1) - t) / (type(1) - y + EPSILON)) / static_cast<type>(n);
    else
        input_gradient.as_vector().array() = (y - t) / static_cast<type>(n);
#else
    const Index num_classes = input.shape.back();
    const float scale = 1.0f / static_cast<float>(input.shape[0]);

    if(num_classes == 1)
        calculate_binary_cross_entropy_delta_cuda(input.size(), input_gradient.data, target.data, input.data, EPSILON, scale);
    else
        calculate_multiple_cross_entropy_delta_cuda(input.size(), input_gradient.data, target.data, input.data, scale);
#endif
}

void minkowski_error(const TensorView& input, const TensorView& target, type p, type& error, float* workspace_device)
{
#ifndef CUDA
    const Index size = input.size();
    error = (input.as_vector() - target.as_vector()).array().abs().pow(p).sum() / static_cast<type>(size);
#else
    (void)input; (void)target; (void)p; (void)error; (void)workspace_device;
#endif
}

void minkowski_error_gradient(const TensorView& input, const TensorView& target, type p, TensorView& input_gradient)
{
#ifndef CUDA
    const Index size = input.size();
    const auto diff = input.as_vector().array() - target.as_vector().array();
    input_gradient.as_vector().array() = (p / static_cast<type>(size)) * diff.sign() * (diff.abs() + EPSILON).pow(p - 1.0f);
#else
    (void)input; (void)target; (void)p; (void)input_gradient;
#endif
}

void l1_regularization(const TensorView& parameters, type lambda, type& penalty)
{
#ifndef CUDA
    penalty = lambda * parameters.as_vector().lpNorm<1>();
#else
    float sum_abs = 0.0f;
    CHECK_CUBLAS(cublasSasum(Device::get_cublas_handle(), static_cast<int>(parameters.size()), parameters.data, 1, &sum_abs));
    penalty = lambda * sum_abs;
#endif
}

void l1_regularization_gradient(const TensorView& parameters, type lambda, TensorView& gradient)
{
#ifndef CUDA
    gradient.as_vector().array() += lambda * parameters.as_vector().array().sign();
#else
    apply_l1_gradient_cuda(parameters.size(), gradient.data, parameters.data, lambda);
#endif
}

void l2_regularization(const TensorView& parameters, type lambda, type& penalty)
{
#ifndef CUDA
    penalty = 0.5f * lambda * parameters.as_vector().squaredNorm();
#else
    float dot_product = 0.0f;
    const int n = static_cast<int>(parameters.size());
    CHECK_CUBLAS(cublasSdot(Device::get_cublas_handle(), n, parameters.data, 1, parameters.data, 1, &dot_product));
    penalty = 0.5f * lambda * dot_product;
#endif
}

void l2_regularization_gradient(const TensorView& parameters, type lambda, TensorView& gradient)
{
#ifndef CUDA
    gradient.as_vector().noalias() += lambda * parameters.as_vector();
#else
    const int n = static_cast<int>(parameters.size());
    CHECK_CUBLAS(cublasSaxpy(Device::get_cublas_handle(), n, &lambda, parameters.data, 1, gradient.data, 1));
#endif
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.