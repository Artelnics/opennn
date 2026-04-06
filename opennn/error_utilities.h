//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E R R O R   U T I L I T I E S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"
#include "tensor_utilities.h"

#pragma once

namespace opennn
{

inline void mean_squared_error(const TensorView& input,
                               const TensorView& target,
                               type& error,
                               float* workspace_device)
{
#ifndef CUDA
    const Index size = input.size();
    error = (input.as_vector() - target.as_vector()).squaredNorm() / static_cast<type>(size);
#else
    const int n = static_cast<int>(input.size());
    const float alpha_neg = -1.0f;
    const float alpha_pos = 1.0f;

    CHECK_CUDNN(cudnnOpTensor(get_cudnn_handle(),
                              get_operator_sum_descriptor(),
                              &alpha_pos,
                              input.get_descriptor(), input.data,
                              &alpha_neg,
                              target.get_descriptor(), target.data,
                              &zero, input.get_descriptor(), workspace_device));

    float sse = 0.0f;

    CHECK_CUBLAS(cublasSdot(get_cublas_handle(), n, workspace_device, 1, workspace_device, 1, &sse));

    error = sse / n;
#endif
}

inline void mean_squared_error_gradient(const TensorView& input,
                                        const TensorView& target,
                                        TensorView& input_gradient)
{
#ifndef CUDA
    const Index size = input.size();
    input_gradient.as_vector().array() = (input.as_vector().array() - target.as_vector().array()) * (2.0f / static_cast<type>(size));
#else
    const int n = static_cast<int>(input.size());
    const float alpha_neg = -1.0f;
    const float alpha_pos = 1.0f;
    const float scale = 2.0f / n;

    CHECK_CUDNN(cudnnOpTensor(get_cudnn_handle(), get_operator_sum_descriptor(),
                              &alpha_pos, input.get_descriptor(), input.data,
                              &alpha_neg, target.get_descriptor(), target.data,
                              &zero, input_gradient.get_descriptor(), input_gradient.data));

    CHECK_CUBLAS(cublasSscal(get_cublas_handle(), n, &scale, input_gradient.data, 1));
#endif
}

inline void normalized_squared_error(const TensorView& input, const TensorView& target, type coefficient, type& error, float* workspace_device)
{
#ifndef CUDA
    error = (input.as_vector() - target.as_vector()).squaredNorm() / (coefficient + EPSILON);
#else
    const int n = static_cast<int>(input.size());
    const float alpha_neg = -1.0f;
    const float alpha_pos = 1.0f;

    CHECK_CUDNN(cudnnOpTensor(get_cudnn_handle(), get_operator_sum_descriptor(),
                              &alpha_pos, input.get_descriptor(), input.data,
                              &alpha_neg, target.get_descriptor(), target.data,
                              &zero, input.get_descriptor(), workspace_device));

    float sse = 0.0f;
    CHECK_CUBLAS(cublasSdot(get_cublas_handle(), n, workspace_device, 1, workspace_device, 1, &sse));

    error = sse / (coefficient + EPSILON);
#endif
}



inline void weighted_squared_error(const TensorView& input,
                                   const TensorView& target,
                                   type pos_w,
                                   type neg_w,
                                   type& error,
                                   float* workspace_device)
{
#ifndef CUDA
    const auto inputs = input.as_vector().array();
    const auto targets = target.as_vector().array();

    const auto squared_error = (inputs - targets).square();

    error = (targets == 1.0f).select(squared_error * pos_w,
                                     squared_error * neg_w).sum();
#else
    calculate_weighted_squared_error_cuda(input.size(), workspace_device, target.data, input.data, pos_w, neg_w);

    float result = 0.0f;
    CHECK_CUBLAS(cublasSasum(get_cublas_handle(), static_cast<int>(input.size()), workspace_device, 1, &result));
    error = result;
#endif
}

inline void weighted_squared_error_gradient(const TensorView& input,
                                            const TensorView& target,
                                            type pos_w,
                                            type neg_w,
                                            type coefficient,
                                            TensorView& input_gradient)
{
#ifndef CUDA
    const auto inputs = input.as_vector().array();
    const auto targets = target.as_vector().array();

    input_gradient.as_vector().array()
        = (targets == 1.0f).select(2.0f * pos_w * (inputs - targets), 2.0f * neg_w * (inputs - targets)) * coefficient;
#else
    calculate_weighted_squared_error_delta_cuda(input.size(),
                                                input_gradient.data,
                                                target.data,
                                                input.data,
                                                pos_w,
                                                neg_w,
                                                coefficient);
#endif
}

inline void binary_cross_entropy(const TensorView& input, const TensorView& target, type& error, float* workspace_device)
{
#ifndef CUDA
    const auto y = input.as_vector().array().cwiseMax(EPSILON).cwiseMin(1.0f - EPSILON);
    const auto t = target.as_vector().array();

    // We divide by the number of active elements (excluding NaNs or masked values if any)
    error = -(t * y.log() + (1.0f - t) * (1.0f - y).log()).sum() / static_cast<type>(input.shape[0]);
#else
    calculate_binary_cross_entropy_cuda(input.size(), workspace_device, target.data, input.data, EPSILON);
    float sum_ce = 0.0f;
    CHECK_CUBLAS(cublasSasum(get_cublas_handle(), static_cast<int>(input.size()), workspace_device, 1, &sum_ce));
    error = sum_ce / input.shape[0];
#endif
}


inline void categorical_cross_entropy(const TensorView& input,
                                      const TensorView& target,
                                      type& error,
                                      float* workspace_device)
{
#ifndef CUDA
    const auto y = input.as_vector().array().cwiseMax(EPSILON);
    const auto t = target.as_vector().array();

    error = -(t * y.log()).sum() / static_cast<type>(input.shape[0]);
#else
    calculate_categorical_cross_entropy_cuda(input.size(), workspace_device, target.data, input.data, EPSILON);

    float sum_ce = 0.0f;

    CHECK_CUBLAS(cublasSasum(get_cublas_handle(), static_cast<int>(input.size()), workspace_device, 1, &sum_ce));

    error = sum_ce / input.shape[0];
#endif
}

inline void cross_entropy_gradient(const TensorView& input,
                                   const TensorView& target,
                                   TensorView& input_gradient)
{
#ifndef CUDA
    const Index n = input.shape[0];
    const auto y = input.as_vector().array().cwiseMax(EPSILON).cwiseMin(1.0f - EPSILON);
    const auto t = target.as_vector().array();

    input_gradient.as_vector().array() = (y - t) / (static_cast<type>(n));
#else
    const float scale = 1.0f / static_cast<float>(input.shape[0]);

    calculate_cross_entropy_delta_cuda(input.size(),
                                       input_gradient.data,
                                       target.data,
                                       input.data,
                                       scale);
#endif
}


inline void minkowski_error(const TensorView& input,
                            const TensorView& target,
                            type p,
                            type& error,
                            float* workspace_device)
{
#ifndef CUDA
    const Index size = input.size();

    error = (input.as_vector() - target.as_vector()).array().abs().pow(p).sum() / static_cast<type>(size);
#else
    calculate_minkowski_error_cuda(input.size(),
                                   workspace_device,
                                   target.data,
                                   input.data, p);

    float sum_m = 0.0f;

    CHECK_CUBLAS(cublasSasum(get_cublas_handle(), static_cast<int>(input.size()), workspace_device, 1, &sum_m));

    error = sum_m / input.size();
#endif
}


inline void normalized_squared_error_gradient(const TensorView& input, const TensorView& target, type coefficient, TensorView& input_gradient)
{
#ifndef CUDA
    const VectorMap inputs = input.as_vector();
    const VectorMap targets = target.as_vector();
    VectorMap input_gradients = input_gradient.as_vector();

    // Formula: 2 * (y - t) / coefficient
    input_gradients.array() = (inputs.array() - targets.array()) * (2.0f / (coefficient + EPSILON));
#else
    const int n = static_cast<int>(input.size());
    const float alpha_pos = 1.0f;
    const float alpha_neg = -1.0f;
    const float scale = 2.0f / (static_cast<float>(coefficient) + EPSILON);

    // 1. gradient = input - target
    CHECK_CUDNN(cudnnOpTensor(get_cudnn_handle(), get_operator_sum_descriptor(),
                              &alpha_pos, input.get_descriptor(), input.data,
                              &alpha_neg, target.get_descriptor(), target.data,
                              &zero, input_gradient.get_descriptor(), input_gradient.data));

    // 2. Scale: gradient *= (2 / coefficient)
    CHECK_CUBLAS(cublasSscal(get_cublas_handle(), n, &scale, input_gradient.data, 1));
#endif
}


inline void minkowski_error_gradient(const TensorView& input, const TensorView& target, type p, TensorView& input_gradient)
{
#ifndef CUDA
    const Index size = input.size();
    const auto inputs = input.as_vector().array();
    const auto targets = target.as_vector().array();
    auto input_gradients = input_gradient.as_vector().array();

    const auto diff = inputs - targets;

    // Formula: (p / N) * sign(y - t) * |y - t|^(p - 1)
    // We add EPSILON inside the power to ensure numerical stability if p-1 < 0
    input_gradients = (p / static_cast<type>(size)) * diff.sign() * (diff.abs() + EPSILON).pow(p - 1.0f);
#else
    // Minkowski logic is too complex for standard cuBLAS/cuDNN
    // Call custom kernel for element-wise power derivative
    calculate_minkowski_error_delta_cuda(input.size(), input_gradient.data, target.data, input.data, p);
#endif
}

inline void l1_regularization(const TensorView& parameters, type lambda, type& penalty)
{
#ifndef CUDA
    penalty = lambda * parameters.as_vector().lpNorm<1>();
#else
    float sum_abs = 0.0f;
    CHECK_CUBLAS(cublasSasum(get_cublas_handle(), static_cast<int>(parameters.size()), parameters.data, 1, &sum_abs));
    penalty = lambda * sum_abs;
#endif
}

inline void l1_regularization_gradient(const TensorView& parameters, type lambda, TensorView& gradient)
{
#ifndef CUDA
    gradient.as_vector().array() += lambda * parameters.as_vector().array().sign();
#else
    // Custom kernel required for sign function
    calculate_l1_regularization_gradient_cuda(parameters.size(), gradient.data, parameters.data, lambda);
#endif
}

// --- L2 Regularization (Ridge) ---

inline void l2_regularization(const TensorView& parameters, type lambda, type& penalty)
{
#ifndef CUDA
    penalty = 0.5f * lambda * parameters.as_vector().squaredNorm();
#else
    float dot_product = 0.0f;
    const int n = static_cast<int>(parameters.size());
    CHECK_CUBLAS(cublasSdot(get_cublas_handle(), n, parameters.data, 1, parameters.data, 1, &dot_product));
    penalty = 0.5f * lambda * dot_product;
#endif
}

inline void l2_regularization_gradient(const TensorView& parameters, type lambda, TensorView& gradient)
{
#ifndef CUDA
    gradient.as_vector().noalias() += lambda * parameters.as_vector();
#else
    const int n = static_cast<int>(parameters.size());
    // gradient = lambda * parameters + gradient
    CHECK_CUBLAS(cublasSaxpy(get_cublas_handle(), n, &lambda, parameters.data, 1, gradient.data, 1));
#endif
}

}
