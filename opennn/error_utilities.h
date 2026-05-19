//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E R R O R   U T I L I T I E S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_utilities.h"

namespace opennn
{

/// @brief Computes the mean squared error between predictions and targets.
/// @param input Predicted values.
/// @param target Ground-truth targets with the same shape as input.
/// @param error Output scalar holding the resulting MSE.
/// @param workspace_device Optional device-side reduction workspace (ignored on CPU).
void mean_squared_error(const TensorView& input, const TensorView& target, float& error, float* workspace_device);
/// @brief Writes the MSE gradient with respect to the predictions into input_delta.
/// @param input Predicted values.
/// @param target Ground-truth targets.
/// @param input_delta Output gradient tensor with the same shape as input.
void mean_squared_error_gradient(const TensorView& input, const TensorView& target, const TensorView& input_delta);

/// @brief Computes the squared error normalized by a dataset-level coefficient.
/// @param input Predicted values.
/// @param target Ground-truth targets.
/// @param coefficient Normalization coefficient (typically the target variance).
/// @param error Output scalar holding the normalized squared error.
/// @param workspace_device Optional device-side reduction workspace.
void normalized_squared_error(const TensorView& input, const TensorView& target, float coefficient, float& error, float* workspace_device);
/// @brief Writes the normalized-squared-error gradient with respect to the predictions into input_delta.
void normalized_squared_error_gradient(const TensorView& input, const TensorView& target, float coefficient, const TensorView& input_delta);

/// @brief Computes the binary squared error weighted asymmetrically for positive and negative classes.
/// @param input Predicted probabilities in [0,1].
/// @param target Binary targets (0 or 1).
/// @param pos_w Weight applied to positive samples.
/// @param neg_w Weight applied to negative samples.
/// @param error Output scalar holding the weighted squared error.
/// @param workspace_device Optional device-side reduction workspace.
void weighted_squared_error(const TensorView& input, const TensorView& target, float pos_w, float neg_w, float& error, float* workspace_device);
/// @brief Writes the gradient of the weighted squared error scaled by coefficient into input_delta.
void weighted_squared_error_gradient(const TensorView& input, const TensorView& target, float pos_w, float neg_w, float coefficient, const TensorView& input_delta);

/// @brief Computes the binary cross-entropy between predicted probabilities and binary targets.
void binary_cross_entropy(const TensorView& input, const TensorView& target, float& error, float* workspace_device);
/// @brief Computes the multi-class (categorical) cross-entropy between softmax probabilities and one-hot targets.
void categorical_cross_entropy(const TensorView& input, const TensorView& target, float& error, float* workspace_device);
/// @brief Writes the cross-entropy gradient with respect to the (pre-softmax/logit) predictions into input_delta.
void cross_entropy_gradient(const TensorView& input, const TensorView& target, const TensorView& input_delta);

/// @brief Computes the Minkowski error sum(|input - target|^power) for the given power exponent.
void minkowski_error(const TensorView& input, const TensorView& target, float power, float& error, float* workspace_device);
/// @brief Writes the Minkowski-error gradient with respect to the predictions into input_delta.
void minkowski_error_gradient(const TensorView& input, const TensorView& target, float power, const TensorView& input_delta);

/// @brief Computes 3-D (sequence) cross-entropy used by transformer-style targets, ignoring padded positions.
/// @param input Predicted logits/probabilities with shape (batch, sequence, vocab).
/// @param target Token ids or one-hot targets with the matching sequence layout.
/// @param error Output scalar holding the cumulative error across active tokens.
/// @param active_tokens_out Output count of non-padding tokens that contributed to the error.
/// @param correct_tokens_out Output count of correctly predicted tokens (for accuracy).
/// @param errors_device Optional device-side reduction workspace.
void cross_entropy_3d(const TensorView& input, const TensorView& target, float& error, Index& active_tokens_out, Index& correct_tokens_out, float* errors_device = nullptr);
/// @brief Writes the 3-D cross-entropy gradient into input_delta, normalizing by the host-side active-token count.
void cross_entropy_3d_gradient(const TensorView& input, const TensorView& target, const TensorView& input_delta, Index active_tokens_count);
/// @brief Variant of cross_entropy_3d_gradient that reads the active-token count from device memory.
void cross_entropy_3d_gradient_device_count(const TensorView& input, const TensorView& target, const TensorView& input_delta, const float* active_tokens_count_device);

/// @brief Computes the L1 regularization penalty lambda * sum(|parameters|).
void l1_regularization(const TensorView& parameters, float lambda, float& penalty);
/// @brief Adds the L1 regularization gradient lambda * sign(parameters) into the gradient tensor.
void l1_regularization_gradient(const TensorView& parameters, float lambda, const TensorView& gradient);

/// @brief Computes the L2 regularization penalty lambda * sum(parameters^2).
void l2_regularization(const TensorView& parameters, float lambda, float& penalty);
/// @brief Adds the L2 regularization gradient 2 * lambda * parameters into the gradient tensor.
void l2_regularization_gradient(const TensorView& parameters, float lambda, const TensorView& gradient);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
