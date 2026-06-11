//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E R R O R   F U N C T I O N S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_types.h"

namespace opennn
{

void mean_squared_error(const TensorView& input, const TensorView& target, float& error, float* workspace_device);
void mean_squared_error_gradient(const TensorView& input, const TensorView& target, const TensorView& input_delta);

void normalized_squared_error(const TensorView& input, const TensorView& target, float coefficient, float& error, float* workspace_device);
void normalized_squared_error_gradient(const TensorView& input, const TensorView& target, float coefficient, const TensorView& input_delta);

void weighted_squared_error(const TensorView& input, const TensorView& target, float positive_weight, float negative_weight, float& error, float* workspace_device);
void weighted_squared_error_gradient(const TensorView& input, const TensorView& target, float positive_weight, float negative_weight, float coefficient, const TensorView& input_delta);

void binary_cross_entropy(const TensorView& input, const TensorView& target, float& error, float* workspace_device);
void categorical_cross_entropy(const TensorView& input, const TensorView& target, float& error, float* workspace_device);
void cross_entropy_gradient(const TensorView& input, const TensorView& target, const TensorView& input_delta);

void minkowski_error(const TensorView& input, const TensorView& target, float power, float& error, float* workspace_device);
void minkowski_error_gradient(const TensorView& input, const TensorView& target, float power, const TensorView& input_delta, bool on_gpu = false);

void cross_entropy_3d(const TensorView& input, const TensorView& target, float& error, Index& active_tokens_out, Index& correct_tokens_out, float* errors_device = nullptr);
void cross_entropy_3d_gradient(const TensorView& input, const TensorView& target, const TensorView& input_delta, Index active_tokens_count);
void cross_entropy_3d_gradient_device_count(const TensorView& input, const TensorView& target, const TensorView& input_delta, const float* active_tokens_count_device);

void l1_regularization(const TensorView& parameters, float lambda, float& penalty);
void l1_regularization_gradient(const TensorView& parameters, float lambda, const TensorView& gradient);

void l2_regularization(const TensorView& parameters, float lambda, float& penalty);
void l2_regularization_gradient(const TensorView& parameters, float lambda, const TensorView& gradient);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
