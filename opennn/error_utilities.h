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

void mean_squared_error(const TensorView& input, const TensorView& target, type& error, float* workspace_device);
void mean_squared_error_gradient(const TensorView& input, const TensorView& target, TensorView& input_gradient);

void normalized_squared_error(const TensorView& input, const TensorView& target, type coefficient, type& error, float* workspace_device);
void normalized_squared_error_gradient(const TensorView& input, const TensorView& target, type coefficient, TensorView& input_gradient);

void weighted_squared_error(const TensorView& input, const TensorView& target, type pos_w, type neg_w, type& error, float* workspace_device);
void weighted_squared_error_gradient(const TensorView& input, const TensorView& target, type pos_w, type neg_w, type coefficient, TensorView& input_gradient);

void binary_cross_entropy(const TensorView& input, const TensorView& target, type& error, float* workspace_device);
void categorical_cross_entropy(const TensorView& input, const TensorView& target, type& error, float* workspace_device);
void cross_entropy_gradient(const TensorView& input, const TensorView& target, TensorView& input_gradient);

void minkowski_error(const TensorView& input, const TensorView& target, type p, type& error, float* workspace_device);
void minkowski_error_gradient(const TensorView& input, const TensorView& target, type p, TensorView& input_gradient);

void cross_entropy_3d(const TensorView& input, const TensorView& target, type& error, Index& active_tokens_out, float* errors_device = nullptr);
void cross_entropy_3d_gradient(const TensorView& input, const TensorView& target, TensorView& input_gradient, Index active_tokens_count);

void l1_regularization(const TensorView& parameters, type lambda, type& penalty);
void l1_regularization_gradient(const TensorView& parameters, type lambda, TensorView& gradient);

void l2_regularization(const TensorView& parameters, type lambda, type& penalty);
void l2_regularization_gradient(const TensorView& parameters, type lambda, TensorView& gradient);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.