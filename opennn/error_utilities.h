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

// On GPU these helpers internally pull a singleton FP32 scratch via
// `get_loss_scratch()` (cuda_gemm.cpp). No workspace plumbing in the public
// signature.

void mean_squared_error(const TensorView& input, const TensorView& target, float& error);
void mean_squared_error_gradient(const TensorView& input, const TensorView& target, TensorView& input_delta);

void normalized_squared_error(const TensorView& input, const TensorView& target, float coefficient, float& error);
void normalized_squared_error_gradient(const TensorView& input, const TensorView& target, float coefficient, TensorView& input_delta);

void weighted_squared_error(const TensorView& input, const TensorView& target, float pos_w, float neg_w, float& error);
void weighted_squared_error_gradient(const TensorView& input, const TensorView& target, float pos_w, float neg_w, float coefficient, TensorView& input_delta);

void binary_cross_entropy(const TensorView& input, const TensorView& target, float& error);
void categorical_cross_entropy(const TensorView& input, const TensorView& target, float& error);
void cross_entropy_gradient(const TensorView& input, const TensorView& target, TensorView& input_delta);

void minkowski_error(const TensorView& input, const TensorView& target, float p, float& error);
void minkowski_error_gradient(const TensorView& input, const TensorView& target, float p, TensorView& input_delta);

void cross_entropy_3d(const TensorView& input, const TensorView& target, float& error, Index& active_tokens_out, Index& correct_tokens_out);
void cross_entropy_3d_gradient(const TensorView& input, const TensorView& target, TensorView& input_delta, Index active_tokens_count);

void l1_regularization(const TensorView& parameters, float lambda, float& penalty);
void l1_regularization_gradient(const TensorView& parameters, float lambda, TensorView& gradient);

void l2_regularization(const TensorView& parameters, float lambda, float& penalty);
void l2_regularization_gradient(const TensorView& parameters, float lambda, TensorView& gradient);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.