// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/core/tensor_types.h"

namespace opennn
{

void mean_squared_error(const TensorView&, const TensorView&, float&, float*);
void mean_squared_error_gradient(const TensorView&, const TensorView&, const TensorView&);

void mean_absolute_error(const TensorView&, const TensorView&, float&, float*);
void mean_absolute_error_gradient(const TensorView&, const TensorView&, const TensorView&);

void normalized_squared_error(const TensorView&, const TensorView&, float, float&, float*);
void normalized_squared_error_gradient(const TensorView&, const TensorView&, float, const TensorView&);

void weighted_squared_error(const TensorView&, const TensorView&, float, float, float&, float*);
void weighted_squared_error_gradient(const TensorView&, const TensorView&, float, float, float, const TensorView&);

void binary_cross_entropy(const TensorView&, const TensorView&, float&, float*);
void categorical_cross_entropy(const TensorView&, const TensorView&, float&, float*);
void cross_entropy(const TensorView&, const TensorView&, float&, float*);
void cross_entropy_gradient(const TensorView&, const TensorView&, const TensorView&);

void minkowski_error(const TensorView&, const TensorView&, float, float&, float*);
void minkowski_error_gradient(const TensorView&, const TensorView&, float, const TensorView&, bool on_gpu = false);

void cross_entropy_3d(const TensorView&, const TensorView&, float&, Index&, Index&,
                      float* errors_device = nullptr,
                      float* reduction_device = nullptr);
void cross_entropy_3d_gradient(const TensorView&, const TensorView&, const TensorView&, Index);
void cross_entropy_3d_gradient_device_count(const TensorView&, const TensorView&, const TensorView&, const float*);

void l1_regularization(const TensorView&, float, float&);
void l1_regularization_gradient(const TensorView&, float, const TensorView&);

void l2_regularization(const TensorView&, float, float&);
void l2_regularization_gradient(const TensorView&, float, const TensorView&);

}
