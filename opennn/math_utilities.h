//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M A T H   U T I L I T I E S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "tensor_utilities.h"
#include "enum_map.h"
#include "operators.h"

namespace opennn
{

// Generic

void padding(const TensorView& input, TensorView& output);
void bounding(const TensorView& input, const TensorView& lower_bounds, const TensorView& upper_bounds, TensorView& output);
void scale(const TensorView& input,
           const TensorView& minimums, const TensorView& maximums,
           const TensorView& means, const TensorView& standard_deviations,
           const TensorView& scalers,
           float min_range, float max_range,
           TensorView& output);
void unscale(const TensorView& input,
             const TensorView& minimums, const TensorView& maximums,
             const TensorView& means, const TensorView& standard_deviations,
             const TensorView& scalers,
             float min_range, float max_range,
             TensorView& output);
void copy(const TensorView& source, TensorView& destination);
void addition(const TensorView& input_1, const TensorView& input_2, TensorView& output);
void multiply(const TensorView& input_a, bool transpose_a, const TensorView& input_b, bool transpose_b, TensorView& output, float alpha = 1.0f, float beta = 0.0f);
void softmax(TensorView& output);
void softmax_backward(const TensorView& softmax_out, TensorView& output_delta);

// Pooling 3D

void max_pooling_3d_forward(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training);
void average_pooling_3d_forward(const TensorView& input, TensorView& output);
void max_pooling_3d_backward(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta);
void average_pooling_3d_backward(const TensorView& input, const TensorView& output_delta, TensorView& input_delta);

// Embedding


// Multi-head attention

void split_heads(const TensorView& source, TensorView& destination);
void merge_heads(const TensorView& source, TensorView& destination);

void attention_masks(const TensorView& source_input, TensorView& attention_weights, const MatrixR& causal_mask, bool use_causal_mask, float* padding_mask_scratch);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
