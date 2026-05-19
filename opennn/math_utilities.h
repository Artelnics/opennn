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
/// @brief Pads the input tensor and writes the result into output.
void pad(const TensorView& input, TensorView& output);

/// @brief Clamps each element of input to the [lower_bounds, upper_bounds] range.
/// @param input Source tensor.
/// @param lower_bounds Per-element (or broadcastable) lower limits.
/// @param upper_bounds Per-element (or broadcastable) upper limits.
/// @param output Destination tensor for the bounded values.
void bound(const TensorView& input, const TensorView& lower_bounds, const TensorView& upper_bounds, TensorView& output);
/// @brief CPU implementation of bound().
void bound_cpu(const TensorView& input, const TensorView& lower_bounds, const TensorView& upper_bounds, TensorView& output);
#ifdef OPENNN_HAS_CUDA
/// @brief GPU implementation of bound().
void bound_gpu(const TensorView& input, const TensorView& lower_bounds, const TensorView& upper_bounds, TensorView& output);
#endif

/// @brief Applies per-feature scaling (mean/std, min/max, or other scalers) to a tensor.
/// @param input Source tensor.
/// @param minimums Column-wise minimums used for min-max scaling.
/// @param maximums Column-wise maximums used for min-max scaling.
/// @param means Column-wise means used for standardisation.
/// @param standard_deviations Column-wise standard deviations.
/// @param scalers Per-column scaler kind selector.
/// @param min_range Lower bound of the target range for min-max scaling.
/// @param max_range Upper bound of the target range for min-max scaling.
/// @param output Destination tensor for the scaled values.
void scale(const TensorView& input,
           const TensorView& minimums, const TensorView& maximums,
           const TensorView& means, const TensorView& standard_deviations,
           const TensorView& scalers,
           float min_range, float max_range,
           TensorView& output);
/// @brief CPU implementation of scale().
void scale_cpu(const TensorView& input,
               const TensorView& minimums, const TensorView& maximums,
               const TensorView& means, const TensorView& standard_deviations,
               const TensorView& scalers,
               float min_range, float max_range,
               TensorView& output);
#ifdef OPENNN_HAS_CUDA
/// @brief GPU implementation of scale().
void scale_gpu(const TensorView& input,
               const TensorView& minimums, const TensorView& maximums,
               const TensorView& means, const TensorView& standard_deviations,
               const TensorView& scalers,
               float min_range, float max_range,
               TensorView& output);
#endif

/// @brief Inverse of scale(); reconstructs original values from a previously scaled tensor.
void unscale(const TensorView& input,
             const TensorView& minimums, const TensorView& maximums,
             const TensorView& means, const TensorView& standard_deviations,
             const TensorView& scalers,
             float min_range, float max_range,
             TensorView& output);
/// @brief CPU implementation of unscale().
void unscale_cpu(const TensorView& input,
                 const TensorView& minimums, const TensorView& maximums,
                 const TensorView& means, const TensorView& standard_deviations,
                 const TensorView& scalers,
                 float min_range, float max_range,
                 TensorView& output);
#ifdef OPENNN_HAS_CUDA
/// @brief GPU implementation of unscale().
void unscale_gpu(const TensorView& input,
                 const TensorView& minimums, const TensorView& maximums,
                 const TensorView& means, const TensorView& standard_deviations,
                 const TensorView& scalers,
                 float min_range, float max_range,
                 TensorView& output);
#endif

/// @brief Copies the contents of source into destination, dispatching to CPU or GPU as needed.
void copy(const TensorView& source, TensorView& destination);
/// @brief CPU implementation of copy().
void copy_cpu(const TensorView& source, TensorView& destination);
#ifdef OPENNN_HAS_CUDA
/// @brief GPU implementation of copy().
void copy_gpu(const TensorView& source, TensorView& destination);
#endif

/// @brief Element-wise addition: output = input_1 + input_2.
void add(const TensorView& input_1, const TensorView& input_2, TensorView& output);
/// @brief CPU implementation of add().
void add_cpu(const TensorView& input_1, const TensorView& input_2, TensorView& output);
#ifdef OPENNN_HAS_CUDA
/// @brief GPU implementation of add().
void add_gpu(const TensorView& input_1, const TensorView& input_2, TensorView& output);
#endif

/// @brief General matrix multiply: output = alpha * op(input_a) * op(input_b) + beta * output.
/// @param input_a Left operand.
/// @param transpose_a Transpose input_a before multiplying.
/// @param input_b Right operand.
/// @param transpose_b Transpose input_b before multiplying.
/// @param output Destination tensor (also read when beta != 0).
/// @param alpha Scale factor applied to the matrix product.
/// @param beta Scale factor applied to the existing output.
void multiply(const TensorView& input_a, bool transpose_a, const TensorView& input_b, bool transpose_b, TensorView& output, float alpha = 1.0f, float beta = 0.0f);
/// @brief CPU implementation of multiply().
void multiply_cpu(const TensorView& input_a, bool transpose_a, const TensorView& input_b, bool transpose_b, TensorView& output, float alpha = 1.0f, float beta = 0.0f);
#ifdef OPENNN_HAS_CUDA
/// @brief GPU implementation of multiply().
void multiply_gpu(const TensorView& input_a, bool transpose_a, const TensorView& input_b, bool transpose_b, TensorView& output, float alpha = 1.0f, float beta = 0.0f);
#endif

/// @brief Applies softmax in place along the trailing dimension of output.
void softmax(TensorView& output);
/// @brief CPU implementation of softmax().
void softmax_cpu(TensorView& output);
#ifdef OPENNN_HAS_CUDA
/// @brief GPU implementation of softmax().
void softmax_gpu(TensorView& output);
#endif
/// @brief Forward pass of 3D max pooling; records argmax positions when training.
/// @param input Source activations.
/// @param output Pooled output activations.
/// @param maximal_indices Positions of the maxima within each pooling window.
/// @param is_training If true, populates maximal_indices for use by the backward pass.
void max_pooling_3d_forward(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training);
/// @brief CPU implementation of max_pooling_3d_forward().
void max_pooling_3d_forward_cpu(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training);
#ifdef OPENNN_HAS_CUDA
/// @brief GPU implementation of max_pooling_3d_forward().
void max_pooling_3d_forward_gpu(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training);
#endif

/// @brief Forward pass of 3D average pooling.
void average_pooling_3d_forward(const TensorView& input, TensorView& output);
/// @brief CPU implementation of average_pooling_3d_forward().
void average_pooling_3d_forward_cpu(const TensorView& input, TensorView& output);
#ifdef OPENNN_HAS_CUDA
/// @brief GPU implementation of average_pooling_3d_forward().
void average_pooling_3d_forward_gpu(const TensorView& input, TensorView& output);
#endif

/// @brief Backward pass for 3D max pooling; routes gradients to argmax positions.
void max_pooling_3d_backward(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta);
/// @brief CPU implementation of max_pooling_3d_backward().
void max_pooling_3d_backward_cpu(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta);
#ifdef OPENNN_HAS_CUDA
/// @brief GPU implementation of max_pooling_3d_backward().
void max_pooling_3d_backward_gpu(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta);
#endif

/// @brief Backward pass for 3D average pooling.
void average_pooling_3d_backward(const TensorView& input, const TensorView& output_delta, TensorView& input_delta);
/// @brief CPU implementation of average_pooling_3d_backward().
void average_pooling_3d_backward_cpu(const TensorView& input, const TensorView& output_delta, TensorView& input_delta);
#ifdef OPENNN_HAS_CUDA
/// @brief GPU implementation of average_pooling_3d_backward().
void average_pooling_3d_backward_gpu(const TensorView& input, const TensorView& output_delta, TensorView& input_delta);
#endif
/// @brief Reshapes a multi-head attention tensor by splitting the last axis into heads.
void split_heads(const TensorView& source, TensorView& destination);
/// @brief CPU implementation of split_heads().
void split_heads_cpu(const TensorView& source, TensorView& destination);
#ifdef OPENNN_HAS_CUDA
/// @brief GPU implementation of split_heads().
void split_heads_gpu(const TensorView& source, TensorView& destination);
#endif

/// @brief Inverse of split_heads(); merges per-head tensors back into a single representation.
void merge_heads(const TensorView& source, TensorView& destination);
/// @brief CPU implementation of merge_heads().
void merge_heads_cpu(const TensorView& source, TensorView& destination);
#ifdef OPENNN_HAS_CUDA
/// @brief GPU implementation of merge_heads().
void merge_heads_gpu(const TensorView& source, TensorView& destination);
#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
