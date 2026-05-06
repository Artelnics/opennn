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
void pad(const TensorView& input, TensorView& output);

void bound(const TensorView& input, const TensorView& lower_bounds, const TensorView& upper_bounds, TensorView& output);
void bound_cpu(const TensorView& input, const TensorView& lower_bounds, const TensorView& upper_bounds, TensorView& output);
#ifdef OPENNN_HAS_CUDA
void bound_gpu(const TensorView& input, const TensorView& lower_bounds, const TensorView& upper_bounds, TensorView& output);
#endif

void scale(const TensorView& input,
           const TensorView& minimums, const TensorView& maximums,
           const TensorView& means, const TensorView& standard_deviations,
           const TensorView& scalers,
           float min_range, float max_range,
           TensorView& output);
void scale_cpu(const TensorView& input,
               const TensorView& minimums, const TensorView& maximums,
               const TensorView& means, const TensorView& standard_deviations,
               const TensorView& scalers,
               float min_range, float max_range,
               TensorView& output);
#ifdef OPENNN_HAS_CUDA
void scale_gpu(const TensorView& input,
               const TensorView& minimums, const TensorView& maximums,
               const TensorView& means, const TensorView& standard_deviations,
               const TensorView& scalers,
               float min_range, float max_range,
               TensorView& output);
#endif

void unscale(const TensorView& input,
             const TensorView& minimums, const TensorView& maximums,
             const TensorView& means, const TensorView& standard_deviations,
             const TensorView& scalers,
             float min_range, float max_range,
             TensorView& output);
void unscale_cpu(const TensorView& input,
                 const TensorView& minimums, const TensorView& maximums,
                 const TensorView& means, const TensorView& standard_deviations,
                 const TensorView& scalers,
                 float min_range, float max_range,
                 TensorView& output);
#ifdef OPENNN_HAS_CUDA
void unscale_gpu(const TensorView& input,
                 const TensorView& minimums, const TensorView& maximums,
                 const TensorView& means, const TensorView& standard_deviations,
                 const TensorView& scalers,
                 float min_range, float max_range,
                 TensorView& output);
#endif

void copy(const TensorView& source, TensorView& destination);
void copy_cpu(const TensorView& source, TensorView& destination);
#ifdef OPENNN_HAS_CUDA
void copy_gpu(const TensorView& source, TensorView& destination);
#endif

void add(const TensorView& input_1, const TensorView& input_2, TensorView& output);
void add_cpu(const TensorView& input_1, const TensorView& input_2, TensorView& output);
#ifdef OPENNN_HAS_CUDA
void add_gpu(const TensorView& input_1, const TensorView& input_2, TensorView& output);
#endif

void multiply(const TensorView& input_a, bool transpose_a, const TensorView& input_b, bool transpose_b, TensorView& output, float alpha = 1.0f, float beta = 0.0f);
void multiply_cpu(const TensorView& input_a, bool transpose_a, const TensorView& input_b, bool transpose_b, TensorView& output, float alpha = 1.0f, float beta = 0.0f);
#ifdef OPENNN_HAS_CUDA
void multiply_gpu(const TensorView& input_a, bool transpose_a, const TensorView& input_b, bool transpose_b, TensorView& output, float alpha = 1.0f, float beta = 0.0f);
#endif

void softmax(TensorView& output);
void softmax_cpu(TensorView& output);
#ifdef OPENNN_HAS_CUDA
void softmax_gpu(TensorView& output);
#endif
void max_pooling_3d_forward(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training);
void max_pooling_3d_forward_cpu(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training);
#ifdef OPENNN_HAS_CUDA
void max_pooling_3d_forward_gpu(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training);
#endif

void average_pooling_3d_forward(const TensorView& input, TensorView& output);
void average_pooling_3d_forward_cpu(const TensorView& input, TensorView& output);
#ifdef OPENNN_HAS_CUDA
void average_pooling_3d_forward_gpu(const TensorView& input, TensorView& output);
#endif

void max_pooling_3d_backward(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta);
void max_pooling_3d_backward_cpu(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta);
#ifdef OPENNN_HAS_CUDA
void max_pooling_3d_backward_gpu(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta);
#endif

void average_pooling_3d_backward(const TensorView& input, const TensorView& output_delta, TensorView& input_delta);
void average_pooling_3d_backward_cpu(const TensorView& input, const TensorView& output_delta, TensorView& input_delta);
#ifdef OPENNN_HAS_CUDA
void average_pooling_3d_backward_gpu(const TensorView& input, const TensorView& output_delta, TensorView& input_delta);
#endif
void split_heads(const TensorView& source, TensorView& destination);
void split_heads_cpu(const TensorView& source, TensorView& destination);
#ifdef OPENNN_HAS_CUDA
void split_heads_gpu(const TensorView& source, TensorView& destination);
#endif

void merge_heads(const TensorView& source, TensorView& destination);
void merge_heads_cpu(const TensorView& source, TensorView& destination);
#ifdef OPENNN_HAS_CUDA
void merge_heads_gpu(const TensorView& source, TensorView& destination);
#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
