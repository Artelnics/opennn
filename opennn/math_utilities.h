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

namespace opennn
{
void pad(const TensorView& input, TensorView& output);

void bound(const TensorView& input, const TensorView& lower_bounds, const TensorView& upper_bounds, TensorView& output);

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

void add(const TensorView& input_1, const TensorView& input_2, TensorView& output);

void multiply(const TensorView& input_a, bool transpose_a, const TensorView& input_b, bool transpose_b, TensorView& output, float alpha = 1.0f, float beta = 0.0f);

void softmax(TensorView& output);

void activation_forward(TensorView& output, ActivationFunction function);
void activation_backward(const TensorView& outputs, TensorView& delta, ActivationFunction function);

void dropout_forward(TensorView& output, Buffer& mask, float rate);
void dropout_backward(TensorView& delta, const Buffer& mask, float rate);

void linear_forward(const TensorView& input, const TensorView& weights, const TensorView& bias,
                    TensorView& output, cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS);
void linear_backward(const TensorView& output_delta, const TensorView& input, const TensorView& weights,
                     const TensorView& weight_gradient, const TensorView& bias_gradient,
                     TensorView& input_delta, bool accumulate_input_delta = false);

void layer_norm_forward(const TensorView& input, const TensorView& gamma, const TensorView& beta,
                        TensorView& means, TensorView& standard_deviations,
                        TensorView& normalized, TensorView& output);
void layer_norm_backward(const TensorView& input, const TensorView& output_delta,
                         const TensorView& means, const TensorView& standard_deviations,
                         const TensorView& normalized, const TensorView& gamma,
                         const TensorView& gamma_gradient, const TensorView& beta_gradient,
                         TensorView& input_delta);

void embedding_lookup_forward(const TensorView& indices, const TensorView& weights,
                              const TensorView& positional_encoding, TensorView& output,
                              Index sequence_length, Index embedding_dimension, Index vocabulary_size,
                              bool scale_embedding, bool add_positional_encoding);
void embedding_lookup_backward(const TensorView& indices, const TensorView& output_delta,
                               const TensorView& weight_gradient,
                               Index embedding_dimension, Index vocabulary_size,
                               bool scale_embedding);

void max_pooling_3d_forward(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training);
void average_pooling_3d_forward(const TensorView& input, TensorView& output);
void max_pooling_3d_backward(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta);
void average_pooling_3d_backward(const TensorView& input, const TensorView& output_delta, TensorView& input_delta);

void split_heads(const TensorView& source, TensorView& destination);
void merge_heads(const TensorView& source, TensorView& destination);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
