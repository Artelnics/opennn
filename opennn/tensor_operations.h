//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   O P E R A T I O N S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn_types.h"
#include "tensor_types.h"
#include "enum_map.h"

namespace opennn
{

enum class ActivationFunction { Identity, Sigmoid, Tanh, ReLU, Softmax, LeakyReLU, GELU, GELUTanh, SiLU };

// Negative-side slope for LeakyReLU. 0.1 matches the Darknet/YOLO default.
// The CUDA kernels keep their own mirrored copy in kernel_common.cuh, like the
// activation ids above.
inline constexpr float LEAKY_RELU_SLOPE = 0.1f;

const EnumMap<ActivationFunction>& activation_function_map();
const string& activation_function_to_string(ActivationFunction);
ActivationFunction activation_function_from_string(const string&);

bool activation_needs_input(ActivationFunction function);

inline float activation_forward_value(ActivationFunction function, float x)
{
    using enum ActivationFunction;
    switch (function)
    {
    case Identity:  return x;
    case Sigmoid:   return 1.0f / (1.0f + exp(-x));
    case Tanh:      return tanh(x);
    case ReLU:      return max(0.0f, x);
    case LeakyReLU: return x >= 0.0f ? x : x * LEAKY_RELU_SLOPE;
    case GELU:      return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
    case GELUTanh:
    {
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
    }
    case SiLU:      return x / (1.0f + exp(-x));
    case Softmax:   break;
    }

    throw runtime_error("activation_forward_value: Softmax must be handled separately.");
}

inline float activation_derivative_from_output_value(ActivationFunction function, float y)
{
    using enum ActivationFunction;
    switch (function)
    {
    case Identity:  return 1.0f;
    case Sigmoid:   return y * (1.0f - y);
    case Tanh:      return 1.0f - y * y;
    case ReLU:      return y > 0.0f ? 1.0f : 0.0f;
    case LeakyReLU: return y >= 0.0f ? 1.0f : LEAKY_RELU_SLOPE;
    case Softmax:   break;
    case GELU:
    case GELUTanh:
    case SiLU:      break;
    }

    throw runtime_error("activation_derivative_from_output_value: Softmax/GELU/GELUTanh/SiLU must be handled separately.");
}

VectorR activation_forward_values(ActivationFunction, const VectorR&);
MatrixR activation_forward_values(ActivationFunction, const MatrixR&);
VectorR activation_derivative_from_output_values(ActivationFunction, const VectorR&);
MatrixR activation_derivative_from_output_values(ActivationFunction, const MatrixR&);
MatrixR activation_derivative_from_output_values(ActivationFunction, const MatrixMap&);

void bound(const TensorView&, const TensorView&, const TensorView&, TensorView&);

void scale(const TensorView&,
           const TensorView&, const TensorView&,
           const TensorView&, const TensorView&,
           const TensorView&,
           float, float,
           TensorView&);

void unscale(const TensorView&,
             const TensorView&, const TensorView&,
             const TensorView&, const TensorView&,
             const TensorView&,
             float, float,
             TensorView&);

void copy(const TensorView&, TensorView&);

void add(const TensorView&, const TensorView&, TensorView&);

void multiply(const TensorView&, bool, const TensorView&, bool, TensorView&, float alpha = 1.0f, float beta = 0.0f);

void softmax(TensorView&);

void activation_forward(TensorView&, ActivationFunction);
void activation_backward(const TensorView&, TensorView&, ActivationFunction);

void dropout_forward(TensorView&, Buffer&, float);
void dropout_backward(TensorView&, const Buffer&, float);

void linear_forward(const TensorView&, const TensorView&, const TensorView&,
                    TensorView&, cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS,
                    TensorView* pre_activation = nullptr);
void linear_backward(const TensorView&, const TensorView&, const TensorView&,
                     const TensorView&, const TensorView&,
                     TensorView&, bool accumulate_input_delta = false);

void layer_normalization_forward(const TensorView&, const TensorView&, const TensorView&,
                        TensorView&, TensorView&,
                        TensorView&, TensorView&);
void layer_normalization_add_forward(const TensorView&, const TensorView&,
                            const TensorView&, const TensorView&,
                            TensorView&, TensorView&,
                            TensorView&, TensorView&, TensorView&);
void layer_normalization_backward(const TensorView&, const TensorView&,
                         const TensorView&, const TensorView&,
                         const TensorView&, const TensorView&,
                         const TensorView&, const TensorView&,
                         TensorView&);

void embedding_lookup_forward(const TensorView&, const TensorView&,
                              const TensorView&, TensorView&,
                              Index, Index, Index,
                              bool, bool);
void embedding_lookup_backward(const TensorView&, const TensorView&,
                               const TensorView&, const TensorView&,
                               Index, Index, Index,
                               bool);

void max_pooling_3d_forward(const TensorView&, TensorView&, TensorView&, bool);
void average_pooling_3d_forward(const TensorView&, TensorView&);
void max_pooling_3d_backward(const TensorView&, const TensorView&, TensorView&);
void average_pooling_3d_backward(const TensorView&, const TensorView&, TensorView&);
void first_token_3d_forward(const TensorView&, TensorView&);
void first_token_3d_backward(const TensorView&, TensorView&);

void compute_token_valid_lengths(const TensorView&, Index, vector<Index>&);

void pooling_2d_forward(const TensorView&, TensorView&, TensorView&,
                        Index, Index, Index,
                        Index, Index,
                        Index, Index,
                        Index, Index,
                        bool);
void pooling_2d_backward(const TensorView&, const TensorView&,
                         TensorView&,
                         Index, Index, Index,
                         Index, Index,
                         Index, Index,
                         Index, Index,
                         bool);

void split_heads(const TensorView&, TensorView&);
void merge_heads(const TensorView&, TensorView&);

MatrixR append_rows(const MatrixR&, const MatrixR&);
MatrixR append_columns(const MatrixR&, const MatrixR&);
VectorR slice_rows(const VectorR&, const vector<Index>&);
MatrixR slice_rows(const MatrixR&, const vector<Index>&);
VectorI get_nearest_points(const MatrixR&, const VectorR&, int = 1);
MatrixR calculate_distances(const MatrixR&);
vector<Index> filter_selected_indices_by_column(const MatrixR&, const vector<Index>&, Index, float, float);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
