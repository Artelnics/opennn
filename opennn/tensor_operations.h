//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   O P E R A T I O N S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "tensor_types.h"
#include "enum_map.h"

namespace opennn
{

enum class ActivationFunction { Identity, Sigmoid, Tanh, ReLU, Softmax, LeakyReLU };

const EnumMap<ActivationFunction>& activation_function_map();
const string& activation_function_to_string(ActivationFunction);
ActivationFunction activation_function_from_string(const string&);

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
                    TensorView&, cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS);
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
                               const TensorView&,
                               Index, Index,
                               bool);

void max_pooling_3d_forward(const TensorView&, TensorView&, TensorView&, bool);
void average_pooling_3d_forward(const TensorView&, TensorView&);
void max_pooling_3d_backward(const TensorView&, const TensorView&, TensorView&);
void average_pooling_3d_backward(const TensorView&, const TensorView&, TensorView&);

void pooling_2d_forward(const TensorView&, TensorView&, TensorView&,
                        Index, Index, Index,
                        Index, Index,
                        Index, Index,
                        Index, Index,
                        bool);
void pooling_2d_backward(const TensorView&, const TensorView&,
                         const TensorView&, const TensorView&,
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
vector<Index> filter_selected_indices_by_column(const MatrixR&, const vector<Index>&, Index, float, float);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
