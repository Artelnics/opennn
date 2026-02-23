//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O R R E L A T I O N S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"

#pragma once

namespace opennn
{

struct Correlation
{
    enum class Method{Pearson, Spearman};

    enum class Form{Linear, Sigmoid, Logarithmic, Exponential, Power};

    Correlation() {}

    void set_perfect();

    string write_type() const;

    void print() const;

    type a = type(NAN);
    type b = type(NAN);
    type r = type(NAN);

    type lower_confidence = type(NAN);
    type upper_confidence = type(NAN);

    Method method = Method::Pearson;
    Form form = Form::Linear;
};


Correlation linear_correlation(const VectorR&, const VectorR&);

Correlation logarithmic_correlation(const VectorR&, const VectorR&);

Correlation exponential_correlation(const VectorR&, const VectorR&);

Correlation power_correlation(const VectorR&, const VectorR&);

Correlation logistic_correlation_vector_vector(const VectorR&, const VectorR&);

Correlation logistic_correlation_vector_matrix(const VectorR&, const MatrixR&);

Correlation logistic_correlation_matrix_vector(const MatrixR&, const VectorR&);

Correlation logistic_correlation_matrix_matrix(const MatrixR&, const MatrixR&);

Correlation correlation(const MatrixR&, const MatrixR&);

// Spearman correlation

Correlation linear_correlation_spearman(const VectorR&, const VectorR&);

VectorR calculate_spearman_ranks(const VectorR&);

Correlation logistic_correlation_vector_vector_spearman(const VectorR&, const VectorR&);

Correlation correlation_spearman(const MatrixR&, const MatrixR&);

// Confidence interval

type r_correlation_to_z_correlation(const type);
type z_correlation_to_r_correlation(const type);

VectorR confidence_interval_z_correlation(const type, Index);

// Time series correlation

VectorR autocorrelations(const VectorR&, Index  = 10);

VectorR cross_correlations(const VectorR&, const VectorR&, Index);

MatrixR get_correlation_values(const Tensor<Correlation, 2>&);

// Missing values

pair<VectorR, VectorR> filter_missing_values_vector_vector(const VectorR&, const VectorR&);
pair<VectorR, MatrixR> filter_missing_values_vector_matrix(const VectorR&, const MatrixR&);
pair<VectorR, MatrixR> filter_missing_values_matrix_vector(const MatrixR&, const VectorR&);
pair<MatrixR, MatrixR> filter_missing_values_matrix_matrix(const MatrixR&, const MatrixR&);

void register_optimization_algorithms();

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
