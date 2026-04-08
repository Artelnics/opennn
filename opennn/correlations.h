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

Correlation logistic_correlation(const VectorR&, const VectorR&);

Correlation logistic_correlation(const VectorR&, const MatrixR&);

Correlation logistic_correlation(const MatrixR&, const VectorR&);

Correlation logistic_correlation(const MatrixR&, const MatrixR&);

Correlation point_biserial_correlation(const VectorR&, const VectorR&);

Correlation eta_squared_correlation(const VectorR&, const MatrixR&);

Correlation correlation(const MatrixR&, const MatrixR&);

// Spearman correlation

Correlation linear_correlation_spearman(const VectorR&, const VectorR&);

VectorR calculate_spearman_ranks(const VectorR&);

Correlation logistic_correlation_spearman(const VectorR&, const VectorR&);

Correlation correlation_spearman(const MatrixR&, const MatrixR&);

// Confidence interval

type r_correlation_to_z_correlation(const type);
type z_correlation_to_r_correlation(const type);

pair<type, type> confidence_interval_z_correlation(const type, Index);

// Time series correlation

VectorR autocorrelations(const VectorR&, Index  = 10);

VectorR cross_correlations(const VectorR&, const VectorR&, Index);

MatrixR get_correlation_values(const Tensor<Correlation, 2>&);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
