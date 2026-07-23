//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O R R E L A T I O N S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn_types.h"

namespace opennn
{

struct Correlation
{
    enum class Method{Pearson, Spearman};

    enum class Form{Identity, Sigmoid, Logarithmic, Exponential, Power};

    void set_perfect();

    void print() const;

    float intercept = NAN;
    float slope = NAN;
    float coefficient = NAN;

    float lower_confidence = NAN;
    float upper_confidence = NAN;

    Method method = Method::Pearson;
    Form form = Form::Identity;
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
Correlation correlation_spearman(const MatrixR&, const MatrixR&);
Correlation linear_correlation_spearman(const VectorR&, const VectorR&);

VectorR calculate_spearman_ranks(const VectorR&);

Correlation logistic_correlation_spearman(const VectorR&, const VectorR&);

float r_correlation_to_z_correlation(const float);
float z_correlation_to_r_correlation(const float);

pair<float, float> confidence_interval_z_correlation(const float, Index);
VectorR autocorrelations(const VectorR&, Index  = 10);

VectorR cross_correlations(const VectorR&, const VectorR&, Index);

MatrixR get_correlation_values(const Tensor<Correlation, 2>&);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
