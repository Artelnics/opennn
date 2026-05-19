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

/// @brief Result of a correlation analysis: model parameters, fit quality, and the method/form used.
struct Correlation
{
    /// @brief Underlying coefficient family used to compute r.
    enum class Method{Pearson, Spearman};

    /// @brief Functional form fitted to the data before computing the coefficient.
    enum class Form{Identity, Sigmoid, Logarithmic, Exponential, Power};

    Correlation() {}

    /// @brief Sets the correlation to a perfect linear relationship (a=0, b=1, r=1).
    void set_perfect();

    /// @brief Prints the correlation parameters and method to stdout.
    void print() const;

    float a = NAN;
    float b = NAN;
    float r = NAN;

    float lower_confidence = NAN;
    float upper_confidence = NAN;

    Method method = Method::Pearson;
    Form form = Form::Identity;
};

/// @brief Pearson linear correlation between two equal-length vectors.
/// @param x First variable.
/// @param y Second variable.
/// @return Correlation with intercept a, slope b, and coefficient r.
[[nodiscard]] Correlation linear_correlation(const VectorR&, const VectorR&);

/// @brief Logarithmic correlation: fits y = a + b * log(x) and returns the resulting fit.
[[nodiscard]] Correlation logarithmic_correlation(const VectorR&, const VectorR&);

/// @brief Exponential correlation: fits y = a * exp(b * x) and returns the resulting fit.
[[nodiscard]] Correlation exponential_correlation(const VectorR&, const VectorR&);

/// @brief Power correlation: fits y = a * x^b and returns the resulting fit.
[[nodiscard]] Correlation power_correlation(const VectorR&, const VectorR&);

/// @brief Logistic correlation between two vectors (binary or continuous targets).
[[nodiscard]] Correlation logistic_correlation(const VectorR&, const VectorR&);

/// @brief Logistic correlation between a vector predictor and a one-hot matrix target.
[[nodiscard]] Correlation logistic_correlation(const VectorR&, const MatrixR&);

/// @brief Logistic correlation between a one-hot matrix predictor and a vector target.
[[nodiscard]] Correlation logistic_correlation(const MatrixR&, const VectorR&);

/// @brief Logistic correlation between two one-hot matrices.
[[nodiscard]] Correlation logistic_correlation(const MatrixR&, const MatrixR&);

/// @brief Point-biserial correlation between a binary vector and a continuous vector.
[[nodiscard]] Correlation point_biserial_correlation(const VectorR&, const VectorR&);

/// @brief Eta-squared (effect-size) correlation between a continuous vector and a categorical matrix.
[[nodiscard]] Correlation eta_squared_correlation(const VectorR&, const MatrixR&);

/// @brief Generic correlation between two matrices, dispatching on column types (binary, categorical, continuous).
[[nodiscard]] Correlation correlation(const MatrixR&, const MatrixR&);

/// @brief Spearman rank correlation between two vectors using a linear fit on ranks.
[[nodiscard]] Correlation linear_correlation_spearman(const VectorR&, const VectorR&);

/// @brief Computes Spearman ranks (average rank for ties) for the entries of a vector.
[[nodiscard]] VectorR calculate_spearman_ranks(const VectorR&);

/// @brief Spearman-rank logistic correlation between two vectors.
[[nodiscard]] Correlation logistic_correlation_spearman(const VectorR&, const VectorR&);

/// @brief Generic Spearman correlation between two matrices, dispatching on column types.
[[nodiscard]] Correlation correlation_spearman(const MatrixR&, const MatrixR&);

/// @brief Fisher r-to-z transform of a correlation coefficient.
[[nodiscard]] float r_correlation_to_z_correlation(const float);

/// @brief Inverse Fisher z-to-r transform.
[[nodiscard]] float z_correlation_to_r_correlation(const float);

/// @brief Returns the [lower, upper] confidence interval for a correlation given its sample size.
[[nodiscard]] pair<float, float> confidence_interval_z_correlation(const float, Index);

/// @brief Autocorrelations of a series for lags 0..max_lag.
/// @param series Input time series.
/// @param max_lag Maximum lag to compute (default 10).
[[nodiscard]] VectorR autocorrelations(const VectorR&, Index  = 10);

/// @brief Cross-correlations between two series for lags 0..max_lag.
[[nodiscard]] VectorR cross_correlations(const VectorR&, const VectorR&, Index);

/// @brief Extracts the coefficient r from a 2D tensor of Correlation values.
[[nodiscard]] MatrixR get_correlation_values(const Tensor<Correlation, 2>&);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
