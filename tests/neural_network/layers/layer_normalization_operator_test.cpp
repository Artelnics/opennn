//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   N O R M A L I Z A T I O N   O P E R A T O R   T E S T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// The normalization kernels were reached only through a transformer, so their
// two variants and the statistics they hand to the backward pass had no direct
// coverage. These pin the closed forms and the invariants that make the
// backward pass valid -- a normalized row has zero mean and unit variance for
// LayerNorm, and unit mean-square for RMS -- against an independent reference.

#include "tests/pch.h"

#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/operators/layer_normalization_operator.h"

using namespace opennn;

namespace
{

constexpr Index rows = 3;
constexpr Index dimension = 4;
constexpr float epsilon = 1.0e-6f;


TensorView matrix_view(MatrixR& values)
{
    return TensorView(values.data(), {values.rows(), values.cols()}, Type::FP32, Device::CPU);
}


TensorView vector_view(VectorR& values)
{
    return TensorView(values.data(), {values.size()}, Type::FP32, Device::CPU);
}


// Row-major, which is what the kernels index: element (r, c) is r*D + c.
MatrixR sample_input()
{
    MatrixR input(rows, dimension);

    input << 1.0f, -2.0f,  3.0f, 0.5f,
            -4.0f,  0.25f, 2.0f, 1.5f,
             0.0f,  0.0f,  0.0f, 0.0f;   // the degenerate row: zero variance

    return input;
}

}


TEST(LayerNormalizationOperatorTest, ForwardMatchesTheClosedForm)
{
    MatrixR input = sample_input();

    VectorR gamma(dimension);
    VectorR beta(dimension);
    gamma << 1.0f, 2.0f, 0.5f, -1.0f;
    beta  << 0.0f, -1.0f, 0.25f, 3.0f;

    VectorR means = VectorR::Zero(rows);
    VectorR standard_deviations = VectorR::Zero(rows);
    MatrixR output = MatrixR::Constant(rows, dimension, -999.0f);

    TensorView means_view = vector_view(means);
    TensorView deviations_view = vector_view(standard_deviations);
    TensorView output_view = matrix_view(output);

    // Order is means, standard_deviations, normalized, output: the normalized
    // cache is what the backward pass reads back, and it is not optional.
    MatrixR normalized = MatrixR::Zero(rows, dimension);
    TensorView normalized_view = matrix_view(normalized);

    layer_normalization_forward(matrix_view(input), vector_view(gamma), vector_view(beta),
                                means_view, deviations_view,
                                normalized_view, output_view, epsilon);

    for (Index row = 0; row < rows; ++row)
    {
        SCOPED_TRACE("row " + to_string(row));

        double sum = 0.0;
        for (Index column = 0; column < dimension; ++column) sum += input(row, column);
        const double mean = sum / dimension;

        double squared = 0.0;
        for (Index column = 0; column < dimension; ++column)
            squared += (input(row, column) - mean) * (input(row, column) - mean);
        const double variance = squared / dimension;

        EXPECT_NEAR(means(row), float(mean), 1.0e-5f);

        for (Index column = 0; column < dimension; ++column)
        {
            const double normalized = (input(row, column) - mean) / sqrt(variance + epsilon);
            const double expected = normalized * gamma(column) + beta(column);

            EXPECT_NEAR(output(row, column), float(expected), 1.0e-4f)
                << "column " << column;
        }
    }
}


TEST(LayerNormalizationOperatorTest, NormalizedRowsHaveZeroMeanAndUnitVariance)
{
    MatrixR input = sample_input();

    // Identity scale and shift, so the output is the normalization itself.
    VectorR gamma = VectorR::Constant(dimension, 1.0f);
    VectorR beta = VectorR::Zero(dimension);

    VectorR means = VectorR::Zero(rows);
    VectorR standard_deviations = VectorR::Zero(rows);
    MatrixR output = MatrixR::Zero(rows, dimension);

    TensorView means_view = vector_view(means);
    TensorView deviations_view = vector_view(standard_deviations);
    TensorView output_view = matrix_view(output);

    // Order is means, standard_deviations, normalized, output: the normalized
    // cache is what the backward pass reads back, and it is not optional.
    MatrixR normalized = MatrixR::Zero(rows, dimension);
    TensorView normalized_view = matrix_view(normalized);

    layer_normalization_forward(matrix_view(input), vector_view(gamma), vector_view(beta),
                                means_view, deviations_view,
                                normalized_view, output_view, epsilon);

    for (Index row = 0; row < rows - 1; ++row)   // the zero row is checked below
    {
        SCOPED_TRACE("row " + to_string(row));

        double sum = 0.0;
        double squared = 0.0;

        for (Index column = 0; column < dimension; ++column)
        {
            sum += output(row, column);
            squared += double(output(row, column)) * output(row, column);
        }

        EXPECT_NEAR(sum / dimension, 0.0, 1.0e-4);
        EXPECT_NEAR(squared / dimension, 1.0, 1.0e-3);
    }

    // A constant row has zero variance: epsilon is what keeps this finite, and
    // the whole row must come out at beta rather than NaN.
    for (Index column = 0; column < dimension; ++column)
        EXPECT_TRUE(isfinite(output(rows - 1, column))) << "column " << column;
}


TEST(LayerNormalizationOperatorTest, RmsForwardMatchesTheClosedFormAndSkipsCentring)
{
    MatrixR input = sample_input();

    VectorR weight(dimension);
    weight << 1.0f, 2.0f, 0.5f, -1.0f;

    VectorR inverse_rms = VectorR::Zero(rows);
    MatrixR normalized = MatrixR::Zero(rows, dimension);
    MatrixR output = MatrixR::Constant(rows, dimension, -999.0f);

    TensorView inverse_view = vector_view(inverse_rms);
    TensorView normalized_view = matrix_view(normalized);
    TensorView output_view = matrix_view(output);

    rms_normalization_forward(matrix_view(input), vector_view(weight),
                              inverse_view, normalized_view, output_view, epsilon);

    for (Index row = 0; row < rows; ++row)
    {
        SCOPED_TRACE("row " + to_string(row));

        // RMS divides by the root mean square without subtracting the mean,
        // which is the whole difference from LayerNorm.
        double mean_square = 0.0;
        for (Index column = 0; column < dimension; ++column)
            mean_square += double(input(row, column)) * input(row, column);
        mean_square /= dimension;

        const double inverse = 1.0 / sqrt(mean_square + epsilon);

        EXPECT_NEAR(inverse_rms(row), float(inverse), 1.0e-3f);

        for (Index column = 0; column < dimension; ++column)
            EXPECT_NEAR(output(row, column),
                        float(input(row, column) * inverse * weight(column)), 1.0e-4f)
                << "column " << column;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
