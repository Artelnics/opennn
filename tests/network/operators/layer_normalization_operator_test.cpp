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
#include "opennn/network/operators/layer_normalization_operator.h"

#ifdef OPENNN_HAS_CUDA
#include "opennn/core/cuda/kernel_normalization.cuh"
#endif

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

#ifdef OPENNN_HAS_CUDA
TEST(LayerNormalizationOperatorTest, CudaWarpShapesAndFallbackMatchForwardAndGradientFormulas)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";
    const auto stream = device::get_compute_stream();
    for (const Type precision : {Type::FP32, Type::BF16})
    {
        if (precision == Type::BF16 && device::cuda_compute_capability() < 80) continue;
        SCOPED_TRACE(precision == Type::FP32 ? "FP32" : "BF16");
        visit_type<Type::FP32, Type::BF16>(precision, [&]<typename T>()
        {
            constexpr int width = 32 * 16 / sizeof(T);
            for (const int columns : {width, width * 2, width * 3, width * 4, width + 1})
            {
                SCOPED_TRACE(columns);
                const Index count = rows * columns;
                vector<T> input(size_t(count), T(0.0f));
                for (Index row = 0; row < rows; ++row)
                    for (int column = 0; column < columns; ++column)
                        input[size_t(row * columns + column)] = T((column % 2 ? -1.0f : 1.0f) * (1.0f + 0.5f * row));
                vector<float> weights(size_t(2 * columns), 0.0f);
                fill_n(weights.begin(), columns, 1.0f);
                Buffer x_device(Device::CUDA), y_device(Device::CUDA), dx_device(Device::CUDA);
                Buffer weights_device(Device::CUDA), stats_device(Device::CUDA), gradients_device(Device::CUDA);
                T* x = x_device.ensure<T>(count);
                T* y = y_device.ensure<T>(count);
                T* dx = dx_device.ensure<T>(count);
                float* gamma = weights_device.ensure<float>(2 * columns);
                float* means = stats_device.ensure<float>(2 * rows);
                float* inverse = means + rows;
                float* gradients = gradients_device.ensure<float>(2 * columns);
                device::copy_async(x, input.data(), count * sizeof(T), device::CopyKind::HostToDevice, stream);
                device::copy_async(gamma, weights.data(), Index(weights.size() * sizeof(float)),
                    device::CopyKind::HostToDevice, stream);

                for (const bool rms : {false, true})
                {
                    SCOPED_TRACE(rms ? "RMS" : "LayerNorm");
                    if (rms)
                    {
                        rmsnorm_forward_cuda<T>(int(rows), columns, x, y, inverse, gamma, epsilon);
                        rmsnorm_backward_cuda<T>(int(rows), columns, x, x, inverse, gamma, dx, gradients);
                    }
                    else
                    {
                        layernorm_forward_cuda<T>(int(rows), columns, x, y, means, inverse, gamma, gamma + columns, epsilon);
                        layernorm_backward_cuda<T>(int(rows), columns, x, x, means, inverse, gamma,
                            dx, nullptr, gradients, gradients + columns);
                    }
                    vector<T> output(size_t(count), T(0.0f)), delta(size_t(count), T(0.0f));
                    vector<float> parameter_gradient(size_t(2 * columns), 0.0f);
                    device::copy_async(output.data(), y, count * sizeof(T), device::CopyKind::DeviceToHost, stream);
                    device::copy_async(delta.data(), dx, count * sizeof(T), device::CopyKind::DeviceToHost, stream);
                    device::copy_async(parameter_gradient.data(), gradients, Index(parameter_gradient.size() * sizeof(float)),
                        device::CopyKind::DeviceToHost, stream);
                    device::synchronize(stream);

                    vector<double> expected_gamma(size_t(columns), 0.0), expected_beta(size_t(columns), 0.0);
                    for (Index row = 0; row < rows; ++row)
                    {
                        double mean = 0.0, squared = 0.0, mean_delta = 0.0, mean_delta_xhat = 0.0;
                        for (int column = 0; column < columns; ++column)
                            mean += float(input[size_t(row * columns + column)]);
                        mean = rms ? 0.0 : mean / columns;
                        for (int column = 0; column < columns; ++column)
                            squared += pow(double(float(input[size_t(row * columns + column)])) - mean, 2);
                        const double inv = 1.0 / sqrt(squared / columns + epsilon);
                        for (int column = 0; column < columns; ++column)
                        {
                            const double value = float(input[size_t(row * columns + column)]);
                            mean_delta += value / columns;
                            mean_delta_xhat += value * (value - mean) * inv / columns;
                        }
                        for (int column = 0; column < columns; ++column)
                        {
                            const size_t index = size_t(row * columns + column);
                            const double value = float(input[index]);
                            const double xhat = (value - mean) * inv;
                            const double expected_delta = (value - (rms ? 0.0 : mean_delta) - xhat * mean_delta_xhat) * inv;
                            EXPECT_NEAR(float(output[index]), xhat, precision == Type::BF16 ? 0.005 : 0.00001);
                            EXPECT_NEAR(float(delta[index]), expected_delta, 0.00001);
                            expected_gamma[size_t(column)] += value * xhat;
                            expected_beta[size_t(column)] += value;
                        }
                    }
                    for (int column = 0; column < columns; ++column)
                    {
                        EXPECT_NEAR(parameter_gradient[size_t(column)], expected_gamma[size_t(column)], 0.0001);
                        if (!rms) EXPECT_NEAR(parameter_gradient[size_t(columns + column)], expected_beta[size_t(column)], 0.0001);
                    }
                }
            }
        });
    }
}
#endif

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
