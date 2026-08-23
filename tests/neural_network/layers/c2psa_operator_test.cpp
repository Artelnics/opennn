//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C 2 P S A   O P E R A T O R   T E S T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// C2PSA splits its input in half and attends over one half, so three of its
// four weight matrices are sized on channels/2 and the fourth on the full
// channel count. Nothing asserted on that split, nor on the two different
// Glorot limits it implies -- and a projection sized on the wrong half still
// links, still runs, and produces a network that trains to the wrong answer.
//
// C2PSA.GpuGradientMatchesNumerical covers the maths on the GPU. These cover
// the parameter contract, which is CPU-side and was untested.

#include "tests/pch.h"

#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/operators/c2psa_operator.h"

using namespace opennn;

namespace
{

constexpr Index height = 4;
constexpr Index width = 4;
constexpr Index channels = 8;
constexpr Index half_channels = channels / 2;

}


TEST(C2PSAOperatorTest, QueryKeyValueAreHalfWidthAndTheOutputIsFull)
{
    C2PSAOperator operator_under_test;
    operator_under_test.set(height, width, channels);

    const vector<TensorSpec> specs = operator_under_test.parameter_specs();

    ASSERT_EQ(specs.size(), size_t(4));

    // Wq, Wk, Wv attend over the split half; Wout mixes the whole thing back.
    EXPECT_EQ(specs[0].shape, (Shape{half_channels, half_channels}));
    EXPECT_EQ(specs[1].shape, (Shape{half_channels, half_channels}));
    EXPECT_EQ(specs[2].shape, (Shape{half_channels, half_channels}));
    EXPECT_EQ(specs[3].shape, (Shape{channels, channels}));
}


TEST(C2PSAOperatorTest, NoChannelsMeansNoParameters)
{
    C2PSAOperator operator_under_test;
    operator_under_test.set(height, width, 0);

    EXPECT_TRUE(operator_under_test.parameter_specs().empty());
}


TEST(C2PSAOperatorTest, LinkParametersBindsQkvThenOutput)
{
    MatrixR wq = MatrixR::Zero(half_channels, half_channels);
    MatrixR wk = MatrixR::Zero(half_channels, half_channels);
    MatrixR wv = MatrixR::Zero(half_channels, half_channels);
    MatrixR wout = MatrixR::Zero(channels, channels);

    const auto view = [](MatrixR& values)
    {
        return TensorView(values.data(), {values.rows(), values.cols()}, Type::FP32, Device::CPU);
    };

    C2PSAOperator operator_under_test;
    operator_under_test.set(height, width, channels);

    const vector<TensorView> views = {view(wq), view(wk), view(wv), view(wout)};
    operator_under_test.link_parameters(views);

    EXPECT_EQ(operator_under_test.Wq.get_data(), wq.data());
    EXPECT_EQ(operator_under_test.Wk.get_data(), wk.data());
    EXPECT_EQ(operator_under_test.Wv.get_data(), wv.data());
    EXPECT_EQ(operator_under_test.Wout.get_data(), wout.data());
}


TEST(C2PSAOperatorTest, GlorotUsesTheHalfLimitForQkvAndTheFullOneForOutput)
{
    MatrixR wq = MatrixR::Zero(half_channels, half_channels);
    MatrixR wk = MatrixR::Zero(half_channels, half_channels);
    MatrixR wv = MatrixR::Zero(half_channels, half_channels);
    MatrixR wout = MatrixR::Zero(channels, channels);

    const auto view = [](MatrixR& values)
    {
        return TensorView(values.data(), {values.rows(), values.cols()}, Type::FP32, Device::CPU);
    };

    C2PSAOperator operator_under_test;
    operator_under_test.set(height, width, channels);

    const vector<TensorView> views = {view(wq), view(wk), view(wv), view(wout)};
    operator_under_test.link_parameters(views);

    operator_under_test.set_parameters_glorot();

    // Two different fan-ins, so two different limits: sqrt(6/(c/2 + c/2)) for
    // the attention projections and sqrt(6/(c + c)) for the output mix.
    const float qkv_limit = sqrt(6.0f / float(half_channels + half_channels));
    const float out_limit = sqrt(6.0f / float(channels + channels));

    EXPECT_GT(qkv_limit, out_limit) << "the two limits must actually differ for this to test anything";

    for (const MatrixR* matrix : {&wq, &wk, &wv})
        for (Index i = 0; i < matrix->size(); ++i)
            EXPECT_LE(abs(matrix->data()[i]), qkv_limit + 1.0e-5f) << "at index " << i;

    for (Index i = 0; i < wout.size(); ++i)
        EXPECT_LE(abs(wout.data()[i]), out_limit + 1.0e-5f) << "at index " << i;
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
