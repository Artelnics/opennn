//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U T L A S S   N A R R O W   G E M M   T E S T   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// A dense forward whose contraction is at most 32 takes a CUTLASS kernel
// instantiated for that shape instead of cuBLASLt, which can only promise
// two-element alignment on an input 28 wide and dispatches an `align2` kernel
// for it. Measured 1.03x to 1.48x faster with bit-identical output.
//
// Three things about that path need a test and none of them is the arithmetic.
//
// The threadblock tile is chosen by row count - three of them, with the
// boundaries at 512 and 2,048 and 32,768 - so the rows here are picked to land
// one in each. The kernel also declines shapes it does not cover, and every
// build without CUTLASS declines all of them, so the same assertions have to
// hold when cuBLASLt runs instead: this is a fast path, never a behaviour
// change, and a test that only passed with CUTLASS present would be testing the
// wrong promise.
//
// And it composes with the row chunking above 16,384 rows, where the forward
// calls the kernel once per chunk. The 20,000-row case crosses that gate, so it
// exercises a whole chunk, a partial one, and the offsets between them.

#include "tests/pch.h"

#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/neural_network.h"

using namespace opennn;

namespace
{

const TensorView* find_parameter(const vector<TensorView>& parameters, size_t rank)
{
    for (const TensorView& parameter : parameters)
        if (parameter.get_shape().get_rank() == rank) return &parameter;
    return nullptr;
}

// The HIGGS first layer's contraction, which is what the path exists for.
constexpr Index features = 28;

void check_narrow_forward(Index rows, Index outputs, Type precision, float tolerance)
{
    Configuration::instance().set(Device::CUDA, precision);

    NeuralNetwork network;
    network.add_layer(make_unique<opennn::Dense>(Shape{features}, Shape{outputs}, "ReLU"));
    network.compile();
    network.set_parameters_glorot();
    network.copy_parameters_device();

    const vector<TensorView>& parameters = network.get_layer(0)->get_parameter_views();
    const TensorView* weights = find_parameter(parameters, 2);
    const TensorView* bias = find_parameter(parameters, 1);
    ASSERT_NE(weights, nullptr);
    ASSERT_NE(bias, nullptr);

    vector<float> weight_host(static_cast<size_t>(features * outputs), 0.0f);
    vector<float> bias_host(static_cast<size_t>(outputs), 0.0f);
    copy_device_to_host_float(weights->get_data(), weights->get_type(), features * outputs,
                              weight_host.data(), device::get_compute_stream());
    copy_device_to_host_float(bias->get_data(), bias->get_type(), outputs,
                              bias_host.data(), device::get_compute_stream());
    device::synchronize(device::get_compute_stream());

    // Every row distinct, so a row served from the wrong chunk offset cannot
    // pass by coincidence.
    MatrixR inputs(rows, features);
    for (Index row = 0; row < rows; ++row)
        for (Index feature = 0; feature < features; ++feature)
            inputs(row, feature) = 0.002f * float((row * 11 + feature * 17) % 193) - 0.19f;

    const TensorView input_view(const_cast<float*>(inputs.data()),
                                Shape{rows, features}, Type::FP32);

    ForwardPropagation forward_propagation(rows, &network);
    network.forward_propagate({input_view}, forward_propagation, ForwardPropagationMode::Inference);

    const TensorView& output = forward_propagation.get_outputs();
    vector<float> measured(static_cast<size_t>(rows * outputs), 0.0f);
    copy_device_to_host_float(output.get_data(), output.get_type(), rows * outputs,
                              measured.data(), device::get_compute_stream());
    device::synchronize(device::get_compute_stream());

    Index reported = 0;
    for (Index row = 0; row < rows && reported < 5; ++row)
        for (Index column = 0; column < outputs; ++column)
        {
            float expected = bias_host[static_cast<size_t>(column)];
            for (Index feature = 0; feature < features; ++feature)
                expected += inputs(row, feature)
                          * weight_host[static_cast<size_t>(feature * outputs + column)];
            expected = expected > 0.0f ? expected : 0.0f;

            const float got = measured[static_cast<size_t>(row * outputs + column)];
            if (fabs(got - expected) > tolerance)
            {
                ++reported;
                EXPECT_NEAR(got, expected, tolerance)
                    << "rows " << rows << ", row " << row << ", column " << column;
                break;
            }
        }

    Configuration::instance().set(Device::CUDA, Type::FP32);
}

}

TEST(CutlassNarrowGemm, SmallTileRowCount)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    check_narrow_forward(256, 1024, Type::BF16, 4e-2f);
}

TEST(CutlassNarrowGemm, MediumTileRowCount)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    check_narrow_forward(1024, 1024, Type::BF16, 4e-2f);
}

TEST(CutlassNarrowGemm, LargeTileRowCount)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    check_narrow_forward(6000, 256, Type::BF16, 4e-2f);
}

// 20,000 rows is two chunks of the row-chunked forward, the second partial.
TEST(CutlassNarrowGemm, CrossesTheRowChunkGate)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    check_narrow_forward(20000, 256, Type::BF16, 4e-2f);
}

// fp32 is not a shape this path covers, so it must fall through to cuBLASLt and
// still be right - the same assertion, exercising the decline.
TEST(CutlassNarrowGemm, Fp32FallsThroughAndIsStillCorrect)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    check_narrow_forward(1024, 1024, Type::FP32, 1e-4f);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
