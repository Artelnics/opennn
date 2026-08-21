//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E M M   R O W   C H U N K   T E S T   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// Above 16,384 rows the dense forward issues its GEMM in chunks of rows rather
// than as one cuBLASLt call, because cuBLASLt loses throughput on a very tall
// operand: measured on the HIGGS hidden layer, fp32 41.5 -> 44.3 TFLOP/s at
// 65,536 rows and bf16 82.3 -> 88.5.
//
// Splitting a GEMM by rows is exact - every output row depends only on its own
// input row - so the risk is not arithmetic, it is the offset arithmetic. A
// chunk index applied to the wrong stride, or a tail chunk sized wrong, leaves
// the rows past the first chunk reading from the wrong place, and the
// benchmark's finite-output check would not notice.
//
// Nothing else in the suite runs a batch this tall, so nothing else takes this
// path. The layer here is deliberately narrow: what is being tested is the row
// offsets, and 20,000 rows of 64 features reaches the second and the partial
// third chunk for the cost of 82 MFLOP.

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

// 16,384 is the chunk, so this is one whole chunk, a second whole chunk, and a
// 3,616-row tail.
constexpr Index rows = 20000;
constexpr Index features = 64;
constexpr Index outputs = 64;

const TensorView* find_parameter(const vector<TensorView>& parameters, size_t rank)
{
    for (const TensorView& parameter : parameters)
        if (parameter.get_shape().get_rank() == rank) return &parameter;
    return nullptr;
}

void check_tall_dense_forward(Type precision, float tolerance)
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

    // Every row distinct, so a row served from the wrong offset cannot pass by
    // coincidence: row r is a ramp whose phase is r.
    MatrixR inputs(rows, features);
    for (Index row = 0; row < rows; ++row)
        for (Index feature = 0; feature < features; ++feature)
            inputs(row, feature) = 0.001f * float((row * 7 + feature * 13) % 251) - 0.12f;

    const TensorView input_view(const_cast<float*>(inputs.data()),
                                Shape{rows, features}, Type::FP32);

    ForwardPropagation forward_propagation(rows, &network);
    network.forward_propagate({input_view}, forward_propagation, false);

    const TensorView& output = forward_propagation.get_outputs();
    vector<float> measured(static_cast<size_t>(rows * outputs), 0.0f);
    copy_device_to_host_float(output.get_data(), output.get_type(), rows * outputs,
                              measured.data(), device::get_compute_stream());
    device::synchronize(device::get_compute_stream());

    // Check the whole tensor, but report the first failure with its row, so a
    // broken offset says which chunk it broke in.
    Index mismatches = 0;
    for (Index row = 0; row < rows && mismatches < 5; ++row)
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
                ++mismatches;
                EXPECT_NEAR(got, expected, tolerance)
                    << "row " << row << " (chunk " << row / 16384 << "), column " << column;
                break;
            }
        }

    Configuration::instance().set(Device::CUDA, Type::FP32);
}

}

TEST(GemmRowChunk, TallDenseForwardMatchesTheDefinition)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    check_tall_dense_forward(Type::FP32, 1e-4f);
}

TEST(GemmRowChunk, TallDenseForwardMatchesTheDefinitionBf16)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    check_tall_dense_forward(Type::BF16, 3e-2f);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
