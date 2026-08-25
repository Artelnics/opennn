//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S I N G L E   O U T P U T   A C T I V A T I O N   F U S I O N   T E S T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// A dense layer with one output runs its combination on CUDA as a warp-per-row
// reduction rather than through cuBLASLt, and that kernel now applies the
// layer's activation to the value it has just accumulated instead of leaving it
// to a pass of its own. On the HIGGS classifier's head that removed a graph node
// worth about a microsecond a batch - five per cent of what a batch of 256
// costs - but it also moved where the rounding happens, so the promise needs
// stating: the fused result is the activation of the sum accumulated in fp32,
// rounded once. The unfused order rounds twice, once on the pre-activation
// value and once on the activated one.
//
// The second half of the promise is that asking for the fusion is only ever a
// speed decision. The reduction refuses shapes whose feature count does not
// divide into whole 16-byte vectors, and linear_forward then has to notice that
// the activation did not travel with the GEMM and run it as a pass of its own.
// A head of 1,022 features is refused in both precisions (1022 % 8 and 1022 % 4
// are both non-zero) and is how these tests reach that path - the environment
// switch cannot, because it is read into a function-local static and so is
// fixed for the life of the process.
//
// Without the fallback the outputs would come back as raw pre-activation sums,
// which for a Glorot-initialised head is values around zero where a sigmoid
// gives values around a half: loud, not subtle.

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

constexpr Index rows = 256;

// 1024 is the HIGGS classifier's head; 1022 is the same head with a feature
// count the reduction refuses, which is the cuBLASLt fallback.
constexpr Index fused_features = 1024;
constexpr Index refused_features = 1022;

struct HeadRun
{
    vector<float> measured;
    vector<float> expected;
};

const TensorView* find_parameter(const vector<TensorView>& parameters, size_t rank)
{
    for (const TensorView& parameter : parameters)
        if (parameter.get_shape().get_rank() == rank) return &parameter;
    return nullptr;
}

// Build a one-output sigmoid head, run it, and compute what the definition says
// it should have produced from the very parameters it used - so nothing depends
// on how the parameters were drawn.
HeadRun run_head(Index features)
{
    NeuralNetwork network;
    network.add_layer(make_unique<opennn::Dense>(Shape{features}, Shape{1}, "Sigmoid"));
    network.compile();
    network.set_parameters_glorot();
    network.copy_parameters_device();

    const vector<TensorView>& parameters = network.get_layer(0)->get_parameter_views();
    const TensorView* weights = find_parameter(parameters, 2);
    const TensorView* bias = find_parameter(parameters, 1);
    EXPECT_NE(weights, nullptr);
    EXPECT_NE(bias, nullptr);

    vector<float> weight_host(static_cast<size_t>(features), 0.0f);
    vector<float> bias_host(1, 0.0f);
    copy_device_to_host_float(weights->get_data(), weights->get_type(), features,
                              weight_host.data(), device::get_compute_stream());
    copy_device_to_host_float(bias->get_data(), bias->get_type(), 1,
                              bias_host.data(), device::get_compute_stream());
    device::synchronize(device::get_compute_stream());

    const MatrixR inputs = MatrixR::Random(rows, features) * 0.05f;

    HeadRun run;
    run.expected.resize(static_cast<size_t>(rows));
    for (Index row = 0; row < rows; ++row)
    {
        float sum = bias_host[0];
        for (Index feature = 0; feature < features; ++feature)
            sum += inputs(row, feature) * weight_host[static_cast<size_t>(feature)];
        run.expected[static_cast<size_t>(row)] = 1.0f / (1.0f + expf(-sum));
    }

    const TensorView input_view(const_cast<float*>(inputs.data()),
                                Shape{rows, features}, Type::FP32);

    ForwardPropagation forward_propagation(rows, &network);
    network.forward_propagate({input_view}, forward_propagation, ForwardPropagationMode::Inference);

    const TensorView& outputs = forward_propagation.get_outputs();
    run.measured.assign(static_cast<size_t>(rows), 0.0f);
    copy_device_to_host_float(outputs.get_data(), outputs.get_type(), rows,
                              run.measured.data(), device::get_compute_stream());
    device::synchronize(device::get_compute_stream());

    return run;
}

void expect_near(const HeadRun& run, float tolerance)
{
    for (Index row = 0; row < rows; ++row)
        EXPECT_NEAR(run.measured[static_cast<size_t>(row)],
                    run.expected[static_cast<size_t>(row)], tolerance) << "row " << row;
}

}

TEST(SingleOutputActivationFusion, FusedHeadMatchesTheDefinition)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    Configuration::instance().set(Device::CUDA, Type::FP32);
    expect_near(run_head(fused_features), 1e-5f);
}

TEST(SingleOutputActivationFusion, RefusedShapeStillGetsItsActivation)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    Configuration::instance().set(Device::CUDA, Type::FP32);
    expect_near(run_head(refused_features), 1e-5f);
}

TEST(SingleOutputActivationFusion, Bf16FusedHeadMatchesTheDefinition)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    Configuration::instance().set(Device::CUDA, Type::BF16);
    expect_near(run_head(fused_features), 3e-3f);

    Configuration::instance().set(Device::CUDA, Type::FP32);
}

TEST(SingleOutputActivationFusion, Bf16RefusedShapeStillGetsItsActivation)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    Configuration::instance().set(Device::CUDA, Type::BF16);
    expect_near(run_head(refused_features), 3e-3f);

    Configuration::instance().set(Device::CUDA, Type::FP32);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
