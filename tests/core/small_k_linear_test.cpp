//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S M A L L   K   L I N E A R   T E S T   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// A bf16 dense forward whose contraction is at most 32 runs
// small_k_linear_forward_cuda (opennn/core/cuda/kernel_small_k_linear.cu)
// instead of cuBLASLt: the layer is an output write, and the kernel keeps it
// near the write floor. The arithmetic is the same fp32 accumulation, so the
// test compares against a host reference at bf16 tolerance.
//
// What needs a test is the tiling. Rows are worked 64 at a time, 16 per
// warp, in a grid-stride loop over whole columns of tiles, so the row counts
// here cover a single partial tile, tiles with a ragged last one, more tiles
// than resident blocks, and the row-chunked forward above 16,384 rows. The
// contraction is padded to 32 with zeros, so it is exercised at the HIGGS
// width, at the pad boundary, and at the narrowest even width. The kernel
// declines what it does not cover - fp32, an odd contraction, an output
// width that is not a multiple of 64 - and cuBLASLt must then give the same
// answer, which the last tests check.

#include "tests/pch.h"

#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/tensor_operations.h"
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

void check_forward(Index rows, Index features, Index outputs, const string& activation,
                   Type precision, float tolerance)
{
    Configuration::instance().set(Device::CUDA, precision);

    NeuralNetwork network;
    network.add_layer(make_unique<opennn::Dense>(Shape{features}, Shape{outputs}, activation));
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

    // Every row distinct, so a row written from the wrong tile cannot pass by
    // coincidence, and a mix of signs so the ReLU clamps some of the output.
    MatrixR inputs(rows, features);
    for (Index row = 0; row < rows; ++row)
        for (Index feature = 0; feature < features; ++feature)
            inputs(row, feature) = 0.01f * float((row * 7 + feature * 13) % 101) - 0.5f;

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
            if (activation == "ReLU") expected = expected > 0.0f ? expected : 0.0f;

            const float got = measured[static_cast<size_t>(row * outputs + column)];
            if (fabs(got - expected) > tolerance)
            {
                ++reported;
                EXPECT_NEAR(got, expected, tolerance)
                    << "rows " << rows << ", features " << features << ", outputs " << outputs
                    << ", row " << row << ", column " << column;
                break;
            }
        }

    Configuration::instance().set(Device::CUDA, Type::FP32);
}

#ifdef OPENNN_HAS_CUDA

// One device allocation holding a bf16 or fp32 copy of a host vector.
struct DeviceTensor
{
    void* data = nullptr;
    Index bytes = 0;
    Type type = Type::FP32;

    DeviceTensor(const vector<float>& host, Type type_) : type(type_)
    {
        const size_t element_bytes = type == Type::BF16 ? sizeof(uint16_t) : sizeof(float);
        bytes = Index(host.size() * element_bytes);
        data = device::allocate(Device::CUDA, bytes);

        if (type == Type::BF16)
        {
            vector<uint16_t> staged(host.size());
            for (size_t i = 0; i < host.size(); ++i)
                staged[i] = float_to_bfloat16_host(host[i]);
            device::copy_async(data, staged.data(), bytes, device::CopyKind::HostToDevice);
        }
        else
        {
            device::copy_async(data, host.data(), bytes, device::CopyKind::HostToDevice);
        }

        device::synchronize();
    }

    TensorView view(const Shape& shape) const
    {
        return TensorView(data, shape, type, Device::CUDA);
    }

    ~DeviceTensor() { device::deallocate(Device::CUDA, data, bytes); }

    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;
};

// linear_forward straight from device views: bf16 input, weights and output
// with an fp32 bias, which the Dense layer never produces under bf16 compute
// (its bias is bf16). The kernel reads that bias without the cast launch the
// cuBLASLt path needs, so both bias dtypes must agree with the host reference.
void check_direct_forward(Index rows, Index features, Index outputs, Type bias_type,
                          cublasLtEpilogue_t epilogue, float tolerance)
{
    vector<float> input_host(static_cast<size_t>(rows * features));
    vector<float> weight_host(static_cast<size_t>(features * outputs));
    vector<float> bias_host(static_cast<size_t>(outputs));

    for (Index row = 0; row < rows; ++row)
        for (Index feature = 0; feature < features; ++feature)
            input_host[static_cast<size_t>(row * features + feature)]
                = 0.01f * float((row * 7 + feature * 13) % 101) - 0.5f;

    for (Index feature = 0; feature < features; ++feature)
        for (Index column = 0; column < outputs; ++column)
            weight_host[static_cast<size_t>(feature * outputs + column)]
                = 0.0625f * float((feature * 5 + column * 3) % 17) - 0.5f;

    for (Index column = 0; column < outputs; ++column)
        bias_host[static_cast<size_t>(column)] = 0.125f * float((column * 11) % 13) - 0.75f;

    // The kernel accumulates the bf16-rounded operands; the reference must too.
    for (float& value : input_host) value = bfloat16_to_float_host(float_to_bfloat16_host(value));
    for (float& value : weight_host) value = bfloat16_to_float_host(float_to_bfloat16_host(value));
    if (bias_type == Type::BF16)
        for (float& value : bias_host) value = bfloat16_to_float_host(float_to_bfloat16_host(value));

    const DeviceTensor input(input_host, Type::BF16);
    const DeviceTensor weights(weight_host, Type::BF16);
    const DeviceTensor bias(bias_host, bias_type);
    const DeviceTensor output(vector<float>(static_cast<size_t>(rows * outputs), 0.0f), Type::BF16);

    TensorView output_view = output.view(Shape{rows, outputs});
    linear_forward(input.view(Shape{rows, features}), weights.view(Shape{features, outputs}),
                   bias.view(Shape{outputs}), output_view, epilogue);

    vector<float> measured(static_cast<size_t>(rows * outputs), 0.0f);
    copy_device_to_host_float(output.data, Type::BF16, rows * outputs,
                              measured.data(), device::get_compute_stream());
    device::synchronize(device::get_compute_stream());

    const bool relu = epilogue == CUBLASLT_EPILOGUE_RELU_BIAS;

    Index reported = 0;
    for (Index row = 0; row < rows && reported < 5; ++row)
        for (Index column = 0; column < outputs; ++column)
        {
            float expected = bias_host[static_cast<size_t>(column)];
            for (Index feature = 0; feature < features; ++feature)
                expected += input_host[static_cast<size_t>(row * features + feature)]
                          * weight_host[static_cast<size_t>(feature * outputs + column)];
            if (relu) expected = expected > 0.0f ? expected : 0.0f;

            const float got = measured[static_cast<size_t>(row * outputs + column)];
            if (fabs(got - expected) > tolerance * max(1.0f, fabs(expected)))
            {
                ++reported;
                EXPECT_NEAR(got, expected, tolerance * max(1.0f, fabs(expected)))
                    << "rows " << rows << ", features " << features << ", outputs " << outputs
                    << ", bias " << (bias_type == Type::BF16 ? "bf16" : "fp32")
                    << ", row " << row << ", column " << column;
                break;
            }
        }
}

#endif

}

// 40 rows: one tile, 24 rows of it past the end, three of the four warps idle.
TEST(SmallKLinear, PartialTile)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    check_forward(40, 28, 1024, "ReLU", Type::BF16, 4e-2f);
}

// 1000 rows: fifteen full tiles and a 40-row one; fewer tiles than blocks.
TEST(SmallKLinear, RaggedLastTile)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    check_forward(1000, 28, 1024, "ReLU", Type::BF16, 4e-2f);
}

// The benchmark shape: more tiles than resident blocks, so blocks loop.
TEST(SmallKLinear, GridStrideOverTiles)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    check_forward(8192, 28, 1024, "ReLU", Type::BF16, 4e-2f);
}

// 20,000 rows is two chunks of the row-chunked forward, the second partial.
TEST(SmallKLinear, CrossesTheRowChunkGate)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    check_forward(20000, 28, 256, "ReLU", Type::BF16, 4e-2f);
}

// Contraction at the pad boundary, and the narrowest even one, with a
// bias-only epilogue (the activation is not fused).
TEST(SmallKLinear, ContractionBoundsAndBiasEpilogue)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    check_forward(777, 32, 64, "ReLU", Type::BF16, 4e-2f);
    check_forward(777, 2, 128, "Identity", Type::BF16, 4e-2f);
}

// Shapes the kernel declines fall through to cuBLASLt and are still right.
TEST(SmallKLinear, DeclinedShapesFallThrough)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    check_forward(1000, 27, 1024, "ReLU", Type::BF16, 4e-2f);   // odd contraction
    check_forward(1000, 28, 96, "ReLU", Type::BF16, 4e-2f);     // outputs not a multiple of 64
    check_forward(1000, 28, 1024, "ReLU", Type::FP32, 1e-4f);   // fp32
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

// The fp32-bias instantiations, reached only through linear_forward itself.
TEST(SmallKLinear, Fp32BiasFromDeviceViews)
{
#ifdef OPENNN_HAS_CUDA
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    check_direct_forward(1000, 28, 1024, Type::FP32, CUBLASLT_EPILOGUE_RELU_BIAS, 1e-2f);
    check_direct_forward(1000, 28, 1024, Type::FP32, CUBLASLT_EPILOGUE_BIAS, 1e-2f);
    check_direct_forward(1000, 28, 1024, Type::BF16, CUBLASLT_EPILOGUE_RELU_BIAS, 1e-2f);
    check_direct_forward(200, 32, 64, Type::BF16, CUBLASLT_EPILOGUE_BIAS, 1e-2f);
#else
    GTEST_SKIP() << "Built without CUDA.";
#endif
}
