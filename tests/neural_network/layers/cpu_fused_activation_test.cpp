//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C P U   F U S E D   A C T I V A T I O N   T E S T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// A layer marks its activation operator "fused" from the activation alone, with
// no device in the decision, and the operator then skips its own pass. Every
// path that claims the fusion therefore has to apply the activation on the CPU
// too. Three of them did not, and no test could see it: the pre-activation was
// non-negative everywhere the suite looked, and a ReLU of a non-negative number
// is the number. Every case below feeds inputs whose pre-activation is negative,
// which is the only place the missing activation shows.

#include "tests/pch.h"

#include "opennn/core/tensor_types.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/layers/convolutional_layer.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/neural_network.h"

using namespace opennn;

namespace
{

vector<float> forward_once(NeuralNetwork& neural_network,
                           float* input_data,
                           const Shape& input_shape)
{
    const Index batch_size = input_shape[0];

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(input_data, input_shape) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    const TensorView outputs = forward_propagation.get_outputs();
    const float* data = outputs.as<float>();

    return vector<float>(data, data + outputs.size());
}

}


// A bias-free Dense asks the combination for CUBLASLT_EPILOGUE_RELU, not
// RELU_BIAS. The CPU epilogue honoured only the latter, so the layer behaved as
// Identity.

TEST(CpuFusedActivationTest, BiasFreeDenseAppliesRelu)
{
    NeuralNetwork neural_network;
    auto dense = make_unique<opennn::Dense>(Shape{2}, Shape{3}, "ReLU");
    dense->set_use_bias(false);
    neural_network.add_layer(std::move(dense));
    neural_network.compile();

    neural_network.set_parameters(VectorR::Ones(neural_network.get_parameters_buffer_size()));

    // Weights are all one and there is no bias, so every output is 1 + (-3) = -2
    // before the activation.
    vector<float> inputs = {1.0f, -3.0f};
    const vector<float> outputs = forward_once(neural_network, inputs.data(), Shape{1, 2});

    ASSERT_EQ(outputs.size(), size_t(3));
    for (const float value : outputs)
        EXPECT_FLOAT_EQ(value, 0.0f);
}


// A Dense whose width is a multiple of eight fuses GELUTanh through the
// cuBLASLt GELU_AUX_BIAS epilogue, which writes the activated result to a second
// slot. There is no such epilogue on the CPU: the fall-through wrote the
// combination into the first slot and left the output slot untouched, so the
// layer returned zeros.

TEST(CpuFusedActivationTest, WidthEightDenseAppliesGeluTanh)
{
    constexpr Index output_features = 8;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{2}, Shape{output_features}, "GELUTanh"));
    neural_network.compile();

    neural_network.set_parameters(VectorR::Ones(neural_network.get_parameters_buffer_size()));

    // Weights and bias are one, so the pre-activation is 1 + (-3) + 1 = -1.
    vector<float> inputs = {1.0f, -3.0f};
    const vector<float> outputs = forward_once(neural_network, inputs.data(), Shape{1, 2});

    const float expected = gelu_tanh_value(-1.0f);

    ASSERT_EQ(outputs.size(), size_t(output_features));
    EXPECT_LT(expected, 0.0f);
    for (const float value : outputs)
        EXPECT_NEAR(value, expected, 1.0e-5f);
}


// A Convolutional without batch normalization folds its ReLU into the
// convolution, which on CUDA is a cuDNN epilogue. The CPU im2col path had none,
// so the activation was lost entirely.

TEST(CpuFusedActivationTest, ConvolutionAppliesReluWithoutBatchNorm)
{
    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Convolutional>(
        Shape{3, 3, 1}, Shape{3, 3, 1, 2}, "ReLU", Shape{1, 1}, "Valid", false));
    neural_network.compile();

    neural_network.get_parameters_map().setConstant(1.0f);

    // Nine kernel taps of one over an input of minus one, plus a bias of one:
    // every output is -8 before the activation.
    vector<float> inputs(9, -1.0f);
    const vector<float> outputs = forward_once(neural_network, inputs.data(), Shape{1, 3, 3, 1});

    ASSERT_EQ(outputs.size(), size_t(2));
    for (const float value : outputs)
        EXPECT_FLOAT_EQ(value, 0.0f);
}


// The same convolution with batch normalization: the ReLU moves to the batch
// norm epilogue, which the CPU path also omitted. Inference normalizes with the
// initial running statistics (mean zero, variance one), so the pre-activation is
// the convolution scaled by gamma and shifted by beta - negative here.

TEST(CpuFusedActivationTest, ConvolutionAppliesReluWithBatchNorm)
{
    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Convolutional>(
        Shape{3, 3, 1}, Shape{3, 3, 1, 2}, "ReLU", Shape{1, 1}, "Valid", true));
    neural_network.compile();

    neural_network.get_parameters_map().setConstant(1.0f);

    vector<float> inputs(9, -1.0f);
    const vector<float> outputs = forward_once(neural_network, inputs.data(), Shape{1, 3, 3, 1});

    ASSERT_EQ(outputs.size(), size_t(2));
    for (const float value : outputs)
        EXPECT_FLOAT_EQ(value, 0.0f);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
