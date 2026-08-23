//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N   O P E R A T O R   T E S T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// The convolution's forward maths is checked against cuDNN and against
// numerical gradients elsewhere. What was not checked is its parameter
// contract: the bias comes first when there is one and the slot vanishes when
// there is not, the kernel is (kernels, height, width, channels) rather than
// any of the other five orderings, and Glorot's fan-in and fan-out are the
// kernel area times the channel counts rather than the counts alone.
//
// A wrong kernel axis order still links, still runs, and still trains -- to a
// different model.

#include "tests/pch.h"

#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/operators/convolution_operator.h"

using namespace opennn;

namespace
{

constexpr Index kernels_number = 4;
constexpr Index kernel_height = 3;
constexpr Index kernel_width = 5;      // deliberately not square
constexpr Index kernel_channels = 2;


// Non-copyable, so it is configured in place rather than returned.
void configure(ConvolutionOperator& convolution, bool use_bias)
{
    convolution.kernels_number = kernels_number;
    convolution.kernel_height = kernel_height;
    convolution.kernel_width = kernel_width;
    convolution.kernel_channels = kernel_channels;
    convolution.use_bias = use_bias;
}

}


TEST(ConvolutionOperatorTest, KernelShapeIsKernelsHeightWidthChannels)
{
    ConvolutionOperator convolution;
    configure(convolution, false);

    const vector<TensorSpec> specs = convolution.parameter_specs();

    ASSERT_EQ(specs.size(), size_t(1));

    // Height and width differ, so a transposed pair would be caught rather
    // than silently agreeing.
    EXPECT_EQ(specs[0].shape,
              (Shape{kernels_number, kernel_height, kernel_width, kernel_channels}));
}


TEST(ConvolutionOperatorTest, BiasComesFirstAndIsOnePerKernel)
{
    ConvolutionOperator convolution;
    configure(convolution, true);

    const vector<TensorSpec> specs = convolution.parameter_specs();

    ASSERT_EQ(specs.size(), size_t(2));
    EXPECT_EQ(specs[0].shape, (Shape{kernels_number}));
    EXPECT_EQ(specs[1].shape,
              (Shape{kernels_number, kernel_height, kernel_width, kernel_channels}));
}


TEST(ConvolutionOperatorTest, LinkParametersBindsInSpecOrder)
{
    VectorR bias_storage = VectorR::Zero(kernels_number);
    VectorR weight_storage =
        VectorR::Zero(kernels_number * kernel_height * kernel_width * kernel_channels);

    const TensorView bias_view(bias_storage.data(), {kernels_number}, Type::FP32, Device::CPU);
    const TensorView weight_view(weight_storage.data(),
                                 {kernels_number, kernel_height, kernel_width, kernel_channels},
                                 Type::FP32, Device::CPU);

    ConvolutionOperator with_bias;
    configure(with_bias, true);
    const vector<TensorView> both = {bias_view, weight_view};
    with_bias.link_parameters(both);

    EXPECT_EQ(with_bias.bias.get_data(), bias_storage.data());
    EXPECT_EQ(with_bias.weights.get_data(), weight_storage.data());

    // Without a bias the single view is the weights, and bias must be cleared
    // rather than left pointing at whatever a previous link left behind.
    ConvolutionOperator without_bias;
    configure(without_bias, false);
    const vector<TensorView> weights_only = {weight_view};
    without_bias.link_parameters(weights_only);

    EXPECT_EQ(without_bias.weights.get_data(), weight_storage.data());
    EXPECT_TRUE(without_bias.bias.empty());
}


TEST(ConvolutionOperatorTest, GlorotCountsTheKernelAreaInBothFans)
{
    VectorR bias_storage = VectorR::Constant(kernels_number, 5.0f);
    VectorR weight_storage =
        VectorR::Zero(kernels_number * kernel_height * kernel_width * kernel_channels);

    const TensorView bias_view(bias_storage.data(), {kernels_number}, Type::FP32, Device::CPU);
    const TensorView weight_view(weight_storage.data(),
                                 {kernels_number, kernel_height, kernel_width, kernel_channels},
                                 Type::FP32, Device::CPU);

    ConvolutionOperator convolution;
    configure(convolution, true);
    const vector<TensorView> views = {bias_view, weight_view};
    convolution.link_parameters(views);

    convolution.set_parameters_glorot();

    // fan_in = area * channels, fan_out = area * kernels. Dropping the area --
    // the easy mistake, since a Dense layer has none -- gives a limit several
    // times larger.
    const float area = float(kernel_height * kernel_width);
    const float limit = sqrt(6.0f / (area * kernel_channels + area * kernels_number));

    const float limit_without_area = sqrt(6.0f / float(kernel_channels + kernels_number));
    EXPECT_GT(limit_without_area, limit * 2.0f)
        << "the two candidate limits must differ enough for this to discriminate";

    for (Index i = 0; i < weight_storage.size(); ++i)
        EXPECT_LE(abs(weight_storage(i)), limit + 1.0e-6f) << "at index " << i;

    for (Index i = 0; i < bias_storage.size(); ++i)
        EXPECT_FLOAT_EQ(bias_storage(i), 0.0f) << "at index " << i;
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
