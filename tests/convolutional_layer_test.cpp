#include "pch.h"


TEST(ConvolutionalLayerTest, EigenConvolution)
{
    Tensor<type, 2> input_2;
    Tensor<type, 2> kernel_2;
    Tensor<type, 2> output_2;

    std::cout << "Hello";

    Eigen::array<ptrdiff_t, 2> dimensions_2 = { 0, 1 };

    Tensor<type, 3> input_3;
    Tensor<type, 3> kernel_3;
    Tensor<type, 3> output_3;

    const Eigen::array<ptrdiff_t, 3> dimensions_3 = { 0, 1, 2 };

    Tensor<type, 4> input_4;
    Tensor<type, 3> kernel_4;
    Tensor<type, 4> output_4;

    const Eigen::array<ptrdiff_t, 3> dimensions_4 = { 1, 2, 3 };

    // Convolution 2D, 1 channel

    input_2.resize(3, 3);
    input_2.setRandom();

    kernel_2.resize(2, 2);
    kernel_2.setRandom();

    output_2 = input_2.convolve(kernel_2, dimensions_2);

//    EXPECT_EQ(output_2.dimension(0), dimensions{ 2,1 });

//    EXPECT_EQ( == 2);
//    EXPECT_EQ(output_2.dimension(1) == 2);

    // Convolution 3D, 3 channels

    input_3.resize(5, 5, 3);
    input_3.setRandom();

    kernel_3.resize(2, 2, 3);
    kernel_3.setRandom();

    output_3 = input_3.convolve(kernel_3, dimensions_3);

//    EXPECT_EQ(output_3.dimension(0) == 4);
//    EXPECT_EQ(output_3.dimension(1) == 4);
//    EXPECT_EQ(output_3.dimension(2) == 1);

    // Convolution 2D, 3 channels, multiple images, 1 kernel

    input_4.resize(10, 3, 5, 5);
    input_4.setConstant(type(1));
    input_4.chip(1, 0).setConstant(type(2));
    input_4.chip(2, 0).setConstant(type(3));

    kernel_4.resize(3, 2, 2);

    kernel_4.setConstant(type(1.0 / 12.0));

    output_4 = input_4.convolve(kernel_4, dimensions_4);

//    EXPECT_EQ(output_3.dimension(0) == 10);
//    EXPECT_EQ(output_3.dimension(1) == 1);
//    EXPECT_EQ(output_3.dimension(2) == 4);
//    EXPECT_EQ(output_3.dimension(3) == 4);

    // EXPECT_EQ(abs(output_3(0, 0, 0, 0) - type(1)) < type(NUMERIC_LIMITS_MIN)
    //                 && abs(output_3(0, 0, 0, 1) - type(1)) < type(NUMERIC_LIMITS_MIN)
    //                 && abs(output_3(0, 0, 0, 2) - type(1)) < type(NUMERIC_LIMITS_MIN)
    //                 && abs(output_3(0, 0, 0, 3) - type(1)) < type(NUMERIC_LIMITS_MIN)
    //                 && abs(output_3(0, 0, 1, 0) - type(1)) < type(NUMERIC_LIMITS_MIN)
    //                 && abs(output_3(0, 0, 1, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(0, 0, 1, 2) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(0, 0, 1, 3) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(0, 0, 2, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(0, 0, 2, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(0, 0, 2, 2) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(0, 0, 2, 3) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(0, 0, 3, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(0, 0, 3, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(0, 0, 3, 2) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(0, 0, 3, 3) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(1, 0, 0, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(1, 0, 0, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(1, 0, 0, 2) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(1, 0, 0, 3) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(1, 0, 1, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(1, 0, 1, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(1, 0, 1, 2) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(1, 0, 1, 3) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(1, 0, 2, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(1, 0, 2, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(1, 0, 2, 2) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(1, 0, 2, 3) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(1, 0, 3, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(1, 0, 3, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(1, 0, 3, 2) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(1, 0, 3, 3) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(2, 0, 0, 0) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(2, 0, 0, 1) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(2, 0, 0, 2) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(2, 0, 0, 3) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(2, 0, 1, 0) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(2, 0, 1, 1) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(2, 0, 1, 2) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(2, 0, 1, 3) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(2, 0, 2, 0) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(2, 0, 2, 1) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(2, 0, 2, 2) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(2, 0, 2, 3) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(2, 0, 3, 0) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(2, 0, 3, 1) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(2, 0, 3, 2) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
    //             abs(output_3(2, 0, 3, 3) - type(3)) <= type(NUMERIC_LIMITS_MIN));

    // Convolution 3D, 2 channels

    Tensor<type, 3> input(3, 3, 2);
    Tensor<type, 3> kernel(2, 2, 2);
    Tensor<type, 3> output(2, 2, 1);

    for (int i = 0; i < 3 * 3 * 2; i++)
        *(input.data() + i) = i;

    for (int i = 0; i < 2 * 2 * 2; i++)
        *(kernel.data() + i) = i + 1;

    const Eigen::array<ptrdiff_t, 3> dimensions = { 0, 1, 2 };

    output = input.convolve(kernel, dimensions);

//    EXPECT_EQ(fabs(output(0, 0, 0) - 320) < type(NUMERIC_LIMITS_MIN));
//    EXPECT_EQ(fabs(output(1, 0, 0) - 356) < type(NUMERIC_LIMITS_MIN));
//    EXPECT_EQ(fabs(output(0, 1, 0) - 428) < type(NUMERIC_LIMITS_MIN));
//    EXPECT_EQ(fabs(output(1, 1, 0) - 464) < type(NUMERIC_LIMITS_MIN));
}


TEST(ConvolutionalLayerTest, DefaultConstructor)
{
    const dimensions input_dimensions = { 28, 29, 1 };
    const dimensions kernel_dimensions = { 3, 2, 1, 16 };

    ConvolutionalLayer convolutional_layer(input_dimensions, kernel_dimensions);

    EXPECT_EQ(convolutional_layer.get_input_height(), 28);
    EXPECT_EQ(convolutional_layer.get_input_width(), 29);
    EXPECT_EQ(convolutional_layer.get_input_channels(), 1);

    EXPECT_EQ(convolutional_layer.get_kernel_height(), 3);
    EXPECT_EQ(convolutional_layer.get_kernel_width(), 2);
    EXPECT_EQ(convolutional_layer.get_kernel_channels(), 1);
    EXPECT_EQ(convolutional_layer.get_kernels_number(), 16);
}


/*
namespace opennn
{

void ConvolutionalLayerTest::test_calculate_convolutions()
{
    // 2 images 1 channel 1 kernel

    Tensor<unsigned char, 3> bmp_image_1;
    Tensor<unsigned char, 3> bmp_image_2;

    ConvolutionalLayer convolutional_layer;

    dimensions input_dimensions;
    dimensions kernel_dimensions;

    bmp_image_1 = read_bmp_image("../examples/mnist/data/images/one/1_0.bmp");
    bmp_image_2 = read_bmp_image("../examples/mnist/data/images/one/1_1.bmp");

    const Index input_images = 2;
    const Index kernels_number = 1;

    const Index input_height = bmp_image_1.dimension(0); // 28
    const Index input_width = bmp_image_1.dimension(1); // 28
    const Index kernel_height = 27;
    const Index kernel_width = 27;

    const Index channels = bmp_image_1.dimension(2); // 1

    Tensor<type, 4> inputs(input_images,
                           input_height,
                           input_width,
                           channels);

    Tensor<type, 4> kernels(kernel_height,
                            kernel_width,
                            channels,
                            kernels_number);

    Tensor<type, 1> biases(kernels_number);

    // Copy bmp_image data into inputs

    for (int h = 0; h < input_height; ++h) 
        for (int w = 0; w < input_width; ++w) 
            for (int c = 0; c < channels; ++c) 
                inputs(0, h, w, c) = type(bmp_image_1(h, w, c));

    for (int h = 0; h < input_height; ++h) 
        for (int w = 0; w < input_width; ++w) 
            for (int c = 0; c < channels; ++c) 
                inputs(1, h, w, c) = type(bmp_image_2(h, w, c));

    kernels.setConstant(type(1));

    biases.setConstant(type(1));

    input_dimensions = {input_height, input_width, channels};
    kernel_dimensions = {kernel_height, kernel_width, channels, kernels_number};

    convolutional_layer.set(input_dimensions, kernel_dimensions);

    Tensor<type, 4> convolutions(input_images,
                                 convolutional_layer.get_output_height(),
                                 convolutional_layer.get_output_width(),
                                 kernels_number);

    convolutional_layer.set_biases(biases);
    convolutional_layer.set_synaptic_weights(kernels);

    convolutional_layer.calculate_convolutions(inputs, convolutions);

    EXPECT_EQ(convolutions(0, 0, 0, 0) == type(9871+1) 
             && convolutions(1, 0, 0, 0) == type(13855+1));

}


void ConvolutionalLayerTest::test_calculate_activations()
{
    Tensor<type, 4> inputs;
    Tensor<type, 4> activations;
    Tensor<type, 4> activation_derivatives;
    Tensor<type, 4> result;

    ConvolutionalLayer convolutional_layer;

    activations.resize(2, 2, 2, 2);
    activation_derivatives.resize(2, 2, 2, 2);

    inputs.resize(2, 2, 2, 2);

    inputs(0,0,0,0) = type(-1.111f);
    inputs(0,0,0,1) = type(-1.112);
    inputs(0,0,1,0) = type(-1.121);
    inputs(0,0,1,1) = type(-1.122f);
    inputs(0,1,0,0) = type(1.211f);
    inputs(0,1,0,1) = type(1.212f);
    inputs(0,1,1,0) = type(1.221f);
    inputs(0,1,1,1) = type(1.222f);
    inputs(1,0,0,0) = type(-2.111f);
    inputs(1,0,0,1) = type(-2.112f);
    inputs(1,0,1,0) = type(-2.121f);
    inputs(1,0,1,1) = type(-2.122f);
    inputs(1,1,0,0) = type(2.211f);
    inputs(1,1,0,1) = type(2.212f);
    inputs(1,1,1,0) = type(2.221f);
    inputs(1,1,1,1) = type(2.222f);

    // Test

    convolutional_layer.set_activation_function(opennn::ConvolutionalLayer::ActivationFunction::HyperbolicTangent);

    convolutional_layer.calculate_activations(activations, activation_derivatives);

    result.resize(2, 2, 2, 2);

    result(0,0,0,0) = type(0.352916f);
    result(0,0,0,1) = type(0.352348f);
    result(0,0,1,0) = type(0.347271f);
    result(0,0,1,1) = type(0.346710f);
    result(0,1,0,0) = type(0.299466f);
    result(0,1,0,1) = type(0.298965f);
    result(0,1,1,0) = type(0.294486f);
    result(0,1,1,1) = type(0.293991f);
    result(1,0,0,0) = type(0.056993f);
    result(1,0,0,1) = type(0.056882f);
    result(1,0,1,0) = type(0.055896f);
    result(1,0,1,1) = type(0.055788f);
    result(1,1,0,0) = type(0.046907f);
    result(1,1,0,1) = type(0.046816f);
    result(1,1,1,0) = type(0.046000f);
    result(1,1,1,1) = type(0.045910f);

    EXPECT_EQ(abs(activation_derivatives(0,0,0,0) - result(0,0,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(0,0,0,1) - result(0,0,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(0,0,1,0) - result(0,0,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(0,0,1,1) - result(0,0,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(0,1,0,0) - result(0,1,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(0,1,0,1) - result(0,1,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(0,1,1,0) - result(0,1,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(0,1,1,1) - result(0,1,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(1,0,0,0) - result(1,0,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(1,0,0,1) - result(1,0,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(1,0,1,0) - result(1,0,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(1,0,1,1) - result(1,0,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(1,1,0,0) - result(1,1,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(1,1,0,1) - result(1,1,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(1,1,1,0) - result(1,1,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(1,1,1,1) - result(1,1,1,1)) < type(NUMERIC_LIMITS_MIN));

    // Test

    convolutional_layer.set_activation_function(opennn::ConvolutionalLayer::ActivationFunction::RectifiedLinear);

    convolutional_layer.calculate_activations(activations, activation_derivatives);

    result(0,0,0,0) = type(0);
    result(0,0,0,1) = type(0);
    result(0,0,1,0) = type(0);
    result(0,0,1,1) = type(0);
    result(0,1,0,0) = type(1);
    result(0,1,0,1) = type(1);
    result(0,1,1,0) = type(1);
    result(0,1,1,1) = type(1);
    result(1,0,0,0) = type(0);
    result(1,0,0,1) = type(0);
    result(1,0,1,0) = type(0);
    result(1,0,1,1) = type(0);
    result(1,1,0,0) = type(1);
    result(1,1,0,1) = type(1);
    result(1,1,1,0) = type(1);
    result(1,1,1,1) = type(1);

    EXPECT_EQ((activation_derivatives(0, 0, 0, 0) - result(0, 0, 0, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activation_derivatives(0, 1, 0, 0) - result(0, 1, 0, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activation_derivatives(1, 0, 0, 0) - result(1, 0, 0, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activation_derivatives(1, 1, 0, 0) - result(1, 1, 0, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activation_derivatives(0, 0, 1, 0) - result(0, 0, 1, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activation_derivatives(0, 1, 1, 0) - result(0, 1, 1, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activation_derivatives(1, 0, 1, 0) - result(1, 0, 1, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activation_derivatives(1, 1, 1, 0) - result(1, 1, 1, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activation_derivatives(0, 0, 0, 1) - result(0, 0, 0, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activation_derivatives(0, 1, 0, 1) - result(0, 1, 0, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activation_derivatives(1, 0, 0, 1) - result(1, 0, 0, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activation_derivatives(1, 1, 0, 1) - result(1, 1, 0, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activation_derivatives(0, 0, 1, 1) - result(0, 0, 1, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activation_derivatives(0, 1, 1, 1) - result(0, 1, 1, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activation_derivatives(1, 0, 1, 1) - result(1, 0, 1, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activation_derivatives(1, 1, 1, 1) - result(1, 1, 1, 1)) <= type(NUMERIC_LIMITS_MIN));

    // Test

    convolutional_layer.set_activation_function(opennn::ConvolutionalLayer::ActivationFunction::SoftPlus);

    convolutional_layer.calculate_activations(activations, activation_derivatives);

    result(0,0,0,0) = type(0.247685f);
    result(0,0,0,1) = type(0.247498f);
    result(0,0,1,0) = type(0.245826f);
    result(0,0,1,1) = type(0.245640f);
    result(0,1,0,0) = type(0.770476f);
    result(0,1,0,1) = type(0.770653f);
    result(0,1,1,0) = type(0.772239f);
    result(0,1,1,1) = type(0.772415f);
    result(1,0,0,0) = type(0.108032f);
    result(1,0,0,1) = type(0.107936f);
    result(1,0,1,0) = type(0.107072f);
    result(1,0,1,1) = type(0.106977f);
    result(1,1,0,0) = type(0.901233f);
    result(1,1,0,1) = type(0.901322f);
    result(1,1,1,0) = type(0.902120f);
    result(1,1,1,1) = type(0.902208f);

    EXPECT_EQ(abs(activation_derivatives(0,0,0,0) - result(0,0,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(0,0,0,1) - result(0,0,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(0,0,1,0) - result(0,0,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(0,0,1,1) - result(0,0,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(0,1,0,0) - result(0,1,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(0,1,0,1) - result(0,1,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(0,1,1,0) - result(0,1,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(0,1,1,1) - result(0,1,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(1,0,0,0) - result(1,0,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(1,0,0,1) - result(1,0,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(1,0,1,0) - result(1,0,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(1,0,1,1) - result(1,0,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(1,1,0,0) - result(1,1,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(1,1,0,1) - result(1,1,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(1,1,1,0) - result(1,1,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activation_derivatives(1,1,1,1) - result(1,1,1,1)) < type(NUMERIC_LIMITS_MIN));
}


void ConvolutionalLayerTest::test_insert_padding()
{
    const Index input_images = 2;
    const Index kernels_number = 3;

    const Index channels = 3;

    const Index input_height = 4;
    const Index input_width = 4;
    const Index kernel_height = 3;
    const Index kernel_width = 3;

    Tensor<type, 4> inputs(input_height, input_width, channels, input_images);
    Tensor<type, 4> kernels(kernel_height, kernel_width, channels, kernels_number);
    Tensor<type, 4> padded(input_height, input_width, channels, input_images);

    inputs.setConstant(type(1));

    dimensions input_dimensions({input_height, input_width, channels, input_images});

    dimensions kernels_dimensions({kernel_height, kernel_width, channels, kernels_number});

    ConvolutionalLayer convolutional_layer(input_dimensions, kernels_dimensions);

    convolutional_layer.set_convolution_type(opennn::ConvolutionalLayer::ConvolutionType::Same);
    convolutional_layer.set(input_dimensions, kernels_dimensions);

    convolutional_layer.insert_padding(inputs, padded);

    EXPECT_EQ(padded.dimension(0) == 6 &&
                padded.dimension(1) == 6,LOG);

    EXPECT_EQ((padded(0, 0, 0, 0) - type(0)) < type(NUMERIC_LIMITS_MIN) &&
                (padded(0, 1, 0, 0) - type(0)) < type(NUMERIC_LIMITS_MIN) &&
                (padded(0, 2, 0, 0) - type(0)) < type(NUMERIC_LIMITS_MIN));

}


void ConvolutionalLayerTest::test_forward_propagate()
{
    // 2 images , 3 channels

    Tensor<unsigned char, 3> bmp_image_1;
    Tensor<unsigned char, 3> bmp_image_2;

    const bool is_training = true;

    const Index input_images = 2;
    const Index kernels_number = 3;

    bmp_image_1 = read_bmp_image("../examples/mnist/data/test/4x4_0.bmp");
    bmp_image_2 = read_bmp_image("../examples/mnist/data/test/4x4_1.bmp");

    const Index channels = bmp_image_1.dimension(2); // 3

    const Index input_height = bmp_image_1.dimension(0); // 4
    const Index input_width = bmp_image_1.dimension(1); // 4
    const Index kernel_height = 3;
    const Index kernel_width = 3;

    dimensions input_dimensions;
    input_dimensions = {input_height, input_width, channels};

    dimensions kernel_dimensions;
    kernel_dimensions = {kernel_height, kernel_width, channels, kernels_number };

    Tensor<type,4> inputs(input_images, input_height, input_width, channels);
    Tensor<type,4> kernel(kernel_height, kernel_width, channels, kernels_number);
    Tensor<type, 1> bias(kernels_number);
    
    for (int h = 0; h < input_height; ++h)
        for (int w = 0; w < input_width; ++w)
            for (int c = 0; c < channels; ++c)
                inputs(0, h, w, c) = type(bmp_image_1(h, w, c));

    for (int h = 0; h < input_height; ++h)
        for (int w = 0; w < input_width; ++w)
            for (int c = 0; c < channels; ++c)
                inputs(1, h, w, c) = type(bmp_image_2(h, w, c));

    bias.setValues({1,1,1});

    kernel.setConstant(type(1));

    Tensor<pair<type*, dimensions>, 1> input_pairs[1];
    input_pairs[0] = {inputs.data(),  { input_images, input_height, input_width, channels }};

    // bmp_image_1:
    //    255 255   0   0   255 255   0   0     255 255   0   0
    //    255 255   0   0   255 255   0   0     255 255   0   0
    //    255 255   0   0   255 255   0   0     255 255   0   0
    //    255 255   0   0   255 255   0   0     255 255   0   0

    // bmp_image_2:
    //    0   0 255 255      0   0 255 255       0   0 255 255
    //    0   0 255 255      0   0 255 255       0   0 255 255
    //    0   0   0   0      0   0   0   0       0   0   0   0
    //    0   0   0   0      0   0   0   0       0   0   0   0

    // kernel x3:
    //    1  1  1
    //    1  1  1
    //    1  1  1

    // Expected outputs: (2,2,2,1)
    //    4590+1  2295+1
    //    4590+1  2295+1
    //    1530+1  3060+1   
    //    765+1   1530+1     
    
    // Test

    convolutional_layer.set(input_dimensions, kernel_dimensions);

    convolutional_layer.set_synaptic_weights(kernel);
    convolutional_layer.set_biases(bias);

    forward_propagation.set(input_images, &convolutional_layer);

    convolutional_layer.set_activation_function(ConvolutionalLayer::ActivationFunction::Linear);

    convolutional_layer.forward_propagate(input_pairs, &forward_propagation, is_training);
 
    EXPECT_EQ(forward_propagation.outputs.dimension(0) == input_images
                && forward_propagation.outputs.dimension(1) == convolutional_layer.get_output_dimensions()[0] 
                && forward_propagation.outputs.dimension(2) == convolutional_layer.get_output_dimensions()[1]
                && forward_propagation.outputs.dimension(3) == convolutional_layer.get_output_dimensions()[2]);

    EXPECT_EQ(forward_propagation.activation_derivatives.dimension(0) == input_images
                && forward_propagation.activation_derivatives.dimension(1) == convolutional_layer.get_input_dimensions()[0]
                && forward_propagation.activation_derivatives.dimension(2) == convolutional_layer.get_input_dimensions()[1]
                && forward_propagation.activation_derivatives.dimension(3) == convolutional_layer.get_input_dimensions()[2]);

    EXPECT_EQ(forward_propagation.outputs(0, 0, 0, 0) == type(4590 + 1)
                && forward_propagation.outputs(1, 0, 0, 0) == type(1530 + 1));

}


void ConvolutionalLayerTest::test_back_propagate() 
{
    // 2 images , 3 channels

    Tensor<unsigned char, 3> bmp_image_1;
    Tensor<unsigned char, 3> bmp_image_2;

    const bool is_training = true;

    const Index input_images = 2;
    const Index kernels_number = 3;

    bmp_image_1 = read_bmp_image("../examples/mnist/data/test/4x4_0.bmp");
    bmp_image_2 = read_bmp_image("../examples/mnist/data/test/4x4_1.bmp");

    const Index channels = bmp_image_1.dimension(2); // 3

    const Index input_height = bmp_image_1.dimension(0); // 4
    const Index input_width = bmp_image_1.dimension(1); // 4
    const Index kernel_height = 3;
    const Index kernel_width = 3;

    ConvolutionalLayer convolutional_layer;

    dimensions input_dimensions;
    input_dimensions = { input_height, input_width, channels };

    dimensions kernel_dimensions;
    kernel_dimensions = { kernel_height, kernel_width, channels, kernels_number };

    Tensor<type, 4> inputs(input_images, input_height, input_width, channels);
    Tensor<type, 4> kernel(kernel_height, kernel_width, channels, kernels_number);
    Tensor<type, 1> bias(kernels_number);

    for (int h = 0; h < input_height; ++h)
        for (int w = 0; w < input_width; ++w)
            for (int c = 0; c < channels; ++c)
                inputs(0, h, w, c) = type(bmp_image_1(h, w, c));

    // Copy bmp_image_2 data into inputs
    for (int h = 0; h < input_height; ++h)
        for (int w = 0; w < input_width; ++w)
            for (int c = 0; c < channels; ++c)
                inputs(1, h, w, c) = type(bmp_image_2(h, w, c));

    bias.setValues({ 1,1,1 });

    kernel.setConstant(type(1));

    Tensor<pair<type*, dimensions>, 1> input_pairs[1];
    input_pairs[0].first = inputs.data();
    input_pairs[0].second = { input_images, input_height, input_width, channels };

    Tensor<pair<type*, dimensions>, 1> delta_pairs(1);
    delta_pairs(0).first = inputs.data();
    delta_pairs(0).second = { input_images, input_height, input_width, channels };

    // bmp_image_1:
    //    255 255   0   0   255 255   0   0     255 255   0   0
    //    255 255   0   0   255 255   0   0     255 255   0   0
    //    255 255   0   0   255 255   0   0     255 255   0   0
    //    255 255   0   0   255 255   0   0     255 255   0   0

    // bmp_image_2:
    //    0   0 255 255      0   0 255 255       0   0 255 255
    //    0   0 255 255      0   0 255 255       0   0 255 255
    //    0   0   0   0      0   0   0   0       0   0   0   0
    //    0   0   0   0      0   0   0   0       0   0   0   0

    // kernel x3:
    //    1  1  1
    //    1  1  1
    //    1  1  1

    // Expected outputs: (2,2,2,1)
    //    4590+1  2295+1
    //    4590+1  2295+1
    //    1530+1  3060+1   
    //    765+1   1530+1     

    // Test

    convolutional_layer.set(input_dimensions, kernel_dimensions);

    convolutional_layer.set_synaptic_weights(kernel);
    convolutional_layer.set_biases(bias);

    forward_propagation.set(input_images, &convolutional_layer);
    back_propagation.set(input_images, &convolutional_layer);

    convolutional_layer.set_activation_function(ConvolutionalLayer::ActivationFunction::Linear);

    convolutional_layer.forward_propagate(input_pairs, &forward_propagation, is_training);

    convolutional_layer.back_propagate(input_pairs, delta_pairs, &forward_propagation, &back_propagation);

    // Current layer's values

    const Index images_number = 2;
    const Index kernels_number = 2;
    const Index output_height = 4;
    const Index output_raw_variables_number = 4;

    // Next layer's values

    const Index neurons_perceptron = 3;

    PerceptronLayer perceptronlayer(kernels_number*output_height*output_raw_variables_number,
                                    neurons_perceptron, PerceptronLayer::ActivationFunction::Linear);

    convolutional_layer.set(Tensor<type, 4>(5,5,3,1), Tensor<type, 4>(2, 2, 3, kernels_number), Tensor<type, 1>());

    PerceptronLayerForwardPropagation perceptron_layer_forward_propagate(images_number, &perceptronlayer);
    PerceptronLayerBackPropagation perceptron_layer_backpropagate(images_number, &perceptronlayer);
    ConvolutionalLayerBackPropagation convolutional_layer_backpropagate(images_number, &convolutional_layer);

    // initialize

    Tensor<float,2> synaptic_weights_perceptron(kernels_number * output_height * output_raw_variables_number,
                                                neurons_perceptron);

    for(int i = 0; i<kernels_number * output_height * output_raw_variables_number*neurons_perceptron; i++)
    {
        Index neuron_value = i / (kernels_number * output_height * output_raw_variables_number);

        *(synaptic_weights_perceptron.data() + i) = 1.0 * neuron_value;
    }

    perceptron_layer_backpropagate.deltas.setValues({{1,1,1},
                                                    {2,2,2}});

    perceptronlayer.set_synaptic_weights(synaptic_weights_perceptron);


    convolutional_layer.calculate_hidden_delta_perceptron(&perceptron_layer_forward_propagate,
                                                          &perceptron_layer_backpropagate,
                                                          &convolutional_layer_backpropagate);

    cout << convolutional_layer_backpropagate.deltas << endl;

}

}
*/
