//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "convolutional_layer_test.h"

ConvolutionalLayerTest::ConvolutionalLayerTest() : UnitTesting()
{
}


ConvolutionalLayerTest::~ConvolutionalLayerTest()
{
}


void ConvolutionalLayerTest::test_eigen_convolution()
{
    cout << "test_eigen_convolution\n";

    // Convolution 2D, 1 channel
    Tensor<type, 2> input(3, 3);
    Tensor<type, 2> kernel(2, 2);
    Tensor<type, 2> output;
    input.setRandom();
    kernel.setRandom();


    Eigen::array<ptrdiff_t, 2> dims = {0, 1};

    output = input.convolve(kernel, dims);

    assert_true(output.dimension(0) == 2, LOG);
    assert_true(output.dimension(1) == 2, LOG);

    // Convolution 2D, 3 channels
    Tensor<type, 3> input_2(5, 5, 3);
    Tensor<type, 3> kernel_2(2, 2, 3);
    Tensor<type, 3> output_2;
    input_2.setRandom();
    kernel_2.setRandom();

    Eigen::array<ptrdiff_t, 3> dims_2 = {0, 1, 2};

    output_2 = input_2.convolve(kernel_2, dims_2);

    assert_true(output_2.dimension(0) == 4, LOG);
    assert_true(output_2.dimension(1) == 4, LOG);
    assert_true(output_2.dimension(2) == 1, LOG);


    // Convolution 2D, 3 channels, multiple images, 1 kernel
    Tensor<type, 4> input_3(10, 3, 5, 5);
    Tensor<type, 3> kernel_3(3, 2, 2);
    Tensor<type, 4> output_3;
    input_3.setConstant(type(1));
    input_3.chip(1, 0).setConstant(type(2));
    input_3.chip(2, 0).setConstant(type(3));

    kernel_3.setConstant(type(1.0/12.f));

    Eigen::array<ptrdiff_t, 3> dims_3 = {1, 2, 3};

    output_3 = input_3.convolve(kernel_3, dims_3);

    assert_true(output_3.dimension(0) == 10, LOG);
    assert_true(output_3.dimension(1) == 1, LOG);
    assert_true(output_3.dimension(2) == 4, LOG);
    assert_true(output_3.dimension(3) == 4, LOG);

    assert_true(abs(output_3(0, 0, 0, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 0, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 0, 2) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 0, 3) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 1, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 1, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 1, 2) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 1, 3) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 2, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 2, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 2, 2) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 2, 3) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 3, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 3, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 3, 2) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 3, 3) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 0, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 0, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 0, 2) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 0, 3) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 1, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 1, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 1, 2) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 1, 3) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 2, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 2, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 2, 2) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 2, 3) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 3, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 3, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 3, 2) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 3, 3) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 0, 0) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 0, 1) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 0, 2) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 0, 3) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 1, 0) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 1, 1) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 1, 2) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 1, 3) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 2, 0) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 2, 1) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 2, 2) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 2, 3) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 3, 0) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 3, 1) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 3, 2) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 3, 3) - type(3)) <= type(NUMERIC_LIMITS_MIN), LOG);

}


void ConvolutionalLayerTest::test_constructor()
{
    cout << "test_constructor\n";

    // Test

    assert_true(convolutional_layer.is_empty(), LOG);

    // Test

//    convolutional_layer = OpenNN::ConvolutionalLayer({3,32,64}, {1,2,3});

//    assert_true(convolutional_layer.get_filters_channels_number() == 3 &&
//                convolutional_layer.get_inputs_rows_number() == 32 &&
//                convolutional_layer.get_inputs_columns_number() == 64 &&
//                convolutional_layer.get_filters_number() == 1 &&
//                convolutional_layer.get_filters_rows_number() == 2 &&
//                convolutional_layer.get_filters_columns_number() == 3, LOG);
}


void ConvolutionalLayerTest::test_destructor()
{
   cout << "test_destructor\n";
}


void ConvolutionalLayerTest::test_get_parameters()
{
    cout << "test_get_parameters\n";


    Tensor<type, 4> synaptic_weights(2, 3, 2, 2);
    Tensor<type, 1> biases(2);
    Tensor<type, 1> parameters(26);

    synaptic_weights(0,0,0,0) = type(1);
    synaptic_weights(0,0,0,1) = type(2);
    synaptic_weights(0,0,1,0) = type(3);
    synaptic_weights(0,0,1,1) = type(4);
    synaptic_weights(0,1,0,0) = type(5);
    synaptic_weights(0,1,0,1) = type(6);
    synaptic_weights(0,1,1,0) = type(7);
    synaptic_weights(0,1,1,1) = type(8);
    synaptic_weights(0,2,0,0) = type(9);
    synaptic_weights(0,2,0,1) = type(10);
    synaptic_weights(0,2,1,0) = type(11);
    synaptic_weights(0,2,1,1) = type(12);
    synaptic_weights(1,0,0,0) = type(13);
    synaptic_weights(1,0,0,1) = type(14);
    synaptic_weights(1,0,1,0) = type(15);
    synaptic_weights(1,0,1,1) = type(16);
    synaptic_weights(1,1,0,0) = type(17);
    synaptic_weights(1,1,0,1) = type(18);
    synaptic_weights(1,1,1,0) = type(19);
    synaptic_weights(1,1,1,1) = type(20);
    synaptic_weights(1,2,0,0) = type(21);
    synaptic_weights(1,2,0,1) = type(22);
    synaptic_weights(1,2,1,0) = type(23);
    synaptic_weights(1,2,1,1) = type(24);

    biases(0) = type(400);
    biases(1) = type(800);

    parameters(0) = type(400);
    parameters(1) = type(800);
    parameters(2) = type(1);
    parameters(3) = type(2);
    parameters(4) = type(3);
    parameters(5) = type(4);
    parameters(6) = type(5);
    parameters(7) = type(6);
    parameters(8) = type(7);
    parameters(9) = type(8);
    parameters(10) = type(9);
    parameters(11) = type(10);
    parameters(12) = type(11);
    parameters(13) = type(12);
    parameters(14) = type(13);
    parameters(15) = type(14);
    parameters(16) = type(15);
    parameters(17) = type(16);
    parameters(18) = type(17);
    parameters(19) = type(18);
    parameters(20) = type(19);
    parameters(21) = type(20);
    parameters(22) = type(21);
    parameters(23) = type(22);
    parameters(24) = type(23);
    parameters(25) = type(24);

    convolutional_layer.set_synaptic_weights(synaptic_weights);
    convolutional_layer.set_biases(biases);

    assert_true(abs(convolutional_layer.get_parameters()(0) - parameters(0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(1) - parameters(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(2) - parameters(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(3) - parameters(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(4) - parameters(4)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(5) - parameters(5)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(6) - parameters(6)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(7) - parameters(7)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(8) - parameters(8)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(9) - parameters(9)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(10) - parameters(10)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(11) - parameters(11)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(12) - parameters(12)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(13) - parameters(13)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(14) - parameters(14)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(15) - parameters(15)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(16) - parameters(16)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(17) - parameters(17)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(18) - parameters(18)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(19) - parameters(19)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(20) - parameters(20)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(21) - parameters(21)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(22) - parameters(22)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(23) - parameters(23)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(24) - parameters(24)) < type(NUMERIC_LIMITS_MIN) &&
                abs(convolutional_layer.get_parameters()(25) - parameters(25)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void ConvolutionalLayerTest::test_get_outputs_dimensions()
{
    cout << "test_get_outputs_dimensions\n";

//    ConvolutionalLayer convolutional_layer = OpenNN::ConvolutionalLayer({3,500,600}, {10,2,3});

    // Test

//    convolutional_layer.set_row_stride(1);
//    convolutional_layer.set_column_stride(1);
//    convolutional_layer.set_padding_option(OpenNN::ConvolutionalLayer::NoPadding);

//    assert_true(convolutional_layer.get_outputs_dimensions()[0] == 10 &&
//                convolutional_layer.get_outputs_dimensions()[1] == 499 &&
//                convolutional_layer.get_outputs_dimensions()[2] == 598, LOG);

    // Test

//    convolutional_layer.set_row_stride(2);
//    convolutional_layer.set_column_stride(3);
//    convolutional_layer.set_padding_option(OpenNN::ConvolutionalLayer::NoPadding);

//    assert_true(convolutional_layer.get_outputs_dimensions()[0] == 10 &&
//                convolutional_layer.get_outputs_dimensions()[1] == 250 &&
//                convolutional_layer.get_outputs_dimensions()[2] == 200, LOG);

    // Test

//    convolutional_layer.set_row_stride(1);
//    convolutional_layer.set_column_stride(1);
//    convolutional_layer.set_padding_option(OpenNN::ConvolutionalLayer::Same);

//    assert_true(convolutional_layer.get_outputs_dimensions()[0] == 10 &&
//                convolutional_layer.get_outputs_dimensions()[1] == 500 &&
//                convolutional_layer.get_outputs_dimensions()[2] == 600, LOG);

    // Test

//    convolutional_layer.set_row_stride(2);
//    convolutional_layer.set_column_stride(3);
//    convolutional_layer.set_padding_option(OpenNN::ConvolutionalLayer::Same);

//    assert_true(convolutional_layer.get_outputs_dimensions()[0] == 10 &&
//                convolutional_layer.get_outputs_dimensions()[1] == 500 &&
//                convolutional_layer.get_outputs_dimensions()[2] == 600, LOG);
}


void ConvolutionalLayerTest::test_get_parameters_number()
{
    cout << "test_get_parameters_number\n";

//

    // Test

//    convolutional_layer = OpenNN::ConvolutionalLayer({3,32,64}, {1,2,3});

//    assert_true(convolutional_layer.get_parameters_number() == 19, LOG);

    // Test

//    convolutional_layer = OpenNN::ConvolutionalLayer({3,500,600}, {10,2,3});

//    assert_true(convolutional_layer.get_parameters_number() == 190, LOG);
}


void ConvolutionalLayerTest::test_set()
{
    cout << "test_set\n";



    // Test

    Tensor<Index, 1> inputs_dimensions(4);
    inputs_dimensions.setValues({1, 3, 256, 128});
    Tensor<Index, 1> kernels_dimensions(4);
    kernels_dimensions.setValues({2, 3, 2, 2});

    convolutional_layer.set(inputs_dimensions, kernels_dimensions);

    assert_true(!convolutional_layer.is_empty() &&
                convolutional_layer.get_inputs_channels_number() == 3 &&
                convolutional_layer.get_inputs_rows_number() == 256 &&
                convolutional_layer.get_inputs_columns_number() == 128 &&
                convolutional_layer.get_kernels_number() == 2 &&
                convolutional_layer.get_kernels_channels_number() == 3 &&
                convolutional_layer.get_kernels_rows_number() == 2 &&
                convolutional_layer.get_kernels_columns_number() == 2, LOG);
}


void ConvolutionalLayerTest::test_set_parameters()
{
    cout << "test_set_parameters\n";

    Tensor<type, 4> new_synaptic_weights(2, 2, 2, 2);
    Tensor<type, 1> new_biases(2);
    Tensor<type, 1> parameters(18);

    convolutional_layer.set_biases(new_biases);
    convolutional_layer.set_synaptic_weights(new_synaptic_weights);

    new_synaptic_weights(0,0,0,0) = type(1);
    new_synaptic_weights(0,0,0,1) = type(2);
    new_synaptic_weights(0,0,1,0) = type(3);
    new_synaptic_weights(0,0,1,1) = type(4);
    new_synaptic_weights(0,1,0,0) = type(5);
    new_synaptic_weights(0,1,0,1) = type(6);
    new_synaptic_weights(0,1,1,0) = type(7);
    new_synaptic_weights(0,1,1,1) = type(8);
    new_synaptic_weights(1,0,0,0) = type(9);
    new_synaptic_weights(1,0,0,1) = type(10);
    new_synaptic_weights(1,0,1,0) = type(11);
    new_synaptic_weights(1,0,1,1) = type(12);
    new_synaptic_weights(1,1,0,0) = type(13);
    new_synaptic_weights(1,1,0,1) = type(14);
    new_synaptic_weights(1,1,1,0) = type(15);
    new_synaptic_weights(1,1,1,1) = type(16);

    new_biases(0) = type(100);
    new_biases(1) = type(200);

    parameters(0) = type(100);
    parameters(1) = type(200);
    parameters(2) = type(1);
    parameters(3) = type(2);
    parameters(4) = type(3);
    parameters(5) = type(4);
    parameters(6) = type(5);
    parameters(7) = type(6);
    parameters(8) = type(7);
    parameters(9) = type(8);
    parameters(10) = type(9);
    parameters(11) = type(10);
    parameters(12) = type(11);
    parameters(13) = type(12);
    parameters(14) = type(13);
    parameters(15) = type(14);
    parameters(16) = type(15);
    parameters(17) = type(16);

    convolutional_layer.set_parameters(parameters, 0);

    const Tensor<type, 4> synaptic_weight = convolutional_layer.get_synaptic_weights();
    const Tensor<type, 1> biases = convolutional_layer.get_biases();

    assert_true(abs(biases(0) - new_biases(0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(biases(1) - new_biases(1)) < type(NUMERIC_LIMITS_MIN),LOG);

    assert_true(abs(synaptic_weight(0,0,0,0) - new_synaptic_weights(0,0,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(synaptic_weight(0,0,0,1) - new_synaptic_weights(0,0,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(synaptic_weight(0,0,1,0) - new_synaptic_weights(0,0,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(synaptic_weight(0,0,1,1) - new_synaptic_weights(0,0,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(synaptic_weight(0,1,0,0) - new_synaptic_weights(0,1,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(synaptic_weight(0,1,0,1) - new_synaptic_weights(0,1,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(synaptic_weight(0,1,1,0) - new_synaptic_weights(0,1,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(synaptic_weight(0,1,1,1) - new_synaptic_weights(0,1,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(synaptic_weight(1,0,0,0) - new_synaptic_weights(1,0,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(synaptic_weight(1,0,0,1) - new_synaptic_weights(1,0,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(synaptic_weight(1,0,1,0) - new_synaptic_weights(1,0,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(synaptic_weight(1,0,1,1) - new_synaptic_weights(1,0,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(synaptic_weight(1,1,0,0) - new_synaptic_weights(1,1,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(synaptic_weight(1,1,0,1) - new_synaptic_weights(1,1,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(synaptic_weight(1,1,1,0) - new_synaptic_weights(1,1,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(synaptic_weight(1,1,1,1) - new_synaptic_weights(1,1,1,1)) < type(NUMERIC_LIMITS_MIN), LOG);


//    assert_true(!convolutional_layer.is_empty() &&
//                convolutional_layer.get_parameters_number() == 18 &&
//                convolutional_layer.get_synaptic_weights() == new_synaptic_weights &&
//                convolutional_layer.get_biases() == new_biases, LOG);
}


void ConvolutionalLayerTest::test_calculate_combinations()
{

    cout << "test_calculate_combinations\n";

    Tensor<type, 4> inputs;
    Tensor<type, 4> kernels;
    Tensor<type, 1> biases;
    Tensor<type, 4> combinations;



    inputs.resize(1, 3, 5, 5);
    kernels.resize(3, 3, 2, 2);
    biases.resize(3);

    combinations.resize(1, 3, 4, 4);

    inputs.setConstant(type(1));
    kernels.setConstant(type(1/12));

    biases(0) = type(0);
    biases(1) = type(1);
    biases(2) = type(2);

    convolutional_layer.set_biases(biases);
    convolutional_layer.set_synaptic_weights(kernels);
    convolutional_layer.calculate_convolutions(inputs, combinations);

    assert_true(abs(combinations(0, 0, 0, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 0, 0, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 0, 0, 2) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 0, 0, 3) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 0, 1, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 0, 1, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 0, 1, 2) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 0, 1, 3) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 0, 2, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 0, 2, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 0, 2, 2) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 0, 2, 3) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 0, 3, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 0, 3, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 0, 3, 2) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 0, 3, 3) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 1, 0, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 1, 0, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 1, 0, 2) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 1, 0, 3) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 1, 1, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 1, 1, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 1, 1, 2) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 1, 1, 3) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 1, 2, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 1, 2, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 1, 2, 2) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 1, 2, 3) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 1, 3, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 1, 3, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 1, 3, 2) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 1, 3, 3) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 2, 0, 0) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 2, 0, 1) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 2, 0, 2) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 2, 0, 3) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 2, 1, 0) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 2, 1, 1) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 2, 1, 2) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 2, 1, 3) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 2, 2, 0) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 2, 2, 1) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 2, 2, 2) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 2, 2, 3) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 2, 3, 0) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 2, 3, 1) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 2, 3, 2) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(combinations(0, 2, 3, 3) - type(3)) <= type(NUMERIC_LIMITS_MIN), LOG);

    inputs.resize(2, 2, 5, 5);
    kernels.resize(2, 2, 2, 2);
    combinations.resize(2, 2, 4, 4);
    biases.resize(2);

    inputs.chip(0, 0).setConstant(type(1));
    inputs.chip(1, 0).setConstant(type(2));
    kernels.setConstant(type(1/8));
    biases(0) = type(1);
    biases(1) = type(2.0);

    convolutional_layer.set_biases(biases);
    convolutional_layer.set_synaptic_weights(kernels);
    convolutional_layer.calculate_convolutions(inputs, combinations);

    assert_true(combinations(0, 0, 0, 0) == type(2) &&
                combinations(0, 0, 0, 1) == type(2) &&
                combinations(0, 0, 0, 2) == type(2) &&
                combinations(0, 0, 0, 3) == type(2) &&
                combinations(0, 0, 1, 0) == type(2) &&
                combinations(0, 0, 1, 1) == type(2) &&
                combinations(0, 0, 1, 2) == type(2) &&
                combinations(0, 0, 1, 3) == type(2) &&
                combinations(0, 0, 2, 0) == type(2) &&
                combinations(0, 0, 2, 1) == type(2) &&
                combinations(0, 0, 2, 2) == type(2) &&
                combinations(0, 0, 2, 3) == type(2) &&
                combinations(0, 0, 3, 0) == type(2) &&
                combinations(0, 0, 3, 1) == type(2) &&
                combinations(0, 0, 3, 2) == type(2) &&
                combinations(0, 0, 3, 3) == type(2) &&
                combinations(0, 1, 0, 0) == type(3) &&
                combinations(0, 1, 0, 1) == type(3) &&
                combinations(0, 1, 0, 2) == type(3) &&
                combinations(0, 1, 0, 3) == type(3) &&
                combinations(0, 1, 1, 0) == type(3) &&
                combinations(0, 1, 1, 1) == type(3) &&
                combinations(0, 1, 1, 2) == type(3) &&
                combinations(0, 1, 1, 3) == type(3) &&
                combinations(0, 1, 2, 0) == type(3) &&
                combinations(0, 1, 2, 1) == type(3) &&
                combinations(0, 1, 2, 2) == type(3) &&
                combinations(0, 1, 2, 3) == type(3) &&
                combinations(0, 1, 3, 0) == type(3) &&
                combinations(0, 1, 3, 1) == type(3) &&
                combinations(0, 1, 3, 2) == type(3) &&
                combinations(0, 1, 3, 3) == type(3) &&
                combinations(1, 0, 0, 0) == type(3) &&
                combinations(1, 0, 0, 1) == type(3) &&
                combinations(1, 0, 0, 2) == type(3) &&
                combinations(1, 0, 0, 3) == type(3) &&
                combinations(1, 0, 1, 0) == type(3) &&
                combinations(1, 0, 1, 1) == type(3) &&
                combinations(1, 0, 1, 2) == type(3) &&
                combinations(1, 0, 1, 3) == type(3) &&
                combinations(1, 0, 2, 0) == type(3) &&
                combinations(1, 0, 2, 1) == type(3) &&
                combinations(1, 0, 2, 2) == type(3) &&
                combinations(1, 0, 2, 3) == type(3) &&
                combinations(1, 0, 3, 0) == type(3) &&
                combinations(1, 0, 3, 1) == type(3) &&
                combinations(1, 0, 3, 2) == type(3) &&
                combinations(1, 0, 3, 3) == type(3) &&
                combinations(1, 1, 0, 0) == type(4) &&
                combinations(1, 1, 0, 1) == type(4) &&
                combinations(1, 1, 0, 2) == type(4) &&
                combinations(1, 1, 0, 3) == type(4) &&
                combinations(1, 1, 1, 0) == type(4) &&
                combinations(1, 1, 1, 1) == type(4) &&
                combinations(1, 1, 1, 2) == type(4) &&
                combinations(1, 1, 1, 3) == type(4) &&
                combinations(1, 1, 2, 0) == type(4) &&
                combinations(1, 1, 2, 1) == type(4) &&
                combinations(1, 1, 2, 2) == type(4) &&
                combinations(1, 1, 2, 3) == type(4) &&
                combinations(1, 1, 3, 0) == type(4) &&
                combinations(1, 1, 3, 1) == type(4) &&
                combinations(1, 1, 3, 2) == type(4) &&
                combinations(1, 1, 3, 3) == type(4), LOG);
}


void ConvolutionalLayerTest::test_calculate_activations()
{
    cout << "test_calculate_activations\n";


    Tensor<type, 4> inputs;
    Tensor<type, 4> activations_4d;
    Tensor<type, 4> result;

    result.resize(2, 2, 2, 2);

    // Test

    inputs.resize(2,2,2,2);
    inputs(0,0,0,0) = type(type(-1.111f));
    inputs(0,0,0,1) = type(type(-1.112));
    inputs(0,0,1,0) = type(type(-1.121));
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

    activations_4d.resize(2,2,2,2);
    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::Threshold);
    convolutional_layer.calculate_activations(inputs, activations_4d);

//    result.resize(2,2,2,2);
//    result(0,0,0,0) = 0;
//    result(0,0,0,1) = 0;
//    result(0,0,1,0) = 0;
//    result(0,0,1,1) = 0;
//    result(0,1,0,0) = 1;
//    result(0,1,0,1) = 1;
//    result(0,1,1,0) = 1;
//    result(0,1,1,1) = 1;
//    result(1,0,0,0) = 0;
//    result(1,0,0,1) = 0;
//    result(1,0,1,0) = 0;
//    result(1,0,1,1) = 0;
//    result(1,1,0,0) = 1;
//    result(1,1,0,1) = 1;
//    result(1,1,1,0) = 1;
//    result(1,1,1,1) = 1;


    assert_true(activations_4d(0,0,0,0) == type(0) &&
                activations_4d(0,0,0,1) == type(0) &&
                activations_4d(0,0,1,0) == type(0) &&
                activations_4d(0,0,1,1) == type(0) &&
                activations_4d(0,1,0,0) == type(1) &&
                activations_4d(0,1,0,1) == type(1) &&
                activations_4d(0,1,1,0) == type(1) &&
                activations_4d(0,1,1,1) == type(1) &&
                activations_4d(1,0,0,0) == type(0) &&
                activations_4d(1,0,0,1) == type(0) &&
                activations_4d(1,0,1,0) == type(0) &&
                activations_4d(1,0,1,1) == type(0) &&
                activations_4d(1,1,0,0) == type(1) &&
                activations_4d(1,1,0,1) == type(1) &&
                activations_4d(1,1,1,0) == type(1) &&
                activations_4d(1,1,1,1) == type(1), LOG);

//    assert_true(activations_4d == result, LOG);

    // Test

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

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::SymmetricThreshold);

    convolutional_layer.calculate_activations(inputs, activations_4d);

    //    result.set(Tensor<Index, 1>({2,2,2,2}));
    //    result(0,0,0,0) = -1;
    //    result(0,0,0,1) = -1;
    //    result(0,0,1,0) = -1;
    //    result(0,0,1,1) = -1;
    //    result(0,1,0,0) = 1;
    //    result(0,1,0,1) = 1;
    //    result(0,1,1,0) = 1;
    //    result(0,1,1,1) = 1;
    //    result(1,0,0,0) = -1;
    //    result(1,0,0,1) = -1;
    //    result(1,0,1,0) = -1;
    //    result(1,0,1,1) = -1;
    //    result(1,1,0,0) = 1;
    //    result(1,1,0,1) = 1;
    //    result(1,1,1,0) = 1;
    //    result(1,1,1,1) = 1;

    //    assert_true(activations == result, LOG);

    assert_true(activations_4d(0,0,0,0) == type(-1) &&
                activations_4d(0,0,0,1) == type(-1) &&
                activations_4d(0,0,1,0) == type(-1) &&
                activations_4d(0,0,1,1) == type(-1) &&
                activations_4d(0,1,0,0) == type(1) &&
                activations_4d(0,1,0,1) == type(1) &&
                activations_4d(0,1,1,0) == type(1) &&
                activations_4d(0,1,1,1) == type(1) &&
                activations_4d(1,0,0,0) == type(-1) &&
                activations_4d(1,0,0,1) == type(-1) &&
                activations_4d(1,0,1,0) == type(-1) &&
                activations_4d(1,0,1,1) == type(-1) &&
                activations_4d(1,1,0,0) == type(1) &&
                activations_4d(1,1,0,1) == type(1) &&
                activations_4d(1,1,1,0) == type(1) &&
                activations_4d(1,1,1,1) == type(1), LOG);

    // Test

//    inputs.resize(({2,2,2,2}));
    inputs(0,0,0,0) = type(type(-1.111f));
    inputs(0,0,0,1) = type(type(-1.112));
    inputs(0,0,1,0) = type(type(-1.121));
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

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::HyperbolicTangent);
    convolutional_layer.calculate_activations(inputs, activations_4d);

//    result.set(Tensor<Index, 1>({2,2,2,2}));
//    result(0,0,0,0) = -0.804416;
//    result(0,0,0,1) = -0.804768;
//    result(0,0,1,0) = -0.807916;
//    result(0,0,1,1) = -0.808263;
//    result(0,1,0,0) = 0.836979;
//    result(0,1,0,1) = 0.837278;
//    result(0,1,1,0) = 0.839949;
//    result(0,1,1,1) = 0.840243;
//    result(1,0,0,0) = -0.971086;
//    result(1,0,0,1) = -0.971143;
//    result(1,0,1,0) = -0.971650;
//    result(1,0,1,1) = -0.971706;
//    result(1,1,0,0) = 0.976265;
//    result(1,1,0,1) = 0.976312;
//    result(1,1,1,0) = 0.976729;
//    result(1,1,1,1) = 0.976775;

    assert_true(abs(activations_4d(0,0,0,0) - type(-0.804416f)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(0,0,0,1) - type(-0.804768f)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(0,0,1,0) - type(-0.807916f)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(0,0,1,1) - type(-0.808263f)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(0,1,0,0) - type(0.836979f)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(0,1,0,1) - type(0.837278f)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(0,1,1,0) - type(0.839949f)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(0,1,1,1) - type(0.840243f)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(1,0,0,0) - type(-0.971086f)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(1,0,0,1) - type(-0.971143f)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(1,0,1,0) - type(-0.971650f)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(1,0,1,1) - type(-0.971706f)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(1,1,0,0) - type(0.976265f)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(1,1,0,1) - type(0.976312f)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(1,1,1,0) - type(0.976729f)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(1,1,1,1) - type(0.976775f)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test


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

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::RectifiedLinear);
    convolutional_layer.calculate_activations(inputs, activations_4d);

    assert_true(activations_4d(0,0,0,0) == type(0) &&
                activations_4d(0,0,0,1) == type(0) &&
                activations_4d(0,0,1,0) == type(0) &&
                activations_4d(0,0,1,1) == type(0) &&
                activations_4d(0,1,0,0) == type(1.211f) &&
                activations_4d(0,1,0,1) == type(1.212f) &&
                activations_4d(0,1,1,0) == type(1.221f) &&
                activations_4d(0,1,1,1) == type(1.222f) &&
                activations_4d(1,0,0,0) == type(0) &&
                activations_4d(1,0,0,1) == type(0) &&
                activations_4d(1,0,1,0) == type(0) &&
                activations_4d(1,0,1,1) == type(0) &&
                activations_4d(1,1,0,0) == type(2.211f) &&
                activations_4d(1,1,0,1) == type(2.212f) &&
                activations_4d(1,1,1,0) == type(2.221f) &&
                activations_4d(1,1,1,1) == type(2.222f), LOG);

    // Test


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

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::SoftPlus);

    convolutional_layer.calculate_activations(inputs, activations_4d);

    result(0,0,0,0) = type(0.284600f);
    result(0,0,0,1) = type(0.284352f);
    result(0,0,1,0) = type(0.282132f);
    result(0,0,1,1) = type(0.281886f);
    result(0,1,0,0) = type(1.471747f);
    result(0,1,0,1) = type(1.472518f);
    result(0,1,1,0) = type(1.479461f);
    result(0,1,1,1) = type(1.480233f);
    result(1,0,0,0) = type(0.114325f);
    result(1,0,0,1) = type(0.114217f);
    result(1,0,1,0) = type(0.113250f);
    result(1,0,1,1) = type(0.113143f);
    result(1,1,0,0) = type(2.314991f);
    result(1,1,0,1) = type(2.315893f);
    result(1,1,1,0) = type(2.324008f);
    result(1,1,1,1) = type(2.324910f);

    assert_true(abs(activations_4d(0,0,0,0) - result(0,0,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(0,0,0,1) - result(0,0,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(0,0,1,0) - result(0,0,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(0,0,1,1) - result(0,0,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(0,1,0,0) - result(0,1,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(0,1,0,1) - result(0,1,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(0,1,1,0) - result(0,1,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(0,1,1,1) - result(0,1,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(1,0,0,0) - result(1,0,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(1,0,0,1) - result(1,0,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(1,0,1,0) - result(1,0,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(1,0,1,1) - result(1,0,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(1,1,0,0) - result(1,1,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(1,1,0,1) - result(1,1,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(1,1,1,0) - result(1,1,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_4d(1,1,1,1) - result(1,1,1,1)) <= type(NUMERIC_LIMITS_MIN), LOG);

}


void ConvolutionalLayerTest::test_calculate_activations_derivatives()
{
    cout << "test_calculate_activations_derivatives\n";

    numerical_differentiation.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

    Tensor<type, 4> numerical_activation_derivatives;


    Tensor<type, 4> inputs;
    Tensor<type, 4> activations_derivatives;
    Tensor<type, 4> activations;
    Tensor<type, 4> result;

    activations.resize(2, 2, 2, 2);
    activations_derivatives.resize(2, 2, 2, 2);

    // Test

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

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::Threshold);

    convolutional_layer.calculate_activations_derivatives(inputs,
                                                          activations,
                                                          activations_derivatives);

    result = activations.constant(type(0));

    assert_true((activations_derivatives(0, 0, 0, 0) - result(0, 0, 0, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 1, 0, 0) - result(0, 1, 0, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 0, 0, 0) - result(1, 0, 0, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 1, 0, 0) - result(1, 1, 0, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 0, 1, 0) - result(0, 0, 1, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 1, 1, 0) - result(0, 1, 1, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 0, 1, 0) - result(1, 0, 1, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 1, 1, 0) - result(1, 1, 1, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 0, 0, 1) - result(0, 0, 0, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 1, 0, 1) - result(0, 1, 0, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 0, 0, 1) - result(1, 0, 0, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 1, 0, 1) - result(1, 1, 0, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 0, 1, 1) - result(0, 0, 1, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 1, 1, 1) - result(0, 1, 1, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 0, 1, 1) - result(1, 0, 1, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 1, 1, 1) - result(1, 1, 1, 1)) <= type(NUMERIC_LIMITS_MIN), LOG);


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

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::SymmetricThreshold);

    convolutional_layer.calculate_activations_derivatives(inputs,
                                                          activations,
                                                          activations_derivatives);

    result = activations.constant(type(0));

    assert_true((activations_derivatives(0, 0, 0, 0) - result(0, 0, 0, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 1, 0, 0) - result(0, 1, 0, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 0, 0, 0) - result(1, 0, 0, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 1, 0, 0) - result(1, 1, 0, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 0, 1, 0) - result(0, 0, 1, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 1, 1, 0) - result(0, 1, 1, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 0, 1, 0) - result(1, 0, 1, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 1, 1, 0) - result(1, 1, 1, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 0, 0, 1) - result(0, 0, 0, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 1, 0, 1) - result(0, 1, 0, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 0, 0, 1) - result(1, 0, 0, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 1, 0, 1) - result(1, 1, 0, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 0, 1, 1) - result(0, 0, 1, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 1, 1, 1) - result(0, 1, 1, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 0, 1, 1) - result(1, 0, 1, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 1, 1, 1) - result(1, 1, 1, 1)) <= type(NUMERIC_LIMITS_MIN), LOG);

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

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::HyperbolicTangent);

    convolutional_layer.calculate_activations_derivatives(inputs,
                                                          activations,
                                                          activations_derivatives);

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

//    numerical_activation_derivatives = numerical_differentiation.calculate_derivatives(convolutional_layer, &ConvolutionalLayer::calculate_activations, 0, inputs);
//    cout << endl << "Numerical differentiation derivatives:" << endl << numerical_activation_derivatives << endl << endl;

//    cout << endl << "Result:" << endl << result << endl << endl;

    assert_true(abs(activations_derivatives(0,0,0,0) - result(0,0,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(0,0,0,1) - result(0,0,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(0,0,1,0) - result(0,0,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(0,0,1,1) - result(0,0,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(0,1,0,0) - result(0,1,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(0,1,0,1) - result(0,1,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(0,1,1,0) - result(0,1,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(0,1,1,1) - result(0,1,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(1,0,0,0) - result(1,0,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(1,0,0,1) - result(1,0,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(1,0,1,0) - result(1,0,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(1,0,1,1) - result(1,0,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(1,1,0,0) - result(1,1,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(1,1,0,1) - result(1,1,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(1,1,1,0) - result(1,1,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(1,1,1,1) - result(1,1,1,1)) < type(NUMERIC_LIMITS_MIN), LOG);

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

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::RectifiedLinear);

    convolutional_layer.calculate_activations_derivatives(inputs,
                                                          activations,
                                                          activations_derivatives);

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

    assert_true((activations_derivatives(0, 0, 0, 0) - result(0, 0, 0, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 1, 0, 0) - result(0, 1, 0, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 0, 0, 0) - result(1, 0, 0, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 1, 0, 0) - result(1, 1, 0, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 0, 1, 0) - result(0, 0, 1, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 1, 1, 0) - result(0, 1, 1, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 0, 1, 0) - result(1, 0, 1, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 1, 1, 0) - result(1, 1, 1, 0)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 0, 0, 1) - result(0, 0, 0, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 1, 0, 1) - result(0, 1, 0, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 0, 0, 1) - result(1, 0, 0, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 1, 0, 1) - result(1, 1, 0, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 0, 1, 1) - result(0, 0, 1, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(0, 1, 1, 1) - result(0, 1, 1, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 0, 1, 1) - result(1, 0, 1, 1)) <= type(NUMERIC_LIMITS_MIN) &&
                (activations_derivatives(1, 1, 1, 1) - result(1, 1, 1, 1)) <= type(NUMERIC_LIMITS_MIN), LOG);


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

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::SoftPlus);

    convolutional_layer.calculate_activations_derivatives(inputs,
                                                          activations,
                                                          activations_derivatives);

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

    assert_true(abs(activations_derivatives(0,0,0,0) - result(0,0,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(0,0,0,1) - result(0,0,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(0,0,1,0) - result(0,0,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(0,0,1,1) - result(0,0,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(0,1,0,0) - result(0,1,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(0,1,0,1) - result(0,1,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(0,1,1,0) - result(0,1,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(0,1,1,1) - result(0,1,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(1,0,0,0) - result(1,0,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(1,0,0,1) - result(1,0,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(1,0,1,0) - result(1,0,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(1,0,1,1) - result(1,0,1,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(1,1,0,0) - result(1,1,0,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(1,1,0,1) - result(1,1,0,1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(1,1,1,0) - result(1,1,1,0)) < type(NUMERIC_LIMITS_MIN) &&
                abs(activations_derivatives(1,1,1,1) - result(1,1,1,1)) < type(NUMERIC_LIMITS_MIN), LOG);
}


/// @todo

void ConvolutionalLayerTest::test_calculate_outputs()
{
    cout << "test_calculate_outputs\n";



    Tensor<type, 4> inputs;
    Tensor<type, 4> kernels;
    Tensor<type, 4> outputs;
    Tensor<type, 1> biases;

    // One image, One filter
    inputs.resize(5, 5, 3, 1);
    kernels.resize(2, 2, 3, 1);
    biases.resize(1);

    inputs.setConstant(type(1));
    kernels.setConstant(type(0.25));
    biases.setConstant(type(0));

    Tensor<Index, 1> inputs_dimensions(4);
    inputs_dimensions.setValues({5, 5, 3, 1});
    Tensor<Index, 1> kernels_dimensions(4);
    kernels_dimensions.setValues({2, 2, 3, 1});

    convolutional_layer.set(inputs_dimensions, kernels_dimensions);

    convolutional_layer.set_synaptic_weights(kernels);
    convolutional_layer.set_biases(biases);
    convolutional_layer.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer.calculate_outputs(inputs, outputs);

    assert_true(outputs(0, 0, 0, 0) == type(3) &&
                outputs(0, 1, 0, 0) == type(3) &&
                outputs(0, 2, 0, 0) == type(3) &&
                outputs(0, 3, 0, 0) == type(3) &&
                outputs(1, 0, 0, 0) == type(3) &&
                outputs(1, 1, 0, 0) == type(3) &&
                outputs(1, 2, 0, 0) == type(3) &&
                outputs(1, 3, 0, 0) == type(3) &&
                outputs(2, 0, 0, 0) == type(3) &&
                outputs(2, 1, 0, 0) == type(3) &&
                outputs(2, 2, 0, 0) == type(3) &&
                outputs(2, 3, 0, 0) == type(3) &&
                outputs(3, 0, 0, 0) == type(3) &&
                outputs(3, 1, 0, 0) == type(3) &&
                outputs(3, 2, 0, 0) == type(3) &&
                outputs(3, 3, 0, 0) == type(3), LOG);

    // One image, multiple filters

    inputs.resize(5, 5, 3, 1);
    kernels.resize(2, 2, 3, 2);
    biases.resize(2);

    inputs.setConstant(type(1));
    kernels.setConstant(type(0.25));
    biases.setConstant(type(0));

    inputs_dimensions.setValues({5, 5, 3, 1});
    kernels_dimensions.setValues({2, 2, 3, 2});

    convolutional_layer.set(inputs_dimensions, kernels_dimensions);

    convolutional_layer.set_synaptic_weights(kernels);
    convolutional_layer.set_biases(biases);
    convolutional_layer.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer.calculate_outputs(inputs, outputs);

    assert_true(outputs(0, 0, 0, 0) == type(type(3)) &&
                outputs(0, 1, 0, 0) == type(type(3)) &&
                outputs(0, 2, 0, 0) == type(type(3)) &&
                outputs(0, 3, 0, 0) == type(type(3)) &&
                outputs(1, 0, 0, 0) == type(type(3)) &&
                outputs(1, 1, 0, 0) == type(type(3)) &&
                outputs(1, 2, 0, 0) == type(type(3)) &&
                outputs(1, 3, 0, 0) == type(type(3)) &&
                outputs(2, 0, 0, 0) == type(type(3)) &&
                outputs(2, 1, 0, 0) == type(type(3)) &&
                outputs(2, 2, 0, 0) == type(type(3)) &&
                outputs(2, 3, 0, 0) == type(type(3)) &&
                outputs(3, 0, 0, 0) == type(type(3)) &&
                outputs(3, 1, 0, 0) == type(type(3)) &&
                outputs(3, 2, 0, 0) == type(type(3)) &&
                outputs(3, 3, 0, 0) == type(type(3)) &&
                outputs(0, 0, 1, 0) == type(type(3)) &&
                outputs(0, 1, 1, 0) == type(type(3)) &&
                outputs(0, 2, 1, 0) == type(type(3)) &&
                outputs(0, 3, 1, 0) == type(type(3)) &&
                outputs(1, 0, 1, 0) == type(type(3)) &&
                outputs(1, 1, 1, 0) == type(type(3)) &&
                outputs(1, 2, 1, 0) == type(type(3)) &&
                outputs(1, 3, 1, 0) == type(type(3)) &&
                outputs(2, 0, 1, 0) == type(type(3)) &&
                outputs(2, 1, 1, 0) == type(type(3)) &&
                outputs(2, 2, 1, 0) == type(type(3)) &&
                outputs(2, 3, 1, 0) == type(type(3)) &&
                outputs(3, 0, 1, 0) == type(type(3)) &&
                outputs(3, 1, 1, 0) == type(type(3)) &&
                outputs(3, 2, 1, 0) == type(type(3)) &&
                outputs(3, 3, 1, 0) == type(type(3)), LOG);

    // Multiple images, multiple filters

    inputs.resize(5, 5, 3, 2);
    kernels.resize(2, 2, 3, 2);
    biases.resize(2);

    inputs.setConstant(type(1));
    kernels.setConstant(type(0.25));
    biases(0) = type(0);
    biases(1) = type(1);

    inputs.chip(1, 3).setConstant(type(2));

    inputs_dimensions.setValues({5, 5, 3, 2});
    kernels_dimensions.setValues({2, 2, 3, 2});

    convolutional_layer.set(inputs_dimensions, kernels_dimensions);

    convolutional_layer.set_synaptic_weights(kernels);
    convolutional_layer.set_biases(biases);
    convolutional_layer.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer.calculate_outputs(inputs, outputs);

    assert_true(outputs(0, 0, 0, 0) == type(3) &&
                outputs(0, 1, 0, 0) == type(3) &&
                outputs(0, 2, 0, 0) == type(3) &&
                outputs(0, 3, 0, 0) == type(3) &&
                outputs(1, 0, 0, 0) == type(3) &&
                outputs(1, 1, 0, 0) == type(3) &&
                outputs(1, 2, 0, 0) == type(3) &&
                outputs(1, 3, 0, 0) == type(3) &&
                outputs(2, 0, 0, 0) == type(3) &&
                outputs(2, 1, 0, 0) == type(3) &&
                outputs(2, 2, 0, 0) == type(3) &&
                outputs(2, 3, 0, 0) == type(3) &&
                outputs(3, 0, 0, 0) == type(3) &&
                outputs(3, 1, 0, 0) == type(3) &&
                outputs(3, 2, 0, 0) == type(3) &&
                outputs(3, 3, 0, 0) == type(3) &&
                outputs(0, 0, 1, 0) == type(4) &&
                outputs(0, 1, 1, 0) == type(4) &&
                outputs(0, 2, 1, 0) == type(4) &&
                outputs(0, 3, 1, 0) == type(4) &&
                outputs(1, 0, 1, 0) == type(4) &&
                outputs(1, 1, 1, 0) == type(4) &&
                outputs(1, 2, 1, 0) == type(4) &&
                outputs(1, 3, 1, 0) == type(4) &&
                outputs(2, 0, 1, 0) == type(4) &&
                outputs(2, 1, 1, 0) == type(4) &&
                outputs(2, 2, 1, 0) == type(4) &&
                outputs(2, 3, 1, 0) == type(4) &&
                outputs(3, 0, 1, 0) == type(4) &&
                outputs(3, 1, 1, 0) == type(4) &&
                outputs(3, 2, 1, 0) == type(4) &&
                outputs(3, 3, 1, 0) == type(4) &&
                outputs(0, 0, 0, 1) == type(6) &&
                outputs(0, 1, 0, 1) == type(6) &&
                outputs(0, 2, 0, 1) == type(6) &&
                outputs(0, 3, 0, 1) == type(6) &&
                outputs(1, 0, 0, 1) == type(6) &&
                outputs(1, 1, 0, 1) == type(6) &&
                outputs(1, 2, 0, 1) == type(6) &&
                outputs(1, 3, 0, 1) == type(6) &&
                outputs(2, 0, 0, 1) == type(6) &&
                outputs(2, 1, 0, 1) == type(6) &&
                outputs(2, 2, 0, 1) == type(6) &&
                outputs(2, 3, 0, 1) == type(6) &&
                outputs(3, 0, 0, 1) == type(6) &&
                outputs(3, 1, 0, 1) == type(6) &&
                outputs(3, 2, 0, 1) == type(6) &&
                outputs(3, 3, 0, 1) == type(6) &&
                outputs(0, 0, 1, 1) == type(7) &&
                outputs(0, 1, 1, 1) == type(7) &&
                outputs(0, 2, 1, 1) == type(7) &&
                outputs(0, 3, 1, 1) == type(7) &&
                outputs(1, 0, 1, 1) == type(7) &&
                outputs(1, 1, 1, 1) == type(7) &&
                outputs(1, 2, 1, 1) == type(7) &&
                outputs(1, 3, 1, 1) == type(7) &&
                outputs(2, 0, 1, 1) == type(7) &&
                outputs(2, 1, 1, 1) == type(7) &&
                outputs(2, 2, 1, 1) == type(7) &&
                outputs(2, 3, 1, 1) == type(7) &&
                outputs(3, 0, 1, 1) == type(7) &&
                outputs(3, 1, 1, 1) == type(7) &&
                outputs(3, 2, 1, 1) == type(7) &&
                outputs(3, 3, 1, 1) == type(7), LOG);

    inputs.resize(5, 5, 3, 2);
    kernels.resize(3, 3, 3, 2);
    biases.resize(2);
}


void ConvolutionalLayerTest::test_insert_padding()
{
    cout << "test_insert_padding\n";



    Tensor<type, 4> inputs;
    Tensor<type, 4> kernels;
    Tensor<type, 4> padded;

    // Test

    inputs.resize(5, 5, 3, 1);
    kernels.resize(3, 3, 3, 1);

    inputs.setConstant(type(1));

    Tensor<Index, 1> inputs_dimensions(4);
    inputs_dimensions.setValues({5, 5, 3, 1});
    Tensor<Index, 1> kernels_dimensions(4);
    kernels_dimensions.setValues({3, 3, 3, 1});

    convolutional_layer.set_row_stride(1);
    convolutional_layer.set_column_stride(1);
    convolutional_layer.set_convolution_type(OpenNN::ConvolutionalLayer::Same);
    convolutional_layer.set(inputs_dimensions, kernels_dimensions);

    convolutional_layer.insert_padding(inputs, padded);

    assert_true(padded.dimension(0) == 7 &&
                padded.dimension(1) == 7,LOG);

    assert_true(padded(0, 0, 0, 0) == type(0) &&
                padded(0, 1, 0, 0) == type(0) &&
                padded(0, 2, 0, 0) == type(0) &&
                padded(0, 3, 0, 0) == type(0) &&
                padded(0, 4, 0, 0) == type(0) &&
                padded(0, 5, 0, 0) == type(0) &&
                padded(0, 6, 0, 0) == type(0) &&
                padded(1, 0, 0, 0) == type(0) &&
                padded(1, 1, 0, 0) == type(1) &&
                padded(1, 2, 0, 0) == type(1) &&
                padded(1, 3, 0, 0) == type(1) &&
                padded(1, 4, 0, 0) == type(1) &&
                padded(1, 5, 0, 0) == type(1) &&
                padded(1, 6, 0, 0) == type(0) &&
                padded(2, 0, 0, 0) == type(0) &&
                padded(2, 1, 0, 0) == type(1) &&
                padded(2, 2, 0, 0) == type(1) &&
                padded(2, 3, 0, 0) == type(1) &&
                padded(2, 4, 0, 0) == type(1) &&
                padded(2, 5, 0, 0) == type(1) &&
                padded(2, 6, 0, 0) == type(0) &&
                padded(3, 0, 0, 0) == type(0) &&
                padded(3, 1, 0, 0) == type(1) &&
                padded(3, 2, 0, 0) == type(1) &&
                padded(3, 3, 0, 0) == type(1) &&
                padded(3, 4, 0, 0) == type(1) &&
                padded(3, 5, 0, 0) == type(1) &&
                padded(3, 6, 0, 0) == type(0) &&
                padded(4, 0, 0, 0) == type(0) &&
                padded(4, 1, 0, 0) == type(1) &&
                padded(4, 2, 0, 0) == type(1) &&
                padded(4, 3, 0, 0) == type(1) &&
                padded(4, 4, 0, 0) == type(1) &&
                padded(4, 5, 0, 0) == type(1) &&
                padded(4, 6, 0, 0) == type(0) &&
                padded(5, 0, 0, 0) == type(0) &&
                padded(5, 1, 0, 0) == type(1) &&
                padded(5, 2, 0, 0) == type(1) &&
                padded(5, 3, 0, 0) == type(1) &&
                padded(5, 4, 0, 0) == type(1) &&
                padded(5, 5, 0, 0) == type(1) &&
                padded(5, 6, 0, 0) == type(0) &&
                padded(6, 0, 0, 0) == type(0) &&
                padded(6, 1, 0, 0) == type(0) &&
                padded(6, 2, 0, 0) == type(0) &&
                padded(6, 3, 0, 0) == type(0) &&
                padded(6, 4, 0, 0) == type(0) &&
                padded(6, 5, 0, 0) == type(0) &&
                padded(6, 6, 0, 0) == type(0), LOG);

    convolutional_layer.set_convolution_type(OpenNN::ConvolutionalLayer::Valid);
    convolutional_layer.set(inputs_dimensions, kernels_dimensions);

    convolutional_layer.insert_padding(inputs, padded);

    assert_true(padded.dimension(0) == 5 &&
                padded.dimension(1) == 5,LOG);

}


void ConvolutionalLayerTest::test_forward_propagate()
{

    cout << "test_forward_propagate\n";
    Tensor<type, 4> inputs;
    Tensor<type, 4> activations;
    Tensor<type, 4> activations_derivatives;
    ConvolutionalLayerForwardPropagation forward_propagation;

    Tensor<type, 4> kernels;
    Tensor<type, 1> biases;



//    inputs.resize(3, 3, 3, 2);
//    kernels.resize(2, , 3, 2);
//    biases.resize(2);

    inputs.resize(2, 3, 3, 3);
    kernels.resize(2, 3, 2, 2);
    biases.resize(2);

    Tensor<Index, 1> inputs_dimensions(4);
    inputs_dimensions.setValues({2, 3, 3, 3});
    Tensor<Index, 1> kernels_dimensions(4);
    kernels_dimensions.setValues({2, 3, 2, 2});

    convolutional_layer.set(inputs_dimensions, kernels_dimensions);
    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::RectifiedLinear);

    inputs.setConstant(type(1));
    biases(0) = type(0);
    biases(1) = type(1);
    kernels.setConstant(type(1.0/12.0));

    convolutional_layer.set_biases(biases);
    convolutional_layer.set_synaptic_weights(kernels);

//    convolutional_layer.forward_propagate(inputs, forward_propagation);

    assert_true(forward_propagation.activations.dimension(0) == 2 &&
                forward_propagation.activations.dimension(1) == 2 &&
                forward_propagation.activations.dimension(2) == 2 &&
                forward_propagation.activations.dimension(3) == 2, LOG);

    assert_true(forward_propagation.activations_derivatives.dimension(0) == 2 &&
                forward_propagation.activations_derivatives.dimension(1) == 2 &&
                forward_propagation.activations_derivatives.dimension(2) == 2 &&
                forward_propagation.activations_derivatives.dimension(3) == 2, LOG);

    assert_true(abs(forward_propagation.activations(0, 0, 0, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations(0, 0, 0, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations(0, 0, 1, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations(0, 0, 1, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations(0, 1, 0, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations(0, 1, 0, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations(0, 1, 1, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations(0, 1, 1, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations(1, 0, 0, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations(1, 0, 0, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations(1, 0, 1, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations(1, 0, 1, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations(1, 1, 0, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations(1, 1, 0, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations(1, 1, 1, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations(1, 1, 1, 1) - type(2)) < type(NUMERIC_LIMITS_MIN), LOG);

    assert_true(abs(forward_propagation.activations_derivatives(0, 0, 0, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations_derivatives(0, 0, 0, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations_derivatives(0, 0, 1, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations_derivatives(0, 0, 1, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations_derivatives(0, 1, 0, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations_derivatives(0, 1, 0, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations_derivatives(0, 1, 1, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations_derivatives(0, 1, 1, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations_derivatives(1, 0, 0, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations_derivatives(1, 0, 0, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations_derivatives(1, 0, 1, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations_derivatives(1, 0, 1, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations_derivatives(1, 1, 0, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations_derivatives(1, 1, 0, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations_derivatives(1, 1, 1, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations_derivatives(1, 1, 1, 1) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    assert_true(abs(forward_propagation.combinations(0, 0, 0, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations(0, 0, 0, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                forward_propagation.activations_derivatives(0, 0, 0, 0) == type(1), LOG);

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::HyperbolicTangent);

//    convolutional_layer.forward_propagate(inputs, forward_propagation);

    assert_true(abs(forward_propagation.combinations(0, 0, 0, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations(0, 0, 0, 0) - type(0.761594f)) < type(NUMERIC_LIMITS_MIN) &&
                abs(forward_propagation.activations_derivatives(0, 0, 0, 0) - type(0.419974f)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void ConvolutionalLayerTest::run_test_case()
{
   cout << "Running convolutional layer test case...\n";

   // Constructor and destructor

   test_constructor();
   test_destructor();

   // Set methods

   test_set();
   test_set_parameters();

   test_get_parameters();

   // Convolutions

   test_eigen_convolution();

   // Combinations

   test_calculate_combinations();

   // Activation

   test_calculate_activations();
   test_calculate_activations_derivatives();

   // Outputs

   test_calculate_outputs();

   // Padding

   test_insert_padding();

   // Get methods

   test_get_parameters();
   test_get_outputs_dimensions();
   test_get_parameters_number();

   // Forward propagate

   test_forward_propagate();

   cout << "End of convolutional layer test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
