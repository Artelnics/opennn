//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "convolutional_layer_test.h"

using namespace OpenNN;


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
    Tensor<float, 2> input(3, 3);
    Tensor<float, 2> kernel(2, 2);
    Tensor<float, 2> output;
    input.setRandom();
    kernel.setRandom();


    Eigen::array<ptrdiff_t, 2> dims = {0, 1};

    output = input.convolve(kernel, dims);

    assert_true(output.dimension(0) == 2, LOG);
    assert_true(output.dimension(1) == 2, LOG);

    // Convolution 2D, 3 channels
    Tensor<float, 3> input_2(5, 5, 3);
    Tensor<float, 3> kernel_2(2, 2, 3);
    Tensor<float, 3> output_2;
    input_2.setRandom();
    kernel_2.setRandom();

    Eigen::array<ptrdiff_t, 3> dims_2 = {0, 1, 2};

    output_2 = input_2.convolve(kernel_2, dims_2);

    assert_true(output_2.dimension(0) == 4, LOG);
    assert_true(output_2.dimension(1) == 4, LOG);
    assert_true(output_2.dimension(2) == 1, LOG);


    // Convolution 2D, 3 channels, multiple images, 1 kernel
    Tensor<float, 4> input_3(5, 5, 3, 10);
    Tensor<float, 3> kernel_3(2, 2, 3);
    Tensor<float, 4> output_3;
    input_3.setRandom();
    kernel_3.setRandom();

    Eigen::array<ptrdiff_t, 3> dims_3 = {0, 1, 2};

    output_3 = input_3.convolve(kernel_3, dims_3);

    assert_true(output_3.dimension(0) == 4, LOG);
    assert_true(output_3.dimension(1) == 4, LOG);
    assert_true(output_3.dimension(2) == 1, LOG);
    assert_true(output_3.dimension(3) == 10, LOG);

}


void ConvolutionalLayerTest::test_constructor()
{
    cout << "test_constructor\n";

    ConvolutionalLayer convolutional_layer;

    // Test

    assert_true(convolutional_layer.is_empty() == true, LOG);

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


void ConvolutionalLayerTest::test_get_parameters() // @todo
{
    cout << "test_get_parameters\n";

//    ConvolutionalLayer convolutional_layer;
//    Tensor<type, 2> new_synaptic_weights;
//    Tensor<type, 1> new_biases;
//    Tensor<type, 1> new_parameters;

//    // Test

//    new_synaptic_weights.resize(2,3,2,2);
//    new_synaptic_weights(0,0,0,0) = 1.111;
//    new_synaptic_weights(0,0,0,1) = 1.112;
//    new_synaptic_weights(0,0,1,0) = 1.121;
//    new_synaptic_weights(0,0,1,1) = 1.122;
//    new_synaptic_weights(0,1,0,0) = 1.211;
//    new_synaptic_weights(0,1,0,1) = 1.212;
//    new_synaptic_weights(0,1,1,0) = 1.221;
//    new_synaptic_weights(0,1,1,1) = 1.222;
//    new_synaptic_weights(0,2,0,0) = 1.311;
//    new_synaptic_weights(0,2,0,1) = 1.312;
//    new_synaptic_weights(0,2,1,0) = 1.321;
//    new_synaptic_weights(0,2,1,1) = 1.322;
//    new_synaptic_weights(1,0,0,0) = 2.111;
//    new_synaptic_weights(1,0,0,1) = 2.112;
//    new_synaptic_weights(1,0,1,0) = 2.121;
//    new_synaptic_weights(1,0,1,1) = 2.122;
//    new_synaptic_weights(1,1,0,0) = 2.211;
//    new_synaptic_weights(1,1,0,1) = 2.212;
//    new_synaptic_weights(1,1,1,0) = 2.221;
//    new_synaptic_weights(1,1,1,1) = 2.222;
//    new_synaptic_weights(1,2,0,0) = 2.311;
//    new_synaptic_weights(1,2,0,1) = 2.312;
//    new_synaptic_weights(1,2,1,0) = 2.321;
//    new_synaptic_weights(1,2,1,1) = 2.322;
//    new_biases = Tensor<type, 1>({4,8});
//    new_parameters = Tensor<type, 1>({1.111,2.111,1.211,2.211,1.311,2.311,1.121,2.121,1.221,2.221,1.321,2.321,
//                                     1.112,2.112,1.212,2.212,1.312,2.312,1.122,2.122,1.222,2.222,1.322,2.322,
//                                     4,8});

//    convolutional_layer.set_synaptic_weights(new_synaptic_weights);
//    convolutional_layer.set_biases(new_biases);

//    assert_true(convolutional_layer.get_parameters() == new_parameters, LOG);
}


void ConvolutionalLayerTest::test_get_outputs_dimensions() // @todo
{
    cout << "test_get_outputs_dimensions\n";

//    ConvolutionalLayer convolutional_layer = OpenNN::ConvolutionalLayer({3,500,600}, {10,2,3});

//    // Test

//    convolutional_layer.set_row_stride(1);
//    convolutional_layer.set_column_stride(1);
//    convolutional_layer.set_padding_option(OpenNN::ConvolutionalLayer::NoPadding);

//    assert_true(convolutional_layer.get_outputs_dimensions()[0] == 10 &&
//                convolutional_layer.get_outputs_dimensions()[1] == 499 &&
//                convolutional_layer.get_outputs_dimensions()[2] == 598, LOG);

//    // Test

//    convolutional_layer.set_row_stride(2);
//    convolutional_layer.set_column_stride(3);
//    convolutional_layer.set_padding_option(OpenNN::ConvolutionalLayer::NoPadding);

//    assert_true(convolutional_layer.get_outputs_dimensions()[0] == 10 &&
//                convolutional_layer.get_outputs_dimensions()[1] == 250 &&
//                convolutional_layer.get_outputs_dimensions()[2] == 200, LOG);

//    // Test

//    convolutional_layer.set_row_stride(1);
//    convolutional_layer.set_column_stride(1);
//    convolutional_layer.set_padding_option(OpenNN::ConvolutionalLayer::Same);

//    assert_true(convolutional_layer.get_outputs_dimensions()[0] == 10 &&
//                convolutional_layer.get_outputs_dimensions()[1] == 500 &&
//                convolutional_layer.get_outputs_dimensions()[2] == 600, LOG);

//    // Test

//    convolutional_layer.set_row_stride(2);
//    convolutional_layer.set_column_stride(3);
//    convolutional_layer.set_padding_option(OpenNN::ConvolutionalLayer::Same);

//    assert_true(convolutional_layer.get_outputs_dimensions()[0] == 10 &&
//                convolutional_layer.get_outputs_dimensions()[1] == 500 &&
//                convolutional_layer.get_outputs_dimensions()[2] == 600, LOG);
}


void ConvolutionalLayerTest::test_get_parameters_number() // @todo
{
    cout << "test_get_parameters_number\n";

//    ConvolutionalLayer convolutional_layer;

//    // Test

//    convolutional_layer = OpenNN::ConvolutionalLayer({3,32,64}, {1,2,3});

//    assert_true(convolutional_layer.get_parameters_number() == 19, LOG);

//    // Test

//    convolutional_layer = OpenNN::ConvolutionalLayer({3,500,600}, {10,2,3});

//    assert_true(convolutional_layer.get_parameters_number() == 190, LOG);
}


void ConvolutionalLayerTest::test_set() // @todo
{
    cout << "test_set\n";

    ConvolutionalLayer convolutional_layer;

    // Test

    Tensor<Index, 1> inputs_dimensions(4);
    inputs_dimensions.setValues({256, 128, 3, 1});
    Tensor<Index, 1> kernels_dimensions(4);
    kernels_dimensions.setValues({2, 2, 3, 2});

    convolutional_layer.set(inputs_dimensions, kernels_dimensions);

    assert_true(convolutional_layer.is_empty() == false &&
                convolutional_layer.get_inputs_channels_number() == 3 &&
                convolutional_layer.get_inputs_rows_number() == 256 &&
                convolutional_layer.get_inputs_columns_number() == 128 &&
                convolutional_layer.get_filters_number() == 2 &&
                convolutional_layer.get_filters_channels_number() == 3 &&
                convolutional_layer.get_filters_rows_number() == 2 &&
                convolutional_layer.get_filters_columns_number() == 2, LOG);
}


void ConvolutionalLayerTest::test_set_parameters()
{
    cout << "test_set_parameters\n";

//    ConvolutionalLayer convolutional_layer;
//    Tensor<type, 2> new_synaptic_weights;
//    Tensor<type, 1> new_biases;

//    // Test

//    new_synaptic_weights.resize(({2,2,2,2}));
//    new_synaptic_weights(0,0,0,0) = 1.111;
//    new_synaptic_weights(0,0,0,1) = 1.112;
//    new_synaptic_weights(0,0,1,0) = 1.121;
//    new_synaptic_weights(0,0,1,1) = 1.122;
//    new_synaptic_weights(0,1,0,0) = 1.211;
//    new_synaptic_weights(0,1,0,1) = 1.212;
//    new_synaptic_weights(0,1,1,0) = 1.221;
//    new_synaptic_weights(0,1,1,1) = 1.222;
//    new_synaptic_weights(1,0,0,0) = 2.111;
//    new_synaptic_weights(1,0,0,1) = 2.112;
//    new_synaptic_weights(1,0,1,0) = 2.121;
//    new_synaptic_weights(1,0,1,1) = 2.122;
//    new_synaptic_weights(1,1,0,0) = 2.211;
//    new_synaptic_weights(1,1,0,1) = 2.212;
//    new_synaptic_weights(1,1,1,0) = 2.221;
//    new_synaptic_weights(1,1,1,1) = 2.222;
//    new_biases = Tensor<type, 1>({-1,1});

//    convolutional_layer.set({2,3,3}, {2,2,2});
//    convolutional_layer.set_parameters({1.111,2.111,1.211,2.211,1.121,2.121,1.221,2.221,1.112,2.112,1.212,2.212,1.122,2.122,1.222,2.222,-1,1});

//    assert_true(convolutional_layer.is_empty() == false &&
//                convolutional_layer.get_parameters_number() == 18 &&
//                convolutional_layer.get_synaptic_weights() == new_synaptic_weights &&
//                convolutional_layer.get_biases() == new_biases, LOG);
}


void ConvolutionalLayerTest::test_calculate_combinations()
{
    Tensor<type, 4> inputs;
    Tensor<type, 4> kernels;
    Tensor<type, 1> biases;
    Tensor<type, 4> combinations;

    ConvolutionalLayer convolutional_layer;

    inputs.resize(5, 5, 3, 1);
    kernels.resize(2, 2, 3, 3);
    biases.resize(3);
    combinations.resize(4, 4, 3, 1);

    inputs.setConstant(1.f);
    kernels.setConstant(1.f/12);

    biases(0) = 0.f;
    biases(1) = 1.f;
    biases(2) = 2.f;

    convolutional_layer.set_biases(biases);
    convolutional_layer.set_synaptic_weights(kernels);
    convolutional_layer.calculate_combinations(inputs, combinations);

//    cout << combinations << endl;
//    cout << combinations(0, 0, 0, 0) << endl;
//    cout << combinations(0, 0, 1, 0) << endl;
//    cout << combinations(0, 0, 2, 0) << endl;

    assert_true(abs(combinations(0, 0, 0, 0) - 1.f) < 1e-6f &&
                abs(combinations(0, 1, 0, 0) - 1.f) < 1e-6f &&
                abs(combinations(0, 2, 0, 0) - 1.f) < 1e-6f &&
                abs(combinations(0, 3, 0, 0) - 1.f) < 1e-6f &&
                abs(combinations(1, 0, 0, 0) - 1.f) < 1e-6f &&
                abs(combinations(1, 1, 0, 0) - 1.f) < 1e-6f &&
                abs(combinations(1, 2, 0, 0) - 1.f) < 1e-6f &&
                abs(combinations(1, 3, 0, 0) - 1.f) < 1e-6f &&
                abs(combinations(2, 0, 0, 0) - 1.f) < 1e-6f &&
                abs(combinations(2, 1, 0, 0) - 1.f) < 1e-6f &&
                abs(combinations(2, 2, 0, 0) - 1.f) < 1e-6f &&
                abs(combinations(2, 3, 0, 0) - 1.f) < 1e-6f &&
                abs(combinations(3, 0, 0, 0) - 1.f) < 1e-6f &&
                abs(combinations(3, 1, 0, 0) - 1.f) < 1e-6f &&
                abs(combinations(3, 2, 0, 0) - 1.f) < 1e-6f &&
                abs(combinations(3, 3, 0, 0) - 1.f) < 1e-6f, LOG);

    assert_true(abs(combinations(0, 0, 1, 0) - 2.f) < 1e-6f &&
                abs(combinations(0, 1, 1, 0) - 2.f) < 1e-6f &&
                abs(combinations(0, 2, 1, 0) - 2.f) < 1e-6f &&
                abs(combinations(0, 3, 1, 0) - 2.f) < 1e-6f &&
                abs(combinations(1, 0, 1, 0) - 2.f) < 1e-6f &&
                abs(combinations(1, 1, 1, 0) - 2.f) < 1e-6f &&
                abs(combinations(1, 2, 1, 0) - 2.f) < 1e-6f &&
                abs(combinations(1, 3, 1, 0) - 2.f) < 1e-6f &&
                abs(combinations(2, 0, 1, 0) - 2.f) < 1e-6f &&
                abs(combinations(2, 1, 1, 0) - 2.f) < 1e-6f &&
                abs(combinations(2, 2, 1, 0) - 2.f) < 1e-6f &&
                abs(combinations(2, 3, 1, 0) - 2.f) < 1e-6f &&
                abs(combinations(3, 0, 1, 0) - 2.f) < 1e-6f &&
                abs(combinations(3, 1, 1, 0) - 2.f) < 1e-6f &&
                abs(combinations(3, 2, 1, 0) - 2.f) < 1e-6f &&
                abs(combinations(3, 3, 1, 0) - 2.f) < 1e-6f, LOG);

    assert_true(abs(combinations(0, 0, 2, 0) - 3.f) < 1e-6f &&
                abs(combinations(0, 1, 2, 0) - 3.f) < 1e-6f &&
                abs(combinations(0, 2, 2, 0) - 3.f) < 1e-6f &&
                abs(combinations(0, 3, 2, 0) - 3.f) < 1e-6f &&
                abs(combinations(1, 0, 2, 0) - 3.f) < 1e-6f &&
                abs(combinations(1, 1, 2, 0) - 3.f) < 1e-6f &&
                abs(combinations(1, 2, 2, 0) - 3.f) < 1e-6f &&
                abs(combinations(1, 3, 2, 0) - 3.f) < 1e-6f &&
                abs(combinations(2, 0, 2, 0) - 3.f) < 1e-6f &&
                abs(combinations(2, 1, 2, 0) - 3.f) < 1e-6f &&
                abs(combinations(2, 2, 2, 0) - 3.f) < 1e-6f &&
                abs(combinations(2, 3, 2, 0) - 3.f) < 1e-6f &&
                abs(combinations(3, 0, 2, 0) - 3.f) < 1e-6f &&
                abs(combinations(3, 1, 2, 0) - 3.f) < 1e-6f &&
                abs(combinations(3, 2, 2, 0) - 3.f) < 1e-6f &&
                abs(combinations(3, 3, 2, 0) - 3.f) < 1e-6f, LOG);


    inputs.resize(5, 5, 2, 2);
    kernels.resize(2, 2, 2, 2);
    combinations.resize(4, 4, 2, 2);
    biases.resize(2);

    inputs.chip(0, 3).setConstant(1.f);
    inputs.chip(1, 3).setConstant(2.f);
    kernels.setConstant(1.f/8);
    biases(0) = 1.0;
    biases(1) = 2.0;

    convolutional_layer.set_biases(biases);
    convolutional_layer.set_synaptic_weights(kernels);
    convolutional_layer.calculate_combinations(inputs, combinations);

    assert_true(combinations(0, 0, 0, 0) == 2.f &&
                combinations(0, 1, 0, 0) == 2.f &&
                combinations(0, 2, 0, 0) == 2.f &&
                combinations(0, 3, 0, 0) == 2.f &&
                combinations(1, 0, 0, 0) == 2.f &&
                combinations(1, 1, 0, 0) == 2.f &&
                combinations(1, 2, 0, 0) == 2.f &&
                combinations(1, 3, 0, 0) == 2.f &&
                combinations(2, 0, 0, 0) == 2.f &&
                combinations(2, 1, 0, 0) == 2.f &&
                combinations(2, 2, 0, 0) == 2.f &&
                combinations(2, 3, 0, 0) == 2.f &&
                combinations(3, 0, 0, 0) == 2.f &&
                combinations(3, 1, 0, 0) == 2.f &&
                combinations(3, 2, 0, 0) == 2.f &&
                combinations(3, 3, 0, 0) == 2.f &&
                combinations(0, 0, 1, 0) == 3.f &&
                combinations(0, 1, 1, 0) == 3.f &&
                combinations(0, 2, 1, 0) == 3.f &&
                combinations(0, 3, 1, 0) == 3.f &&
                combinations(1, 0, 1, 0) == 3.f &&
                combinations(1, 1, 1, 0) == 3.f &&
                combinations(1, 2, 1, 0) == 3.f &&
                combinations(1, 3, 1, 0) == 3.f &&
                combinations(2, 0, 1, 0) == 3.f &&
                combinations(2, 1, 1, 0) == 3.f &&
                combinations(2, 2, 1, 0) == 3.f &&
                combinations(2, 3, 1, 0) == 3.f &&
                combinations(3, 0, 1, 0) == 3.f &&
                combinations(3, 1, 1, 0) == 3.f &&
                combinations(3, 2, 1, 0) == 3.f &&
                combinations(3, 3, 1, 0) == 3.f &&
                combinations(0, 0, 0, 1) == 3.f &&
                combinations(0, 1, 0, 1) == 3.f &&
                combinations(0, 2, 0, 1) == 3.f &&
                combinations(0, 3, 0, 1) == 3.f &&
                combinations(1, 0, 0, 1) == 3.f &&
                combinations(1, 1, 0, 1) == 3.f &&
                combinations(1, 2, 0, 1) == 3.f &&
                combinations(1, 3, 0, 1) == 3.f &&
                combinations(2, 0, 0, 1) == 3.f &&
                combinations(2, 1, 0, 1) == 3.f &&
                combinations(2, 2, 0, 1) == 3.f &&
                combinations(2, 3, 0, 1) == 3.f &&
                combinations(3, 0, 0, 1) == 3.f &&
                combinations(3, 1, 0, 1) == 3.f &&
                combinations(3, 2, 0, 1) == 3.f &&
                combinations(3, 3, 0, 1) == 3.f &&
                combinations(0, 0, 1, 1) == 4.f &&
                combinations(0, 1, 1, 1) == 4.f &&
                combinations(0, 2, 1, 1) == 4.f &&
                combinations(0, 3, 1, 1) == 4.f &&
                combinations(1, 0, 1, 1) == 4.f &&
                combinations(1, 1, 1, 1) == 4.f &&
                combinations(1, 2, 1, 1) == 4.f &&
                combinations(1, 3, 1, 1) == 4.f &&
                combinations(2, 0, 1, 1) == 4.f &&
                combinations(2, 1, 1, 1) == 4.f &&
                combinations(2, 2, 1, 1) == 4.f &&
                combinations(2, 3, 1, 1) == 4.f &&
                combinations(3, 0, 1, 1) == 4.f &&
                combinations(3, 1, 1, 1) == 4.f &&
                combinations(3, 2, 1, 1) == 4.f &&
                combinations(3, 3, 1, 1) == 4.f, LOG);

}


void ConvolutionalLayerTest::test_calculate_activations()
{
    cout << "test_calculate_activations\n";

    ConvolutionalLayer convolutional_layer;
    Tensor<type, 4> inputs;
    Tensor<type, 4> activations_4d;
    Tensor<type, 4> result;

    result.resize(2, 2, 2, 2);

    // Test

    inputs.resize(2,2,2,2);
    inputs(0,0,0,0) = -1.111f;
    inputs(0,0,0,1) = -1.112f;
    inputs(0,0,1,0) = -1.121f;
    inputs(0,0,1,1) = -1.122f;
    inputs(0,1,0,0) = 1.211f;
    inputs(0,1,0,1) = 1.212f;
    inputs(0,1,1,0) = 1.221f;
    inputs(0,1,1,1) = 1.222f;
    inputs(1,0,0,0) = -2.111f;
    inputs(1,0,0,1) = -2.112f;
    inputs(1,0,1,0) = -2.121f;
    inputs(1,0,1,1) = -2.122f;
    inputs(1,1,0,0) = 2.211f;
    inputs(1,1,0,1) = 2.212f;
    inputs(1,1,1,0) = 2.221f;
    inputs(1,1,1,1) = 2.222f;

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


    assert_true(activations_4d(0,0,0,0) == 0.f &&
                activations_4d(0,0,0,1) == 0.f &&
                activations_4d(0,0,1,0) == 0.f &&
                activations_4d(0,0,1,1) == 0.f &&
                activations_4d(0,1,0,0) == 1.f &&
                activations_4d(0,1,0,1) == 1.f &&
                activations_4d(0,1,1,0) == 1.f &&
                activations_4d(0,1,1,1) == 1.f &&
                activations_4d(1,0,0,0) == 0.f &&
                activations_4d(1,0,0,1) == 0.f &&
                activations_4d(1,0,1,0) == 0.f &&
                activations_4d(1,0,1,1) == 0.f &&
                activations_4d(1,1,0,0) == 1.f &&
                activations_4d(1,1,0,1) == 1.f &&
                activations_4d(1,1,1,0) == 1.f &&
                activations_4d(1,1,1,1) == 1.f, LOG);

//    assert_true(activations_4d == result, LOG);

    // Test

    inputs(0,0,0,0) = -1.111f;
    inputs(0,0,0,1) = -1.112f;
    inputs(0,0,1,0) = -1.121f;
    inputs(0,0,1,1) = -1.122f;
    inputs(0,1,0,0) = 1.211f;
    inputs(0,1,0,1) = 1.212f;
    inputs(0,1,1,0) = 1.221f;
    inputs(0,1,1,1) = 1.222f;
    inputs(1,0,0,0) = -2.111f;
    inputs(1,0,0,1) = -2.112f;
    inputs(1,0,1,0) = -2.121f;
    inputs(1,0,1,1) = -2.122f;
    inputs(1,1,0,0) = 2.211f;
    inputs(1,1,0,1) = 2.212f;
    inputs(1,1,1,0) = 2.221f;
    inputs(1,1,1,1) = 2.222f;

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

    //    assert_true(activations_2d == result, LOG);

    assert_true(activations_4d(0,0,0,0) == -1 &&
                activations_4d(0,0,0,1) == -1 &&
                activations_4d(0,0,1,0) == -1 &&
                activations_4d(0,0,1,1) == -1 &&
                activations_4d(0,1,0,0) == 1.f &&
                activations_4d(0,1,0,1) == 1.f &&
                activations_4d(0,1,1,0) == 1.f &&
                activations_4d(0,1,1,1) == 1.f &&
                activations_4d(1,0,0,0) == -1 &&
                activations_4d(1,0,0,1) == -1 &&
                activations_4d(1,0,1,0) == -1 &&
                activations_4d(1,0,1,1) == -1 &&
                activations_4d(1,1,0,0) == 1.f &&
                activations_4d(1,1,0,1) == 1.f &&
                activations_4d(1,1,1,0) == 1.f &&
                activations_4d(1,1,1,1) == 1.f, LOG);

//    // Test

//    inputs.resize(({2,2,2,2}));
    inputs(0,0,0,0) = -1.111f;
    inputs(0,0,0,1) = -1.112f;
    inputs(0,0,1,0) = -1.121f;
    inputs(0,0,1,1) = -1.122f;
    inputs(0,1,0,0) = 1.211f;
    inputs(0,1,0,1) = 1.212f;
    inputs(0,1,1,0) = 1.221f;
    inputs(0,1,1,1) = 1.222f;
    inputs(1,0,0,0) = -2.111f;
    inputs(1,0,0,1) = -2.112f;
    inputs(1,0,1,0) = -2.121f;
    inputs(1,0,1,1) = -2.122f;
    inputs(1,1,0,0) = 2.211f;
    inputs(1,1,0,1) = 2.212f;
    inputs(1,1,1,0) = 2.221f;
    inputs(1,1,1,1) = 2.222f;

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

    assert_true(abs(activations_4d(0,0,0,0) - (-0.804416f)) < 1e-6f &&
                abs(activations_4d(0,0,0,1) - (-0.804768f)) < 1e-6f &&
                abs(activations_4d(0,0,1,0) - (-0.807916f)) < 1e-6f &&
                abs(activations_4d(0,0,1,1) - (-0.808263f)) < 1e-6f &&
                abs(activations_4d(0,1,0,0) - 0.836979f) < 1e-6f &&
                abs(activations_4d(0,1,0,1) - 0.837278f) < 1e-6f &&
                abs(activations_4d(0,1,1,0) - 0.839949f) < 1e-6f &&
                abs(activations_4d(0,1,1,1) - 0.840243f) < 1e-6f &&
                abs(activations_4d(1,0,0,0) - (-0.971086f)) < 1e-6f &&
                abs(activations_4d(1,0,0,1) - (-0.971143f)) < 1e-6f &&
                abs(activations_4d(1,0,1,0) - (-0.971650f)) < 1e-6f &&
                abs(activations_4d(1,0,1,1) - (-0.971706f)) < 1e-6f &&
                abs(activations_4d(1,1,0,0) - 0.976265f) < 1e-6f &&
                abs(activations_4d(1,1,0,1) - 0.976312f) < 1e-6f &&
                abs(activations_4d(1,1,1,0) - 0.976729f) < 1e-6f &&
                abs(activations_4d(1,1,1,1) - 0.976775f) < 1e-6f, LOG);

//    // Test


    inputs(0,0,0,0) = -1 * 1.111f;
    inputs(0,0,0,1) = -1 * 1.112f;
    inputs(0,0,1,0) = -1 * 1.121f;
    inputs(0,0,1,1) = -1 * 1.122f;
    inputs(0,1,0,0) = 1.211f;
    inputs(0,1,0,1) = 1.212f;
    inputs(0,1,1,0) = 1.221f;
    inputs(0,1,1,1) = 1.222f;
    inputs(1,0,0,0) = -1 * 2.111f;
    inputs(1,0,0,1) = -1 * 2.112f;
    inputs(1,0,1,0) = -1 * 2.121f;
    inputs(1,0,1,1) = -1 * 2.122f;
    inputs(1,1,0,0) = 2.211f;
    inputs(1,1,0,1) = 2.212f;
    inputs(1,1,1,0) = 2.221f;
    inputs(1,1,1,1) = 2.222f;

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::RectifiedLinear);
    convolutional_layer.calculate_activations(inputs, activations_4d);

    assert_true(activations_4d(0,0,0,0) == 0.f &&
                activations_4d(0,0,0,1) == 0.f &&
                activations_4d(0,0,1,0) == 0.f &&
                activations_4d(0,0,1,1) == 0.f &&
                activations_4d(0,1,0,0) == 1.211f &&
                activations_4d(0,1,0,1) == 1.212f &&
                activations_4d(0,1,1,0) == 1.221f &&
                activations_4d(0,1,1,1) == 1.222f &&
                activations_4d(1,0,0,0) == 0.f &&
                activations_4d(1,0,0,1) == 0.f &&
                activations_4d(1,0,1,0) == 0.f &&
                activations_4d(1,0,1,1) == 0.f &&
                activations_4d(1,1,0,0) == 2.211f &&
                activations_4d(1,1,0,1) == 2.212f &&
                activations_4d(1,1,1,0) == 2.221f &&
                activations_4d(1,1,1,1) == 2.222f, LOG);

    // Test


    inputs(0,0,0,0) = -1.111f;
    inputs(0,0,0,1) = -1.112f;
    inputs(0,0,1,0) = -1.121f;
    inputs(0,0,1,1) = -1.122f;
    inputs(0,1,0,0) = 1.211f;
    inputs(0,1,0,1) = 1.212f;
    inputs(0,1,1,0) = 1.221f;
    inputs(0,1,1,1) = 1.222f;
    inputs(1,0,0,0) = -2.111f;
    inputs(1,0,0,1) = -2.112f;
    inputs(1,0,1,0) = -2.121f;
    inputs(1,0,1,1) = -2.122f;
    inputs(1,1,0,0) = 2.211f;
    inputs(1,1,0,1) = 2.212f;
    inputs(1,1,1,0) = 2.221f;
    inputs(1,1,1,1) = 2.222f;

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::SoftPlus);

    convolutional_layer.calculate_activations(inputs, activations_4d);

    result(0,0,0,0) = 0.284600f;
    result(0,0,0,1) = 0.284352f;
    result(0,0,1,0) = 0.282132f;
    result(0,0,1,1) = 0.281886f;
    result(0,1,0,0) = 1.471747f;
    result(0,1,0,1) = 1.472518f;
    result(0,1,1,0) = 1.479461f;
    result(0,1,1,1) = 1.480233f;
    result(1,0,0,0) = 0.114325f;
    result(1,0,0,1) = 0.114217f;
    result(1,0,1,0) = 0.113250f;
    result(1,0,1,1) = 0.113143f;
    result(1,1,0,0) = 2.314991f;
    result(1,1,0,1) = 2.315893f;
    result(1,1,1,0) = 2.324008f;
    result(1,1,1,1) = 2.324910f;

    assert_true(abs(activations_4d(0,0,0,0) - result(0,0,0,0)) <= 1e-6f &&
                abs(activations_4d(0,0,0,1) - result(0,0,0,1)) <= 1e-6f &&
                abs(activations_4d(0,0,1,0) - result(0,0,1,0)) <= 1e-6f &&
                abs(activations_4d(0,0,1,1) - result(0,0,1,1)) <= 1e-6f &&
                abs(activations_4d(0,1,0,0) - result(0,1,0,0)) <= 1e-6f &&
                abs(activations_4d(0,1,0,1) - result(0,1,0,1)) <= 1e-6f &&
                abs(activations_4d(0,1,1,0) - result(0,1,1,0)) <= 1e-6f &&
                abs(activations_4d(0,1,1,1) - result(0,1,1,1)) <= 1e-6f &&
                abs(activations_4d(1,0,0,0) - result(1,0,0,0)) <= 1e-6f &&
                abs(activations_4d(1,0,0,1) - result(1,0,0,1)) <= 1e-6f &&
                abs(activations_4d(1,0,1,0) - result(1,0,1,0)) <= 1e-6f &&
                abs(activations_4d(1,0,1,1) - result(1,0,1,1)) <= 1e-6f &&
                abs(activations_4d(1,1,0,0) - result(1,1,0,0)) <= 1e-6f &&
                abs(activations_4d(1,1,0,1) - result(1,1,0,1)) <= 1e-6f &&
                abs(activations_4d(1,1,1,0) - result(1,1,1,0)) <= 1e-6f &&
                abs(activations_4d(1,1,1,1) - result(1,1,1,1)) <= 1e-6f, LOG);

}


void ConvolutionalLayerTest::test_calculate_activations_derivatives() // @todo
{
    cout << "test_calculate_activations_derivatives\n";

    NumericalDifferentiation numerical_differentiation;
    numerical_differentiation.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

    Tensor<type, 4> numerical_activation_derivatives;

    ConvolutionalLayer convolutional_layer;
    Tensor<type, 4> inputs;
    Tensor<type, 4> activations_derivatives;
    Tensor<type, 4> activations;
    Tensor<type, 4> result;

    activations.resize(2, 2, 2, 2);
    activations_derivatives.resize(2, 2, 2, 2);

//    // Test

    inputs.resize(2, 2, 2, 2);
    inputs(0,0,0,0) = -1.111f;
    inputs(0,0,0,1) = -1.112f;
    inputs(0,0,1,0) = -1.121f;
    inputs(0,0,1,1) = -1.122f;
    inputs(0,1,0,0) = 1.211f;
    inputs(0,1,0,1) = 1.212f;
    inputs(0,1,1,0) = 1.221f;
    inputs(0,1,1,1) = 1.222f;
    inputs(1,0,0,0) = -2.111f;
    inputs(1,0,0,1) = -2.112f;
    inputs(1,0,1,0) = -2.121f;
    inputs(1,0,1,1) = -2.122f;
    inputs(1,1,0,0) = 2.211f;
    inputs(1,1,0,1) = 2.212f;
    inputs(1,1,1,0) = 2.221f;
    inputs(1,1,1,1) = 2.222f;

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::Threshold);

    convolutional_layer.calculate_activations_derivatives(inputs,
                                                          activations,
                                                          activations_derivatives);

    result = activations.constant(0.f);

    assert_true(activations_derivatives(0, 0, 0, 0) == result(0, 0, 0, 0) &&
                activations_derivatives(0, 1, 0, 0) == result(0, 1, 0, 0) &&
                activations_derivatives(1, 0, 0, 0) == result(1, 0, 0, 0) &&
                activations_derivatives(1, 1, 0, 0) == result(1, 1, 0, 0) &&
                activations_derivatives(0, 0, 1, 0) == result(0, 0, 1, 0) &&
                activations_derivatives(0, 1, 1, 0) == result(0, 1, 1, 0) &&
                activations_derivatives(1, 0, 1, 0) == result(1, 0, 1, 0) &&
                activations_derivatives(1, 1, 1, 0) == result(1, 1, 1, 0) &&
                activations_derivatives(0, 0, 0, 1) == result(0, 0, 0, 1) &&
                activations_derivatives(0, 1, 0, 1) == result(0, 1, 0, 1) &&
                activations_derivatives(1, 0, 0, 1) == result(1, 0, 0, 1) &&
                activations_derivatives(1, 1, 0, 1) == result(1, 1, 0, 1) &&
                activations_derivatives(0, 0, 1, 1) == result(0, 0, 1, 1) &&
                activations_derivatives(0, 1, 1, 1) == result(0, 1, 1, 1) &&
                activations_derivatives(1, 0, 1, 1) == result(1, 0, 1, 1) &&
                activations_derivatives(1, 1, 1, 1) == result(1, 1, 1, 1), LOG);


    inputs(0,0,0,0) = -1.111f;
    inputs(0,0,0,1) = -1.112f;
    inputs(0,0,1,0) = -1.121f;
    inputs(0,0,1,1) = -1.122f;
    inputs(0,1,0,0) = 1.211f;
    inputs(0,1,0,1) = 1.212f;
    inputs(0,1,1,0) = 1.221f;
    inputs(0,1,1,1) = 1.222f;
    inputs(1,0,0,0) = -2.111f;
    inputs(1,0,0,1) = -2.112f;
    inputs(1,0,1,0) = -2.121f;
    inputs(1,0,1,1) = -2.122f;
    inputs(1,1,0,0) = 2.211f;
    inputs(1,1,0,1) = 2.212f;
    inputs(1,1,1,0) = 2.221f;
    inputs(1,1,1,1) = 2.222f;

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::SymmetricThreshold);

    convolutional_layer.calculate_activations_derivatives(inputs,
                                                          activations,
                                                          activations_derivatives);

    result = activations.constant(0.f);

    assert_true(activations_derivatives(0, 0, 0, 0) == result(0, 0, 0, 0) &&
                activations_derivatives(0, 1, 0, 0) == result(0, 1, 0, 0) &&
                activations_derivatives(1, 0, 0, 0) == result(1, 0, 0, 0) &&
                activations_derivatives(1, 1, 0, 0) == result(1, 1, 0, 0) &&
                activations_derivatives(0, 0, 1, 0) == result(0, 0, 1, 0) &&
                activations_derivatives(0, 1, 1, 0) == result(0, 1, 1, 0) &&
                activations_derivatives(1, 0, 1, 0) == result(1, 0, 1, 0) &&
                activations_derivatives(1, 1, 1, 0) == result(1, 1, 1, 0) &&
                activations_derivatives(0, 0, 0, 1) == result(0, 0, 0, 1) &&
                activations_derivatives(0, 1, 0, 1) == result(0, 1, 0, 1) &&
                activations_derivatives(1, 0, 0, 1) == result(1, 0, 0, 1) &&
                activations_derivatives(1, 1, 0, 1) == result(1, 1, 0, 1) &&
                activations_derivatives(0, 0, 1, 1) == result(0, 0, 1, 1) &&
                activations_derivatives(0, 1, 1, 1) == result(0, 1, 1, 1) &&
                activations_derivatives(1, 0, 1, 1) == result(1, 0, 1, 1) &&
                activations_derivatives(1, 1, 1, 1) == result(1, 1, 1, 1), LOG);

    inputs(0,0,0,0) = -1.111f;
    inputs(0,0,0,1) = -1.112f;
    inputs(0,0,1,0) = -1.121f;
    inputs(0,0,1,1) = -1.122f;
    inputs(0,1,0,0) = 1.211f;
    inputs(0,1,0,1) = 1.212f;
    inputs(0,1,1,0) = 1.221f;
    inputs(0,1,1,1) = 1.222f;
    inputs(1,0,0,0) = -2.111f;
    inputs(1,0,0,1) = -2.112f;
    inputs(1,0,1,0) = -2.121f;
    inputs(1,0,1,1) = -2.122f;
    inputs(1,1,0,0) = 2.211f;
    inputs(1,1,0,1) = 2.212f;
    inputs(1,1,1,0) = 2.221f;
    inputs(1,1,1,1) = 2.222f;

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::HyperbolicTangent);

    convolutional_layer.calculate_activations_derivatives(inputs,
                                                          activations,
                                                          activations_derivatives);

    result(0,0,0,0) = 0.352916f;
    result(0,0,0,1) = 0.352348f;
    result(0,0,1,0) = 0.347271f;
    result(0,0,1,1) = 0.346710f;
    result(0,1,0,0) = 0.299466f;
    result(0,1,0,1) = 0.298965f;
    result(0,1,1,0) = 0.294486f;
    result(0,1,1,1) = 0.293991f;
    result(1,0,0,0) = 0.056993f;
    result(1,0,0,1) = 0.056882f;
    result(1,0,1,0) = 0.055896f;
    result(1,0,1,1) = 0.055788f;
    result(1,1,0,0) = 0.046907f;
    result(1,1,0,1) = 0.046816f;
    result(1,1,1,0) = 0.046000f;
    result(1,1,1,1) = 0.045910f;

//    numerical_activation_derivatives = numerical_differentiation.calculate_derivatives(convolutional_layer, &ConvolutionalLayer::calculate_activations, 0, inputs);
//    cout << endl << "Numerical differentiation derivatives:" << endl << numerical_activation_derivatives << endl << endl;

//    cout << endl << "Result:" << endl << result << endl << endl;

    assert_true(abs(activations_derivatives(0,0,0,0) - result(0,0,0,0)) < 1e-6f &&
                abs(activations_derivatives(0,0,0,1) - result(0,0,0,1)) < 1e-6f &&
                abs(activations_derivatives(0,0,1,0) - result(0,0,1,0)) < 1e-6f &&
                abs(activations_derivatives(0,0,1,1) - result(0,0,1,1)) < 1e-6f &&
                abs(activations_derivatives(0,1,0,0) - result(0,1,0,0)) < 1e-6f &&
                abs(activations_derivatives(0,1,0,1) - result(0,1,0,1)) < 1e-6f &&
                abs(activations_derivatives(0,1,1,0) - result(0,1,1,0)) < 1e-6f &&
                abs(activations_derivatives(0,1,1,1) - result(0,1,1,1)) < 1e-6f &&
                abs(activations_derivatives(1,0,0,0) - result(1,0,0,0)) < 1e-6f &&
                abs(activations_derivatives(1,0,0,1) - result(1,0,0,1)) < 1e-6f &&
                abs(activations_derivatives(1,0,1,0) - result(1,0,1,0)) < 1e-6f &&
                abs(activations_derivatives(1,0,1,1) - result(1,0,1,1)) < 1e-6f &&
                abs(activations_derivatives(1,1,0,0) - result(1,1,0,0)) < 1e-6f &&
                abs(activations_derivatives(1,1,0,1) - result(1,1,0,1)) < 1e-6f &&
                abs(activations_derivatives(1,1,1,0) - result(1,1,1,0)) < 1e-6f &&
                abs(activations_derivatives(1,1,1,1) - result(1,1,1,1)) < 1e-6f, LOG);

    inputs(0,0,0,0) = -1.111f;
    inputs(0,0,0,1) = -1.112f;
    inputs(0,0,1,0) = -1.121f;
    inputs(0,0,1,1) = -1.122f;
    inputs(0,1,0,0) = 1.211f;
    inputs(0,1,0,1) = 1.212f;
    inputs(0,1,1,0) = 1.221f;
    inputs(0,1,1,1) = 1.222f;
    inputs(1,0,0,0) = -2.111f;
    inputs(1,0,0,1) = -2.112f;
    inputs(1,0,1,0) = -2.121f;
    inputs(1,0,1,1) = -2.122f;
    inputs(1,1,0,0) = 2.211f;
    inputs(1,1,0,1) = 2.212f;
    inputs(1,1,1,0) = 2.221f;
    inputs(1,1,1,1) = 2.222f;

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::RectifiedLinear);

    convolutional_layer.calculate_activations_derivatives(inputs,
                                                          activations,
                                                          activations_derivatives);

    result(0,0,0,0) = 0;
    result(0,0,0,1) = 0;
    result(0,0,1,0) = 0;
    result(0,0,1,1) = 0;
    result(0,1,0,0) = 1;
    result(0,1,0,1) = 1;
    result(0,1,1,0) = 1;
    result(0,1,1,1) = 1;
    result(1,0,0,0) = 0;
    result(1,0,0,1) = 0;
    result(1,0,1,0) = 0;
    result(1,0,1,1) = 0;
    result(1,1,0,0) = 1;
    result(1,1,0,1) = 1;
    result(1,1,1,0) = 1;
    result(1,1,1,1) = 1;

    assert_true(activations_derivatives(0, 0, 0, 0) == result(0, 0, 0, 0) &&
                activations_derivatives(0, 1, 0, 0) == result(0, 1, 0, 0) &&
                activations_derivatives(1, 0, 0, 0) == result(1, 0, 0, 0) &&
                activations_derivatives(1, 1, 0, 0) == result(1, 1, 0, 0) &&
                activations_derivatives(0, 0, 1, 0) == result(0, 0, 1, 0) &&
                activations_derivatives(0, 1, 1, 0) == result(0, 1, 1, 0) &&
                activations_derivatives(1, 0, 1, 0) == result(1, 0, 1, 0) &&
                activations_derivatives(1, 1, 1, 0) == result(1, 1, 1, 0) &&
                activations_derivatives(0, 0, 0, 1) == result(0, 0, 0, 1) &&
                activations_derivatives(0, 1, 0, 1) == result(0, 1, 0, 1) &&
                activations_derivatives(1, 0, 0, 1) == result(1, 0, 0, 1) &&
                activations_derivatives(1, 1, 0, 1) == result(1, 1, 0, 1) &&
                activations_derivatives(0, 0, 1, 1) == result(0, 0, 1, 1) &&
                activations_derivatives(0, 1, 1, 1) == result(0, 1, 1, 1) &&
                activations_derivatives(1, 0, 1, 1) == result(1, 0, 1, 1) &&
                activations_derivatives(1, 1, 1, 1) == result(1, 1, 1, 1), LOG);


    inputs(0,0,0,0) = -1.111f;
    inputs(0,0,0,1) = -1.112f;
    inputs(0,0,1,0) = -1.121f;
    inputs(0,0,1,1) = -1.122f;
    inputs(0,1,0,0) = 1.211f;
    inputs(0,1,0,1) = 1.212f;
    inputs(0,1,1,0) = 1.221f;
    inputs(0,1,1,1) = 1.222f;
    inputs(1,0,0,0) = -2.111f;
    inputs(1,0,0,1) = -2.112f;
    inputs(1,0,1,0) = -2.121f;
    inputs(1,0,1,1) = -2.122f;
    inputs(1,1,0,0) = 2.211f;
    inputs(1,1,0,1) = 2.212f;
    inputs(1,1,1,0) = 2.221f;
    inputs(1,1,1,1) = 2.222f;

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::SoftPlus);

    convolutional_layer.calculate_activations_derivatives(inputs,
                                                          activations,
                                                          activations_derivatives);

    result(0,0,0,0) = 0.247685f;
    result(0,0,0,1) = 0.247498f;
    result(0,0,1,0) = 0.245826f;
    result(0,0,1,1) = 0.245640f;
    result(0,1,0,0) = 0.770476f;
    result(0,1,0,1) = 0.770653f;
    result(0,1,1,0) = 0.772239f;
    result(0,1,1,1) = 0.772415f;
    result(1,0,0,0) = 0.108032f;
    result(1,0,0,1) = 0.107936f;
    result(1,0,1,0) = 0.107072f;
    result(1,0,1,1) = 0.106977f;
    result(1,1,0,0) = 0.901233f;
    result(1,1,0,1) = 0.901322f;
    result(1,1,1,0) = 0.902120f;
    result(1,1,1,1) = 0.902208f;

    assert_true(abs(activations_derivatives(0,0,0,0) - result(0,0,0,0)) < 1e-6f &&
                abs(activations_derivatives(0,0,0,1) - result(0,0,0,1)) < 1e-6f &&
                abs(activations_derivatives(0,0,1,0) - result(0,0,1,0)) < 1e-6f &&
                abs(activations_derivatives(0,0,1,1) - result(0,0,1,1)) < 1e-6f &&
                abs(activations_derivatives(0,1,0,0) - result(0,1,0,0)) < 1e-6f &&
                abs(activations_derivatives(0,1,0,1) - result(0,1,0,1)) < 1e-6f &&
                abs(activations_derivatives(0,1,1,0) - result(0,1,1,0)) < 1e-6f &&
                abs(activations_derivatives(0,1,1,1) - result(0,1,1,1)) < 1e-6f &&
                abs(activations_derivatives(1,0,0,0) - result(1,0,0,0)) < 1e-6f &&
                abs(activations_derivatives(1,0,0,1) - result(1,0,0,1)) < 1e-6f &&
                abs(activations_derivatives(1,0,1,0) - result(1,0,1,0)) < 1e-6f &&
                abs(activations_derivatives(1,0,1,1) - result(1,0,1,1)) < 1e-6f &&
                abs(activations_derivatives(1,1,0,0) - result(1,1,0,0)) < 1e-6f &&
                abs(activations_derivatives(1,1,0,1) - result(1,1,0,1)) < 1e-6f &&
                abs(activations_derivatives(1,1,1,0) - result(1,1,1,0)) < 1e-6f &&
                abs(activations_derivatives(1,1,1,1) - result(1,1,1,1)) < 1e-6f, LOG);
}


void ConvolutionalLayerTest::test_calculate_outputs()
{
    cout << "test_calculate_outputs\n";

    ConvolutionalLayer convolutional_layer;

    Tensor<type, 4> inputs;
    Tensor<type, 4> kernels;
    Tensor<type, 4> outputs;
    Tensor<type, 1> biases;

    // One image, One filter
    inputs.resize(5, 5, 3, 1);
    kernels.resize(2, 2, 3, 1);
    biases.resize(1);

    inputs.setConstant(1.0);
    kernels.setConstant(0.25);
    biases.setConstant(0.f);

    Tensor<Index, 1> inputs_dimensions(4);
    inputs_dimensions.setValues({5, 5, 3, 1});
    Tensor<Index, 1> kernels_dimensions(4);
    kernels_dimensions.setValues({2, 2, 3, 1});

    convolutional_layer.set(inputs_dimensions, kernels_dimensions);

    convolutional_layer.set_synaptic_weights(kernels);
    convolutional_layer.set_biases(biases);
    convolutional_layer.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer.calculate_outputs(inputs, outputs);

    assert_true(outputs(0, 0, 0, 0) == 3.f &&
                outputs(0, 1, 0, 0) == 3.f &&
                outputs(0, 2, 0, 0) == 3.f &&
                outputs(0, 3, 0, 0) == 3.f &&
                outputs(1, 0, 0, 0) == 3.f &&
                outputs(1, 1, 0, 0) == 3.f &&
                outputs(1, 2, 0, 0) == 3.f &&
                outputs(1, 3, 0, 0) == 3.f &&
                outputs(2, 0, 0, 0) == 3.f &&
                outputs(2, 1, 0, 0) == 3.f &&
                outputs(2, 2, 0, 0) == 3.f &&
                outputs(2, 3, 0, 0) == 3.f &&
                outputs(3, 0, 0, 0) == 3.f &&
                outputs(3, 1, 0, 0) == 3.f &&
                outputs(3, 2, 0, 0) == 3.f &&
                outputs(3, 3, 0, 0) == 3.f, LOG);

    // One image, multiple filters
    inputs.resize(5, 5, 3, 1);
    kernels.resize(2, 2, 3, 2);
    biases.resize(2);

    inputs.setConstant(1.0);
    kernels.setConstant(0.25);
    biases.setConstant(0.f);

    inputs_dimensions.setValues({5, 5, 3, 1});
    kernels_dimensions.setValues({2, 2, 3, 2});

    convolutional_layer.set(inputs_dimensions, kernels_dimensions);

    convolutional_layer.set_synaptic_weights(kernels);
    convolutional_layer.set_biases(biases);
    convolutional_layer.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer.calculate_outputs(inputs, outputs);

    assert_true(outputs(0, 0, 0, 0) == 3.f &&
                outputs(0, 1, 0, 0) == 3.f &&
                outputs(0, 2, 0, 0) == 3.f &&
                outputs(0, 3, 0, 0) == 3.f &&
                outputs(1, 0, 0, 0) == 3.f &&
                outputs(1, 1, 0, 0) == 3.f &&
                outputs(1, 2, 0, 0) == 3.f &&
                outputs(1, 3, 0, 0) == 3.f &&
                outputs(2, 0, 0, 0) == 3.f &&
                outputs(2, 1, 0, 0) == 3.f &&
                outputs(2, 2, 0, 0) == 3.f &&
                outputs(2, 3, 0, 0) == 3.f &&
                outputs(3, 0, 0, 0) == 3.f &&
                outputs(3, 1, 0, 0) == 3.f &&
                outputs(3, 2, 0, 0) == 3.f &&
                outputs(3, 3, 0, 0) == 3.f &&
                outputs(0, 0, 1, 0) == 3.f &&
                outputs(0, 1, 1, 0) == 3.f &&
                outputs(0, 2, 1, 0) == 3.f &&
                outputs(0, 3, 1, 0) == 3.f &&
                outputs(1, 0, 1, 0) == 3.f &&
                outputs(1, 1, 1, 0) == 3.f &&
                outputs(1, 2, 1, 0) == 3.f &&
                outputs(1, 3, 1, 0) == 3.f &&
                outputs(2, 0, 1, 0) == 3.f &&
                outputs(2, 1, 1, 0) == 3.f &&
                outputs(2, 2, 1, 0) == 3.f &&
                outputs(2, 3, 1, 0) == 3.f &&
                outputs(3, 0, 1, 0) == 3.f &&
                outputs(3, 1, 1, 0) == 3.f &&
                outputs(3, 2, 1, 0) == 3.f &&
                outputs(3, 3, 1, 0) == 3.f , LOG);


    // Multiple images, multiple filters
    inputs.resize(5, 5, 3, 2);
    kernels.resize(2, 2, 3, 2);
    biases.resize(2);

    inputs.setConstant(1.0);
    kernels.setConstant(0.25);
    biases(0) = 0;
    biases(1) = 1;

    inputs.chip(1, 3).setConstant(2.0);

    inputs_dimensions.setValues({5, 5, 3, 2});
    kernels_dimensions.setValues({2, 2, 3, 2});

    convolutional_layer.set(inputs_dimensions, kernels_dimensions);

    convolutional_layer.set_synaptic_weights(kernels);
    convolutional_layer.set_biases(biases);
    convolutional_layer.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer.calculate_outputs(inputs, outputs);

    assert_true(outputs(0, 0, 0, 0) == 3.f &&
                outputs(0, 1, 0, 0) == 3.f &&
                outputs(0, 2, 0, 0) == 3.f &&
                outputs(0, 3, 0, 0) == 3.f &&
                outputs(1, 0, 0, 0) == 3.f &&
                outputs(1, 1, 0, 0) == 3.f &&
                outputs(1, 2, 0, 0) == 3.f &&
                outputs(1, 3, 0, 0) == 3.f &&
                outputs(2, 0, 0, 0) == 3.f &&
                outputs(2, 1, 0, 0) == 3.f &&
                outputs(2, 2, 0, 0) == 3.f &&
                outputs(2, 3, 0, 0) == 3.f &&
                outputs(3, 0, 0, 0) == 3.f &&
                outputs(3, 1, 0, 0) == 3.f &&
                outputs(3, 2, 0, 0) == 3.f &&
                outputs(3, 3, 0, 0) == 3.f &&
                outputs(0, 0, 1, 0) == 4.f &&
                outputs(0, 1, 1, 0) == 4.f &&
                outputs(0, 2, 1, 0) == 4.f &&
                outputs(0, 3, 1, 0) == 4.f &&
                outputs(1, 0, 1, 0) == 4.f &&
                outputs(1, 1, 1, 0) == 4.f &&
                outputs(1, 2, 1, 0) == 4.f &&
                outputs(1, 3, 1, 0) == 4.f &&
                outputs(2, 0, 1, 0) == 4.f &&
                outputs(2, 1, 1, 0) == 4.f &&
                outputs(2, 2, 1, 0) == 4.f &&
                outputs(2, 3, 1, 0) == 4.f &&
                outputs(3, 0, 1, 0) == 4.f &&
                outputs(3, 1, 1, 0) == 4.f &&
                outputs(3, 2, 1, 0) == 4.f &&
                outputs(3, 3, 1, 0) == 4.f &&
                outputs(0, 0, 0, 1) == 6.f &&
                outputs(0, 1, 0, 1) == 6.f &&
                outputs(0, 2, 0, 1) == 6.f &&
                outputs(0, 3, 0, 1) == 6.f &&
                outputs(1, 0, 0, 1) == 6.f &&
                outputs(1, 1, 0, 1) == 6.f &&
                outputs(1, 2, 0, 1) == 6.f &&
                outputs(1, 3, 0, 1) == 6.f &&
                outputs(2, 0, 0, 1) == 6.f &&
                outputs(2, 1, 0, 1) == 6.f &&
                outputs(2, 2, 0, 1) == 6.f &&
                outputs(2, 3, 0, 1) == 6.f &&
                outputs(3, 0, 0, 1) == 6.f &&
                outputs(3, 1, 0, 1) == 6.f &&
                outputs(3, 2, 0, 1) == 6.f &&
                outputs(3, 3, 0, 1) == 6.f &&
                outputs(0, 0, 1, 1) == 7.f &&
                outputs(0, 1, 1, 1) == 7.f &&
                outputs(0, 2, 1, 1) == 7.f &&
                outputs(0, 3, 1, 1) == 7.f &&
                outputs(1, 0, 1, 1) == 7.f &&
                outputs(1, 1, 1, 1) == 7.f &&
                outputs(1, 2, 1, 1) == 7.f &&
                outputs(1, 3, 1, 1) == 7.f &&
                outputs(2, 0, 1, 1) == 7.f &&
                outputs(2, 1, 1, 1) == 7.f &&
                outputs(2, 2, 1, 1) == 7.f &&
                outputs(2, 3, 1, 1) == 7.f &&
                outputs(3, 0, 1, 1) == 7.f &&
                outputs(3, 1, 1, 1) == 7.f &&
                outputs(3, 2, 1, 1) == 7.f &&
                outputs(3, 3, 1, 1) == 7.f, LOG);

}


void ConvolutionalLayerTest::test_insert_padding() // @todo
{
    cout << "test_insert_padding\n";

    ConvolutionalLayer convolutional_layer;

    Tensor<type, 4> inputs;
    Tensor<type, 4> kernels;
    Tensor<type, 4> padded;

    // Test

    inputs.resize(5, 5, 3, 1);
    kernels.resize(3, 3, 3, 1);

    inputs.setConstant(1.0);

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

    assert_true(padded(0, 0, 0, 0) == 0.f &&
                padded(0, 1, 0, 0) == 0.f &&
                padded(0, 2, 0, 0) == 0.f &&
                padded(0, 3, 0, 0) == 0.f &&
                padded(0, 4, 0, 0) == 0.f &&
                padded(0, 5, 0, 0) == 0.f &&
                padded(0, 6, 0, 0) == 0.f &&
                padded(1, 0, 0, 0) == 0.f &&
                padded(1, 1, 0, 0) == 1.f &&
                padded(1, 2, 0, 0) == 1.f &&
                padded(1, 3, 0, 0) == 1.f &&
                padded(1, 4, 0, 0) == 1.f &&
                padded(1, 5, 0, 0) == 1.f &&
                padded(1, 6, 0, 0) == 0.f &&
                padded(2, 0, 0, 0) == 0.f &&
                padded(2, 1, 0, 0) == 1.f &&
                padded(2, 2, 0, 0) == 1.f &&
                padded(2, 3, 0, 0) == 1.f &&
                padded(2, 4, 0, 0) == 1.f &&
                padded(2, 5, 0, 0) == 1.f &&
                padded(2, 6, 0, 0) == 0.f &&
                padded(3, 0, 0, 0) == 0.f &&
                padded(3, 1, 0, 0) == 1.f &&
                padded(3, 2, 0, 0) == 1.f &&
                padded(3, 3, 0, 0) == 1.f &&
                padded(3, 4, 0, 0) == 1.f &&
                padded(3, 5, 0, 0) == 1.f &&
                padded(3, 6, 0, 0) == 0.f &&
                padded(4, 0, 0, 0) == 0.f &&
                padded(4, 1, 0, 0) == 1.f &&
                padded(4, 2, 0, 0) == 1.f &&
                padded(4, 3, 0, 0) == 1.f &&
                padded(4, 4, 0, 0) == 1.f &&
                padded(4, 5, 0, 0) == 1.f &&
                padded(4, 6, 0, 0) == 0.f &&
                padded(5, 0, 0, 0) == 0.f &&
                padded(5, 1, 0, 0) == 1.f &&
                padded(5, 2, 0, 0) == 1.f &&
                padded(5, 3, 0, 0) == 1.f &&
                padded(5, 4, 0, 0) == 1.f &&
                padded(5, 5, 0, 0) == 1.f &&
                padded(5, 6, 0, 0) == 0.f &&
                padded(6, 0, 0, 0) == 0.f &&
                padded(6, 1, 0, 0) == 0.f &&
                padded(6, 2, 0, 0) == 0.f &&
                padded(6, 3, 0, 0) == 0.f &&
                padded(6, 4, 0, 0) == 0.f &&
                padded(6, 5, 0, 0) == 0.f &&
                padded(6, 6, 0, 0) == 0.f, LOG);

    convolutional_layer.set_convolution_type(OpenNN::ConvolutionalLayer::Valid);
    convolutional_layer.set(inputs_dimensions, kernels_dimensions);

    convolutional_layer.insert_padding(inputs, padded);

    assert_true(padded.dimension(0) == 5 &&
                padded.dimension(1) == 5,LOG);

}


void ConvolutionalLayerTest::test_forward_propagate()
{
    Tensor<type, 4> inputs;
    Tensor<type, 4> activations;
    Tensor<type, 4> activations_derivatives;
    ConvolutionalLayer::ForwardPropagation forward_propagation;

    Tensor<type, 4> kernels;
    Tensor<type, 1> biases;

    ConvolutionalLayer convolutional_layer;

    inputs.resize(3, 3, 3, 2);
    kernels.resize(2, 2, 3, 2);
    biases.resize(2);

    Tensor<Index, 1> inputs_dimensions(4);
    inputs_dimensions.setValues({3, 3, 3, 2});
    Tensor<Index, 1> kernels_dimensions(4);
    kernels_dimensions.setValues({2, 2, 3, 2});

    convolutional_layer.set(inputs_dimensions, kernels_dimensions);
    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::RectifiedLinear);

    inputs.setConstant(1.f);
    biases.setConstant(0.f);
    kernels.setConstant(1.f/12);

    convolutional_layer.set_biases(biases);
    convolutional_layer.set_synaptic_weights(kernels);

    convolutional_layer.forward_propagate(inputs, forward_propagation);

    assert_true(abs(forward_propagation.combinations_4d(0, 0, 0, 0) - 1.f) < 1.e-6f &&
                abs(forward_propagation.activations_4d(0, 0, 0, 0) - 1.f) < 1.e-6f &&
                forward_propagation.activations_derivatives_4d(0, 0, 0, 0) == 1.f, LOG);

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::HyperbolicTangent);
    convolutional_layer.forward_propagate(inputs, forward_propagation);

    assert_true(abs(forward_propagation.combinations_4d(0, 0, 0, 0) - 1.f) < 1.e-6f &&
                abs(forward_propagation.activations_4d(0, 0, 0, 0) - 0.761594f) < 1.e-6f &&
                abs(forward_propagation.activations_derivatives_4d(0, 0, 0, 0) - 0.419974f) < 1.e-6f, LOG);
}


void ConvolutionalLayerTest::run_test_case() // @todo
{
   cout << "Running convolutional layer test case...\n";

//   // Constructor and destructor

   test_constructor();
   test_destructor();

   // Set methods

   test_set();
//   test_set_parameters();

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

/*
   // Get methods

   test_get_parameters();
   test_get_outputs_dimensions();
   test_get_parameters_number();

   // Combinations

   test_calculate_image_convolution();

*/

   // Forward propagate

   test_forward_propagate();

   cout << "End of convolutional layer test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2020 Artificial Intelligence Techniques, SL.
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
