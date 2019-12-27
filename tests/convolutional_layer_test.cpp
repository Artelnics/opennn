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


void ConvolutionalLayerTest::test_constructor()
{
    cout << "test_constructor\n";

    ConvolutionalLayer convolutional_layer;

    // Test

    assert_true(convolutional_layer.is_empty() == true, LOG);

    // Test

    convolutional_layer = OpenNN::ConvolutionalLayer({3,32,64}, {1,2,3});

    assert_true(convolutional_layer.get_filters_channels_number() == 3 &&
                convolutional_layer.get_inputs_rows_number() == 32 &&
                convolutional_layer.get_inputs_columns_number() == 64 &&
                convolutional_layer.get_filters_number() == 1 &&
                convolutional_layer.get_filters_rows_number() == 2 &&
                convolutional_layer.get_filters_columns_number() == 3, LOG);
}


void ConvolutionalLayerTest::test_destructor()
{
   cout << "test_destructor\n";
}


void ConvolutionalLayerTest::test_get_parameters()
{
    cout << "test_get_parameters\n";

    ConvolutionalLayer convolutional_layer;
    Tensor<double> new_synaptic_weights;
    Vector<double> new_biases;
    Vector<double> new_parameters;

    // Test

    new_synaptic_weights.set(Vector<size_t>({2,3,2,2}));
    new_synaptic_weights(0,0,0,0) = 1.111;
    new_synaptic_weights(0,0,0,1) = 1.112;
    new_synaptic_weights(0,0,1,0) = 1.121;
    new_synaptic_weights(0,0,1,1) = 1.122;
    new_synaptic_weights(0,1,0,0) = 1.211;
    new_synaptic_weights(0,1,0,1) = 1.212;
    new_synaptic_weights(0,1,1,0) = 1.221;
    new_synaptic_weights(0,1,1,1) = 1.222;
    new_synaptic_weights(0,2,0,0) = 1.311;
    new_synaptic_weights(0,2,0,1) = 1.312;
    new_synaptic_weights(0,2,1,0) = 1.321;
    new_synaptic_weights(0,2,1,1) = 1.322;
    new_synaptic_weights(1,0,0,0) = 2.111;
    new_synaptic_weights(1,0,0,1) = 2.112;
    new_synaptic_weights(1,0,1,0) = 2.121;
    new_synaptic_weights(1,0,1,1) = 2.122;
    new_synaptic_weights(1,1,0,0) = 2.211;
    new_synaptic_weights(1,1,0,1) = 2.212;
    new_synaptic_weights(1,1,1,0) = 2.221;
    new_synaptic_weights(1,1,1,1) = 2.222;
    new_synaptic_weights(1,2,0,0) = 2.311;
    new_synaptic_weights(1,2,0,1) = 2.312;
    new_synaptic_weights(1,2,1,0) = 2.321;
    new_synaptic_weights(1,2,1,1) = 2.322;
    new_biases = Vector<double>({4,8});
    new_parameters = Vector<double>({1.111,2.111,1.211,2.211,1.311,2.311,1.121,2.121,1.221,2.221,1.321,2.321,
                                     1.112,2.112,1.212,2.212,1.312,2.312,1.122,2.122,1.222,2.222,1.322,2.322,
                                     4,8});

    convolutional_layer.set_synaptic_weights(new_synaptic_weights);
    convolutional_layer.set_biases(new_biases);

    assert_true(convolutional_layer.get_parameters() == new_parameters, LOG);
}


void ConvolutionalLayerTest::test_get_outputs_dimensions()
{
    cout << "test_get_outputs_dimensions\n";

    ConvolutionalLayer convolutional_layer = OpenNN::ConvolutionalLayer({3,500,600}, {10,2,3});

    // Test

    convolutional_layer.set_row_stride(1);
    convolutional_layer.set_column_stride(1);
    convolutional_layer.set_padding_option(OpenNN::ConvolutionalLayer::NoPadding);

    assert_true(convolutional_layer.get_outputs_dimensions()[0] == 10 &&
                convolutional_layer.get_outputs_dimensions()[1] == 499 &&
                convolutional_layer.get_outputs_dimensions()[2] == 598, LOG);

    // Test

    convolutional_layer.set_row_stride(2);
    convolutional_layer.set_column_stride(3);
    convolutional_layer.set_padding_option(OpenNN::ConvolutionalLayer::NoPadding);

    assert_true(convolutional_layer.get_outputs_dimensions()[0] == 10 &&
                convolutional_layer.get_outputs_dimensions()[1] == 250 &&
                convolutional_layer.get_outputs_dimensions()[2] == 200, LOG);

    // Test

    convolutional_layer.set_row_stride(1);
    convolutional_layer.set_column_stride(1);
    convolutional_layer.set_padding_option(OpenNN::ConvolutionalLayer::Same);

    assert_true(convolutional_layer.get_outputs_dimensions()[0] == 10 &&
                convolutional_layer.get_outputs_dimensions()[1] == 500 &&
                convolutional_layer.get_outputs_dimensions()[2] == 600, LOG);

    // Test

    convolutional_layer.set_row_stride(2);
    convolutional_layer.set_column_stride(3);
    convolutional_layer.set_padding_option(OpenNN::ConvolutionalLayer::Same);

    assert_true(convolutional_layer.get_outputs_dimensions()[0] == 10 &&
                convolutional_layer.get_outputs_dimensions()[1] == 500 &&
                convolutional_layer.get_outputs_dimensions()[2] == 600, LOG);
}


void ConvolutionalLayerTest::test_get_parameters_number()
{
    cout << "test_get_parameters_number\n";

    ConvolutionalLayer convolutional_layer;

    // Test

    convolutional_layer = OpenNN::ConvolutionalLayer({3,32,64}, {1,2,3});

    assert_true(convolutional_layer.get_parameters_number() == 19, LOG);

    // Test

    convolutional_layer = OpenNN::ConvolutionalLayer({3,500,600}, {10,2,3});

    assert_true(convolutional_layer.get_parameters_number() == 190, LOG);
}


void ConvolutionalLayerTest::test_set()
{
    cout << "test_set\n";

    ConvolutionalLayer convolutional_layer;

    // Test

    convolutional_layer.set({3,256,300}, {10,5,5});

    assert_true(convolutional_layer.is_empty() == false &&
                convolutional_layer.get_inputs_channels_number() == 3 &&
                convolutional_layer.get_inputs_rows_number() == 256 &&
                convolutional_layer.get_inputs_columns_number() == 300 &&
                convolutional_layer.get_filters_number() == 10 &&
                convolutional_layer.get_filters_channels_number() == 3 &&
                convolutional_layer.get_filters_rows_number() == 5 &&
                convolutional_layer.get_filters_columns_number() == 5, LOG);
}


void ConvolutionalLayerTest::test_set_parameters()
{
    cout << "test_set_parameters\n";

    ConvolutionalLayer convolutional_layer;
    Tensor<double> new_synaptic_weights;
    Vector<double> new_biases;

    // Test

    new_synaptic_weights.set(Vector<size_t>({2,2,2,2}));
    new_synaptic_weights(0,0,0,0) = 1.111;
    new_synaptic_weights(0,0,0,1) = 1.112;
    new_synaptic_weights(0,0,1,0) = 1.121;
    new_synaptic_weights(0,0,1,1) = 1.122;
    new_synaptic_weights(0,1,0,0) = 1.211;
    new_synaptic_weights(0,1,0,1) = 1.212;
    new_synaptic_weights(0,1,1,0) = 1.221;
    new_synaptic_weights(0,1,1,1) = 1.222;
    new_synaptic_weights(1,0,0,0) = 2.111;
    new_synaptic_weights(1,0,0,1) = 2.112;
    new_synaptic_weights(1,0,1,0) = 2.121;
    new_synaptic_weights(1,0,1,1) = 2.122;
    new_synaptic_weights(1,1,0,0) = 2.211;
    new_synaptic_weights(1,1,0,1) = 2.212;
    new_synaptic_weights(1,1,1,0) = 2.221;
    new_synaptic_weights(1,1,1,1) = 2.222;
    new_biases = Vector<double>({-1,1});

    convolutional_layer.set({2,3,3}, {2,2,2});
    convolutional_layer.set_parameters({1.111,2.111,1.211,2.211,1.121,2.121,1.221,2.221,1.112,2.112,1.212,2.212,1.122,2.122,1.222,2.222,-1,1});

    assert_true(convolutional_layer.is_empty() == false &&
                convolutional_layer.get_parameters_number() == 18 &&
                convolutional_layer.get_synaptic_weights() == new_synaptic_weights &&
                convolutional_layer.get_biases() == new_biases, LOG);
}


void ConvolutionalLayerTest::test_calculate_image_convolution()
{
   cout << "test_calculate_image_convolution\n";

   ConvolutionalLayer convolutional_layer;

   Tensor<double> image;
   Tensor<double> filter;
   Matrix<double> convolutions;
   Matrix<double> result;

   // Test

   image.set(Vector<size_t>({1, 2, 2}));

   filter.set(Vector<size_t>({1, 1, 1}));

   convolutions = convolutional_layer.calculate_image_convolution(image, filter);

   result.set(2, 2);

   assert_true(convolutions == result, LOG);

   // Test

   image.set({1, 5, 5}, 1.0);

   filter.set(Vector<size_t>({1, 2, 2}));
   filter(0,0,0) = 1;
   filter(0,0,1) = 2;
   filter(0,1,0) = 3;
   filter(0,1,1) = 4;

   convolutional_layer.set_row_stride(1);
   convolutional_layer.set_column_stride(1);

   convolutions = convolutional_layer.calculate_image_convolution(image, filter);

   result.set(4, 4, 10);

   assert_true(convolutions == result, LOG);

   // Test

   image.set(Vector<size_t>({1,4,4}));
   image(0,0,0) = 3;
   image(0,0,1) = 4;
   image(0,0,2) = 4;
   image(0,0,3) = 4;
   image(0,1,0) = 5;
   image(0,1,1) = 1;
   image(0,1,2) = 4;
   image(0,1,3) = 1;
   image(0,2,0) = 4;
   image(0,2,1) = 5;
   image(0,2,2) = 4;
   image(0,2,3) = 4;
   image(0,3,0) = 5;
   image(0,3,1) = 5;
   image(0,3,2) = 2;
   image(0,3,3) = 1;

   filter.set(Vector<size_t>({1, 2, 2}));
   filter(0,0,0) = 1;
   filter(0,0,1) = 2;
   filter(0,1,0) = 3;
   filter(0,1,1) = 4;

   convolutional_layer.set_row_stride(1);
   convolutional_layer.set_column_stride(1);

   convolutions = convolutional_layer.calculate_image_convolution(image, filter);

   result.set(3, 3);
   result(0,0) = 30;
   result(0,1) = 31;
   result(0,2) = 28;
   result(1,0) = 39;
   result(1,1) = 40;
   result(1,2) = 34;
   result(2,0) = 49;
   result(2,1) = 36;
   result(2,2) = 22;

   assert_true(convolutions == result, LOG);

   // Test

   image.set(Vector<size_t>({1,5,4}));
   image(0,0,0) = 3;
   image(0,0,1) = 4;
   image(0,0,2) = 4;
   image(0,0,3) = 4;
   image(0,1,0) = 5;
   image(0,1,1) = 1;
   image(0,1,2) = 4;
   image(0,1,3) = 1;
   image(0,2,0) = 4;
   image(0,2,1) = 5;
   image(0,2,2) = 4;
   image(0,2,3) = 4;
   image(0,3,0) = 5;
   image(0,3,1) = 5;
   image(0,3,2) = 2;
   image(0,3,3) = 1;
   image(0,4,0) = 1;
   image(0,4,1) = 1;
   image(0,4,2) = 1;
   image(0,4,3) = 1;

   filter.set(Vector<size_t>({1, 2, 2}));
   filter(0,0,0) = 1;
   filter(0,0,1) = 0;
   filter(0,1,0) = 0;
   filter(0,1,1) = 1;

   convolutional_layer.set_row_stride(3);
   convolutional_layer.set_column_stride(1);

   convolutions = convolutional_layer.calculate_image_convolution(image, filter);

   result.set(2, 3);
   result(0,0) = 4;
   result(0,1) = 8;
   result(0,2) = 5;
   result(1,0) = 6;
   result(1,1) = 6;
   result(1,2) = 3;

   assert_true(convolutions == result, LOG);

   // Test

   image.set(Vector<size_t>({1,5,4}));
   image(0,0,0) = 3;
   image(0,0,1) = 4;
   image(0,0,2) = 4;
   image(0,0,3) = 4;
   image(0,1,0) = 5;
   image(0,1,1) = 1;
   image(0,1,2) = 4;
   image(0,1,3) = 1;
   image(0,2,0) = 4;
   image(0,2,1) = 5;
   image(0,2,2) = 4;
   image(0,2,3) = 4;
   image(0,3,0) = 5;
   image(0,3,1) = 5;
   image(0,3,2) = 2;
   image(0,3,3) = 1;
   image(0,4,0) = 1;
   image(0,4,1) = 1;
   image(0,4,2) = 1;
   image(0,4,3) = 1;

   filter.set(Vector<size_t>({1, 2, 2}));
   filter(0,0,0) = 1;
   filter(0,0,1) = 0;
   filter(0,1,0) = 0;
   filter(0,1,1) = 1;

   convolutional_layer.set_row_stride(1);
   convolutional_layer.set_column_stride(2);

   convolutions = convolutional_layer.calculate_image_convolution(image, filter);

   result.set(4, 2);
   result(0,0) = 4;
   result(0,1) = 5;
   result(1,0) = 10;
   result(1,1) = 8;
   result(2,0) = 9;
   result(2,1) = 5;
   result(3,0) = 6;
   result(3,1) = 3;

   assert_true(convolutions == result, LOG);

   // Test

   image.set(Vector<size_t>({2,3,3}));
   image(0,0,0) = 1;
   image(0,0,1) = 2;
   image(0,0,2) = 3;
   image(0,1,0) = 1;
   image(0,1,1) = 3;
   image(0,1,2) = 2;
   image(0,2,0) = 3;
   image(0,2,1) = 2;
   image(0,2,2) = 1;
   image(1,0,0) = 1;
   image(1,0,1) = 1;
   image(1,0,2) = 1;
   image(1,1,0) = 2;
   image(1,1,1) = 2;
   image(1,1,2) = 2;
   image(1,2,0) = 3;
   image(1,2,1) = 3;
   image(1,2,2) = 3;

   filter.set(Vector<size_t>({2, 2, 2}));
   filter(0,0,0) = 1;
   filter(0,0,1) = 2;
   filter(0,1,0) = 2;
   filter(0,1,1) = 1;
   filter(1,0,0) = 1;
   filter(1,0,1) = 3;
   filter(1,1,0) = 3;
   filter(1,1,1) = 1;

   convolutional_layer.set_row_stride(1);
   convolutional_layer.set_column_stride(1);

   convolutions = convolutional_layer.calculate_image_convolution(image, filter);

   result.set(2, 2);
   result(0,0) = 22;
   result(0,1) = 28;
   result(1,0) = 35;
   result(1,1) = 32;

   assert_true(convolutions == result, LOG);
}


void ConvolutionalLayerTest::test_calculate_convolutions()
{
    cout << "test_calculate_convolutions\n";

    ConvolutionalLayer convolutional_layer;

    Tensor<double> images;
    Tensor<double> filters;
    Tensor<double> convolutions;
    Tensor<double> result;

    // Test

    images.set(Vector<size_t>({2,2,3,3}));
    images(0,0,0,0) = 1.1;
    images(0,0,0,1) = 1.1;
    images(0,0,0,2) = 1.1;
    images(0,0,1,0) = 1.1;
    images(0,0,1,1) = 1.1;
    images(0,0,1,2) = 1.1;
    images(0,0,2,0) = 1.1;
    images(0,0,2,1) = 1.1;
    images(0,0,2,2) = 1.1;
    images(0,1,0,0) = 1.2;
    images(0,1,0,1) = 1.2;
    images(0,1,0,2) = 1.2;
    images(0,1,1,0) = 1.2;
    images(0,1,1,1) = 1.2;
    images(0,1,1,2) = 1.2;
    images(0,1,2,0) = 1.2;
    images(0,1,2,1) = 1.2;
    images(0,1,2,2) = 1.2;
    images(1,0,0,0) = 2.1;
    images(1,0,0,1) = 2.1;
    images(1,0,0,2) = 2.1;
    images(1,0,1,0) = 2.1;
    images(1,0,1,1) = 2.1;
    images(1,0,1,2) = 2.1;
    images(1,0,2,0) = 2.1;
    images(1,0,2,1) = 2.1;
    images(1,0,2,2) = 2.1;
    images(1,1,0,0) = 2.2;
    images(1,1,0,1) = 2.2;
    images(1,1,0,2) = 2.2;
    images(1,1,1,0) = 2.2;
    images(1,1,1,1) = 2.2;
    images(1,1,1,2) = 2.2;
    images(1,1,2,0) = 2.2;
    images(1,1,2,1) = 2.2;
    images(1,1,2,2) = 2.2;

    filters.set({2,2,2,2}, 0);

    convolutional_layer.set({2,3,3}, {2,2,2});
    convolutional_layer.set_synaptic_weights(filters);
    convolutional_layer.set_biases({-1,1});

    convolutions = convolutional_layer.calculate_convolutions(images);

    result.set(Vector<size_t>({2,2,2,2}));
    result(0,0,0,0) = -1;
    result(0,0,0,1) = -1;
    result(0,0,1,0) = -1;
    result(0,0,1,1) = -1;
    result(0,1,0,0) = 1;
    result(0,1,0,1) = 1;
    result(0,1,1,0) = 1;
    result(0,1,1,1) = 1;
    result(1,0,0,0) = -1;
    result(1,0,0,1) = -1;
    result(1,0,1,0) = -1;
    result(1,0,1,1) = -1;
    result(1,1,0,0) = 1;
    result(1,1,0,1) = 1;
    result(1,1,1,0) = 1;
    result(1,1,1,1) = 1;

    assert_true(convolutions == result, LOG);

    // Test

    images.set(Vector<size_t>({2,1,6,6}));
    images(0,0,0,0) = 1;
    images(0,0,0,1) = 2;
    images(0,0,0,2) = 3;
    images(0,0,0,3) = 4;
    images(0,0,0,4) = 5;
    images(0,0,0,5) = 6;
    images(0,0,1,0) = 7;
    images(0,0,1,1) = 8;
    images(0,0,1,2) = 9;
    images(0,0,1,3) = 10;
    images(0,0,1,4) = 11;
    images(0,0,1,5) = 12;
    images(0,0,2,0) = 13;
    images(0,0,2,1) = 14;
    images(0,0,2,2) = 15;
    images(0,0,2,3) = 16;
    images(0,0,2,4) = 17;
    images(0,0,2,5) = 18;
    images(0,0,3,0) = 19;
    images(0,0,3,1) = 20;
    images(0,0,3,2) = 21;
    images(0,0,3,3) = 22;
    images(0,0,3,4) = 23;
    images(0,0,3,5) = 24;
    images(0,0,4,0) = 25;
    images(0,0,4,1) = 26;
    images(0,0,4,2) = 27;
    images(0,0,4,3) = 28;
    images(0,0,4,4) = 29;
    images(0,0,4,5) = 30;
    images(0,0,5,0) = 31;
    images(0,0,5,1) = 32;
    images(0,0,5,2) = 33;
    images(0,0,5,3) = 34;
    images(0,0,5,4) = 35;
    images(0,0,5,5) = 36;
    images(1,0,0,0) = -1;
    images(1,0,0,1) = -2;
    images(1,0,0,2) = -3;
    images(1,0,0,3) = -4;
    images(1,0,0,4) = -5;
    images(1,0,0,5) = -6;
    images(1,0,1,0) = -7;
    images(1,0,1,1) = -8;
    images(1,0,1,2) = -9;
    images(1,0,1,3) = -10;
    images(1,0,1,4) = -11;
    images(1,0,1,5) = -12;
    images(1,0,2,0) = -13;
    images(1,0,2,1) = -14;
    images(1,0,2,2) = -15;
    images(1,0,2,3) = -16;
    images(1,0,2,4) = -17;
    images(1,0,2,5) = -18;
    images(1,0,3,0) = -19;
    images(1,0,3,1) = -20;
    images(1,0,3,2) = -21;
    images(1,0,3,3) = -22;
    images(1,0,3,4) = -23;
    images(1,0,3,5) = -24;
    images(1,0,4,0) = -25;
    images(1,0,4,1) = -26;
    images(1,0,4,2) = -27;
    images(1,0,4,3) = -28;
    images(1,0,4,4) = -29;
    images(1,0,4,5) = -30;
    images(1,0,5,0) = -31;
    images(1,0,5,1) = -32;
    images(1,0,5,2) = -33;
    images(1,0,5,3) = -34;
    images(1,0,5,4) = -35;
    images(1,0,5,5) = -36;

    filters.set(Vector<size_t>({3,1,3,3}));
    filters(0,0,0,0) = 1;
    filters(0,0,0,1) = 1;
    filters(0,0,0,2) = 1;
    filters(0,0,1,0) = 1;
    filters(0,0,1,1) = 1;
    filters(0,0,1,2) = 1;
    filters(0,0,2,0) = 1;
    filters(0,0,2,1) = 1;
    filters(0,0,2,2) = 1;
    filters(1,0,0,0) = 2;
    filters(1,0,0,1) = 2;
    filters(1,0,0,2) = 2;
    filters(1,0,1,0) = 2;
    filters(1,0,1,1) = 2;
    filters(1,0,1,2) = 2;
    filters(1,0,2,0) = 2;
    filters(1,0,2,1) = 2;
    filters(1,0,2,2) = 2;
    filters(2,0,0,0) = 3;
    filters(2,0,0,1) = 3;
    filters(2,0,0,2) = 3;
    filters(2,0,1,0) = 3;
    filters(2,0,1,1) = 3;
    filters(2,0,1,2) = 3;
    filters(2,0,2,0) = 3;
    filters(2,0,2,1) = 3;
    filters(2,0,2,2) = 3;

    convolutional_layer.set({1,6,6}, {3,3,3});
    convolutional_layer.set_synaptic_weights(filters);
    convolutional_layer.set_biases({-1,0,1});

    convolutions = convolutional_layer.calculate_convolutions(images);

    result.set(Vector<size_t>({2,3,4,4}));
    result(0,0,0,0) = 71;
    result(0,0,0,1) = 80;
    result(0,0,0,2) = 89;
    result(0,0,0,3) = 98;
    result(0,0,1,0) = 125;
    result(0,0,1,1) = 134;
    result(0,0,1,2) = 143;
    result(0,0,1,3) = 152;
    result(0,0,2,0) = 179;
    result(0,0,2,1) = 188;
    result(0,0,2,2) = 197;
    result(0,0,2,3) = 206;
    result(0,0,3,0) = 233;
    result(0,0,3,1) = 242;
    result(0,0,3,2) = 251;
    result(0,0,3,3) = 260;
    result(0,1,0,0) = 144;
    result(0,1,0,1) = 162;
    result(0,1,0,2) = 180;
    result(0,1,0,3) = 198;
    result(0,1,1,0) = 252;
    result(0,1,1,1) = 270;
    result(0,1,1,2) = 288;
    result(0,1,1,3) = 306;
    result(0,1,2,0) = 360;
    result(0,1,2,1) = 378;
    result(0,1,2,2) = 396;
    result(0,1,2,3) = 414;
    result(0,1,3,0) = 468;
    result(0,1,3,1) = 486;
    result(0,1,3,2) = 504;
    result(0,1,3,3) = 522;
    result(0,2,0,0) = 217;
    result(0,2,0,1) = 244;
    result(0,2,0,2) = 271;
    result(0,2,0,3) = 298;
    result(0,2,1,0) = 379;
    result(0,2,1,1) = 406;
    result(0,2,1,2) = 433;
    result(0,2,1,3) = 460;
    result(0,2,2,0) = 541;
    result(0,2,2,1) = 568;
    result(0,2,2,2) = 595;
    result(0,2,2,3) = 622;
    result(0,2,3,0) = 703;
    result(0,2,3,1) = 730;
    result(0,2,3,2) = 757;
    result(0,2,3,3) = 784;
    result(1,0,0,0) = -73;
    result(1,0,0,1) = -82;
    result(1,0,0,2) = -91;
    result(1,0,0,3) = -100;
    result(1,0,1,0) = -127;
    result(1,0,1,1) = -136;
    result(1,0,1,2) = -145;
    result(1,0,1,3) = -154;
    result(1,0,2,0) = -181;
    result(1,0,2,1) = -190;
    result(1,0,2,2) = -199;
    result(1,0,2,3) = -208;
    result(1,0,3,0) = -235;
    result(1,0,3,1) = -244;
    result(1,0,3,2) = -253;
    result(1,0,3,3) = -262;
    result(1,1,0,0) = -144;
    result(1,1,0,1) = -162;
    result(1,1,0,2) = -180;
    result(1,1,0,3) = -198;
    result(1,1,1,0) = -252;
    result(1,1,1,1) = -270;
    result(1,1,1,2) = -288;
    result(1,1,1,3) = -306;
    result(1,1,2,0) = -360;
    result(1,1,2,1) = -378;
    result(1,1,2,2) = -396;
    result(1,1,2,3) = -414;
    result(1,1,3,0) = -468;
    result(1,1,3,1) = -486;
    result(1,1,3,2) = -504;
    result(1,1,3,3) = -522;
    result(1,2,0,0) = -215;
    result(1,2,0,1) = -242;
    result(1,2,0,2) = -269;
    result(1,2,0,3) = -296;
    result(1,2,1,0) = -377;
    result(1,2,1,1) = -404;
    result(1,2,1,2) = -431;
    result(1,2,1,3) = -458;
    result(1,2,2,0) = -539;
    result(1,2,2,1) = -566;
    result(1,2,2,2) = -593;
    result(1,2,2,3) = -620;
    result(1,2,3,0) = -701;
    result(1,2,3,1) = -728;
    result(1,2,3,2) = -755;
    result(1,2,3,3) = -782;

    assert_true(convolutions == result, LOG);
}


void ConvolutionalLayerTest::test_calculate_activations()
{
    cout << "test_calculate_activations\n";

    ConvolutionalLayer convolutional_layer;
    Tensor<double> inputs;
    Tensor<double> activations;
    Tensor<double> result;

    // Test

    inputs.set(Vector<size_t>({2,2,2,2}));
    inputs(0,0,0,0) = -1.111;
    inputs(0,0,0,1) = -1.112;
    inputs(0,0,1,0) = -1.121;
    inputs(0,0,1,1) = -1.122;
    inputs(0,1,0,0) = 1.211;
    inputs(0,1,0,1) = 1.212;
    inputs(0,1,1,0) = 1.221;
    inputs(0,1,1,1) = 1.222;
    inputs(1,0,0,0) = -2.111;
    inputs(1,0,0,1) = -2.112;
    inputs(1,0,1,0) = -2.121;
    inputs(1,0,1,1) = -2.122;
    inputs(1,1,0,0) = 2.211;
    inputs(1,1,0,1) = 2.212;
    inputs(1,1,1,0) = 2.221;
    inputs(1,1,1,1) = 2.222;

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::Threshold);

    activations = convolutional_layer.calculate_activations(inputs);

    result.set(Vector<size_t>({2,2,2,2}));
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

    assert_true(activations == result, LOG);

    // Test

    inputs.set(Vector<size_t>({2,2,2,2}));
    inputs(0,0,0,0) = -1.111;
    inputs(0,0,0,1) = -1.112;
    inputs(0,0,1,0) = -1.121;
    inputs(0,0,1,1) = -1.122;
    inputs(0,1,0,0) = 1.211;
    inputs(0,1,0,1) = 1.212;
    inputs(0,1,1,0) = 1.221;
    inputs(0,1,1,1) = 1.222;
    inputs(1,0,0,0) = -2.111;
    inputs(1,0,0,1) = -2.112;
    inputs(1,0,1,0) = -2.121;
    inputs(1,0,1,1) = -2.122;
    inputs(1,1,0,0) = 2.211;
    inputs(1,1,0,1) = 2.212;
    inputs(1,1,1,0) = 2.221;
    inputs(1,1,1,1) = 2.222;

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::SymmetricThreshold);

    activations = convolutional_layer.calculate_activations(inputs);

    result.set(Vector<size_t>({2,2,2,2}));
    result(0,0,0,0) = -1;
    result(0,0,0,1) = -1;
    result(0,0,1,0) = -1;
    result(0,0,1,1) = -1;
    result(0,1,0,0) = 1;
    result(0,1,0,1) = 1;
    result(0,1,1,0) = 1;
    result(0,1,1,1) = 1;
    result(1,0,0,0) = -1;
    result(1,0,0,1) = -1;
    result(1,0,1,0) = -1;
    result(1,0,1,1) = -1;
    result(1,1,0,0) = 1;
    result(1,1,0,1) = 1;
    result(1,1,1,0) = 1;
    result(1,1,1,1) = 1;

    assert_true(activations == result, LOG);

    // Test

    inputs.set(Vector<size_t>({2,2,2,2}));
    inputs(0,0,0,0) = -1.111;
    inputs(0,0,0,1) = -1.112;
    inputs(0,0,1,0) = -1.121;
    inputs(0,0,1,1) = -1.122;
    inputs(0,1,0,0) = 1.211;
    inputs(0,1,0,1) = 1.212;
    inputs(0,1,1,0) = 1.221;
    inputs(0,1,1,1) = 1.222;
    inputs(1,0,0,0) = -2.111;
    inputs(1,0,0,1) = -2.112;
    inputs(1,0,1,0) = -2.121;
    inputs(1,0,1,1) = -2.122;
    inputs(1,1,0,0) = 2.211;
    inputs(1,1,0,1) = 2.212;
    inputs(1,1,1,0) = 2.221;
    inputs(1,1,1,1) = 2.222;

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::HyperbolicTangent);

    activations = convolutional_layer.calculate_activations(inputs);

    result.set(Vector<size_t>({2,2,2,2}));
    result(0,0,0,0) = -0.804416;
    result(0,0,0,1) = -0.804768;
    result(0,0,1,0) = -0.807916;
    result(0,0,1,1) = -0.808263;
    result(0,1,0,0) = 0.836979;
    result(0,1,0,1) = 0.837278;
    result(0,1,1,0) = 0.839949;
    result(0,1,1,1) = 0.840243;
    result(1,0,0,0) = -0.971086;
    result(1,0,0,1) = -0.971143;
    result(1,0,1,0) = -0.971650;
    result(1,0,1,1) = -0.971706;
    result(1,1,0,0) = 0.976265;
    result(1,1,0,1) = 0.976312;
    result(1,1,1,0) = 0.976729;
    result(1,1,1,1) = 0.976775;

    assert_true(abs(activations(0,0,0,0) - result(0,0,0,0)) < 1e-6 &&
                abs(activations(0,0,0,1) - result(0,0,0,1)) < 1e-6 &&
                abs(activations(0,0,1,0) - result(0,0,1,0)) < 1e-6 &&
                abs(activations(0,0,1,1) - result(0,0,1,1)) < 1e-6 &&
                abs(activations(0,1,0,0) - result(0,1,0,0)) < 1e-6 &&
                abs(activations(0,1,0,1) - result(0,1,0,1)) < 1e-6 &&
                abs(activations(0,1,1,0) - result(0,1,1,0)) < 1e-6 &&
                abs(activations(0,1,1,1) - result(0,1,1,1)) < 1e-6 &&
                abs(activations(1,0,0,0) - result(1,0,0,0)) < 1e-6 &&
                abs(activations(1,0,0,1) - result(1,0,0,1)) < 1e-6 &&
                abs(activations(1,0,1,0) - result(1,0,1,0)) < 1e-6 &&
                abs(activations(1,0,1,1) - result(1,0,1,1)) < 1e-6 &&
                abs(activations(1,1,0,0) - result(1,1,0,0)) < 1e-6 &&
                abs(activations(1,1,0,1) - result(1,1,0,1)) < 1e-6 &&
                abs(activations(1,1,1,0) - result(1,1,1,0)) < 1e-6 &&
                abs(activations(1,1,1,1) - result(1,1,1,1)) < 1e-6, LOG);

    // Test

    inputs.set(Vector<size_t>({2,2,2,2}));
    inputs(0,0,0,0) = -1.111;
    inputs(0,0,0,1) = -1.112;
    inputs(0,0,1,0) = -1.121;
    inputs(0,0,1,1) = -1.122;
    inputs(0,1,0,0) = 1.211;
    inputs(0,1,0,1) = 1.212;
    inputs(0,1,1,0) = 1.221;
    inputs(0,1,1,1) = 1.222;
    inputs(1,0,0,0) = -2.111;
    inputs(1,0,0,1) = -2.112;
    inputs(1,0,1,0) = -2.121;
    inputs(1,0,1,1) = -2.122;
    inputs(1,1,0,0) = 2.211;
    inputs(1,1,0,1) = 2.212;
    inputs(1,1,1,0) = 2.221;
    inputs(1,1,1,1) = 2.222;

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::RectifiedLinear);

    activations = convolutional_layer.calculate_activations(inputs);

    result.set(Vector<size_t>({2,2,2,2}));
    result(0,0,0,0) = 0;
    result(0,0,0,1) = 0;
    result(0,0,1,0) = 0;
    result(0,0,1,1) = 0;
    result(0,1,0,0) = 1.211;
    result(0,1,0,1) = 1.212;
    result(0,1,1,0) = 1.221;
    result(0,1,1,1) = 1.222;
    result(1,0,0,0) = 0;
    result(1,0,0,1) = 0;
    result(1,0,1,0) = 0;
    result(1,0,1,1) = 0;
    result(1,1,0,0) = 2.211;
    result(1,1,0,1) = 2.212;
    result(1,1,1,0) = 2.221;
    result(1,1,1,1) = 2.222;

    assert_true(activations == result, LOG);

    // Test

    inputs.set(Vector<size_t>({2,2,2,2}));
    inputs(0,0,0,0) = -1.111;
    inputs(0,0,0,1) = -1.112;
    inputs(0,0,1,0) = -1.121;
    inputs(0,0,1,1) = -1.122;
    inputs(0,1,0,0) = 1.211;
    inputs(0,1,0,1) = 1.212;
    inputs(0,1,1,0) = 1.221;
    inputs(0,1,1,1) = 1.222;
    inputs(1,0,0,0) = -2.111;
    inputs(1,0,0,1) = -2.112;
    inputs(1,0,1,0) = -2.121;
    inputs(1,0,1,1) = -2.122;
    inputs(1,1,0,0) = 2.211;
    inputs(1,1,0,1) = 2.212;
    inputs(1,1,1,0) = 2.221;
    inputs(1,1,1,1) = 2.222;

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::SoftPlus);

    activations = convolutional_layer.calculate_activations(inputs);

    result.set(Vector<size_t>({2,2,2,2}));
    result(0,0,0,0) = 0.284600;
    result(0,0,0,1) = 0.284352;
    result(0,0,1,0) = 0.282132;
    result(0,0,1,1) = 0.281886;
    result(0,1,0,0) = 1.471747;
    result(0,1,0,1) = 1.472518;
    result(0,1,1,0) = 1.479461;
    result(0,1,1,1) = 1.480233;
    result(1,0,0,0) = 0.114325;
    result(1,0,0,1) = 0.114217;
    result(1,0,1,0) = 0.113250;
    result(1,0,1,1) = 0.113143;
    result(1,1,0,0) = 2.314991;
    result(1,1,0,1) = 2.315893;
    result(1,1,1,0) = 2.324008;
    result(1,1,1,1) = 2.324910;

    assert_true(abs(activations(0,0,0,0) - result(0,0,0,0)) < 1e-6 &&
                abs(activations(0,0,0,1) - result(0,0,0,1)) < 1e-6 &&
                abs(activations(0,0,1,0) - result(0,0,1,0)) < 1e-6 &&
                abs(activations(0,0,1,1) - result(0,0,1,1)) < 1e-6 &&
                abs(activations(0,1,0,0) - result(0,1,0,0)) < 1e-6 &&
                abs(activations(0,1,0,1) - result(0,1,0,1)) < 1e-6 &&
                abs(activations(0,1,1,0) - result(0,1,1,0)) < 1e-6 &&
                abs(activations(0,1,1,1) - result(0,1,1,1)) < 1e-6 &&
                abs(activations(1,0,0,0) - result(1,0,0,0)) < 1e-6 &&
                abs(activations(1,0,0,1) - result(1,0,0,1)) < 1e-6 &&
                abs(activations(1,0,1,0) - result(1,0,1,0)) < 1e-6 &&
                abs(activations(1,0,1,1) - result(1,0,1,1)) < 1e-6 &&
                abs(activations(1,1,0,0) - result(1,1,0,0)) < 1e-6 &&
                abs(activations(1,1,0,1) - result(1,1,0,1)) < 1e-6 &&
                abs(activations(1,1,1,0) - result(1,1,1,0)) < 1e-6 &&
                abs(activations(1,1,1,1) - result(1,1,1,1)) < 1e-6, LOG);
}


void ConvolutionalLayerTest::test_calculate_activations_derivatives()
{
    cout << "test_calculate_activations_derivatives\n";

    ConvolutionalLayer convolutional_layer;
    Tensor<double> inputs;
    Tensor<double> activations_derivatives;
    Tensor<double> result;

    // Test

    inputs.set(Vector<size_t>({2,2,2,2}));
    inputs(0,0,0,0) = -1.111;
    inputs(0,0,0,1) = -1.112;
    inputs(0,0,1,0) = -1.121;
    inputs(0,0,1,1) = -1.122;
    inputs(0,1,0,0) = 1.211;
    inputs(0,1,0,1) = 1.212;
    inputs(0,1,1,0) = 1.221;
    inputs(0,1,1,1) = 1.222;
    inputs(1,0,0,0) = -2.111;
    inputs(1,0,0,1) = -2.112;
    inputs(1,0,1,0) = -2.121;
    inputs(1,0,1,1) = -2.122;
    inputs(1,1,0,0) = 2.211;
    inputs(1,1,0,1) = 2.212;
    inputs(1,1,1,0) = 2.221;
    inputs(1,1,1,1) = 2.222;

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::Threshold);

    activations_derivatives = convolutional_layer.calculate_activations_derivatives(inputs);

    result.set({2,2,2,2}, 0);

    assert_true(activations_derivatives == result, LOG);

    // Test

    inputs.set(Vector<size_t>({2,2,2,2}));
    inputs(0,0,0,0) = -1.111;
    inputs(0,0,0,1) = -1.112;
    inputs(0,0,1,0) = -1.121;
    inputs(0,0,1,1) = -1.122;
    inputs(0,1,0,0) = 1.211;
    inputs(0,1,0,1) = 1.212;
    inputs(0,1,1,0) = 1.221;
    inputs(0,1,1,1) = 1.222;
    inputs(1,0,0,0) = -2.111;
    inputs(1,0,0,1) = -2.112;
    inputs(1,0,1,0) = -2.121;
    inputs(1,0,1,1) = -2.122;
    inputs(1,1,0,0) = 2.211;
    inputs(1,1,0,1) = 2.212;
    inputs(1,1,1,0) = 2.221;
    inputs(1,1,1,1) = 2.222;

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::SymmetricThreshold);

    activations_derivatives = convolutional_layer.calculate_activations_derivatives(inputs);

    result.set({2,2,2,2}, 0);

    assert_true(activations_derivatives == result, LOG);

    // Test

    inputs.set(Vector<size_t>({2,2,2,2}));
    inputs(0,0,0,0) = -1.111;
    inputs(0,0,0,1) = -1.112;
    inputs(0,0,1,0) = -1.121;
    inputs(0,0,1,1) = -1.122;
    inputs(0,1,0,0) = 1.211;
    inputs(0,1,0,1) = 1.212;
    inputs(0,1,1,0) = 1.221;
    inputs(0,1,1,1) = 1.222;
    inputs(1,0,0,0) = -2.111;
    inputs(1,0,0,1) = -2.112;
    inputs(1,0,1,0) = -2.121;
    inputs(1,0,1,1) = -2.122;
    inputs(1,1,0,0) = 2.211;
    inputs(1,1,0,1) = 2.212;
    inputs(1,1,1,0) = 2.221;
    inputs(1,1,1,1) = 2.222;

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::HyperbolicTangent);

    activations_derivatives = convolutional_layer.calculate_activations_derivatives(inputs);

    result.set(Vector<size_t>({2,2,2,2}));
    result(0,0,0,0) = 0.352916;
    result(0,0,0,1) = 0.352348;
    result(0,0,1,0) = 0.347271;
    result(0,0,1,1) = 0.346710;
    result(0,1,0,0) = 0.299466;
    result(0,1,0,1) = 0.298965;
    result(0,1,1,0) = 0.294486;
    result(0,1,1,1) = 0.293991;
    result(1,0,0,0) = 0.056993;
    result(1,0,0,1) = 0.056882;
    result(1,0,1,0) = 0.055896;
    result(1,0,1,1) = 0.055788;
    result(1,1,0,0) = 0.046907;
    result(1,1,0,1) = 0.046816;
    result(1,1,1,0) = 0.046000;
    result(1,1,1,1) = 0.045910;

    assert_true(abs(activations_derivatives(0,0,0,0) - result(0,0,0,0)) < 1e-6 &&
                abs(activations_derivatives(0,0,0,1) - result(0,0,0,1)) < 1e-6 &&
                abs(activations_derivatives(0,0,1,0) - result(0,0,1,0)) < 1e-6 &&
                abs(activations_derivatives(0,0,1,1) - result(0,0,1,1)) < 1e-6 &&
                abs(activations_derivatives(0,1,0,0) - result(0,1,0,0)) < 1e-6 &&
                abs(activations_derivatives(0,1,0,1) - result(0,1,0,1)) < 1e-6 &&
                abs(activations_derivatives(0,1,1,0) - result(0,1,1,0)) < 1e-6 &&
                abs(activations_derivatives(0,1,1,1) - result(0,1,1,1)) < 1e-6 &&
                abs(activations_derivatives(1,0,0,0) - result(1,0,0,0)) < 1e-6 &&
                abs(activations_derivatives(1,0,0,1) - result(1,0,0,1)) < 1e-6 &&
                abs(activations_derivatives(1,0,1,0) - result(1,0,1,0)) < 1e-6 &&
                abs(activations_derivatives(1,0,1,1) - result(1,0,1,1)) < 1e-6 &&
                abs(activations_derivatives(1,1,0,0) - result(1,1,0,0)) < 1e-6 &&
                abs(activations_derivatives(1,1,0,1) - result(1,1,0,1)) < 1e-6 &&
                abs(activations_derivatives(1,1,1,0) - result(1,1,1,0)) < 1e-6 &&
                abs(activations_derivatives(1,1,1,1) - result(1,1,1,1)) < 1e-6, LOG);

    // Test

    inputs.set(Vector<size_t>({2,2,2,2}));
    inputs(0,0,0,0) = -1.111;
    inputs(0,0,0,1) = -1.112;
    inputs(0,0,1,0) = -1.121;
    inputs(0,0,1,1) = -1.122;
    inputs(0,1,0,0) = 1.211;
    inputs(0,1,0,1) = 1.212;
    inputs(0,1,1,0) = 1.221;
    inputs(0,1,1,1) = 1.222;
    inputs(1,0,0,0) = -2.111;
    inputs(1,0,0,1) = -2.112;
    inputs(1,0,1,0) = -2.121;
    inputs(1,0,1,1) = -2.122;
    inputs(1,1,0,0) = 2.211;
    inputs(1,1,0,1) = 2.212;
    inputs(1,1,1,0) = 2.221;
    inputs(1,1,1,1) = 2.222;

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::RectifiedLinear);

    activations_derivatives = convolutional_layer.calculate_activations_derivatives(inputs);

    result.set(Vector<size_t>({2,2,2,2}));
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

    assert_true(activations_derivatives == result, LOG);

    // Test

    inputs.set(Vector<size_t>({2,2,2,2}));
    inputs(0,0,0,0) = -1.111;
    inputs(0,0,0,1) = -1.112;
    inputs(0,0,1,0) = -1.121;
    inputs(0,0,1,1) = -1.122;
    inputs(0,1,0,0) = 1.211;
    inputs(0,1,0,1) = 1.212;
    inputs(0,1,1,0) = 1.221;
    inputs(0,1,1,1) = 1.222;
    inputs(1,0,0,0) = -2.111;
    inputs(1,0,0,1) = -2.112;
    inputs(1,0,1,0) = -2.121;
    inputs(1,0,1,1) = -2.122;
    inputs(1,1,0,0) = 2.211;
    inputs(1,1,0,1) = 2.212;
    inputs(1,1,1,0) = 2.221;
    inputs(1,1,1,1) = 2.222;

    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::SoftPlus);

    activations_derivatives = convolutional_layer.calculate_activations_derivatives(inputs);

    result.set(Vector<size_t>({2,2,2,2}));
    result(0,0,0,0) = 0.247685;
    result(0,0,0,1) = 0.247498;
    result(0,0,1,0) = 0.245826;
    result(0,0,1,1) = 0.245640;
    result(0,1,0,0) = 0.770476;
    result(0,1,0,1) = 0.770653;
    result(0,1,1,0) = 0.772239;
    result(0,1,1,1) = 0.772415;
    result(1,0,0,0) = 0.108032;
    result(1,0,0,1) = 0.107936;
    result(1,0,1,0) = 0.107072;
    result(1,0,1,1) = 0.106977;
    result(1,1,0,0) = 0.901233;
    result(1,1,0,1) = 0.901322;
    result(1,1,1,0) = 0.902120;
    result(1,1,1,1) = 0.902208;

    assert_true(abs(activations_derivatives(0,0,0,0) - result(0,0,0,0)) < 1e-6 &&
                abs(activations_derivatives(0,0,0,1) - result(0,0,0,1)) < 1e-6 &&
                abs(activations_derivatives(0,0,1,0) - result(0,0,1,0)) < 1e-6 &&
                abs(activations_derivatives(0,0,1,1) - result(0,0,1,1)) < 1e-6 &&
                abs(activations_derivatives(0,1,0,0) - result(0,1,0,0)) < 1e-6 &&
                abs(activations_derivatives(0,1,0,1) - result(0,1,0,1)) < 1e-6 &&
                abs(activations_derivatives(0,1,1,0) - result(0,1,1,0)) < 1e-6 &&
                abs(activations_derivatives(0,1,1,1) - result(0,1,1,1)) < 1e-6 &&
                abs(activations_derivatives(1,0,0,0) - result(1,0,0,0)) < 1e-6 &&
                abs(activations_derivatives(1,0,0,1) - result(1,0,0,1)) < 1e-6 &&
                abs(activations_derivatives(1,0,1,0) - result(1,0,1,0)) < 1e-6 &&
                abs(activations_derivatives(1,0,1,1) - result(1,0,1,1)) < 1e-6 &&
                abs(activations_derivatives(1,1,0,0) - result(1,1,0,0)) < 1e-6 &&
                abs(activations_derivatives(1,1,0,1) - result(1,1,0,1)) < 1e-6 &&
                abs(activations_derivatives(1,1,1,0) - result(1,1,1,0)) < 1e-6 &&
                abs(activations_derivatives(1,1,1,1) - result(1,1,1,1)) < 1e-6, LOG);
}


void ConvolutionalLayerTest::test_calculate_outputs()
{
    cout << "test_calculate_outputs\n";

    ConvolutionalLayer convolutional_layer;

    Tensor<double> images;
    Tensor<double> filters;
    Tensor<double> outputs;

    // Test

    images.set(Vector<size_t>({2,2,3,3}));
    images(0,0,0,0) = 1.1;
    images(0,0,0,1) = 1.1;
    images(0,0,0,2) = 1.1;
    images(0,0,1,0) = 1.1;
    images(0,0,1,1) = 1.1;
    images(0,0,1,2) = 1.1;
    images(0,0,2,0) = 1.1;
    images(0,0,2,1) = 1.1;
    images(0,0,2,2) = 1.1;
    images(0,1,0,0) = 1.2;
    images(0,1,0,1) = 1.2;
    images(0,1,0,2) = 1.2;
    images(0,1,1,0) = 1.2;
    images(0,1,1,1) = 1.2;
    images(0,1,1,2) = 1.2;
    images(0,1,2,0) = 1.2;
    images(0,1,2,1) = 1.2;
    images(0,1,2,2) = 1.2;
    images(1,0,0,0) = 2.1;
    images(1,0,0,1) = 2.1;
    images(1,0,0,2) = 2.1;
    images(1,0,1,0) = 2.1;
    images(1,0,1,1) = 2.1;
    images(1,0,1,2) = 2.1;
    images(1,0,2,0) = 2.1;
    images(1,0,2,1) = 2.1;
    images(1,0,2,2) = 2.1;
    images(1,1,0,0) = 2.2;
    images(1,1,0,1) = 2.2;
    images(1,1,0,2) = 2.2;
    images(1,1,1,0) = 2.2;
    images(1,1,1,1) = 2.2;
    images(1,1,1,2) = 2.2;
    images(1,1,2,0) = 2.2;
    images(1,1,2,1) = 2.2;
    images(1,1,2,2) = 2.2;

    filters.set({2,2,2,2}, 0);

    convolutional_layer.set({2,3,3}, {2,2,2});
    convolutional_layer.set_synaptic_weights(filters);
    convolutional_layer.set_biases({-1,1});
    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::Logistic);

    outputs = convolutional_layer.calculate_outputs(images);

    assert_true(abs(outputs(0,0,0,0) - 0.268941) < 1e-6 &&
                abs(outputs(0,0,0,1) - 0.268941) < 1e-6 &&
                abs(outputs(0,0,1,0) - 0.268941) < 1e-6 &&
                abs(outputs(0,0,1,1) - 0.268941) < 1e-6 &&
                abs(outputs(0,1,0,0) - 0.731059) < 1e-6 &&
                abs(outputs(0,1,0,1) - 0.731059) < 1e-6 &&
                abs(outputs(0,1,1,0) - 0.731059) < 1e-6 &&
                abs(outputs(0,1,1,1) - 0.731059) < 1e-6 &&
                abs(outputs(1,0,0,0) - 0.268941) < 1e-6 &&
                abs(outputs(1,0,0,1) - 0.268941) < 1e-6 &&
                abs(outputs(1,0,1,0) - 0.268941) < 1e-6 &&
                abs(outputs(1,0,1,1) - 0.268941) < 1e-6 &&
                abs(outputs(1,1,0,0) - 0.731059) < 1e-6 &&
                abs(outputs(1,1,0,1) - 0.731059) < 1e-6 &&
                abs(outputs(1,1,1,0) - 0.731059) < 1e-6 &&
                abs(outputs(1,1,1,1) - 0.731059) < 1e-6, LOG);

    // Test

    images.set(Vector<size_t>({2,1,6,6}));
    images(0,0,0,0) = 1;
    images(0,0,0,1) = 2;
    images(0,0,0,2) = 3;
    images(0,0,0,3) = 4;
    images(0,0,0,4) = 5;
    images(0,0,0,5) = 6;
    images(0,0,1,0) = 7;
    images(0,0,1,1) = 8;
    images(0,0,1,2) = 9;
    images(0,0,1,3) = 10;
    images(0,0,1,4) = 11;
    images(0,0,1,5) = 12;
    images(0,0,2,0) = 13;
    images(0,0,2,1) = 14;
    images(0,0,2,2) = 15;
    images(0,0,2,3) = 16;
    images(0,0,2,4) = 17;
    images(0,0,2,5) = 18;
    images(0,0,3,0) = 19;
    images(0,0,3,1) = 20;
    images(0,0,3,2) = 21;
    images(0,0,3,3) = 22;
    images(0,0,3,4) = 23;
    images(0,0,3,5) = 24;
    images(0,0,4,0) = 25;
    images(0,0,4,1) = 26;
    images(0,0,4,2) = 27;
    images(0,0,4,3) = 28;
    images(0,0,4,4) = 29;
    images(0,0,4,5) = 30;
    images(0,0,5,0) = 31;
    images(0,0,5,1) = 32;
    images(0,0,5,2) = 33;
    images(0,0,5,3) = 34;
    images(0,0,5,4) = 35;
    images(0,0,5,5) = 36;
    images(1,0,0,0) = -1;
    images(1,0,0,1) = -2;
    images(1,0,0,2) = -3;
    images(1,0,0,3) = -4;
    images(1,0,0,4) = -5;
    images(1,0,0,5) = -6;
    images(1,0,1,0) = -7;
    images(1,0,1,1) = -8;
    images(1,0,1,2) = -9;
    images(1,0,1,3) = -10;
    images(1,0,1,4) = -11;
    images(1,0,1,5) = -12;
    images(1,0,2,0) = -13;
    images(1,0,2,1) = -14;
    images(1,0,2,2) = -15;
    images(1,0,2,3) = -16;
    images(1,0,2,4) = -17;
    images(1,0,2,5) = -18;
    images(1,0,3,0) = -19;
    images(1,0,3,1) = -20;
    images(1,0,3,2) = -21;
    images(1,0,3,3) = -22;
    images(1,0,3,4) = -23;
    images(1,0,3,5) = -24;
    images(1,0,4,0) = -25;
    images(1,0,4,1) = -26;
    images(1,0,4,2) = -27;
    images(1,0,4,3) = -28;
    images(1,0,4,4) = -29;
    images(1,0,4,5) = -30;
    images(1,0,5,0) = -31;
    images(1,0,5,1) = -32;
    images(1,0,5,2) = -33;
    images(1,0,5,3) = -34;
    images(1,0,5,4) = -35;
    images(1,0,5,5) = -36;

    filters.set(Vector<size_t>({3,1,3,3}));
    filters(0,0,0,0) = 1;
    filters(0,0,0,1) = 1;
    filters(0,0,0,2) = 1;
    filters(0,0,1,0) = 1;
    filters(0,0,1,1) = 1;
    filters(0,0,1,2) = 1;
    filters(0,0,2,0) = 1;
    filters(0,0,2,1) = 1;
    filters(0,0,2,2) = 1;
    filters(1,0,0,0) = 2;
    filters(1,0,0,1) = 2;
    filters(1,0,0,2) = 2;
    filters(1,0,1,0) = 2;
    filters(1,0,1,1) = 2;
    filters(1,0,1,2) = 2;
    filters(1,0,2,0) = 2;
    filters(1,0,2,1) = 2;
    filters(1,0,2,2) = 2;
    filters(2,0,0,0) = 3;
    filters(2,0,0,1) = 3;
    filters(2,0,0,2) = 3;
    filters(2,0,1,0) = 3;
    filters(2,0,1,1) = 3;
    filters(2,0,1,2) = 3;
    filters(2,0,2,0) = 3;
    filters(2,0,2,1) = 3;
    filters(2,0,2,2) = 3;

    convolutional_layer.set({1,6,6}, {3,3,3});
    convolutional_layer.set_synaptic_weights(filters);
    convolutional_layer.set_biases({-1,0,1});
    convolutional_layer.set_activation_function(OpenNN::ConvolutionalLayer::HardSigmoid);

    outputs = convolutional_layer.calculate_outputs(images);

    assert_true(outputs(0,0,0,0) == 1.0 &&
                outputs(0,0,0,0) == 1.0 &&
                outputs(0,0,0,1) == 1.0 &&
                outputs(0,0,0,2) == 1.0 &&
                outputs(0,0,0,3) == 1.0 &&
                outputs(0,0,1,0) == 1.0 &&
                outputs(0,0,1,1) == 1.0 &&
                outputs(0,0,1,2) == 1.0 &&
                outputs(0,0,1,3) == 1.0 &&
                outputs(0,0,2,0) == 1.0 &&
                outputs(0,0,2,1) == 1.0 &&
                outputs(0,0,2,2) == 1.0 &&
                outputs(0,0,2,3) == 1.0 &&
                outputs(0,0,3,0) == 1.0 &&
                outputs(0,0,3,1) == 1.0 &&
                outputs(0,0,3,2) == 1.0 &&
                outputs(0,0,3,3) == 1.0 &&
                outputs(0,1,0,0) == 1.0 &&
                outputs(0,1,0,1) == 1.0 &&
                outputs(0,1,0,2) == 1.0 &&
                outputs(0,1,0,3) == 1.0 &&
                outputs(0,1,1,0) == 1.0 &&
                outputs(0,1,1,1) == 1.0 &&
                outputs(0,1,1,2) == 1.0 &&
                outputs(0,1,1,3) == 1.0 &&
                outputs(0,1,2,0) == 1.0 &&
                outputs(0,1,2,1) == 1.0 &&
                outputs(0,1,2,2) == 1.0 &&
                outputs(0,1,2,3) == 1.0 &&
                outputs(0,1,3,0) == 1.0 &&
                outputs(0,1,3,1) == 1.0 &&
                outputs(0,1,3,2) == 1.0 &&
                outputs(0,1,3,3) == 1.0 &&
                outputs(0,2,0,0) == 1.0 &&
                outputs(0,2,0,1) == 1.0 &&
                outputs(0,2,0,2) == 1.0 &&
                outputs(0,2,0,3) == 1.0 &&
                outputs(0,2,1,0) == 1.0 &&
                outputs(0,2,1,1) == 1.0 &&
                outputs(0,2,1,2) == 1.0 &&
                outputs(0,2,1,3) == 1.0 &&
                outputs(0,2,2,0) == 1.0 &&
                outputs(0,2,2,1) == 1.0 &&
                outputs(0,2,2,2) == 1.0 &&
                outputs(0,2,2,3) == 1.0 &&
                outputs(0,2,3,0) == 1.0 &&
                outputs(0,2,3,1) == 1.0 &&
                outputs(0,2,3,2) == 1.0 &&
                outputs(0,2,3,3) == 1.0 &&
                outputs(1,0,0,0) == 0.0 &&
                outputs(1,0,0,1) == 0.0 &&
                outputs(1,0,0,2) == 0.0 &&
                outputs(1,0,0,3) == 0.0 &&
                outputs(1,0,1,0) == 0.0 &&
                outputs(1,0,1,1) == 0.0 &&
                outputs(1,0,1,2) == 0.0 &&
                outputs(1,0,1,3) == 0.0 &&
                outputs(1,0,2,0) == 0.0 &&
                outputs(1,0,2,1) == 0.0 &&
                outputs(1,0,2,2) == 0.0 &&
                outputs(1,0,2,3) == 0.0 &&
                outputs(1,0,3,0) == 0.0 &&
                outputs(1,0,3,1) == 0.0 &&
                outputs(1,0,3,2) == 0.0 &&
                outputs(1,0,3,3) == 0.0 &&
                outputs(1,1,0,0) == 0.0 &&
                outputs(1,1,0,1) == 0.0 &&
                outputs(1,1,0,2) == 0.0 &&
                outputs(1,1,0,3) == 0.0 &&
                outputs(1,1,1,0) == 0.0 &&
                outputs(1,1,1,1) == 0.0 &&
                outputs(1,1,1,2) == 0.0 &&
                outputs(1,1,1,3) == 0.0 &&
                outputs(1,1,2,0) == 0.0 &&
                outputs(1,1,2,1) == 0.0 &&
                outputs(1,1,2,2) == 0.0 &&
                outputs(1,1,2,3) == 0.0 &&
                outputs(1,1,3,0) == 0.0 &&
                outputs(1,1,3,1) == 0.0 &&
                outputs(1,1,3,2) == 0.0 &&
                outputs(1,1,3,3) == 0.0 &&
                outputs(1,2,0,0) == 0.0 &&
                outputs(1,2,0,1) == 0.0 &&
                outputs(1,2,0,2) == 0.0 &&
                outputs(1,2,0,3) == 0.0 &&
                outputs(1,2,1,0) == 0.0 &&
                outputs(1,2,1,1) == 0.0 &&
                outputs(1,2,1,2) == 0.0 &&
                outputs(1,2,1,3) == 0.0 &&
                outputs(1,2,2,0) == 0.0 &&
                outputs(1,2,2,1) == 0.0 &&
                outputs(1,2,2,2) == 0.0 &&
                outputs(1,2,3,0) == 0.0 &&
                outputs(1,2,2,3) == 0.0 &&
                outputs(1,2,3,1) == 0.0 &&
                outputs(1,2,3,2) == 0.0 &&
                outputs(1,2,3,3) == 0.0, LOG);
}


void ConvolutionalLayerTest::test_insert_padding()
{
    cout << "test_insert_padding\n";

    ConvolutionalLayer convolutional_layer;

    Tensor<double> image;
    Tensor<double> padded_image;

    // Test

    image.set({1,4,4}, 1);

    convolutional_layer.set_row_stride(1);
    convolutional_layer.set_column_stride(1);
    convolutional_layer.set_padding_option(OpenNN::ConvolutionalLayer::Same);
    convolutional_layer.set({1,4,4}, {1,3,3});

    padded_image = convolutional_layer.insert_padding(image);

    assert_true(padded_image(0,0,0) == 0.0 &&
                padded_image(0,0,1) == 0.0 &&
                padded_image(0,0,2) == 0.0 &&
                padded_image(0,0,3) == 0.0 &&
                padded_image(0,0,4) == 0.0 &&
                padded_image(0,0,5) == 0.0 &&
                padded_image(0,1,0) == 0.0 &&
                padded_image(0,1,1) == 1.0 &&
                padded_image(0,1,2) == 1.0 &&
                padded_image(0,1,3) == 1.0 &&
                padded_image(0,1,4) == 1.0 &&
                padded_image(0,1,5) == 0.0 &&
                padded_image(0,2,0) == 0.0 &&
                padded_image(0,2,1) == 1.0 &&
                padded_image(0,2,2) == 1.0 &&
                padded_image(0,2,3) == 1.0 &&
                padded_image(0,2,4) == 1.0 &&
                padded_image(0,2,5) == 0.0 &&
                padded_image(0,3,0) == 0.0 &&
                padded_image(0,3,1) == 1.0 &&
                padded_image(0,3,2) == 1.0 &&
                padded_image(0,3,3) == 1.0 &&
                padded_image(0,3,4) == 1.0 &&
                padded_image(0,3,5) == 0.0 &&
                padded_image(0,4,0) == 0.0 &&
                padded_image(0,4,1) == 1.0 &&
                padded_image(0,4,2) == 1.0 &&
                padded_image(0,4,3) == 1.0 &&
                padded_image(0,4,4) == 1.0 &&
                padded_image(0,4,5) == 0.0 &&
                padded_image(0,5,0) == 0.0 &&
                padded_image(0,5,1) == 0.0 &&
                padded_image(0,5,2) == 0.0 &&
                padded_image(0,5,3) == 0.0 &&
                padded_image(0,5,4) == 0.0 &&
                padded_image(0,5,5) == 0.0, LOG);
}


void ConvolutionalLayerTest::run_test_case()
{
   cout << "Running convolutional layer test case...\n";

   // Constructor and destructor

   test_constructor();
   test_destructor();

   // Get methods

   test_get_parameters();
   test_get_outputs_dimensions();
   test_get_parameters_number();

   // Set methods

   test_set();
   test_set_parameters();

   // Combinations

   test_calculate_image_convolution();
   test_calculate_convolutions();

   // Activation

   test_calculate_activations();
   test_calculate_activations_derivatives();

   // Outputs

   test_calculate_outputs();
   test_insert_padding();

   cout << "End of convolutional layer test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
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
