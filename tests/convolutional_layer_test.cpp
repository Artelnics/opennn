#include "pch.h"

#include "../opennn/convolutional_layer.h"
#include "../opennn/tensor_utilities.h"
#include "../opennn/neural_network.h"

using namespace opennn;


struct ConvolutionalLayerConfig {
    Shape input_shape;
    Shape kernel_shape;
    Shape stride_shape;
    string activation_function;
    string convolution_type;
    bool batch_normalization;
    string test_name;
};


class ConvolutionalLayerTest : public ::testing::TestWithParam<ConvolutionalLayerConfig> {};

INSTANTIATE_TEST_SUITE_P(ConvolutionalLayerTests, ConvolutionalLayerTest, ::testing::Values(

                                                                              ConvolutionalLayerConfig{
                                                                                  {28, 28, 1},
                                                                                  {3, 3, 1, 16},
                                                                                  {1, 1},
                                                                                  "Linear",
                                                                                  "Valid",
                                                                                  false,
                                                                                  "ValidPaddingWithoutBN"
                                                                              },

                                                                              ConvolutionalLayerConfig{
                                                                                  {32, 32, 3},
                                                                                  {5, 5, 3, 32},
                                                                                  {1, 1},
                                                                                  "RectifiedLinear",
                                                                                  "Same",
                                                                                  false,                 // Batch Normalization
                                                                                  "SamePaddingWithoutBN"
                                                                              }
                                                                              ));


TEST_P(ConvolutionalLayerTest, Constructor) {

    ConvolutionalLayerConfig parameters = GetParam();

    Convolutional convolutional_layer(parameters.input_shape,
                                      parameters.kernel_shape,
                                      parameters.activation_function,
                                      parameters.stride_shape,
                                      parameters.convolution_type,
                                      parameters.batch_normalization,
                                      parameters.test_name);

    EXPECT_EQ(convolutional_layer.get_input_shape(), parameters.input_shape);
    EXPECT_EQ(convolutional_layer.get_kernel_height(), parameters.kernel_shape[0]);
    EXPECT_EQ(convolutional_layer.get_kernel_width(), parameters.kernel_shape[1]);
    EXPECT_EQ(convolutional_layer.get_kernel_channels(), parameters.kernel_shape[2]);
    EXPECT_EQ(convolutional_layer.get_kernels_number(), parameters.kernel_shape[3]);
    EXPECT_EQ(convolutional_layer.get_row_stride(), parameters.stride_shape[0]);
    EXPECT_EQ(convolutional_layer.get_column_stride(), parameters.stride_shape[1]);
    EXPECT_EQ(convolutional_layer.get_activation_function(), string_to_activation(parameters.activation_function));
    EXPECT_EQ(convolutional_layer.get_batch_normalization(), parameters.batch_normalization);
    EXPECT_EQ(convolutional_layer.get_convolution_type(), string_to_convolution_type(parameters.convolution_type));
}


TEST_P(ConvolutionalLayerTest, ForwardPropagate)
{
    ConvolutionalLayerConfig parameters = GetParam();

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Convolutional>(
        parameters.input_shape,
        parameters.kernel_shape,
        parameters.activation_function,
        parameters.stride_shape,
        parameters.convolution_type,
        parameters.batch_normalization,
        parameters.test_name));
    neural_network.compile();

    // Set all parameters to 0.5
    VectorMap(neural_network.get_parameters_data(), neural_network.get_parameters_size()).setConstant(type(0.5));

    const Index batch_size = 2;

    Tensor4 input_data(batch_size,
                       parameters.input_shape[0],
                       parameters.input_shape[1],
                       parameters.input_shape[2]);
    input_data.setConstant(1.0);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(input_data.data(),
                          { batch_size,
                           parameters.input_shape[0],
                           parameters.input_shape[1],
                           parameters.input_shape[2] }) };

    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();
    const Shape expected_output_dims = neural_network.get_layer(0)->get_output_shape();

    ASSERT_EQ(output_view.shape.rank, 4);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], expected_output_dims[0]);
    EXPECT_EQ(output_view.shape[2], expected_output_dims[1]);
    EXPECT_EQ(output_view.shape[3], expected_output_dims[2]);

    if(!parameters.batch_normalization && parameters.activation_function == "Linear")
    {
        const Index kernel_height = parameters.kernel_shape[0];
        const Index kernel_width = parameters.kernel_shape[1];
        const Index kernel_channels = parameters.kernel_shape[2];

        const type expected_value = (kernel_height * kernel_width * kernel_channels * 1.0 * 0.5) + 0.5;

        const type* output_data = output_view.as<type>();
        for (Index i = 0; i < output_view.size(); ++i)
            EXPECT_NEAR(output_data[i], expected_value, 1e-5);
    }
}


TEST_P(ConvolutionalLayerTest, BackPropagate)
{
    ConvolutionalLayerConfig parameters = GetParam();

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Convolutional>(
        parameters.input_shape,
        parameters.kernel_shape,
        parameters.activation_function,
        parameters.stride_shape,
        parameters.convolution_type,
        parameters.batch_normalization,
        parameters.test_name));
    neural_network.compile();

    VectorMap(neural_network.get_parameters_data(), neural_network.get_parameters_size()).setConstant(type(0.5));

    const Index batch_size = 2;

    Tensor4 input_data(batch_size,
                       parameters.input_shape[0],
                       parameters.input_shape[1],
                       parameters.input_shape[2]);
    input_data.setConstant(1.0);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(input_data.data(),
                          { batch_size, parameters.input_shape[0], parameters.input_shape[1], parameters.input_shape[2] }) };

    neural_network.forward_propagate(input_views, forward_propagation, true);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 4);
    EXPECT_EQ(output_view.shape[0], batch_size);
}
