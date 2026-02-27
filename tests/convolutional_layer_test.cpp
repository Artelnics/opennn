#include "pch.h"

#include "../opennn/convolutional_layer.h"
#include "../opennn/tensors.h"

using namespace opennn;

void set_layer_parameters_constant(Layer& layer, type value)
{
    const vector<TensorView*> parameter_views = layer.get_parameter_views();

    for (TensorView* view : parameter_views)
    {
        if (view && view->size() > 0)
        {
            TensorMap1 parameters_map(view->data, view->size());

            parameters_map.setConstant(value);
        }
    }
}


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
    EXPECT_EQ(convolutional_layer.get_activation_function(), parameters.activation_function);
    EXPECT_EQ(convolutional_layer.get_batch_normalization(), parameters.batch_normalization);
    EXPECT_EQ(convolutional_layer.get_convolution_type(), parameters.convolution_type);
}


TEST_P(ConvolutionalLayerTest, ForwardPropagate)
{
    ConvolutionalLayerConfig parameters = GetParam();

    Convolutional convolutional_layer(parameters.input_shape,
        parameters.kernel_shape,
        parameters.activation_function,
        parameters.stride_shape,
        parameters.convolution_type,
        parameters.batch_normalization,
        parameters.test_name);

    set_layer_parameters_constant(convolutional_layer, 0.5);

    const Index batch_size = 2;

    Tensor4 input_data(batch_size,
        parameters.input_shape[0],
        parameters.input_shape[1],
        parameters.input_shape[2]);
    input_data.setConstant(1.0);

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<ConvolutionalForwardPropagation>(batch_size, &convolutional_layer);

    TensorView input_view(input_data.data(),
        { batch_size,
         parameters.input_shape[0],
         parameters.input_shape[1],
         parameters.input_shape[2] });

    vector<TensorView> input_views = { input_view };

    convolutional_layer.forward_propagate(input_views, forward_propagation, true);

    const TensorView output_view = forward_propagation->get_outputs();
    const Shape expected_output_dims = convolutional_layer.get_output_shape();

    ASSERT_EQ(output_view.shape.size(), 4);
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

        TensorMap<const Tensor<const type, 4>> output_tensor(output_view.data,
            output_view.shape[0],
            output_view.shape[1],
            output_view.shape[2],
            output_view.shape[3]);

        for (Index i = 0; i < output_tensor.size(); ++i) {
            EXPECT_NEAR(output_tensor(i), expected_value, 1e-5);
        }
    }
}


TEST_P(ConvolutionalLayerTest, BackPropagate)
{
    ConvolutionalLayerConfig parameters = GetParam();

    Convolutional convolutional_layer(parameters.input_shape,
                                      parameters.kernel_shape,
                                      parameters.activation_function,
                                      parameters.stride_shape,
                                      parameters.convolution_type,
                                      parameters.batch_normalization,
                                      parameters.test_name);

    set_layer_parameters_constant(convolutional_layer, 0.5);

    const Index batch_size = 2;

    Tensor4 input_data(batch_size,
                       parameters.input_shape[0],
                       parameters.input_shape[1],
                       parameters.input_shape[2]);
    input_data.setConstant(1.0);

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<ConvolutionalForwardPropagation>(batch_size, &convolutional_layer);

    forward_propagation->initialize();

    TensorView input_view(input_data.data(), { batch_size, parameters.input_shape[0], parameters.input_shape[1], parameters.input_shape[2] });

    // Forward

    convolutional_layer.forward_propagate({ input_view }, forward_propagation, true);

    unique_ptr<LayerBackPropagation> back_propagation_base =
        make_unique<ConvolutionalBackPropagation>(batch_size, &convolutional_layer);

    back_propagation_base->initialize();

    ConvolutionalBackPropagation* back_propagation =
        static_cast<ConvolutionalBackPropagation*>(back_propagation_base.get());

    TensorView output_view = forward_propagation->get_outputs();

    ASSERT_EQ(output_view.shape.size(), 4);

    Tensor4 deltas(output_view.shape[0], output_view.shape[1], output_view.shape[2], output_view.shape[3]);
    deltas.setConstant(1.0);

    TensorView delta_view(deltas.data(), output_view.shape);

    // Backpropagate

    convolutional_layer.back_propagate({ input_view }, { delta_view }, forward_propagation, back_propagation_base);

    const TensorView bias_gradients = back_propagation->bias_gradients;
    EXPECT_EQ(bias_gradients.size(), convolutional_layer.get_kernels_number());

    const vector<TensorView>& input_deltas_vector = back_propagation->input_gradients;

    ASSERT_FALSE(input_deltas_vector.empty());

    const TensorView& input_deltas_view = input_deltas_vector[0];

    ASSERT_EQ(input_deltas_view.shape.size(), 4);

    EXPECT_EQ(input_deltas_view.shape[0], batch_size);
    EXPECT_EQ(input_deltas_view.shape[1], parameters.input_shape[0]);
    EXPECT_EQ(input_deltas_view.shape[2], parameters.input_shape[1]);
    EXPECT_EQ(input_deltas_view.shape[3], parameters.input_shape[2]);

    // @todo Calculate numeric gradients and compare with analytical gradients.
}
