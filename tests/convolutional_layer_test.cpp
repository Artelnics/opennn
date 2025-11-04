#include "pch.h"

#include "../opennn/convolutional_layer.h"
#include "../opennn/tensors.h"

using namespace opennn;


void set_layer_parameters_constant(Layer& layer, const type& value)
{
    const vector<ParameterView> parameter_views = layer.get_parameter_views();

    for (const auto& view : parameter_views)
    {
        TensorMap<Tensor<type, 1>> parameters_map(view.data, view.size);
        parameters_map.setConstant(value);
    }
}


struct ConvolutionalLayerConfig {
    dimensions input_dimensions;
    dimensions kernel_dimensions;
    dimensions stride_dimensions;
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

    Convolutional convolutional_layer(parameters.input_dimensions,
        parameters.kernel_dimensions,
        parameters.activation_function,
        parameters.stride_dimensions,
        parameters.convolution_type,
        parameters.batch_normalization,
        parameters.test_name);

    EXPECT_EQ(convolutional_layer.get_input_dimensions(), parameters.input_dimensions);
    EXPECT_EQ(convolutional_layer.get_kernel_height(), parameters.kernel_dimensions[0]);
    EXPECT_EQ(convolutional_layer.get_kernel_width(), parameters.kernel_dimensions[1]);
    EXPECT_EQ(convolutional_layer.get_kernel_channels(), parameters.kernel_dimensions[2]);
    EXPECT_EQ(convolutional_layer.get_kernels_number(), parameters.kernel_dimensions[3]);
    EXPECT_EQ(convolutional_layer.get_row_stride(), parameters.stride_dimensions[0]);
    EXPECT_EQ(convolutional_layer.get_column_stride(), parameters.stride_dimensions[1]);
    EXPECT_EQ(convolutional_layer.get_activation_function(), parameters.activation_function);
    EXPECT_EQ(convolutional_layer.get_batch_normalization(), parameters.batch_normalization);
    EXPECT_EQ(convolutional_layer.get_convolution_type(), parameters.convolution_type);
}


TEST_P(ConvolutionalLayerTest, ForwardPropagate)
{
    ConvolutionalLayerConfig parameters = GetParam();

    Convolutional convolutional_layer(parameters.input_dimensions,
        parameters.kernel_dimensions,
        parameters.activation_function,
        parameters.stride_dimensions,
        parameters.convolution_type,
        parameters.batch_normalization,
        parameters.test_name);

    set_layer_parameters_constant(convolutional_layer, 0.5);

    const Index batch_size = 2;

    Tensor<type, 4> input_data(batch_size,
        parameters.input_dimensions[0],
        parameters.input_dimensions[1],
        parameters.input_dimensions[2]);
    input_data.setConstant(1.0);

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<ConvolutionalForwardPropagation>(batch_size, &convolutional_layer);

    TensorView input_view(input_data.data(),
        { batch_size,
         parameters.input_dimensions[0],
         parameters.input_dimensions[1],
         parameters.input_dimensions[2] });

    vector<TensorView> input_views = { input_view };

    convolutional_layer.forward_propagate(input_views, forward_propagation, true);

    const TensorView output_view = forward_propagation->get_output_pair();
    const dimensions expected_output_dims = convolutional_layer.get_output_dimensions();

    ASSERT_EQ(output_view.dims.size(), 4);
    EXPECT_EQ(output_view.dims[0], batch_size);
    EXPECT_EQ(output_view.dims[1], expected_output_dims[0]);
    EXPECT_EQ(output_view.dims[2], expected_output_dims[1]);
    EXPECT_EQ(output_view.dims[3], expected_output_dims[2]);

    if (!parameters.batch_normalization && parameters.activation_function == "Linear")
    {
        const Index kernel_height = parameters.kernel_dimensions[0];
        const Index kernel_width = parameters.kernel_dimensions[1];
        const Index kernel_channels = parameters.kernel_dimensions[2];

        const type expected_value = (kernel_height * kernel_width * kernel_channels * 1.0 * 0.5) + 0.5;

        TensorMap<const Tensor<const type, 4>> output_tensor(output_view.data,
            output_view.dims[0],
            output_view.dims[1],
            output_view.dims[2],
            output_view.dims[3]);

        for (Index i = 0; i < output_tensor.size(); ++i) {
            EXPECT_NEAR(output_tensor(i), expected_value, 1e-5);
        }
    }
}


TEST_P(ConvolutionalLayerTest, BackPropagate)
{
    ConvolutionalLayerConfig parameters = GetParam();

    Convolutional convolutional_layer(parameters.input_dimensions,
        parameters.kernel_dimensions,
        parameters.activation_function,
        parameters.stride_dimensions,
        parameters.convolution_type,
        parameters.batch_normalization,
        parameters.test_name);

    set_layer_parameters_constant(convolutional_layer, 0.5);

    const Index batch_size = 2;

    Tensor<type, 4> input_data(batch_size,
        parameters.input_dimensions[0],
        parameters.input_dimensions[1],
        parameters.input_dimensions[2]);
    input_data.setConstant(1.0);

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<ConvolutionalForwardPropagation>(batch_size, &convolutional_layer);

    TensorView input_view(input_data.data(), { batch_size, parameters.input_dimensions[0], parameters.input_dimensions[1], parameters.input_dimensions[2] });
    convolutional_layer.forward_propagate({ input_view }, forward_propagation, true);

    unique_ptr<LayerBackPropagation> back_propagation_base =
        make_unique<ConvolutionalBackPropagation>(batch_size, &convolutional_layer);

    ConvolutionalBackPropagation* back_propagation =
        static_cast<ConvolutionalBackPropagation*>(back_propagation_base.get());

    TensorView output_view = forward_propagation->get_output_pair();

    ASSERT_EQ(output_view.dims.size(), 4);

    Tensor<type, 4> deltas(output_view.dims[0], output_view.dims[1], output_view.dims[2], output_view.dims[3]);
    deltas.setConstant(1.0);

    TensorView delta_view(deltas.data(), output_view.dims);

    convolutional_layer.back_propagate({ input_view }, { delta_view }, forward_propagation, back_propagation_base);

    const Tensor<type, 1>& bias_deltas = back_propagation->bias_deltas;
    EXPECT_EQ(bias_deltas.size(), convolutional_layer.get_kernels_number());

    const Tensor<type, 4>& input_derivatives = back_propagation->input_deltas;

    EXPECT_EQ(input_derivatives.dimension(0), batch_size);
    EXPECT_EQ(input_derivatives.dimension(1), input_data.dimension(1));
    EXPECT_EQ(input_derivatives.dimension(2), input_data.dimension(2));
    EXPECT_EQ(input_derivatives.dimension(3), input_data.dimension(3));

    // @todo Calculate numeric gradients and compare with analytical gradients.
}
