#include "pch.h"

#include "../opennn/convolutional_layer.h"
#include "../opennn/tensors.h"


Tensor<type, 4> generate_input_tensor_convolution(const Tensor<type, 2>& data,
    const vector<Index>& row_indices,
    const vector<Index>& column_indices,
    const dimensions& input_dimensions) {
    Tensor<type, 4> input_tensor(row_indices.size(),
                                 input_dimensions[0],
                                 input_dimensions[1],
                                 input_dimensions[2]);
    type* tensor_data = input_tensor.data();
    fill_tensor_data(data, row_indices, column_indices, tensor_data);
    return input_tensor;
}


struct ConvolutionalLayerConfig {
    dimensions input_dimensions;
    dimensions kernel_dimensions;
    dimensions stride_dimensions;
    ConvolutionalLayer::ActivationFunction activation_function;
    ConvolutionalLayer::ConvolutionType convolution_type;
    string test_name;
    Tensor<type, 4> input_data;
    Tensor<type, 4> expected_output;
};

class ConvolutionalLayerTest : public ::testing::TestWithParam<ConvolutionalLayerConfig> {};

INSTANTIATE_TEST_CASE_P(ConvolutionalLayerTests, ConvolutionalLayerTest, ::testing::Values(
    ConvolutionalLayerConfig{
        {4, 4, 1}, {3, 3, 1, 1}, {1, 1}, ConvolutionalLayer::ActivationFunction::Linear, ConvolutionalLayer::ConvolutionType::Valid, "ConvolutionLayer",
        ([] {
            Tensor<type, 2> data(1, 16);
            data.setValues({
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
            });
            return generate_input_tensor_convolution(data, {0}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {4, 4, 1});
        })(),
        ([] {
            Tensor<type, 4> expected_output(1, 2, 2, 1);
            expected_output.setValues({
                {{{55}, {91}},
                 {{64}, {100}}}
            });
            return expected_output;
        })()
    }
));


TEST_P(ConvolutionalLayerTest, Constructor) {
    ConvolutionalLayerConfig parameters = GetParam();

    ConvolutionalLayer conv_layer(parameters.input_dimensions,
                                  parameters.kernel_dimensions,
                                  parameters.activation_function,
                                  parameters.stride_dimensions,
                                  parameters.convolution_type,
                                  parameters.test_name);

    EXPECT_EQ(conv_layer.get_input_dimensions(), parameters.input_dimensions);
    EXPECT_EQ(conv_layer.get_kernel_height(), parameters.kernel_dimensions[0]);
    EXPECT_EQ(conv_layer.get_kernel_width(), parameters.kernel_dimensions[1]);
    EXPECT_EQ(conv_layer.get_kernels_number(), parameters.kernel_dimensions[3]);
    EXPECT_EQ(conv_layer.get_row_stride(), parameters.stride_dimensions[0]);
    EXPECT_EQ(conv_layer.get_column_stride(), parameters.stride_dimensions[1]);
    EXPECT_EQ(conv_layer.get_activation_function(), parameters.activation_function);
    EXPECT_EQ(conv_layer.get_convolution_type(), parameters.convolution_type);
}


TEST_P(ConvolutionalLayerTest, ForwardPropagate) {
    /*
    ConvolutionalLayerConfig parameters = GetParam();

    ConvolutionalLayer convolutional_layer(parameters.input_dimensions,
                                           parameters.kernel_dimensions,
                                           parameters.activation_function,
                                           parameters.stride_dimensions,
                                           parameters.convolution_type,
                                           parameters.test_name);

    const Index batch_samples_number = parameters.input_data.dimension(0);

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<ConvolutionalLayerForwardPropagation>(batch_samples_number, &convolutional_layer);

    pair<type*, dimensions> input_pair(parameters.input_data.data(),
                                       {batch_samples_number,
                                        parameters.input_dimensions[0],
                                        parameters.input_dimensions[1],
                                        parameters.input_dimensions[2]});

    convolutional_layer.set_parameters_constant(1.0);

    convolutional_layer.forward_propagate({input_pair}, forward_propagation, true);

    pair<type*, dimensions> output_pair = forward_propagation->get_outputs_pair();

    EXPECT_EQ(output_pair.second[0], batch_samples_number);
    EXPECT_EQ(output_pair.second[1], parameters.expected_output.dimension(1));
    EXPECT_EQ(output_pair.second[2], parameters.expected_output.dimension(2));
    EXPECT_EQ(output_pair.second[3], parameters.expected_output.dimension(3));

    TensorMap<Tensor<type, 4>> output_tensor(output_pair.first,
                                             batch_samples_number,
                                             parameters.expected_output.dimension(1),
                                             parameters.expected_output.dimension(2),
                                             parameters.expected_output.dimension(3));

    for (Index b = 0; b < batch_samples_number; ++b) {
        for (Index h = 0; h < parameters.expected_output.dimension(1); ++h) {
            for (Index w = 0; w < parameters.expected_output.dimension(2); ++w) {
                for (Index c = 0; c < parameters.expected_output.dimension(3); ++c) {
                    EXPECT_NEAR(output_tensor(b, h, w, c), parameters.expected_output(b, h, w, c), 1e-5)
                        << "Mismatch at batch=" << b << ", height=" << h
                        << ", width=" << w << ", channel=" << c;
                }
            }
        }
    }
    */
}


TEST_P(ConvolutionalLayerTest, BackPropagate) 
{
    /*
    ConvolutionalLayerConfig parameters = GetParam();

    ConvolutionalLayer convolutional_layer(parameters.input_dimensions,
                                           parameters.kernel_dimensions,
                                           parameters.activation_function,
                                           parameters.stride_dimensions,
                                           parameters.convolution_type,
                                           parameters.test_name);

    const Index batch_samples_number = parameters.input_data.dimension(0);

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<ConvolutionalLayerForwardPropagation>(batch_samples_number, &convolutional_layer);

    unique_ptr<LayerBackPropagation> back_propagation =
        make_unique<ConvolutionalLayerBackPropagation>(batch_samples_number, &convolutional_layer);

    pair<type*, dimensions> input_pair(parameters.input_data.data(),
                                       {batch_samples_number,
                                        parameters.input_dimensions[0],
                                        parameters.input_dimensions[1],
                                        parameters.input_dimensions[2]});
    
    convolutional_layer.set_parameters_constant(1.0);

    convolutional_layer.forward_propagate({input_pair}, forward_propagation, true);

    pair<type*, dimensions> output_pair = forward_propagation->get_outputs_pair();

    // Initialize deltas with ones (mock values for testing backpropagation)
    Tensor<type, 4> deltas(batch_samples_number,
                           parameters.expected_output.dimension(1),
                           parameters.expected_output.dimension(2),
                           parameters.expected_output.dimension(3));

    deltas.setConstant(1.0);

    pair<type*, dimensions> delta_pair(deltas.data(),
                                       {batch_samples_number,
                                        parameters.expected_output.dimension(1),
                                        parameters.expected_output.dimension(2),
                                        parameters.expected_output.dimension(3)});
    /*
    convolutional_layer.back_propagate({input_pair}, {delta_pair}, forward_propagation, back_propagation);

    vector<pair<type*, dimensions>> input_derivatives_pair = back_propagation->get_input_derivative_pairs();

    EXPECT_EQ(input_derivatives_pair[0].second[0], batch_samples_number);
    EXPECT_EQ(input_derivatives_pair[0].second[1], parameters.input_data.dimension(1));
    EXPECT_EQ(input_derivatives_pair[0].second[2], parameters.input_data.dimension(2));
    EXPECT_EQ(input_derivatives_pair[0].second[3], parameters.input_data.dimension(3));

    TensorMap<Tensor<type, 4>> input_derivatives(input_derivatives_pair[0].first,
                                                 batch_samples_number,
                                                 parameters.input_data.dimension(1),
                                                 parameters.input_data.dimension(2),
                                                 parameters.input_data.dimension(3));

    // Validate input derivatives (mock expected values for now)
    Tensor<type, 4> expected_input_derivatives(batch_samples_number,
                                               parameters.input_data.dimension(1),
                                               parameters.input_data.dimension(2),
                                               parameters.input_data.dimension(3));
    expected_input_derivatives.setConstant(1.0);  // Replace with actual expected derivatives logic if known.

    for (Index b = 0; b < batch_samples_number; ++b) {
        for (Index h = 0; h < parameters.input_data.dimension(1); ++h) {
            for (Index w = 0; w < parameters.input_data.dimension(2); ++w) {
                for (Index c = 0; c < parameters.input_data.dimension(3); ++c) {
                    EXPECT_NEAR(input_derivatives(b, h, w, c), expected_input_derivatives(b, h, w, c), 1e-5)
                        << "Mismatch in input derivatives at batch=" << b << ", height=" << h
                        << ", width=" << w << ", channel=" << c;
                }
            }
        }
    }

    // Validate bias derivatives
    const Tensor<type, 1>& bias_derivatives = static_cast<ConvolutionalLayerBackPropagation*>(back_propagation.get())->bias_derivatives;
    EXPECT_EQ(bias_derivatives.size(), convolutional_layer.get_kernels_number());
    //for (Index k = 0; k < convolutional_layer.get_kernels_number(); ++k)
    //{
    //    EXPECT_NEAR(bias_derivatives(k), 1.0 * batch_samples_number * parameters.expected_output.dimension(1) * parameters.expected_output.dimension(2), 1e-5)
    //        << "Mismatch in bias derivative for kernel=" << k;
    //}

    // Validate synaptic weight derivatives
    const Tensor<type, 4>& weight_derivatives = static_cast<ConvolutionalLayerBackPropagation*>(back_propagation.get())->synaptic_weight_derivatives;
    EXPECT_EQ(weight_derivatives.dimension(0), parameters.kernel_dimensions[3]);
    EXPECT_EQ(weight_derivatives.dimension(1), parameters.kernel_dimensions[0]);
    EXPECT_EQ(weight_derivatives.dimension(2), parameters.kernel_dimensions[1]);
    EXPECT_EQ(weight_derivatives.dimension(3), parameters.kernel_dimensions[2]);
*/
}