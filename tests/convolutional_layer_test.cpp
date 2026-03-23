#include "pch.h"

#include "../opennn/convolutional_layer.h"
#include "../opennn/tensor_utilities.h"

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

    vector<TensorView*> param_views = convolutional_layer.get_parameter_views();
    VectorR layer_parameters(get_size(param_views));
    link(layer_parameters.data(), param_views);

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

    vector<TensorView*> workspace_views = forward_propagation->get_workspace_views();
    VectorR layer_workspace(get_size(workspace_views));
    link(layer_workspace.data(), workspace_views);


    TensorView input_view(input_data.data(),
                          { batch_size,
                           parameters.input_shape[0],
                           parameters.input_shape[1],
                           parameters.input_shape[2] });

    forward_propagation->inputs = { input_view };

    convolutional_layer.forward_propagate(forward_propagation, true);

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

        for (Index i = 0; i < output_tensor.size(); ++i)
            EXPECT_NEAR(output_tensor(i), expected_value, 1e-5);
    }

#ifdef OPENNN_CUDA

    vector<TensorViewCuda*> param_views_device = convolutional_layer.get_parameter_views_device();
    TensorCuda layer_parameters_device({get_size(param_views_device)});
    link(layer_parameters_device.data, param_views_device);

    CHECK_CUDA(cudaMemcpy(layer_parameters_device.data, layer_parameters.data(), layer_parameters.size() * sizeof(type), cudaMemcpyHostToDevice));

    TensorCuda input_data_device({batch_size, parameters.input_shape[0], parameters.input_shape[1], parameters.input_shape[2]});
    CHECK_CUDA(cudaMemcpy(input_data_device.data, input_data.data(), input_data.size() * sizeof(type), cudaMemcpyHostToDevice));

    unique_ptr<LayerForwardPropagationCuda> forward_propagation_cuda =
        make_unique<ConvolutionalForwardPropagationCuda>(batch_size, &convolutional_layer);
    forward_propagation_cuda->initialize();

    vector<TensorViewCuda*> workspace_views_device = forward_propagation_cuda->get_workspace_views();
    TensorCuda layer_workspace_device({get_size(workspace_views_device)});
    link(layer_workspace_device.data, workspace_views_device);

    forward_propagation_cuda->inputs = { input_data_device.view() };
    convolutional_layer.forward_propagate(forward_propagation_cuda, true);

    // CPU vs GPU

    TensorViewCuda output_view_device = forward_propagation_cuda->outputs;
    EXPECT_EQ(output_view_device.size(), output_view.size());

    vector<type> host_output_from_gpu(output_view.size());
    CHECK_CUDA(cudaMemcpy(host_output_from_gpu.data(), output_view_device.data, output_view.size() * sizeof(type), cudaMemcpyDeviceToHost));

    for (Index i = 0; i < output_view.size(); ++i) {
        EXPECT_NEAR(output_view.data[i], host_output_from_gpu[i], 1e-4);
    }
#endif
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

    vector<TensorView*> param_views = convolutional_layer.get_parameter_views();
    VectorR layer_parameters(get_size(param_views));
    link(layer_parameters.data(), param_views);

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
    vector<TensorView*> workspace_views = forward_propagation->get_workspace_views();
    VectorR layer_workspace(get_size(workspace_views));
    link(layer_workspace.data(), workspace_views);

    TensorView input_view(input_data.data(), { batch_size, parameters.input_shape[0], parameters.input_shape[1], parameters.input_shape[2] });
    forward_propagation->inputs = { input_view };

    convolutional_layer.forward_propagate(forward_propagation, true);

    unique_ptr<LayerBackPropagation> back_propagation_base =
        make_unique<ConvolutionalBackPropagation>(batch_size, &convolutional_layer);

    back_propagation_base->initialize();
    vector<TensorView*> gradient_views = back_propagation_base->get_gradient_views();
    VectorR layer_gradients(get_size(gradient_views));
    link(layer_gradients.data(), gradient_views);

    ConvolutionalBackPropagation* back_propagation =
        static_cast<ConvolutionalBackPropagation*>(back_propagation_base.get());

    TensorView output_view = forward_propagation->get_outputs();

    Tensor4 deltas(output_view.shape[0], output_view.shape[1], output_view.shape[2], output_view.shape[3]);
    deltas.setConstant(1.0);
    TensorView delta_view(deltas.data(), output_view.shape);

#ifdef OPENNN_CUDA

    vector<TensorViewCuda*> param_views_device = convolutional_layer.get_parameter_views_device();
    TensorCuda layer_parameters_device({get_size(param_views_device)});
    link(layer_parameters_device.data, param_views_device);
    CHECK_CUDA(cudaMemcpy(layer_parameters_device.data, layer_parameters.data(), layer_parameters.size() * sizeof(type), cudaMemcpyHostToDevice));

    TensorCuda input_data_device({batch_size, parameters.input_shape[0], parameters.input_shape[1], parameters.input_shape[2]});
    CHECK_CUDA(cudaMemcpy(input_data_device.data, input_data.data(), input_data.size() * sizeof(type), cudaMemcpyHostToDevice));

    unique_ptr<LayerForwardPropagationCuda> forward_propagation_cuda =
        make_unique<ConvolutionalForwardPropagationCuda>(batch_size, &convolutional_layer);
    forward_propagation_cuda->initialize();
    vector<TensorViewCuda*> workspace_views_device = forward_propagation_cuda->get_workspace_views();
    TensorCuda layer_workspace_device({get_size(workspace_views_device)});
    link(layer_workspace_device.data, workspace_views_device);

    forward_propagation_cuda->inputs = { input_data_device.view() };
    convolutional_layer.forward_propagate(forward_propagation_cuda, true);

    unique_ptr<LayerBackPropagationCuda> back_propagation_cuda_base =
        make_unique<ConvolutionalBackPropagationCuda>(batch_size, &convolutional_layer);
    back_propagation_cuda_base->initialize();
    vector<TensorViewCuda*> gradient_views_device = back_propagation_cuda_base->get_gradient_views();
    TensorCuda layer_gradients_device({get_size(gradient_views_device)});
    link(layer_gradients_device.data, gradient_views_device);

    ConvolutionalBackPropagationCuda* back_propagation_cuda =
        static_cast<ConvolutionalBackPropagationCuda*>(back_propagation_cuda_base.get());

    TensorCuda delta_device({output_view.shape[0], output_view.shape[1], output_view.shape[2], output_view.shape[3]});
    CHECK_CUDA(cudaMemcpy(delta_device.data, deltas.data(), deltas.size() * sizeof(type), cudaMemcpyHostToDevice));
    vector<TensorViewCuda> delta_views_device = { delta_device.view() };
#endif

    convolutional_layer.back_propagate(forward_propagation, back_propagation_base);

#ifdef OPENNN_CUDA

    convolutional_layer.back_propagate(forward_propagation_cuda, back_propagation_cuda_base);

    // CPU vs GPU
    const TensorView bias_gradients = back_propagation->bias_gradients;
    const TensorView weight_gradients = back_propagation->weight_gradients;
    const TensorView& input_deltas_view = back_propagation->input_gradients[0];

    vector<type> host_bias_grads(back_propagation_cuda->bias_gradients.size());
    CHECK_CUDA(cudaMemcpy(host_bias_grads.data(), back_propagation_cuda->bias_gradients.data, back_propagation_cuda->bias_gradients.size() * sizeof(type), cudaMemcpyDeviceToHost));
    for (Index i = 0; i < bias_gradients.size(); ++i) {
        EXPECT_NEAR(bias_gradients.data[i], host_bias_grads[i], 1e-4);
    }

    vector<type> host_weight_grads(back_propagation_cuda->weight_gradients.size());
    CHECK_CUDA(cudaMemcpy(host_weight_grads.data(), back_propagation_cuda->weight_gradients.data, back_propagation_cuda->weight_gradients.size() * sizeof(type), cudaMemcpyDeviceToHost));
    for (Index i = 0; i < weight_gradients.size(); ++i) {
        EXPECT_NEAR(weight_gradients.data[i], host_weight_grads[i], 1e-3);
    }

    vector<type> host_input_grads(back_propagation_cuda->input_gradients[0].size());
    CHECK_CUDA(cudaMemcpy(host_input_grads.data(), back_propagation_cuda->input_gradients[0].data, back_propagation_cuda->input_gradients[0].size() * sizeof(type), cudaMemcpyDeviceToHost));
    for (Index i = 0; i < input_deltas_view.size(); ++i) {
        EXPECT_NEAR(input_deltas_view.data[i], host_input_grads[i], 1e-4);
    }
#endif
}
