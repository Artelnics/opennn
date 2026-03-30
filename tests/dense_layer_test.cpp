#include "pch.h"
#include "../opennn/tensor_utilities.h"
#include "../opennn/layer.h"
#include "../opennn/dense_layer.h"
#include "../opennn/neural_network.h"

using namespace opennn;


TEST(Dense2dTest, DefaultConstructor)
{
    opennn::Dense<2> dense_layer;

    EXPECT_EQ(dense_layer.get_name(), "Dense2d");
    EXPECT_EQ(dense_layer.get_input_shape().size(), 1);
    EXPECT_EQ(dense_layer.get_output_shape().size(), 1);
}


TEST(Dense2dTest, GeneralConstructor)
{
    opennn::Dense<2> dense_layer({10}, {3}, "Linear");

    EXPECT_EQ(dense_layer.get_activation_function(), "Linear");

    ASSERT_EQ(dense_layer.get_input_shape().size(), 1);
    EXPECT_EQ(dense_layer.get_input_shape()[0], 10);

    ASSERT_EQ(dense_layer.get_output_shape().size(), 1);
    EXPECT_EQ(dense_layer.get_output_shape()[0], 3);

    EXPECT_EQ(dense_layer.get_parameters_number(), 33);
}


TEST(Dense2dTest, ForwardPropagate)
{
    const Index batch_size = 2;
    const Index inputs_number = 3;
    const Index outputs_number = 4;
    const bool is_training = true;

    opennn::Dense<2> dense_layer({inputs_number}, {outputs_number}, "Linear");
    vector<TensorView*> param_views = dense_layer.get_parameter_views();
    VectorR layer_parameters(get_size(param_views));
    layer_parameters.setZero();
    link(layer_parameters.data(), param_views);
    dense_layer.set_parameters_random();

    unique_ptr<LayerForwardPropagation> forward_propagation = make_unique<DenseForwardPropagation<2>>(batch_size, &dense_layer);
    forward_propagation->initialize();

    Tensor1 workspace_fw(get_size(forward_propagation->get_workspace_views()));
    workspace_fw.setZero();
    link(workspace_fw.data(), forward_propagation->get_workspace_views());

    MatrixR input_data(batch_size, inputs_number);
    input_data.setConstant(type(1.0));

    forward_propagation->inputs = { TensorView(input_data.data(), {batch_size, inputs_number}) };

    ASSERT_NO_THROW(dense_layer.forward_propagate(forward_propagation, is_training));

    EXPECT_EQ(dense_layer.get_name(), "Dense2d");

    const TensorView output_view = forward_propagation->get_outputs();

    ASSERT_EQ(output_view.shape.size(), 2);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], outputs_number);

#ifdef CUDA
    vector<TensorView*> param_views_device = dense_layer.get_parameter_views_device();
    TensorCuda layer_parameters_device({get_size(param_views_device)});
    link(layer_parameters_device.data, param_views_device);
    CHECK_CUDA(cudaMemcpy(layer_parameters_device.data, layer_parameters.data(), layer_parameters.size() * sizeof(type), cudaMemcpyHostToDevice));

    unique_ptr<LayerForwardPropagationCuda> forward_propagation_cuda = make_unique<DenseForwardPropagationCuda<2>>(batch_size, &dense_layer);
    forward_propagation_cuda->initialize();

    vector<TensorView*> workspace_fw_views_device = forward_propagation_cuda->get_workspace_views();
    TensorCuda layer_workspace_fw_device({get_size(workspace_fw_views_device)});
    link(layer_workspace_fw_device.data, workspace_fw_views_device);

    TensorCuda input_device({batch_size, inputs_number});
    CHECK_CUDA(cudaMemcpy(input_device.data, input_data.data(), input_data.size() * sizeof(type), cudaMemcpyHostToDevice));

    forward_propagation_cuda->inputs = { input_device.view() };

    dense_layer.forward_propagate(forward_propagation_cuda, is_training);

    vector<type> host_outputs(forward_propagation_cuda->outputs.size());
    CHECK_CUDA(cudaMemcpy(host_outputs.data(), forward_propagation_cuda->outputs.data, forward_propagation_cuda->outputs.size() * sizeof(type), cudaMemcpyDeviceToHost));

    for (Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_view.data[i], host_outputs[i], 1e-4);
#endif
}


TEST(Dense2dTest, BackPropagate)
{
    const Index batch_size = 2;
    const Index inputs_number = 3;
    const Index outputs_number = 4;

    opennn::Dense<2> dense_layer({inputs_number}, {outputs_number}, "Linear");
    vector<TensorView*> param_views = dense_layer.get_parameter_views();
    VectorR layer_parameters(get_size(param_views));
    layer_parameters.setZero();
    link(layer_parameters.data(), param_views);
    dense_layer.set_parameters_random();

    MatrixR input_data(batch_size, inputs_number);
    input_data.setConstant(type(1.0));

    unique_ptr<LayerForwardPropagation> forward_propagation = make_unique<DenseForwardPropagation<2>>(batch_size, &dense_layer);
    forward_propagation->initialize();

    Tensor1 workspace_fw(get_size(forward_propagation->get_workspace_views()));
    workspace_fw.setZero();
    link(workspace_fw.data(), forward_propagation->get_workspace_views());

    forward_propagation->inputs = { TensorView(input_data.data(), {batch_size, inputs_number}) };
    dense_layer.forward_propagate(forward_propagation, true);
/*
    unique_ptr<LayerBackPropagation> back_propagation = make_unique<DenseBackPropagation<2>>(batch_size, &dense_layer);
    back_propagation->initialize();

    vector<TensorView*> gradient_views = back_propagation->get_gradient_views();
    VectorR layer_gradients(get_size(gradient_views));
    layer_gradients.setZero();
    link(layer_gradients.data(), gradient_views);

    vector<TensorView*> bp_workspace_views = back_propagation->get_workspace_views();
    VectorR bp_workspace(get_size(bp_workspace_views));
    bp_workspace.setZero();
    if (bp_workspace.size() > 0)
        link(bp_workspace.data(), bp_workspace_views);

    MatrixR deltas(batch_size, outputs_number);
    deltas.setConstant(type(1.0));

    back_propagation->output_gradients = { TensorView(deltas.data(), {batch_size, outputs_number}) };

    dense_layer.back_propagate(forward_propagation, back_propagation);
*/
#ifdef CUDA
    vector<TensorView*> param_views_device = dense_layer.get_parameter_views_device();
    TensorCuda layer_parameters_device({get_size(param_views_device)});
    link(layer_parameters_device.data, param_views_device);
    CHECK_CUDA(cudaMemcpy(layer_parameters_device.data, layer_parameters.data(), layer_parameters.size() * sizeof(type), cudaMemcpyHostToDevice));

    unique_ptr<LayerForwardPropagationCuda> forward_propagation_cuda = make_unique<DenseForwardPropagationCuda<2>>(batch_size, &dense_layer);
    forward_propagation_cuda->initialize();

    vector<TensorView*> workspace_fw_views_device = forward_propagation_cuda->get_workspace_views();
    TensorCuda layer_workspace_fw_device({get_size(workspace_fw_views_device)});
    link(layer_workspace_fw_device.data, workspace_fw_views_device);

    TensorCuda input_device({batch_size, inputs_number});
    CHECK_CUDA(cudaMemcpy(input_device.data, input_data.data(), input_data.size() * sizeof(type), cudaMemcpyHostToDevice));
    forward_propagation_cuda->inputs = { input_device.view() };

    dense_layer.forward_propagate(forward_propagation_cuda, true);

    unique_ptr<LayerBackPropagationCuda> back_propagation_cuda = make_unique<DenseBackPropagationCuda<2>>(batch_size, &dense_layer);
    back_propagation_cuda->initialize();

    vector<TensorView*> gradient_views_device = back_propagation_cuda->get_gradient_views();
    TensorCuda layer_gradients_device({get_size(gradient_views_device)});
    link(layer_gradients_device.data, gradient_views_device);

    vector<TensorView*> bp_workspace_views_device = back_propagation_cuda->get_workspace_views();
    TensorCuda bp_workspace_device({get_size(bp_workspace_views_device)});
    if (bp_workspace_device.size() > 0)
        link(bp_workspace_device.data, bp_workspace_views_device);

    TensorCuda delta_device({batch_size, outputs_number});
    CHECK_CUDA(cudaMemcpy(delta_device.data, deltas.data(), deltas.size() * sizeof(type), cudaMemcpyHostToDevice));

    back_propagation_cuda->output_gradients = { delta_device.view() };

    dense_layer.back_propagate(forward_propagation_cuda, back_propagation_cuda);

    DenseBackPropagation<2>* bp_cpu = static_cast<DenseBackPropagation<2>*>(back_propagation.get());
    DenseBackPropagationCuda<2>* bp_gpu = static_cast<DenseBackPropagationCuda<2>*>(back_propagation_cuda.get());

    vector<type> host_bias_grads(bp_gpu->bias_gradients.size());
    CHECK_CUDA(cudaMemcpy(host_bias_grads.data(), bp_gpu->bias_gradients.data, host_bias_grads.size() * sizeof(type), cudaMemcpyDeviceToHost));
    for (Index i = 0; i < bp_cpu->bias_gradients.size(); ++i)
        EXPECT_NEAR(bp_cpu->bias_gradients.data[i], host_bias_grads[i], 1e-4);

    vector<type> host_weight_grads(bp_gpu->weight_gradients.size());
    CHECK_CUDA(cudaMemcpy(host_weight_grads.data(), bp_gpu->weight_gradients.data, host_weight_grads.size() * sizeof(type), cudaMemcpyDeviceToHost));
    for (Index i = 0; i < bp_cpu->weight_gradients.size(); ++i)
        EXPECT_NEAR(bp_cpu->weight_gradients.data[i], host_weight_grads[i], 1e-4);

    vector<type> host_input_grads(bp_gpu->input_gradients[0].size());
    CHECK_CUDA(cudaMemcpy(host_input_grads.data(), bp_gpu->input_gradients[0].data, host_input_grads.size() * sizeof(type), cudaMemcpyDeviceToHost));
    for (Index i = 0; i < bp_cpu->input_gradients[0].size(); ++i)
        EXPECT_NEAR(bp_cpu->input_gradients[0].data[i], host_input_grads[i], 1e-4);
#endif
}


TEST(Dense3dTest, DefaultConstructor)
{
    opennn::Dense<3> dense_layer({1, 1}, {1});

    EXPECT_EQ(dense_layer.get_name(), "Dense3d");
    ASSERT_EQ(dense_layer.get_input_shape().size(), 2);
    EXPECT_EQ(dense_layer.get_input_shape()[0], 1);
    EXPECT_EQ(dense_layer.get_input_shape()[1], 1);
    ASSERT_EQ(dense_layer.get_output_shape().size(), 2);
    EXPECT_EQ(dense_layer.get_output_shape()[0], 1);
    EXPECT_EQ(dense_layer.get_output_shape()[1], 1);
}


TEST(Dense3dTest, GeneralConstructor)
{
    const Index sequence_length = 5;
    const Index input_embedding = 4;
    const Index output_embedding = 3;

    opennn::Dense<3> dense_layer({sequence_length, input_embedding}, {output_embedding}, "Linear");

    EXPECT_EQ(dense_layer.get_activation_function(), "Linear");

    ASSERT_EQ(dense_layer.get_input_shape().size(), 2);
    EXPECT_EQ(dense_layer.get_input_shape()[0], sequence_length);
    EXPECT_EQ(dense_layer.get_input_shape()[1], input_embedding);

    ASSERT_EQ(dense_layer.get_output_shape().size(), 2);
    EXPECT_EQ(dense_layer.get_output_shape()[0], sequence_length);
    EXPECT_EQ(dense_layer.get_output_shape()[1], output_embedding);

    EXPECT_EQ(dense_layer.get_parameters_number(), input_embedding * output_embedding + output_embedding);
}


TEST(Dense3dTest, ForwardPropagate)
{
    const Index batch_size = 2;
    const Index sequence_length = 3;
    const Index input_embedding = 4;
    const Index output_embedding = 5;
    const bool is_training = true;

    opennn::Dense<3> dense_layer({sequence_length, input_embedding}, {output_embedding}, "Linear");
    vector<TensorView*> param_views = dense_layer.get_parameter_views();
    VectorR layer_parameters(get_size(param_views));
    layer_parameters.setZero();
    link(layer_parameters.data(), param_views);
    dense_layer.set_parameters_random();

    unique_ptr<LayerForwardPropagation> forward_propagation = make_unique<DenseForwardPropagation<3>>(batch_size, &dense_layer);
    forward_propagation->initialize();

    Tensor1 workspace_fw(get_size(forward_propagation->get_workspace_views()));
    workspace_fw.setZero();
    link(workspace_fw.data(), forward_propagation->get_workspace_views());

    Tensor3 input_data(batch_size, sequence_length, input_embedding);
    input_data.setConstant(type(0.5));

    forward_propagation->inputs = { TensorView(input_data.data(), {batch_size, sequence_length, input_embedding}) };

    ASSERT_NO_THROW(dense_layer.forward_propagate(forward_propagation, is_training));

    const TensorView output_view = forward_propagation->get_outputs();

    ASSERT_EQ(output_view.shape.size(), 3);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], sequence_length);
    EXPECT_EQ(output_view.shape[2], output_embedding);

#ifdef CUDA
    vector<TensorView*> param_views_device = dense_layer.get_parameter_views_device();
    TensorCuda layer_parameters_device({get_size(param_views_device)});
    link(layer_parameters_device.data, param_views_device);
    CHECK_CUDA(cudaMemcpy(layer_parameters_device.data, layer_parameters.data(), layer_parameters.size() * sizeof(type), cudaMemcpyHostToDevice));

    unique_ptr<LayerForwardPropagationCuda> forward_propagation_cuda = make_unique<DenseForwardPropagationCuda<3>>(batch_size, &dense_layer);
    forward_propagation_cuda->initialize();

    vector<TensorView*> workspace_fw_views_device = forward_propagation_cuda->get_workspace_views();
    TensorCuda layer_workspace_fw_device({get_size(workspace_fw_views_device)});
    link(layer_workspace_fw_device.data, workspace_fw_views_device);

    TensorCuda input_device({batch_size, sequence_length, input_embedding});
    CHECK_CUDA(cudaMemcpy(input_device.data, input_data.data(), input_data.size() * sizeof(type), cudaMemcpyHostToDevice));

    forward_propagation_cuda->inputs = { input_device.view() };

    dense_layer.forward_propagate(forward_propagation_cuda, is_training);

    vector<type> host_outputs(forward_propagation_cuda->outputs.size());
    CHECK_CUDA(cudaMemcpy(host_outputs.data(), forward_propagation_cuda->outputs.data, forward_propagation_cuda->outputs.size() * sizeof(type), cudaMemcpyDeviceToHost));

    for (Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_view.data[i], host_outputs[i], 1e-4);
#endif
}


TEST(Dense3dTest, BackPropagate)
{
    const Index batch_size = 2;
    const Index sequence_length = 3;
    const Index input_embedding = 4;
    const Index output_embedding = 5;

    opennn::Dense<3> dense_layer({sequence_length, input_embedding}, {output_embedding}, "Linear");
    vector<TensorView*> param_views = dense_layer.get_parameter_views();
    VectorR layer_parameters(get_size(param_views));
    layer_parameters.setZero();
    link(layer_parameters.data(), param_views);
    dense_layer.set_parameters_random();

    Tensor3 input_data(batch_size, sequence_length, input_embedding);
    input_data.setConstant(type(0.5));

    unique_ptr<LayerForwardPropagation> forward_propagation = make_unique<DenseForwardPropagation<3>>(batch_size, &dense_layer);
    forward_propagation->initialize();

    Tensor1 workspace_fw(get_size(forward_propagation->get_workspace_views()));
    workspace_fw.setZero();
    link(workspace_fw.data(), forward_propagation->get_workspace_views());

    forward_propagation->inputs = { TensorView(input_data.data(), {batch_size, sequence_length, input_embedding}) };
    dense_layer.forward_propagate(forward_propagation, true);

    unique_ptr<LayerBackPropagation> back_propagation = make_unique<DenseBackPropagation<3>>(batch_size, &dense_layer);
    back_propagation->initialize();

    vector<TensorView*> gradient_views = back_propagation->get_gradient_views();
    VectorR layer_gradients(get_size(gradient_views));
    layer_gradients.setZero();
    link(layer_gradients.data(), gradient_views);

    vector<TensorView*> bp_workspace_views = back_propagation->get_workspace_views();
    VectorR bp_workspace(get_size(bp_workspace_views));
    bp_workspace.setZero();
    if (bp_workspace.size() > 0)
        link(bp_workspace.data(), bp_workspace_views);

    Tensor3 deltas(batch_size, sequence_length, output_embedding);
    deltas.setConstant(type(1.0));

    back_propagation->output_gradients = { TensorView(deltas.data(), {batch_size, sequence_length, output_embedding}) };

    dense_layer.back_propagate(forward_propagation, back_propagation);

    DenseBackPropagation<3>* bp_cpu = static_cast<DenseBackPropagation<3>*>(back_propagation.get());

    ASSERT_EQ(bp_cpu->bias_gradients.shape.size(), 1);
    EXPECT_EQ(bp_cpu->bias_gradients.shape[0], output_embedding);
    ASSERT_EQ(bp_cpu->weight_gradients.shape.size(), 2);
    EXPECT_EQ(bp_cpu->weight_gradients.shape[0], input_embedding);
    EXPECT_EQ(bp_cpu->weight_gradients.shape[1], output_embedding);
    ASSERT_EQ(bp_cpu->input_gradients[0].shape.size(), 3);
    EXPECT_EQ(bp_cpu->input_gradients[0].shape[0], batch_size);
    EXPECT_EQ(bp_cpu->input_gradients[0].shape[1], sequence_length);
    EXPECT_EQ(bp_cpu->input_gradients[0].shape[2], input_embedding);

#ifdef CUDA
    vector<TensorView*> param_views_device = dense_layer.get_parameter_views_device();
    TensorCuda layer_parameters_device({get_size(param_views_device)});
    link(layer_parameters_device.data, param_views_device);
    CHECK_CUDA(cudaMemcpy(layer_parameters_device.data, layer_parameters.data(), layer_parameters.size() * sizeof(type), cudaMemcpyHostToDevice));

    unique_ptr<LayerForwardPropagationCuda> forward_propagation_cuda = make_unique<DenseForwardPropagationCuda<3>>(batch_size, &dense_layer);
    forward_propagation_cuda->initialize();

    vector<TensorView*> workspace_fw_views_device = forward_propagation_cuda->get_workspace_views();
    TensorCuda layer_workspace_fw_device({get_size(workspace_fw_views_device)});
    link(layer_workspace_fw_device.data, workspace_fw_views_device);

    TensorCuda input_device({batch_size, sequence_length, input_embedding});
    CHECK_CUDA(cudaMemcpy(input_device.data, input_data.data(), input_data.size() * sizeof(type), cudaMemcpyHostToDevice));
    forward_propagation_cuda->inputs = { input_device.view() };

    dense_layer.forward_propagate(forward_propagation_cuda, true);

    unique_ptr<LayerBackPropagationCuda> back_propagation_cuda = make_unique<DenseBackPropagationCuda<3>>(batch_size, &dense_layer);
    back_propagation_cuda->initialize();

    vector<TensorView*> gradient_views_device = back_propagation_cuda->get_gradient_views();
    TensorCuda layer_gradients_device({get_size(gradient_views_device)});
    link(layer_gradients_device.data, gradient_views_device);

    vector<TensorView*> bp_workspace_views_device = back_propagation_cuda->get_workspace_views();
    TensorCuda bp_workspace_device({get_size(bp_workspace_views_device)});
    if (bp_workspace_device.size() > 0)
        link(bp_workspace_device.data, bp_workspace_views_device);

    TensorCuda delta_device({batch_size, sequence_length, output_embedding});
    CHECK_CUDA(cudaMemcpy(delta_device.data, deltas.data(), deltas.size() * sizeof(type), cudaMemcpyHostToDevice));

    back_propagation_cuda->output_gradients = { delta_device.view() };

    dense_layer.back_propagate(forward_propagation_cuda, back_propagation_cuda);

    DenseBackPropagationCuda<3>* bp_gpu = static_cast<DenseBackPropagationCuda<3>*>(back_propagation_cuda.get());

    vector<type> host_bias_grads(bp_gpu->bias_gradients.size());
    CHECK_CUDA(cudaMemcpy(host_bias_grads.data(), bp_gpu->bias_gradients.data, host_bias_grads.size() * sizeof(type), cudaMemcpyDeviceToHost));
    for (Index i = 0; i < bp_cpu->bias_gradients.size(); ++i)
        EXPECT_NEAR(bp_cpu->bias_gradients.data[i], host_bias_grads[i], 1e-4);

    vector<type> host_weight_grads(bp_gpu->weight_gradients.size());
    CHECK_CUDA(cudaMemcpy(host_weight_grads.data(), bp_gpu->weight_gradients.data, host_weight_grads.size() * sizeof(type), cudaMemcpyDeviceToHost));
    for (Index i = 0; i < bp_cpu->weight_gradients.size(); ++i)
        EXPECT_NEAR(bp_cpu->weight_gradients.data[i], host_weight_grads[i], 1e-4);

    vector<type> host_input_grads(bp_gpu->input_gradients[0].size());
    CHECK_CUDA(cudaMemcpy(host_input_grads.data(), bp_gpu->input_gradients[0].data, host_input_grads.size() * sizeof(type), cudaMemcpyDeviceToHost));
    for (Index i = 0; i < bp_cpu->input_gradients[0].size(); ++i)
        EXPECT_NEAR(bp_cpu->input_gradients[0].data[i], host_input_grads[i], 1e-4);
#endif
}
