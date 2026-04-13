#include "pch.h"

#include "../opennn/tensor_utilities.h"
#include "../opennn/random_utilities.h"
#include "../opennn/normalization_layer_3d.h"
#include "../opennn/neural_network.h"

using namespace opennn;


struct Normalization3dLayerConfig {
    Index batch_size;
    Index sequence_length;
    Index embedding_dimension;
    string test_name;
};

class Normalization3dLayerTest : public ::testing::TestWithParam<Normalization3dLayerConfig> {};

INSTANTIATE_TEST_SUITE_P(Normalization3dTests, Normalization3dLayerTest, ::testing::Values(
                                                                             Normalization3dLayerConfig{ 2, 5, 8, "Small" },
                                                                             Normalization3dLayerConfig{ 4, 10, 16, "Medium" },
                                                                             Normalization3dLayerConfig{ 1, 32, 64, "SingleBatch" },
                                                                             Normalization3dLayerConfig{ 8, 4, 12, "OddDimensions" }
                                                                             ));


TEST(Normalization3dTest, DefaultConstructor)
{
    Normalization3d normalization_3d;

    EXPECT_EQ(normalization_3d.get_input_shape().size(), 2);
    EXPECT_EQ(normalization_3d.get_input_shape()[0], 0);
    EXPECT_EQ(normalization_3d.get_output_shape().size(), 2);
    EXPECT_EQ(normalization_3d.get_output_shape()[0], 0);
}


TEST(Normalization3dTest, GeneralConstructor)
{
    const Index sequence_length = 15;
    const Index embedding_dimension = 32;

    Normalization3d normalization_3d({sequence_length, embedding_dimension});

    EXPECT_EQ(normalization_3d.get_sequence_length(), sequence_length);
    EXPECT_EQ(normalization_3d.get_embedding_dimension(), embedding_dimension);
    EXPECT_EQ(normalization_3d.get_input_shape()[0], sequence_length);
    EXPECT_EQ(normalization_3d.get_input_shape()[1], embedding_dimension);
}


TEST_P(Normalization3dLayerTest, ForwardPropagate)
{
    Normalization3dLayerConfig parameters = GetParam();
    const Index batch_size = parameters.batch_size;
    const Index seq = parameters.sequence_length;
    const Index dim = parameters.embedding_dimension;

    Normalization3d norm_layer({seq, dim}, parameters.test_name);

    vector<TensorView*> param_views = norm_layer.get_parameter_views();
    VectorR layer_parameters(get_size(param_views));
    link(layer_parameters.data(), param_views);

    norm_layer.set_parameters_random();

    Tensor3 inputs_tensor(batch_size, seq, dim);
    for (Index i = 0; i < inputs_tensor.size(); ++i) {
        inputs_tensor.data()[i] = static_cast<type>(random_normal(0.0, 5.0));
    }

    unique_ptr<LayerForwardPropagation> forward_propagation_base =
        make_unique<Normalization3dForwardPropagation>(batch_size, &norm_layer);
    forward_propagation_base->initialize();

    Tensor1 workspace(get_size(forward_propagation_base->get_workspace_views()));
    link(workspace.data(), forward_propagation_base->get_workspace_views());

    forward_propagation_base->inputs = { TensorView(inputs_tensor.data(), {batch_size, seq, dim}) };

    norm_layer.forward_propagate(forward_propagation_base, false);
    TensorView output_view = forward_propagation_base->get_outputs();

    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], seq);
    EXPECT_EQ(output_view.shape[2], dim);

#ifdef OPENNN_CUDA

    vector<TensorViewCuda*> param_views_device = norm_layer.get_parameter_views_device();
    TensorCuda layer_parameters_device({ get_size(param_views_device) });
    link(layer_parameters_device.data, param_views_device);

    CHECK_CUDA(cudaMemcpy(layer_parameters_device.data, layer_parameters.data(), layer_parameters.size() * sizeof(type), cudaMemcpyHostToDevice));

    unique_ptr<LayerForwardPropagationCuda> forward_propagation_cuda_base =
        make_unique<Normalization3dForwardPropagationCuda>(batch_size, &norm_layer);
    forward_propagation_cuda_base->initialize();

    vector<TensorViewCuda*> workspace_views_device = forward_propagation_cuda_base->get_workspace_views();
    TensorCuda layer_workspace_device({get_size(workspace_views_device)});
    link(layer_workspace_device.data, workspace_views_device);

    TensorCuda inputs_device({batch_size, seq, dim});
    CHECK_CUDA(cudaMemcpy(inputs_device.data, inputs_tensor.data(), inputs_tensor.size() * sizeof(type), cudaMemcpyHostToDevice));
    forward_propagation_cuda_base->inputs = { inputs_device.view() };

    norm_layer.forward_propagate(forward_propagation_cuda_base, false);

    TensorViewCuda output_view_device = forward_propagation_cuda_base->outputs;
    EXPECT_EQ(output_view_device.size(), output_view.size());

    vector<type> host_output_from_gpu(output_view_device.size());
    CHECK_CUDA(cudaMemcpy(host_output_from_gpu.data(), output_view_device.data, output_view_device.size() * sizeof(type), cudaMemcpyDeviceToHost));

    for (Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_view.data[i], host_output_from_gpu[i], 1e-3);

#endif
}


TEST_P(Normalization3dLayerTest, BackPropagate)
{
    Normalization3dLayerConfig parameters = GetParam();
    const Index batch_size = parameters.batch_size;
    const Index seq = parameters.sequence_length;
    const Index dim = parameters.embedding_dimension;

    Normalization3d norm_layer({seq, dim}, parameters.test_name);

    vector<TensorView*> param_views = norm_layer.get_parameter_views();
    VectorR layer_parameters(get_size(param_views));
    link(layer_parameters.data(), param_views);
    norm_layer.set_parameters_random();

    Tensor3 inputs_tensor(batch_size, seq, dim);
    for (Index i = 0; i < inputs_tensor.size(); ++i) inputs_tensor.data()[i] = static_cast<type>(random_normal(0.0, 5.0));

    unique_ptr<LayerForwardPropagation> forward_propagation_base = make_unique<Normalization3dForwardPropagation>(batch_size, &norm_layer);
    forward_propagation_base->initialize();

    Tensor1 workspace_fw(get_size(forward_propagation_base->get_workspace_views()));
    link(workspace_fw.data(), forward_propagation_base->get_workspace_views());

    forward_propagation_base->inputs = { TensorView(inputs_tensor.data(), {batch_size, seq, dim}) };

    norm_layer.forward_propagate(forward_propagation_base, true);
    TensorView output_view = forward_propagation_base->get_outputs();

    Tensor1 deltas(output_view.size());
    for(Index i = 0; i < deltas.size(); ++i) deltas(i) = static_cast<type>(random_normal(0.0, 1.0));
    TensorView delta_view(deltas.data(), output_view.shape);

    unique_ptr<LayerBackPropagation> back_propagation_base = make_unique<Normalization3dBackPropagation>(batch_size, &norm_layer);
    back_propagation_base->initialize();

    vector<TensorView*> gradient_views = back_propagation_base->get_gradient_views();
    VectorR layer_gradients(get_size(gradient_views));
    link(layer_gradients.data(), gradient_views);

    vector<TensorView*> bp_workspace_views = back_propagation_base->get_workspace_views();
    VectorR bp_workspace(get_size(bp_workspace_views));
    if (bp_workspace.size() > 0)
        link(bp_workspace.data(), bp_workspace_views);

    back_propagation_base->output_gradients = { delta_view };

    norm_layer.back_propagate(forward_propagation_base, back_propagation_base);

    Normalization3dBackPropagation* bp_cpu = static_cast<Normalization3dBackPropagation*>(back_propagation_base.get());
    vector<TensorView> input_derivatives_pair = bp_cpu->get_input_gradients();

    EXPECT_EQ(input_derivatives_pair[0].shape[0], batch_size);
    EXPECT_EQ(input_derivatives_pair[0].shape[1], seq);
    EXPECT_EQ(input_derivatives_pair[0].shape[2], dim);

#ifdef OPENNN_CUDA

    vector<TensorViewCuda*> param_views_device = norm_layer.get_parameter_views_device();
    TensorCuda layer_parameters_device({ get_size(param_views_device) });
    link(layer_parameters_device.data, param_views_device);
    CHECK_CUDA(cudaMemcpy(layer_parameters_device.data, layer_parameters.data(), layer_parameters.size() * sizeof(type), cudaMemcpyHostToDevice));

    TensorCuda delta_device({ batch_size, seq, dim });
    CHECK_CUDA(cudaMemcpy(delta_device.data, deltas.data(), deltas.size() * sizeof(type), cudaMemcpyHostToDevice));

    unique_ptr<LayerForwardPropagationCuda> forward_propagation_cuda_base = make_unique<Normalization3dForwardPropagationCuda>(batch_size, &norm_layer);
    forward_propagation_cuda_base->initialize();

    vector<TensorViewCuda*> workspace_fw_views_device = forward_propagation_cuda_base->get_workspace_views();
    TensorCuda layer_workspace_fw_device({get_size(workspace_fw_views_device)});
    link(layer_workspace_fw_device.data, workspace_fw_views_device);

    TensorCuda inputs_device({batch_size, seq, dim});
    CHECK_CUDA(cudaMemcpy(inputs_device.data, inputs_tensor.data(), inputs_tensor.size() * sizeof(type), cudaMemcpyHostToDevice));
    forward_propagation_cuda_base->inputs = { inputs_device.view() };

    norm_layer.forward_propagate(forward_propagation_cuda_base, true);

    unique_ptr<LayerBackPropagationCuda> back_propagation_cuda_base = make_unique<Normalization3dBackPropagationCuda>(batch_size, &norm_layer);
    back_propagation_cuda_base->initialize();

    vector<TensorViewCuda*> gradient_views_device = back_propagation_cuda_base->get_gradient_views();
    TensorCuda layer_gradients_device({get_size(gradient_views_device)});
    link(layer_gradients_device.data, gradient_views_device);

    vector<TensorViewCuda*> bp_workspace_views_device = back_propagation_cuda_base->get_workspace_views();
    TensorCuda bp_workspace_device({get_size(bp_workspace_views_device)});
    if (bp_workspace_device.size() > 0)
        link(bp_workspace_device.data, bp_workspace_views_device);

    back_propagation_cuda_base->output_gradients = { delta_device.view() };

    norm_layer.back_propagate(forward_propagation_cuda_base, back_propagation_cuda_base);

    Normalization3dBackPropagationCuda* bp_gpu = static_cast<Normalization3dBackPropagationCuda*>(back_propagation_cuda_base.get());

    vector<type> host_input_grads(bp_gpu->input_gradients[0].size());
    CHECK_CUDA(cudaMemcpy(host_input_grads.data(), bp_gpu->input_gradients[0].data, host_input_grads.size() * sizeof(type), cudaMemcpyDeviceToHost));

    for (Index i = 0; i < input_derivatives_pair[0].size(); ++i)
        EXPECT_NEAR(input_derivatives_pair[0].data[i], host_input_grads[i], 1e-2);

    vector<type> host_gamma_grads(bp_gpu->gamma_gradients.size());
    CHECK_CUDA(cudaMemcpy(host_gamma_grads.data(), bp_gpu->gamma_gradients.data, host_gamma_grads.size() * sizeof(type), cudaMemcpyDeviceToHost));

    vector<type> host_beta_grads(bp_gpu->beta_gradients.size());
    CHECK_CUDA(cudaMemcpy(host_beta_grads.data(), bp_gpu->beta_gradients.data, host_beta_grads.size() * sizeof(type), cudaMemcpyDeviceToHost));

    for (Index i = 0; i < dim; ++i)
    {
        EXPECT_NEAR(bp_cpu->gamma_gradients.data[i], host_gamma_grads[i], 1e-2);
        EXPECT_NEAR(bp_cpu->beta_gradients.data[i], host_beta_grads[i], 1e-2);
    }

#endif

}
