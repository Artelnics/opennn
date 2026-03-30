#include "pch.h"

#include "../opennn/tensor_utilities.h"
#include "../opennn/embedding_layer.h"
#include "../opennn/random_utilities.h"
#include <iostream>


using namespace opennn;


struct EmbeddingLayerConfig {
    Index batch_size;
    Index vocabulary_size;
    Index sequence_length;
    Index embedding_dimension;
    bool scale_embedding;
    bool add_positional_encoding;
    string test_name;
};

class EmbeddingLayerTest : public ::testing::TestWithParam<EmbeddingLayerConfig> {};

INSTANTIATE_TEST_SUITE_P(EmbeddingLayerTests, EmbeddingLayerTest, ::testing::Values(
                                                                      EmbeddingLayerConfig{ 2, 10, 5, 8, false, false, "Basic" },
                                                                      EmbeddingLayerConfig{ 3, 20, 7, 16, true, false, "Scaled" },
                                                                      EmbeddingLayerConfig{ 4, 15, 6, 12, false, true, "PositionalEncoding" },
                                                                      EmbeddingLayerConfig{ 2, 12, 8, 32, true, true, "ScaledAndPositionalEncoding" }
                                                                      ));


TEST(Embedding, DefaultConstructor)
{
    Embedding embedding_layer;

    EXPECT_EQ(embedding_layer.get_vocabulary_size(), 0);
    EXPECT_EQ(embedding_layer.get_sequence_length(), 0);
    EXPECT_EQ(embedding_layer.get_embedding_dimension(), 0);
}


TEST(Embedding, GeneralConstructor)
{
    const Shape input_shape{ 15, 5 };
    const Index embedding_dimension = 8;

    Embedding embedding_layer(input_shape, embedding_dimension);

    EXPECT_EQ(embedding_layer.get_vocabulary_size(), input_shape[0]);
    EXPECT_EQ(embedding_layer.get_sequence_length(), input_shape[1]);
    EXPECT_EQ(embedding_layer.get_embedding_dimension(), embedding_dimension);
}


TEST(EmbeddingForwardPropagationTest, GetOutputPairReturnsCorrectDataAndShape)
{
    const Index batch_size = 2;
    const Index vocabulary_size = 15;
    const Index sequence_length = 5;
    const Index embedding_dimension = 6;

    Embedding layer({ vocabulary_size, sequence_length }, embedding_dimension, "test_embedding");

    EmbeddingForwardPropagation forward(batch_size, &layer);
    forward.initialize();

    VectorR fake_memory(forward.outputs.size());
    forward.outputs.data = fake_memory.data();

    const TensorView output_view = forward.get_outputs();

    EXPECT_EQ(output_view.data, forward.outputs.data);
    ASSERT_EQ(output_view.shape.size(), 3);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], sequence_length);
    EXPECT_EQ(output_view.shape[2], embedding_dimension);
}



TEST_P(EmbeddingLayerTest, ForwardPropagate)
{
    EmbeddingLayerConfig parameters = GetParam();

    Embedding embedding_layer({ parameters.vocabulary_size, parameters.sequence_length }, parameters.embedding_dimension);
    embedding_layer.set_scale_embedding(parameters.scale_embedding);
    embedding_layer.set_add_positional_encoding(parameters.add_positional_encoding);

    vector<TensorView*> param_views = embedding_layer.get_parameter_views();
    VectorR layer_parameters(get_size(param_views));
    link(layer_parameters.data(), param_views);
    embedding_layer.set_parameters_random();

    const Index batch_size = parameters.batch_size;

    MatrixR inputs_mat(batch_size, parameters.sequence_length);
    for (Index i = 0; i < batch_size; ++i)
        for (Index j = 0; j < parameters.sequence_length; ++j)
            inputs_mat(i, j) = static_cast<type>(random_integer(0, parameters.vocabulary_size - 1));

    unique_ptr<LayerForwardPropagation> forward_propagation_base =
        make_unique<EmbeddingForwardPropagation>(batch_size, &embedding_layer);
    forward_propagation_base->initialize();

    Tensor1 workspace(get_size(forward_propagation_base->get_workspace_views()));
    link(workspace.data(), forward_propagation_base->get_workspace_views());

    forward_propagation_base->inputs = { TensorView(inputs_mat.data(), {batch_size, parameters.sequence_length}) };

    embedding_layer.forward_propagate(forward_propagation_base, false);
    const TensorView output_view = forward_propagation_base->get_outputs();

#ifdef OPENNN_CUDA

    vector<TensorView*> param_views_device = embedding_layer.get_parameter_views_device();
    TensorCuda layer_parameters_device({ get_size(param_views_device) });
    link(layer_parameters_device.data, param_views_device);
    CHECK_CUDA(cudaMemcpy(layer_parameters_device.data, layer_parameters.data(), layer_parameters.size() * sizeof(type), cudaMemcpyHostToDevice));

    unique_ptr<LayerForwardPropagationCuda> forward_propagation_cuda_base =
        make_unique<EmbeddingForwardPropagationCuda>(batch_size, &embedding_layer);
    forward_propagation_cuda_base->initialize();

    vector<TensorView*> workspace_views_device = forward_propagation_cuda_base->get_workspace_views();
    TensorCuda layer_workspace_device({get_size(workspace_views_device)});
    link(layer_workspace_device.data, workspace_views_device);

    TensorCuda inputs_device({batch_size, parameters.sequence_length});
    CHECK_CUDA(cudaMemcpy(inputs_device.data, inputs_mat.data(), inputs_mat.size() * sizeof(type), cudaMemcpyHostToDevice));
    forward_propagation_cuda_base->inputs = { inputs_device.view() };

    embedding_layer.forward_propagate(forward_propagation_cuda_base, false);

    vector<type> host_output_from_gpu(forward_propagation_cuda_base->outputs.size());
    CHECK_CUDA(cudaMemcpy(host_output_from_gpu.data(), forward_propagation_cuda_base->outputs.data, forward_propagation_cuda_base->outputs.size() * sizeof(type), cudaMemcpyDeviceToHost));

    for (Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_view.data[i], host_output_from_gpu[i], 1e-4);

#endif
}


TEST_P(EmbeddingLayerTest, BackPropagate)
{
    EmbeddingLayerConfig parameters = GetParam();

    Embedding embedding_layer({ parameters.vocabulary_size, parameters.sequence_length }, parameters.embedding_dimension);
    embedding_layer.set_scale_embedding(parameters.scale_embedding);
    embedding_layer.set_add_positional_encoding(parameters.add_positional_encoding);

    vector<TensorView*> param_views = embedding_layer.get_parameter_views();
    VectorR layer_parameters(get_size(param_views));
    link(layer_parameters.data(), param_views);
    embedding_layer.set_parameters_random();

    const Index batch_size = parameters.batch_size;

    MatrixR inputs_mat(batch_size, parameters.sequence_length);
    for (Index i = 0; i < batch_size; ++i)
        for (Index j = 0; j < parameters.sequence_length; ++j)
            inputs_mat(i, j) = static_cast<type>(random_integer(0, parameters.vocabulary_size - 1));

    unique_ptr<LayerForwardPropagation> forward_propagation_base =
        make_unique<EmbeddingForwardPropagation>(batch_size, &embedding_layer);
    forward_propagation_base->initialize();

    Tensor1 workspace_fw(get_size(forward_propagation_base->get_workspace_views()));
    link(workspace_fw.data(), forward_propagation_base->get_workspace_views());

    forward_propagation_base->inputs = { TensorView(inputs_mat.data(), {batch_size, parameters.sequence_length}) };

    embedding_layer.forward_propagate(forward_propagation_base, true);
    TensorView output_view = forward_propagation_base->get_outputs();

    unique_ptr<LayerBackPropagation> back_propagation_base =
        make_unique<EmbeddingBackPropagation>(batch_size, &embedding_layer);
    back_propagation_base->initialize();

    vector<TensorView*> gradient_views = back_propagation_base->get_gradient_views();
    VectorR layer_gradients(get_size(gradient_views));
    link(layer_gradients.data(), gradient_views);

    vector<TensorView*> bp_workspace_views = back_propagation_base->get_workspace_views();
    VectorR bp_workspace(get_size(bp_workspace_views));
    if (bp_workspace.size() > 0)
        link(bp_workspace.data(), bp_workspace_views);

    Tensor1 deltas(output_view.size());
    for(Index i = 0; i < deltas.size(); ++i) deltas(i) = static_cast<type>(random_normal(0.0, 1.0));
    TensorView delta_view(deltas.data(), output_view.shape);

    back_propagation_base->output_gradients = { delta_view };

#ifdef OPENNN_CUDA
    TensorCuda delta_device({ output_view.shape[0], output_view.shape[1], output_view.shape[2] });
    CHECK_CUDA(cudaMemcpy(delta_device.data, deltas.data(), deltas.size() * sizeof(type), cudaMemcpyHostToDevice));
#endif

    embedding_layer.back_propagate(forward_propagation_base, back_propagation_base);

    EmbeddingBackPropagation* back_propagation = static_cast<EmbeddingBackPropagation*>(back_propagation_base.get());
    const TensorView weight_gradients_cpu = back_propagation->weight_gradients;

#ifdef OPENNN_CUDA
    vector<TensorView*> param_views_device = embedding_layer.get_parameter_views_device();
    TensorCuda layer_parameters_device({ get_size(param_views_device) });
    link(layer_parameters_device.data, param_views_device);
    CHECK_CUDA(cudaMemcpy(layer_parameters_device.data, layer_parameters.data(), layer_parameters.size() * sizeof(type), cudaMemcpyHostToDevice));

    unique_ptr<LayerForwardPropagationCuda> forward_propagation_cuda_base =
        make_unique<EmbeddingForwardPropagationCuda>(batch_size, &embedding_layer);
    forward_propagation_cuda_base->initialize();

    vector<TensorView*> workspace_fw_views_device = forward_propagation_cuda_base->get_workspace_views();
    TensorCuda layer_workspace_fw_device({get_size(workspace_fw_views_device)});
    link(layer_workspace_fw_device.data, workspace_fw_views_device);

    TensorCuda inputs_device({batch_size, parameters.sequence_length});
    CHECK_CUDA(cudaMemcpy(inputs_device.data, inputs_mat.data(), inputs_mat.size() * sizeof(type), cudaMemcpyHostToDevice));
    forward_propagation_cuda_base->inputs = { inputs_device.view() };

    embedding_layer.forward_propagate(forward_propagation_cuda_base, true);

    unique_ptr<LayerBackPropagationCuda> back_propagation_cuda_base =
        make_unique<EmbeddingBackPropagationCuda>(batch_size, &embedding_layer);
    back_propagation_cuda_base->initialize();

    vector<TensorView*> gradient_views_device = back_propagation_cuda_base->get_gradient_views();
    TensorCuda layer_gradients_device({get_size(gradient_views_device)});
    link(layer_gradients_device.data, gradient_views_device);

    vector<TensorView*> bp_workspace_views_device = back_propagation_cuda_base->get_workspace_views();
    TensorCuda bp_workspace_device({get_size(bp_workspace_views_device)});
    if (bp_workspace_device.size() > 0)
        link(bp_workspace_device.data, bp_workspace_views_device);

    back_propagation_cuda_base->output_gradients = { delta_device.view() };

    embedding_layer.back_propagate(forward_propagation_cuda_base, back_propagation_cuda_base);

    EmbeddingBackPropagationCuda* back_propagation_cuda = static_cast<EmbeddingBackPropagationCuda*>(back_propagation_cuda_base.get());

    vector<type> host_weight_grads(back_propagation_cuda->weight_gradients.size());
    CHECK_CUDA(cudaMemcpy(host_weight_grads.data(), back_propagation_cuda->weight_gradients.data, back_propagation_cuda->weight_gradients.size() * sizeof(type), cudaMemcpyDeviceToHost));

    for (Index i = 0; i < weight_gradients_cpu.size(); ++i)
        EXPECT_NEAR(weight_gradients_cpu.data[i], host_weight_grads[i], 1e-3);

#endif
}
