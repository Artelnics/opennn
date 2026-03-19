#include "pch.h"

#include "../opennn/tensor_utilities.h"
#include "../opennn/multihead_attention_layer.h"
#include "../opennn/random_utilities.h"
#include <iostream>
#include <cmath>

using namespace opennn;

struct MultiHeadAttentionConfig
{
    Index batch_size;
    Index query_sequence_length;
    Index source_sequence_length;
    Index embedding_dimension;
    Index heads_number;
    bool use_causal_mask;
    bool is_cross_attention;
    string test_name;
};

class MultiHeadAttentionTest : public ::testing::TestWithParam<MultiHeadAttentionConfig> {};

INSTANTIATE_TEST_SUITE_P(MultiHeadAttentionTest, MultiHeadAttentionTest, ::testing::Values(
                         MultiHeadAttentionConfig{ 2, 5, 5, 16, 4, false, false, "SelfAttention" },
                         MultiHeadAttentionConfig{ 3, 6, 6, 32, 8, true, false, "SelfAttentionCausalMask" },
                         MultiHeadAttentionConfig{ 2, 4, 7, 12, 3, false, true, "CrossAttention" },
                         MultiHeadAttentionConfig{ 8, 3, 3, 8, 2, false, false, "LargeBatchSmallDims" }));

TEST(MultiHeadAttentionTest, DefaultConstructors)
{
    MultiHeadAttention mha_self;
    EXPECT_EQ(mha_self.get_query_sequence_length(), 0);
    EXPECT_EQ(mha_self.get_source_sequence_length(), 0);
    EXPECT_EQ(mha_self.get_embedding_dimension(), 0);
}

TEST(MultiHeadAttentionTest, GeneralConstructors)
{
    MultiHeadAttention mha_self_config({ 10, 32 }, 4);
    EXPECT_EQ(mha_self_config.get_query_sequence_length(), 10);
    EXPECT_EQ(mha_self_config.get_source_sequence_length(), 10);
    EXPECT_EQ(mha_self_config.get_embedding_dimension(), 32);
    EXPECT_EQ(mha_self_config.get_heads_number(), 4);

    MultiHeadAttention mha_cross({ 5, 16 }, { 8, 16 }, 2);
    EXPECT_EQ(mha_cross.get_query_sequence_length(), 5);
    EXPECT_EQ(mha_cross.get_source_sequence_length(), 8);
    EXPECT_EQ(mha_cross.get_embedding_dimension(), 16);
    EXPECT_EQ(mha_cross.get_heads_number(), 2);
}

TEST_P(MultiHeadAttentionTest, ForwardPropagate)
{
    MultiHeadAttentionConfig params = GetParam();

    unique_ptr<MultiHeadAttention> layer;
    if (params.is_cross_attention)
        layer = make_unique<MultiHeadAttention>(Shape{ params.query_sequence_length, params.embedding_dimension }, Shape{ params.source_sequence_length, params.embedding_dimension }, params.heads_number);
    else
        layer = make_unique<MultiHeadAttention>(Shape{ params.query_sequence_length, params.embedding_dimension }, params.heads_number);

    layer->set(params.query_sequence_length, params.source_sequence_length, params.embedding_dimension, params.heads_number, params.use_causal_mask);

    vector<TensorView*> param_views = layer->get_parameter_views();
    VectorR layer_parameters(get_size(param_views));
    link(layer_parameters.data(), param_views);
    layer->set_parameters_random();

    Tensor3 query_input(params.batch_size, params.query_sequence_length, params.embedding_dimension);
    query_input.setRandom();
    TensorView query_view(query_input.data(), { params.batch_size, params.query_sequence_length, params.embedding_dimension });

    Tensor3 source_input;
    vector<TensorView> input_views;

    if (params.is_cross_attention)
    {
        source_input.resize(params.batch_size, params.source_sequence_length, params.embedding_dimension);
        source_input.setRandom();
        input_views = { query_view, TensorView(source_input.data(), { params.batch_size, params.source_sequence_length, params.embedding_dimension }) };
    }
    else
        input_views = { query_view };

    unique_ptr<LayerForwardPropagation> forward_base = make_unique<MultiHeadAttentionForwardPropagation>(params.batch_size, layer.get());
    forward_base->initialize();

    MultiHeadAttentionForwardPropagation* forward = static_cast<MultiHeadAttentionForwardPropagation*>(forward_base.get());
    VectorR fp_outputs(forward->outputs.size());
    forward->outputs.data = fp_outputs.data();

    layer->forward_propagate(input_views, forward_base, false);
    const TensorView output_view = forward->get_outputs();

#ifdef OPENNN_CUDA

    vector<TensorViewCuda*> param_views_device = layer->get_parameter_views_device();
    TensorCuda layer_parameters_device({get_size(param_views_device)});
    link(layer_parameters_device.data, param_views_device);
    CHECK_CUDA(cudaMemcpy(layer_parameters_device.data, layer_parameters.data(), layer_parameters.size() * sizeof(type), cudaMemcpyHostToDevice));

    TensorCuda query_input_device({params.batch_size, params.query_sequence_length, params.embedding_dimension});
    CHECK_CUDA(cudaMemcpy(query_input_device.data, query_input.data(), query_input.size() * sizeof(type), cudaMemcpyHostToDevice));

    TensorCuda source_input_device;
    vector<TensorViewCuda> input_views_device;
    if (params.is_cross_attention)
    {
        source_input_device.resize({params.batch_size, params.source_sequence_length, params.embedding_dimension});
        CHECK_CUDA(cudaMemcpy(source_input_device.data, source_input.data(), source_input.size() * sizeof(type), cudaMemcpyHostToDevice));
        input_views_device = {query_input_device.view(), source_input_device.view()};
    }
    else
        input_views_device = {query_input_device.view()};

    unique_ptr<LayerForwardPropagationCuda> forward_cuda_base = make_unique<MultiHeadAttentionForwardPropagationCuda>(params.batch_size, layer.get());
    forward_cuda_base->initialize();

    MultiHeadAttentionForwardPropagationCuda* forward_cuda = static_cast<MultiHeadAttentionForwardPropagationCuda*>(forward_cuda_base.get());
    TensorCuda fp_outputs_device({params.batch_size * params.query_sequence_length * params.embedding_dimension});
    forward_cuda->outputs.data = fp_outputs_device.data;

    layer->forward_propagate(input_views_device, forward_cuda_base, false);

    vector<type> host_output_from_gpu(forward_cuda->outputs.size());
    CHECK_CUDA(cudaMemcpy(host_output_from_gpu.data(), forward_cuda->outputs.data, forward_cuda->outputs.size() * sizeof(type), cudaMemcpyDeviceToHost));

    for (Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_view.data[i], host_output_from_gpu[i], 1e-3);

#endif
}

TEST_P(MultiHeadAttentionTest, BackPropagate)
{
    MultiHeadAttentionConfig params = GetParam();

    unique_ptr<MultiHeadAttention> layer;
    if (params.is_cross_attention)
        layer = make_unique<MultiHeadAttention>(Shape{ params.query_sequence_length, params.embedding_dimension }, Shape{ params.source_sequence_length, params.embedding_dimension }, params.heads_number);
    else
        layer = make_unique<MultiHeadAttention>(Shape{ params.query_sequence_length, params.embedding_dimension }, params.heads_number);

    layer->set(params.query_sequence_length, params.source_sequence_length, params.embedding_dimension, params.heads_number, params.use_causal_mask);

    vector<TensorView*> param_views = layer->get_parameter_views();
    VectorR layer_parameters(get_size(param_views));
    link(layer_parameters.data(), param_views);
    layer->set_parameters_random();

    Tensor3 query_input(params.batch_size, params.query_sequence_length, params.embedding_dimension);
    query_input.setRandom();
    TensorView query_view(query_input.data(), { params.batch_size, params.query_sequence_length, params.embedding_dimension });

    Tensor3 source_input;
    vector<TensorView> input_views;

    if (params.is_cross_attention)
    {
        source_input.resize(params.batch_size, params.source_sequence_length, params.embedding_dimension);
        source_input.setRandom();
        input_views = { query_view, TensorView(source_input.data(), { params.batch_size, params.source_sequence_length, params.embedding_dimension }) };
    }
    else
        input_views = { query_view };

    unique_ptr<LayerForwardPropagation> forward_base = make_unique<MultiHeadAttentionForwardPropagation>(params.batch_size, layer.get());
    forward_base->initialize();

    MultiHeadAttentionForwardPropagation* forward = static_cast<MultiHeadAttentionForwardPropagation*>(forward_base.get());
    VectorR fp_outputs(forward->outputs.size());
    forward->outputs.data = fp_outputs.data();

    layer->forward_propagate(input_views, forward_base, true);
    TensorView output_view = forward->get_outputs();

    unique_ptr<LayerBackPropagation> back_base = make_unique<MultiHeadAttentionBackPropagation>(params.batch_size, layer.get());
    back_base->initialize();

    vector<TensorView*> gradient_views = back_base->get_gradient_views();
    VectorR layer_gradients(get_size(gradient_views));
    layer_gradients.setZero();
    link(layer_gradients.data(), gradient_views);

    Tensor1 deltas(output_view.size());
    for(Index i = 0; i < deltas.size(); ++i) deltas(i) = static_cast<type>(random_normal(0.0, 1.0));
    TensorView delta_view(deltas.data(), output_view.shape);
    vector<TensorView> delta_views = { delta_view };

#ifdef OPENNN_CUDA

    TensorCuda delta_device({params.batch_size, params.query_sequence_length, params.embedding_dimension});
    CHECK_CUDA(cudaMemcpy(delta_device.data, deltas.data(), deltas.size() * sizeof(type), cudaMemcpyHostToDevice));
    vector<TensorViewCuda> delta_views_device = { delta_device.view() };

#endif

    layer->back_propagate(input_views, delta_views, forward_base, back_base);
    MultiHeadAttentionBackPropagation* back = static_cast<MultiHeadAttentionBackPropagation*>(back_base.get());

#ifdef OPENNN_CUDA

    vector<TensorViewCuda*> param_views_device = layer->get_parameter_views_device();
    TensorCuda layer_parameters_device({get_size(param_views_device)});
    link(layer_parameters_device.data, param_views_device);
    CHECK_CUDA(cudaMemcpy(layer_parameters_device.data, layer_parameters.data(), layer_parameters.size() * sizeof(type), cudaMemcpyHostToDevice));

    TensorCuda query_input_device({params.batch_size, params.query_sequence_length, params.embedding_dimension});
    CHECK_CUDA(cudaMemcpy(query_input_device.data, query_input.data(), query_input.size() * sizeof(type), cudaMemcpyHostToDevice));

    TensorCuda source_input_device;
    vector<TensorViewCuda> input_views_device;
    if (params.is_cross_attention)
    {
        source_input_device.resize({params.batch_size, params.source_sequence_length, params.embedding_dimension});
        CHECK_CUDA(cudaMemcpy(source_input_device.data, source_input.data(), source_input.size() * sizeof(type), cudaMemcpyHostToDevice));
        input_views_device = {query_input_device.view(), source_input_device.view()};
    }
    else
        input_views_device = {query_input_device.view()};

    unique_ptr<LayerForwardPropagationCuda> forward_cuda_base = make_unique<MultiHeadAttentionForwardPropagationCuda>(params.batch_size, layer.get());
    forward_cuda_base->initialize();
    MultiHeadAttentionForwardPropagationCuda* forward_cuda = static_cast<MultiHeadAttentionForwardPropagationCuda*>(forward_cuda_base.get());
    TensorCuda fp_outputs_device({params.batch_size * params.query_sequence_length * params.embedding_dimension});
    forward_cuda->outputs.data = fp_outputs_device.data;

    unique_ptr<LayerBackPropagationCuda> back_cuda_base = make_unique<MultiHeadAttentionBackPropagationCuda>(params.batch_size, layer.get());
    back_cuda_base->initialize();

    vector<TensorViewCuda*> gradient_views_device = back_cuda_base->get_gradient_views();
    TensorCuda layer_gradients_device({get_size(gradient_views_device)});
    CHECK_CUDA(cudaMemset(layer_gradients_device.data, 0, layer_gradients_device.size() * sizeof(type)));
    link(layer_gradients_device.data, gradient_views_device);

    MultiHeadAttentionBackPropagationCuda* back_cuda = static_cast<MultiHeadAttentionBackPropagationCuda*>(back_cuda_base.get());

    layer->forward_propagate(input_views_device, forward_cuda_base, true);

    layer->back_propagate(input_views_device, delta_views_device, forward_cuda_base, back_cuda_base);

    vector<type> host_layer_gradients(layer_gradients_device.size());
    CHECK_CUDA(cudaMemcpy(host_layer_gradients.data(), layer_gradients_device.data, layer_gradients_device.size() * sizeof(type), cudaMemcpyDeviceToHost));

    for (Index i = 0; i < layer_gradients.size(); ++i)
        EXPECT_NEAR(layer_gradients[i], host_layer_gradients[i], 1e-2);

    vector<type> host_q_input_grads(back_cuda->input_gradients[0].size());
    CHECK_CUDA(cudaMemcpy(host_q_input_grads.data(), back_cuda->input_gradients[0].data, host_q_input_grads.size() * sizeof(type), cudaMemcpyDeviceToHost));

    for (Index i = 0; i < back->input_gradients[0].size(); ++i)
        EXPECT_NEAR(back->input_gradients[0].data[i], host_q_input_grads[i], 1e-2);

    if (params.is_cross_attention)
    {
        vector<type> host_s_input_grads(back_cuda->input_gradients[1].size());
        CHECK_CUDA(cudaMemcpy(host_s_input_grads.data(), back_cuda->input_gradients[1].data, host_s_input_grads.size() * sizeof(type), cudaMemcpyDeviceToHost));

        for (Index i = 0; i < back->input_gradients[1].size(); ++i)
            EXPECT_NEAR(back->input_gradients[1].data[i], host_s_input_grads[i], 1e-2);
    }

#endif
}
