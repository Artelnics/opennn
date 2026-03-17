#include "pch.h"

#include "../opennn/tensor_utilities.h"
#include "../opennn/multihead_attention_layer.h"
#include "../opennn/random_utilities.h"
#include <iostream>
#include <cmath>

using namespace opennn;

// --- FUNCIONES AUXILIARES ---

void copy_to_gpu(const type* host_data, float* device_data, size_t size)
{
    vector<float> temp(size);
    for (size_t i = 0; i < size; ++i) temp[i] = static_cast<float>(host_data[i]);
    CHECK_CUDA(cudaMemcpy(device_data, temp.data(), size * sizeof(float), cudaMemcpyHostToDevice));
}

void copy_from_gpu(const float* device_data, type* host_data, size_t size)
{
    vector<float> temp(size);
    CHECK_CUDA(cudaMemcpy(temp.data(), device_data, size * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < size; ++i) host_data[i] = static_cast<type>(temp[i]);
}

void debug_compare(const string& step_name, const type* cpu_data, const float* gpu_data, size_t size)
{
    vector<float> gpu_host(size);
    cudaMemcpy(gpu_host.data(), gpu_data, size * sizeof(float), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float diff = std::abs(static_cast<float>(cpu_data[i]) - gpu_host[i]);
        if (diff > max_diff) max_diff = diff;
    }

    cout << "\n--- [DEBUG] " << step_name << " ---" << endl;
    cout << std::fixed << std::setprecision(5);
    cout << "CPU : ";
    for (size_t i = 0; i < std::min((size_t)5, size); ++i) cout << cpu_data[i] << "  ";
    cout << "..." << endl;

    cout << "GPU : ";
    for (size_t i = 0; i < std::min((size_t)5, size); ++i) cout << gpu_host[i] << "  ";
    cout << "..." << endl;

    cout << ">>> Max Diff: " << max_diff << (max_diff > 1e-2 ? "  <-- ¡FALLO AQUÍ!" : "  (OK)") << endl;
}

// -----------------------------------------------------------------------------------

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

INSTANTIATE_TEST_SUITE_P(MHATests, MultiHeadAttentionTest, ::testing::Values(
                                                               MultiHeadAttentionConfig{ 2, 5, 5, 16, 4, false, false, "SelfAttention" },
                                                               MultiHeadAttentionConfig{ 3, 6, 6, 32, 8, true, false, "SelfAttentionCausalMask" },
                                                               MultiHeadAttentionConfig{ 2, 4, 7, 12, 3, false, true, "CrossAttention" },
                                                               MultiHeadAttentionConfig{ 8, 3, 3, 8, 2, false, false, "LargeBatchSmallDims" }));

TEST(MultiHeadAttention, DefaultConstructors)
{
    MultiHeadAttention mha_self;
    EXPECT_EQ(mha_self.get_query_sequence_length(), 0);
    EXPECT_EQ(mha_self.get_source_sequence_length(), 0);
    EXPECT_EQ(mha_self.get_embedding_dimension(), 0);
}


TEST(MultiHeadAttention, GeneralConstructors)
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
    link(layer_gradients.data(), gradient_views);

    Tensor1 deltas(output_view.size());
    for(Index i = 0; i < deltas.size(); ++i) deltas(i) = static_cast<type>(random_normal(0.0, 1.0));
    TensorView delta_view(deltas.data(), output_view.shape);
    vector<TensorView> delta_views = { delta_view };

#ifdef OPENNN_CUDA
    TensorCuda delta_device({params.batch_size, params.query_sequence_length, params.embedding_dimension});
    copy_to_gpu(deltas.data(), delta_device.data, deltas.size());
    vector<TensorViewCuda> delta_views_device = { delta_device.view() };
#endif

    // EJECUCIÓN CPU
    layer->back_propagate(input_views, delta_views, forward_base, back_base);
    MultiHeadAttentionBackPropagation* back = static_cast<MultiHeadAttentionBackPropagation*>(back_base.get());
    const TensorView weight_gradients_cpu = back->query_weight_gradients;

#ifdef OPENNN_CUDA
    vector<TensorViewCuda*> param_views_device = layer->get_parameter_views_device();
    TensorCuda layer_parameters_device({get_size(param_views_device)});
    link(layer_parameters_device.data, param_views_device);
    copy_to_gpu(layer_parameters.data(), layer_parameters_device.data, layer_parameters.size());

    TensorCuda query_input_device({params.batch_size, params.query_sequence_length, params.embedding_dimension});
    copy_to_gpu(query_input.data(), query_input_device.data, query_input.size());

    TensorCuda source_input_device;
    vector<TensorViewCuda> input_views_device;
    if (params.is_cross_attention)
    {
        source_input_device.resize({params.batch_size, params.source_sequence_length, params.embedding_dimension});
        copy_to_gpu(source_input.data(), source_input_device.data, source_input.size());
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
    link(layer_gradients_device.data, gradient_views_device);

    MultiHeadAttentionBackPropagationCuda* back_cuda = static_cast<MultiHeadAttentionBackPropagationCuda*>(back_cuda_base.get());

    layer->forward_propagate(input_views_device, forward_cuda_base, true);

    // EJECUCIÓN GPU
    layer->back_propagate(input_views_device, delta_views_device, forward_cuda_base, back_cuda_base);

    // ========================================================================
    // IMPRESIONES DE DEBUG PASO A PASO (BACKWARD)
    // ========================================================================
    if (params.test_name == "SelfAttention") {
        cout << "\n[ INICIANDO COMPARACION DE TENSORES INTERMEDIOS BACKWARD ]" << endl;
        debug_compare("Projection Weight Gradients", back->projection_weight_gradients.data, back_cuda->projection_weight_gradients.data, back->projection_weight_gradients.size());
        debug_compare("Concatenated Attention Output Gradients", back->concatenated_attention_output_gradients.data(), back_cuda->concatenated_attention_output_gradients.data, back->concatenated_attention_output_gradients.size());
        debug_compare("Value Gradients Transposed", back->value_gradients.data(), back_cuda->value_gradients_transposed.data, back->value_gradients.size());
        debug_compare("Query Weight Gradients (El fallo reportado)", back->query_weight_gradients.data, back_cuda->query_weight_gradients.data, back->query_weight_gradients.size());
        debug_compare("Final Input Gradients", back->input_gradients[0].data, back_cuda->input_gradients[0].data, back->input_gradients[0].size());
        cout << "--------------------------------------------------------\n" << endl;
    }

    // Aserciones formales
    vector<type> host_q_weight_grads(back_cuda->query_weight_gradients.size());
    copy_from_gpu(back_cuda->query_weight_gradients.data, host_q_weight_grads.data(), host_q_weight_grads.size());

    for (Index i = 0; i < weight_gradients_cpu.size(); ++i)
        EXPECT_NEAR(weight_gradients_cpu.data[i], host_q_weight_grads[i], 1e-2);

    vector<type> host_q_input_grads(back_cuda->input_gradients[0].size());
    copy_from_gpu(back_cuda->input_gradients[0].data, host_q_input_grads.data(), host_q_input_grads.size());

    //for (Index i = 0; i < back->input_gradients[0].size(); ++i)
    //    EXPECT_NEAR(back->input_gradients[0].data[i], host_q_input_grads[i], 1e-2);

#endif
}
