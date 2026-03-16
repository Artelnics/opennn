#include "pch.h"

#include "../opennn/tensor_utilities.h"
#include "../opennn/multihead_attention_layer.h"
#include "../opennn/random_utilities.h"
#include <iostream>
#include <cmath>

using namespace opennn;

// ----------------------------------------------------------------------------
// ESTRUCTURA DE CONFIGURACIÓN PARAMETRIZADA
// ----------------------------------------------------------------------------

struct MultiHeadAttentionConfig {
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

// Instanciación de pruebas variadas para garantizar que toda la matemática funciona
INSTANTIATE_TEST_SUITE_P(MHATests, MultiHeadAttentionTest, ::testing::Values(
                                                               // Self-Attention clásico (ej: Encoder de Transformer)
                                                               MultiHeadAttentionConfig{ 2, 5, 5, 16, 4, false, false, "SelfAttention" },
                                                               // Self-Attention con máscara causal (ej: Decoder autoregresivo)
                                                               MultiHeadAttentionConfig{ 3, 6, 6, 32, 8, true, false, "SelfAttentionCausalMask" },
                                                               // Cross-Attention (Query diferente a Source, ej: Decoder mirando al Encoder)
                                                               MultiHeadAttentionConfig{ 2, 4, 7, 12, 3, false, true, "CrossAttention" },
                                                               // Reducción al absurdo (batch grande, pocas dimensiones)
                                                               MultiHeadAttentionConfig{ 8, 3, 3, 8, 2, false, false, "LargeBatchSmallDims" }
                                                               ));


// ----------------------------------------------------------------------------
// TEST: Constructores básicos
// ----------------------------------------------------------------------------

TEST(MultiHeadAttention, DefaultConstructors)
{
    // 1. Constructor por defecto Self-Attention
    MultiHeadAttention mha_self;
    EXPECT_EQ(mha_self.get_query_sequence_length(), 0);
    EXPECT_EQ(mha_self.get_source_sequence_length(), 0);
    EXPECT_EQ(mha_self.get_embedding_dimension(), 0);

    // 2. Constructor de configuración Self-Attention
    MultiHeadAttention mha_self_config({ 10, 32 }, 4); // {seq_length, emb_dim}, heads
    EXPECT_EQ(mha_self_config.get_query_sequence_length(), 10);
    EXPECT_EQ(mha_self_config.get_source_sequence_length(), 10);
    EXPECT_EQ(mha_self_config.get_embedding_dimension(), 32);
    EXPECT_EQ(mha_self_config.get_heads_number(), 4);

    // 3. Constructor de configuración Cross-Attention
    MultiHeadAttention mha_cross({ 5, 16 }, { 8, 16 }, 2); // {query_len, emb_dim}, {source_len, emb_dim}, heads
    EXPECT_EQ(mha_cross.get_query_sequence_length(), 5);
    EXPECT_EQ(mha_cross.get_source_sequence_length(), 8);
    EXPECT_EQ(mha_cross.get_embedding_dimension(), 16);
    EXPECT_EQ(mha_cross.get_heads_number(), 2);
}


// ----------------------------------------------------------------------------
// TEST: FORWARD PROPAGATE (Aislamiento de capa en CPU)
// ----------------------------------------------------------------------------

TEST_P(MultiHeadAttentionTest, ForwardPropagate)
{
    MultiHeadAttentionConfig params = GetParam();

    // 1. Inicializar Capa
    unique_ptr<MultiHeadAttention> layer;
    if (params.is_cross_attention) {
        layer = make_unique<MultiHeadAttention>(
            Shape{ params.query_sequence_length, params.embedding_dimension },
            Shape{ params.source_sequence_length, params.embedding_dimension },
            params.heads_number);
    } else {
        layer = make_unique<MultiHeadAttention>(
            Shape{ params.query_sequence_length, params.embedding_dimension },
            params.heads_number);
    }

    // Configurar máscara causal si se requiere
    layer->set(params.query_sequence_length, params.source_sequence_length,
               params.embedding_dimension, params.heads_number, params.use_causal_mask);

    // Asignar memoria para los pesos y aleatorizar
    vector<TensorView*> param_views = layer->get_parameter_views();
    VectorR layer_parameters(get_size(param_views));
    link(layer_parameters.data(), param_views);
    layer->set_parameters_random();

    // 2. Crear Entradas Fake
    Tensor3 query_input(params.batch_size, params.query_sequence_length, params.embedding_dimension);
    query_input.setRandom();
    TensorView query_view(query_input.data(), { params.batch_size, params.query_sequence_length, params.embedding_dimension });

    Tensor3 source_input; // Solo se usará si es cross_attention
    vector<TensorView> input_views;

    if (params.is_cross_attention) {
        source_input.resize(params.batch_size, params.source_sequence_length, params.embedding_dimension);
        source_input.setRandom();
        input_views = { query_view, TensorView(source_input.data(), { params.batch_size, params.source_sequence_length, params.embedding_dimension }) };
    } else {
        input_views = { query_view };
    }

    // 3. Crear Workspace de Forward
    unique_ptr<LayerForwardPropagation> forward_base =
        make_unique<MultiHeadAttentionForwardPropagation>(params.batch_size, layer.get());
    forward_base->initialize();

    // Vincular memoria del workspace
    MultiHeadAttentionForwardPropagation* forward = static_cast<MultiHeadAttentionForwardPropagation*>(forward_base.get());
    VectorR fp_outputs(forward->outputs.size());
    forward->outputs.data = fp_outputs.data();

    // 4. Ejecutar Forward
    layer->forward_propagate(input_views, forward_base, false);

    // 5. Validaciones
    const TensorView output_view = forward->get_outputs();

    // A) Verificar Dimensiones de salida (Siempre es igual a las dimensiones de Query)
    ASSERT_EQ(output_view.shape.size(), 3);
    EXPECT_EQ(output_view.shape[0], params.batch_size);
    EXPECT_EQ(output_view.shape[1], params.query_sequence_length);
    EXPECT_EQ(output_view.shape[2], params.embedding_dimension);

    // B) Verificar que el modelo no ha explotado matemáticamente produciendo NaNs
    for (Index i = 0; i < output_view.size(); ++i) {
        EXPECT_FALSE(std::isnan(output_view.data[i]));
        EXPECT_FALSE(std::isinf(output_view.data[i]));
    }
}


// ----------------------------------------------------------------------------
// TEST: BACKWARD PROPAGATE (Aislamiento de capa en CPU)
// ----------------------------------------------------------------------------

TEST_P(MultiHeadAttentionTest, BackPropagate)
{
    MultiHeadAttentionConfig params = GetParam();

    unique_ptr<MultiHeadAttention> layer;
    if (params.is_cross_attention) {
        layer = make_unique<MultiHeadAttention>(
            Shape{ params.query_sequence_length, params.embedding_dimension },
            Shape{ params.source_sequence_length, params.embedding_dimension },
            params.heads_number);
    } else {
        layer = make_unique<MultiHeadAttention>(
            Shape{ params.query_sequence_length, params.embedding_dimension },
            params.heads_number);
    }

    layer->set(params.query_sequence_length, params.source_sequence_length,
               params.embedding_dimension, params.heads_number, params.use_causal_mask);

    vector<TensorView*> param_views = layer->get_parameter_views();
    VectorR layer_parameters(get_size(param_views));
    link(layer_parameters.data(), param_views);
    layer->set_parameters_random();

    // 1. Entradas Fake
    Tensor3 query_input(params.batch_size, params.query_sequence_length, params.embedding_dimension);
    query_input.setRandom();
    TensorView query_view(query_input.data(), { params.batch_size, params.query_sequence_length, params.embedding_dimension });

    Tensor3 source_input;
    vector<TensorView> input_views;

    if (params.is_cross_attention) {
        source_input.resize(params.batch_size, params.source_sequence_length, params.embedding_dimension);
        source_input.setRandom();
        input_views = { query_view, TensorView(source_input.data(), { params.batch_size, params.source_sequence_length, params.embedding_dimension }) };
    } else {
        input_views = { query_view };
    }

    // 2. Forward Workspace
    unique_ptr<LayerForwardPropagation> forward_base =
        make_unique<MultiHeadAttentionForwardPropagation>(params.batch_size, layer.get());
    forward_base->initialize();

    MultiHeadAttentionForwardPropagation* forward = static_cast<MultiHeadAttentionForwardPropagation*>(forward_base.get());
    VectorR fp_outputs(forward->outputs.size());
    forward->outputs.data = fp_outputs.data();

    layer->forward_propagate(input_views, forward_base, true);
    TensorView output_view = forward->get_outputs();

    // 3. Backward Workspace
    unique_ptr<LayerBackPropagation> back_base =
        make_unique<MultiHeadAttentionBackPropagation>(params.batch_size, layer.get());
    back_base->initialize();

    vector<TensorView*> gradient_views = back_base->get_gradient_views();
    VectorR layer_gradients(get_size(gradient_views));
    link(layer_gradients.data(), gradient_views);

    // 4. Crear Deltas Fake (Gradientes provenientes de capas superiores)
    Tensor1 deltas(output_view.size());
    for (Index i = 0; i < deltas.size(); ++i) deltas(i) = static_cast<type>(random_normal(0.0, 1.0));
    TensorView delta_view(deltas.data(), output_view.shape);
    vector<TensorView> delta_views = { delta_view };

    // 5. Ejecutar Backward
    layer->back_propagate(input_views, delta_views, forward_base, back_base);

    MultiHeadAttentionBackPropagation* back = static_cast<MultiHeadAttentionBackPropagation*>(back_base.get());

    // 6. Validaciones

    // A) Verificar que los gradientes de entrada se han generado correctamente
    vector<TensorView> input_grads = back->get_input_gradients();

    if (params.is_cross_attention) {
        // En Cross-Attention devolvemos gradientes distintos para Query y Source
        EXPECT_EQ(input_grads.size(), 2);

        EXPECT_EQ(input_grads[0].shape[0], params.batch_size);
        EXPECT_EQ(input_grads[0].shape[1], params.query_sequence_length);
        EXPECT_EQ(input_grads[0].shape[2], params.embedding_dimension);

        EXPECT_EQ(input_grads[1].shape[0], params.batch_size);
        EXPECT_EQ(input_grads[1].shape[1], params.source_sequence_length);
        EXPECT_EQ(input_grads[1].shape[2], params.embedding_dimension);
    } else {
        // En Self-Attention, Source y Query comparten origen, se acumula el gradiente en input_grads[0]
        EXPECT_GE(input_grads.size(), 1); // OpenNN asigna espacio de sobra, pero validamos el 0

        EXPECT_EQ(input_grads[0].shape[0], params.batch_size);
        EXPECT_EQ(input_grads[0].shape[1], params.query_sequence_length);
        EXPECT_EQ(input_grads[0].shape[2], params.embedding_dimension);
    }

    // B) Verificar que los pesos internos han recibido gradientes (no son todo ceros)
    const TensorView q_weight_grads = back->query_weight_gradients;
    bool has_non_zero_grad = false;
    for (Index i = 0; i < q_weight_grads.size(); ++i) {
        if (std::abs(q_weight_grads.data[i]) > 1e-6) {
            has_non_zero_grad = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero_grad) << "Los gradientes de los pesos de Query no se calcularon (son cero).";
}
