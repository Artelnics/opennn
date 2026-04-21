#include "pch.h"

#include "../opennn/tensor_utilities.h"
#include "../opennn/embedding_layer.h"
#include "../opennn/neural_network.h"
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


TEST_P(EmbeddingLayerTest, ForwardPropagate)
{
    EmbeddingLayerConfig parameters = GetParam();

    NeuralNetwork neural_network;
    auto layer = make_unique<Embedding>(
        Shape{parameters.vocabulary_size, parameters.sequence_length},
        parameters.embedding_dimension);
    layer->set_scale_embedding(parameters.scale_embedding);
    layer->set_add_positional_encoding(parameters.add_positional_encoding);
    neural_network.add_layer(std::move(layer));
    neural_network.compile();
    neural_network.set_parameters_random();

    const Index batch_size = parameters.batch_size;

    MatrixR inputs_mat(batch_size, parameters.sequence_length);
    for (Index i = 0; i < batch_size; ++i)
        for (Index j = 0; j < parameters.sequence_length; ++j)
            inputs_mat(i, j) = static_cast<type>(random_integer(0, parameters.vocabulary_size - 1));

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs_mat.data(), {batch_size, parameters.sequence_length}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank(), 3);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], parameters.sequence_length);
    EXPECT_EQ(output_view.shape[2], parameters.embedding_dimension);
}


TEST_P(EmbeddingLayerTest, BackPropagate)
{
    EmbeddingLayerConfig parameters = GetParam();

    NeuralNetwork neural_network;
    auto layer = make_unique<Embedding>(
        Shape{parameters.vocabulary_size, parameters.sequence_length},
        parameters.embedding_dimension);
    layer->set_scale_embedding(parameters.scale_embedding);
    layer->set_add_positional_encoding(parameters.add_positional_encoding);
    neural_network.add_layer(std::move(layer));
    neural_network.compile();
    neural_network.set_parameters_random();

    const Index batch_size = parameters.batch_size;

    MatrixR inputs_mat(batch_size, parameters.sequence_length);
    for (Index i = 0; i < batch_size; ++i)
        for (Index j = 0; j < parameters.sequence_length; ++j)
            inputs_mat(i, j) = static_cast<type>(random_integer(0, parameters.vocabulary_size - 1));

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs_mat.data(), {batch_size, parameters.sequence_length}) };
    neural_network.forward_propagate(input_views, forward_propagation, true);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank(), 3);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], parameters.sequence_length);
    EXPECT_EQ(output_view.shape[2], parameters.embedding_dimension);
}
