#include "pch.h"

#include "opennn/tensor_types.h"
#include "opennn/embedding_layer.h"
#include "opennn/neural_network.h"
#include "opennn/random_utilities.h"
#include <cmath>
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
    EXPECT_EQ(embedding_layer.get_input_shape(), (Shape{input_shape[1]}));
    EXPECT_EQ(embedding_layer.get_output_shape(), (Shape{input_shape[1], embedding_dimension}));
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
    neural_network.add_layer(move(layer));
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

    ASSERT_EQ(output_view.shape.rank, 3);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], parameters.sequence_length);
    EXPECT_EQ(output_view.shape[2], parameters.embedding_dimension);
}


TEST(Embedding, ForwardValuesMatchExpected)
{
    const Index vocabulary_size = 4;
    const Index sequence_length = 2;
    const Index embedding_dimension = 3;
    const Index batch_size = 1;

    NeuralNetwork neural_network;
    auto layer = make_unique<Embedding>(
        Shape{vocabulary_size, sequence_length}, embedding_dimension);
    layer->set_scale_embedding(false);
    layer->set_add_positional_encoding(false);
    neural_network.add_layer(move(layer));
    neural_network.compile();

    const float weights[] = { 0.0f, 0.0f, 0.0f,
                              1.0f, 2.0f, 3.0f,
                              4.0f, 5.0f, 6.0f,
                              7.0f, 8.0f, 9.0f };
    VectorR table = VectorR::Zero(neural_network.get_parameters_size());
    for (Index i = 0; i < vocabulary_size * embedding_dimension; ++i)
        table(i) = weights[i];
    neural_network.set_parameters(table);

    MatrixR inputs_mat(batch_size, sequence_length);
    inputs_mat(0, 0) = 1.0f;
    inputs_mat(0, 1) = 3.0f;

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs_mat.data(), {batch_size, sequence_length}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    const TensorView output_view = forward_propagation.get_outputs();
    ASSERT_EQ(output_view.shape.rank, 3);
    ASSERT_EQ(output_view.size(), batch_size * sequence_length * embedding_dimension);

    const float* output = output_view.as<type>();

    EXPECT_NEAR(output[0], 1.0f, 1.0e-5f);
    EXPECT_NEAR(output[1], 2.0f, 1.0e-5f);
    EXPECT_NEAR(output[2], 3.0f, 1.0e-5f);
    EXPECT_NEAR(output[3], 7.0f, 1.0e-5f);
    EXPECT_NEAR(output[4], 8.0f, 1.0e-5f);
    EXPECT_NEAR(output[5], 9.0f, 1.0e-5f);
}


TEST(Embedding, ScaleEmbeddingForwardValues)
{
    const Index vocabulary_size = 4;
    const Index sequence_length = 2;
    const Index embedding_dimension = 4;
    const Index batch_size = 1;

    NeuralNetwork neural_network;
    auto layer = make_unique<Embedding>(
        Shape{vocabulary_size, sequence_length}, embedding_dimension);
    layer->set_scale_embedding(true);
    layer->set_add_positional_encoding(false);
    neural_network.add_layer(move(layer));
    neural_network.compile();

    const float weights[] = { 0.0f, 0.0f, 0.0f, 0.0f,
                              1.0f, 2.0f, 3.0f, 4.0f,
                              5.0f, 6.0f, 7.0f, 8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f };
    VectorR table = VectorR::Zero(neural_network.get_parameters_size());
    for (Index i = 0; i < vocabulary_size * embedding_dimension; ++i)
        table(i) = weights[i];
    neural_network.set_parameters(table);

    MatrixR inputs_mat(batch_size, sequence_length);
    inputs_mat(0, 0) = 1.0f;
    inputs_mat(0, 1) = 2.0f;

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs_mat.data(), {batch_size, sequence_length}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    const TensorView output_view = forward_propagation.get_outputs();
    ASSERT_EQ(output_view.size(), batch_size * sequence_length * embedding_dimension);

    const float* output = output_view.as<type>();
    const float scale = sqrt(static_cast<float>(embedding_dimension));

    EXPECT_NEAR(output[0], 1.0f * scale, 1.0e-4f);
    EXPECT_NEAR(output[1], 2.0f * scale, 1.0e-4f);
    EXPECT_NEAR(output[2], 3.0f * scale, 1.0e-4f);
    EXPECT_NEAR(output[3], 4.0f * scale, 1.0e-4f);
    EXPECT_NEAR(output[4], 5.0f * scale, 1.0e-4f);
    EXPECT_NEAR(output[5], 6.0f * scale, 1.0e-4f);
    EXPECT_NEAR(output[6], 7.0f * scale, 1.0e-4f);
    EXPECT_NEAR(output[7], 8.0f * scale, 1.0e-4f);
}


TEST(Embedding, PositionalEncodingForwardValues)
{
    const Index vocabulary_size = 4;
    const Index sequence_length = 2;
    const Index embedding_dimension = 4;
    const Index batch_size = 1;

    NeuralNetwork neural_network;
    auto layer = make_unique<Embedding>(
        Shape{vocabulary_size, sequence_length}, embedding_dimension);
    layer->set_scale_embedding(false);
    layer->set_add_positional_encoding(true);
    neural_network.add_layer(move(layer));
    neural_network.compile();

    const float weights[] = { 0.0f, 0.0f, 0.0f, 0.0f,
                              1.0f, 2.0f, 3.0f, 4.0f,
                              5.0f, 6.0f, 7.0f, 8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f };
    VectorR table = VectorR::Zero(neural_network.get_parameters_size());
    for (Index i = 0; i < vocabulary_size * embedding_dimension; ++i)
        table(i) = weights[i];
    neural_network.set_parameters(table);

    MatrixR inputs_mat(batch_size, sequence_length);
    inputs_mat(0, 0) = 1.0f;
    inputs_mat(0, 1) = 2.0f;

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs_mat.data(), {batch_size, sequence_length}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    const TensorView output_view = forward_propagation.get_outputs();
    ASSERT_EQ(output_view.size(), batch_size * sequence_length * embedding_dimension);

    const float* output = output_view.as<type>();

    const Index half = embedding_dimension / 2;
    const float half_f = static_cast<float>(embedding_dimension) / 2.0f;

    auto positional = [&](Index position, Index dimension) -> float
    {
        const float divisor = pow(10000.0f,
            (dimension < half ? static_cast<float>(dimension)
                              : static_cast<float>(dimension - half)) / half_f);
        return (dimension < half)
            ? sin(static_cast<float>(position) / divisor)
            : cos(static_cast<float>(position) / divisor);
    };

    const float row1[embedding_dimension] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float row2[embedding_dimension] = { 5.0f, 6.0f, 7.0f, 8.0f };

    for (Index dimension = 0; dimension < embedding_dimension; ++dimension)
    {
        EXPECT_NEAR(output[dimension], row1[dimension] + positional(0, dimension), 1.0e-5f);
        EXPECT_NEAR(output[embedding_dimension + dimension],
                    row2[dimension] + positional(1, dimension), 1.0e-5f);
    }
}
