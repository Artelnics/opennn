#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/multihead_attention_layer.h"
#include "../opennn/neural_network.h"

using namespace opennn;

TEST(MultiHeadAttention, DefaultConstructorSelfAttention)
{

    MultiHeadAttention multihead_attention_layer;

    EXPECT_EQ(multihead_attention_layer.get_source_sequence_length(), 0);
    EXPECT_EQ(multihead_attention_layer.get_query_sequence_length(), 0);
    EXPECT_EQ(multihead_attention_layer.get_embedding_dimension(), 0);
    EXPECT_EQ(multihead_attention_layer.get_heads_number(), 0);
}


TEST(MultiHeadAttention, GeneralConstructorSelfAttention)
{
    const Index sequence_length = get_random_index(1, 10);
    const Index heads_number = get_random_index(1, 10);
    const Index head_dimension = get_random_index(1, 10);
    const Index embedding_dimension = heads_number * head_dimension;

    MultiHeadAttention multihead_attention_layer({sequence_length, embedding_dimension}, heads_number);

    EXPECT_EQ(multihead_attention_layer.get_source_sequence_length(), sequence_length);
    EXPECT_EQ(multihead_attention_layer.get_query_sequence_length(), sequence_length);
    EXPECT_EQ(multihead_attention_layer.get_embedding_dimension(), embedding_dimension);
    EXPECT_EQ(multihead_attention_layer.get_heads_number(), heads_number);
}


TEST(MultiHeadAttention, ForwardPropagateSelfAttention)
{
    const Index batch_size = get_random_index(1, 10);
    const Index sequence_length = get_random_index(1, 10);
    const Index heads_number = get_random_index(1, 10);
    const Index head_dimension = get_random_index(1, 10);
    const Index embedding_dimension = heads_number * head_dimension;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<MultiHeadAttention>(dimensions({sequence_length, embedding_dimension}), heads_number));

    Tensor<type, 3> inputs_1(batch_size, sequence_length, embedding_dimension);
    inputs_1.setRandom();

    Tensor<type, 3> inputs_2(batch_size, sequence_length, embedding_dimension);
    inputs_1.setRandom();

    Tensor<type, 3> outputs = neural_network.calculate_outputs(inputs_1, inputs_2);

    EXPECT_EQ(outputs.dimension(0), batch_size);
    EXPECT_EQ(outputs.dimension(1), sequence_length);
    EXPECT_EQ(outputs.dimension(2), embedding_dimension);
}

