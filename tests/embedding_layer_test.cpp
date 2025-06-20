#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/embedding_layer.h"
#include "../opennn/neural_network.h"

using namespace opennn;

TEST(Embedding, DefaultConstructor)
{
    Embedding embedding_layer;

    EXPECT_EQ(embedding_layer.get_vocabulary_size(), 0);
    EXPECT_EQ(embedding_layer.get_sequence_length(), 0);
    EXPECT_EQ(embedding_layer.get_embedding_dimension(), 0);
}


TEST(Embedding, GeneralConstructor)
{
    const Index vocabulary_size = 1;
    const Index sequence_length = 2;
    const Index embedding_dimension = 3;

    Embedding embedding_layer({ vocabulary_size, sequence_length }, embedding_dimension);

    EXPECT_EQ(embedding_layer.get_vocabulary_size(), 1);
    EXPECT_EQ(embedding_layer.get_sequence_length(), 2);
    EXPECT_EQ(embedding_layer.get_embedding_dimension(), 3);
}


TEST(Embedding, ForwardPropagate)
{

    const Index samples_number = get_random_index(2, 10);
    const Index vocabulary_size = get_random_index(1, 10);
    const Index sequence_length = get_random_index(1, 10);
    const Index embedding_dimension = get_random_index(1, 10);

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Embedding>(dimensions({ vocabulary_size, sequence_length }), embedding_dimension));

    Embedding embedding_layer({ vocabulary_size, sequence_length }, embedding_dimension);
    embedding_layer.set_parameters_random();

    Tensor<type, 2> inputs(samples_number, sequence_length);
    inputs.setConstant(type(0));

    Tensor<type, 3> outputs = neural_network.calculate_outputs<2,3>(inputs);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), sequence_length);
    EXPECT_EQ(outputs.dimension(2), embedding_dimension);

}
