#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/embedding_layer.h"

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
/*
    Embedding embedding_layer(1,2,3);

    EXPECT_EQ(embedding_layer.get_vocabulary_size(), 1);
    EXPECT_EQ(embedding_layer.get_sequence_length(), 2);
    EXPECT_EQ(embedding_layer.get_embedding_dimension(), 3);
*/
}


TEST(Embedding, ForwardPropagate)
{
/*
    const Index samples_number = get_random_index(1, 10);
    const Index vocabulary_size = get_random_index(1, 10);
    const Index sequence_length = get_random_index(1, 10);
    const Index embedding_dimension = get_random_index(1, 10);

    Embedding embedding_layer(vocabulary_size, sequence_length, embedding_dimension);
    embedding_layer.set_parameters_constant(type(0));

    unique_ptr<LayerForwardPropagation> embedding_forward_propagation
        = make_unique<EmbeddingForwardPropagation>(samples_number, &embedding_layer);

    Tensor<type, 2> inputs(samples_number, sequence_length);
    inputs.setConstant(type(0));

    embedding_layer.forward_propagate({ make_pair(inputs.data(), dimensions{samples_number, sequence_length}) },
        embedding_forward_propagation,
        true);

    EXPECT_EQ(embedding_forward_propagation->batch_size, samples_number);
    EXPECT_EQ(embedding_forward_propagation->get_outputs_pair().second[0], samples_number);
    EXPECT_EQ(embedding_forward_propagation->get_outputs_pair().second[1], sequence_length);
    EXPECT_EQ(embedding_forward_propagation->get_outputs_pair().second[2], embedding_dimension);
*/
}
