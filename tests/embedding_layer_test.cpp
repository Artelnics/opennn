#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/language_dataset.h"
#include "../opennn/embedding_layer.h"
#include "../opennn/flatten_layer.h"
#include "../opennn/dense_layer.h"
#include "../opennn/neural_network.h"
#include "../opennn/mean_squared_error.h"

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
    const dimensions input_dimensions = {1, 2, 3};

    const Index vocabulary_size = input_dimensions[0];
    const Index sequence_length = input_dimensions[1];
    const Index embedding_dimension = input_dimensions[2];

    Embedding embedding_layer({vocabulary_size, sequence_length}, embedding_dimension);

    EXPECT_EQ(embedding_layer.get_vocabulary_size(), vocabulary_size);
    EXPECT_EQ(embedding_layer.get_sequence_length(), sequence_length);
    EXPECT_EQ(embedding_layer.get_embedding_dimension(), embedding_dimension);
}


TEST(Embedding, ForwardPropagate)
{
    const Index samples_number = get_random_index(1, 10);
    const Index vocabulary_size = get_random_index(1, 10);
    const Index sequence_length = get_random_index(1, 10);
    const Index embedding_dimension = get_random_index(1, 10);

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Embedding>(dimensions({ vocabulary_size, sequence_length }), embedding_dimension));

    Embedding embedding_layer({vocabulary_size, sequence_length}, embedding_dimension);
    embedding_layer.set_parameters_random();

    Tensor<type, 2> inputs(samples_number, sequence_length);
    inputs.setConstant(type(0));

    Tensor<type, 3> outputs = neural_network.calculate_outputs<2,3>(inputs);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), sequence_length);
    EXPECT_EQ(outputs.dimension(2), embedding_dimension);
}


 TEST(EmbeddingForwardPropagationTest, GetOutputPairReturnsCorrectDataAndShape)
 {
     const Index batch_size = 2;
     const Index vocab_size = 15;
     const Index sequence_length = 5;
     const Index embedding_dimension = 6;

     Embedding layer({sequence_length}, embedding_dimension, "test_embedding");
     layer.set(vocab_size, sequence_length, embedding_dimension, "test_embedding");

     EmbeddingForwardPropagation forward(batch_size, &layer);

     const TensorView output_view = forward.get_output_pair();

     const TensorMap<Tensor<type, 3>> out = tensor_map<3>(output_view);

     EXPECT_EQ(output_view.data, forward.outputs.data());
     ASSERT_EQ(output_view.dims.size(), 3);
     EXPECT_EQ(output_view.dims[0], batch_size);
     EXPECT_EQ(output_view.dims[1], sequence_length);
     EXPECT_EQ(output_view.dims[2], embedding_dimension);
 }


TEST(Embedding, BackPropagate)
{
    LanguageDataset language_dataset("../examples/amazon_reviews/data/amazon_cells_labelled.txt");
    language_dataset.set_sample_uses("Training");

    const Index embedding_dimension = get_random_index(1,10);
    const Index vocabulary_size = language_dataset.get_input_vocabulary_size();
    const Index sequence_length = language_dataset.get_input_sequence_length();

    dimensions input_dimensions = { vocabulary_size, sequence_length };

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<Embedding>(input_dimensions, embedding_dimension));
    neural_network.add_layer(make_unique<Flatten<3>>(neural_network.get_output_dimensions()));
    neural_network.add_layer(make_unique<Dense2d>(neural_network.get_output_dimensions(), language_dataset.get_target_dimensions(), "Logistic"));

    Tensor<type, 2> inputs  = language_dataset.get_data_variables("Input");
    const Index batch_size = inputs.dimension(0);

    Layer* first_layer = neural_network.get_layer(0).get();

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<EmbeddingForwardPropagation>(batch_size, first_layer);

    dimensions input_dims_vector(inputs.dimensions().begin(), inputs.dimensions().end());
    TensorView input_view(inputs.data(), input_dims_vector);

    first_layer->forward_propagate({ input_view }, forward_propagation, false);

    const TensorView embedding_output_view = forward_propagation->get_output_pair();

    ASSERT_EQ(embedding_output_view.dims.size(), 3);
    EXPECT_EQ(embedding_output_view.dims[0], batch_size); 
    EXPECT_EQ(embedding_output_view.dims[1], sequence_length);
    EXPECT_EQ(embedding_output_view.dims[2], embedding_dimension);
}
