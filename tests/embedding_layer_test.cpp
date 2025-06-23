#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/language_dataset.h"
#include "../opennn/embedding_layer.h"
#include "../opennn/flatten_layer_3d.h"
#include "../opennn/perceptron_layer.h"
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


TEST(Embedding, BackPropagate)
{
    const Index samples_number = get_random_index(2, 10);
    const Index sequence_length = get_random_index(1, 10);
    const Index vocabulary_size = get_random_index(1, 10);
    const Index embedding_dimension = get_random_index(1, 10);

    LanguageDataset language_dataset(samples_number, sequence_length, vocabulary_size);
    language_dataset.set(Dataset::SampleUse::Training);

    //language_dataset.print_data();
    //exit(0);

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Embedding>(language_dataset.get_input_dimensions(), embedding_dimension));
    neural_network.add_layer(make_unique<Flatten3d>(neural_network.get_output_dimensions()));
    neural_network.add_layer(make_unique<Dense2d>(neural_network.get_output_dimensions(), language_dataset.get_target_dimensions(), Dense2d::Activation::Logistic));

    Tensor<type, 2> inputs = language_dataset.get_data(Dataset::VariableUse::Input);
/*
    Tensor<type, 2> outputs = neural_network.calculate_outputs<2,2>(inputs);

    cout << outputs << endl;exit(0);

    MeanSquaredError mean_squared_error(&neural_network, &language_dataset);

    cout << mean_squared_error.calculate_numerical_error() << endl;
/*
    //cout << (mean_squared_error.calculate_gradient().abs() - mean_squared_error.calculate_numerical_gradient().abs()).maximum()<< endl;



    Embedding embedding_layer({ vocabulary_size, sequence_length }, embedding_dimension);
    embedding_layer.set_parameters_random();

    Tensor<type, 2> inputs(samples_number, sequence_length);
    inputs.setConstant(type(0));

    Tensor<type, 3> outputs = neural_network.calculate_outputs<2,3>(inputs);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), sequence_length);
    EXPECT_EQ(outputs.dimension(2), embedding_dimension);
*/
}

