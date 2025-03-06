#include "pch.h"

#include "../opennn/cross_entropy_error.h"
#include "../opennn/forward_propagation.h"
#include "../opennn/back_propagation.h"
#include "../opennn/tensors.h"


TEST(CrossEntropyErrorTest, DefaultConstructor)
{

    CrossEntropyError cross_entropy_error;

    EXPECT_TRUE(!cross_entropy_error.has_data_set());
    EXPECT_TRUE(!cross_entropy_error.has_neural_network());

}


TEST(CrossEntropyErrorTest, BackPropagateEmpty)
{

    DataSet data_set;

    Batch batch;

    NeuralNetwork neural_network;

    ForwardPropagation forward_propagation;

    CrossEntropyError cross_entropy_error(&neural_network, &data_set);
    BackPropagation back_propagation;

    cross_entropy_error.back_propagate(batch, forward_propagation, back_propagation);

}


TEST(CrossEntropyErrorTest, BackPropagate)
{

    const Index samples_number = get_random_index(1, 10);
    const Index inputs_number = get_random_index(1, 10);
    const Index targets_number = get_random_index(1, 10);
    const Index neurons_number = get_random_index(1, 10);

    DataSet data_set(samples_number, { inputs_number }, { targets_number });
    
    data_set.set_data_classification();

    data_set.set(DataSet::SampleUse::Training);

    Batch batch(samples_number, &data_set);
    batch.fill(data_set.get_sample_indices(DataSet::SampleUse::Training),
               data_set.get_variable_indices(DataSet::VariableUse::Input),
               data_set.get_variable_indices(DataSet::VariableUse::Decoder),
               data_set.get_variable_indices(DataSet::VariableUse::Target));

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification,
        { inputs_number }, { neurons_number }, { targets_number });

    neural_network.set_parameters_random();

    ForwardPropagation forward_propagation(samples_number, &neural_network);

    neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, true);

    // Loss index
    
    CrossEntropyError cross_entropy_error(&neural_network, &data_set);

    BackPropagation back_propagation(samples_number, &cross_entropy_error);
    
    cross_entropy_error.back_propagate(batch, forward_propagation, back_propagation);

    const Tensor<type, 1> numerical_gradient = cross_entropy_error.calculate_numerical_gradient();

    EXPECT_EQ(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-1)), true);

}
