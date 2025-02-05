#include "pch.h"

#include "../opennn/data_set.h"
#include "../opennn/mean_squared_error.h"
#include "../opennn/back_propagation.h"
#include "../opennn/convolutional_layer.h"
#include "../opennn/forward_propagation.h"
#include "../opennn/tensors.h"

TEST(MeanSquaredErrorTest, DefaultConstructor)
{
    MeanSquaredError mean_squared_error;

    EXPECT_EQ(mean_squared_error.has_neural_network(), false);
    EXPECT_EQ(mean_squared_error.has_data_set(), false);
}


TEST(MeanSquaredErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    DataSet data_set;
    MeanSquaredError mean_squared_error(&neural_network, &data_set);

    EXPECT_EQ(mean_squared_error.has_neural_network(), true);
    EXPECT_EQ(mean_squared_error.has_data_set(), true);
}


TEST(MeanSquaredErrorTest, BackPropagateEmpty)
{
    ForwardPropagation forward_propagation;

    BackPropagation back_propagation;
}


TEST(MeanSquaredErrorTest, BackPropagate)
{
    /*
    const Index samples_number = get_random_index(1, 10);
    const Index inputs_number = get_random_index(1, 10);
    const Index targets_number = get_random_index(1, 10);
    const Index neurons_number = get_random_index(1, 10);

    DataSet data_set(samples_number, { inputs_number }, { targets_number });
    data_set.set_data_random();
    data_set.set(DataSet::SampleUse::Training);

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, {1}, {1}, {1});

    neural_network.set_parameters_constant(type(0)); 

    Batch batch(samples_number, &data_set);
    batch.fill(data_set.get_sample_indices(DataSet::SampleUse::Training),

    data_set.get_variable_indices(DataSet::VariableUse::Input),
    data_set.get_variable_indices(DataSet::VariableUse::Decoder),
    data_set.get_variable_indices(DataSet::VariableUse::Target));
/*
    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation,
        { inputs_number }, { neurons_number }, { targets_number });
    
    neural_network.set_parameters_random();

    ForwardPropagation forward_propagation(samples_number, &neural_network);
    neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, true);

    // Loss index

    MeanSquaredError mean_squared_error(&neural_network, &data_set);

    BackPropagation back_propagation(samples_number, &mean_squared_error);    
    mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    const Tensor<type, 1> numerical_gradient = mean_squared_error.calculate_numerical_gradient();

    EXPECT_EQ(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)), true);
    */
}


TEST(MeanSquaredErrorTest, BackPropagateLm)
{
    const Index samples_number = get_random_index(1, 10);
    const Index inputs_number = get_random_index(1, 1);
    const Index outputs_number = get_random_index(1, 1);
    const Index neurons_number = get_random_index(1, 1);
/*
    // Data set

    DataSet data_set(samples_number, { inputs_number }, { outputs_number });
    data_set.set_data_random();
    data_set.set(DataSet::SampleUse::Training);

    Batch batch(samples_number, &data_set);
    batch.fill(data_set.get_sample_indices(DataSet::SampleUse::Training),
               data_set.get_variable_indices(DataSet::VariableUse::Input),
               data_set.get_variable_indices(DataSet::VariableUse::Decoder),
               data_set.get_variable_indices(DataSet::VariableUse::Target));

    // Neural network

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, 
                                { inputs_number }, { neurons_number }, { outputs_number });
    neural_network.set_parameters_random();

    ForwardPropagation forward_propagation(samples_number, &neural_network);
    neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, true);

    // Loss index

    MeanSquaredError mean_squared_error(&neural_network, &data_set);

    BackPropagation back_propagation(samples_number, &mean_squared_error);
    mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    BackPropagationLM back_propagation_lm(samples_number, &mean_squared_error);
    mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

    const Tensor<type, 2> numerical_jacobian = mean_squared_error.calculate_numerical_jacobian();
    const Tensor<type, 1> numerical_gradient = mean_squared_error.calculate_numerical_gradient();
    const Tensor<type, 2> numerical_hessian = mean_squared_error.calculate_numerical_hessian();

    EXPECT_NEAR(back_propagation_lm.error(), back_propagation.error(), type(1.0e-3));
    EXPECT_EQ(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian), true);
    EXPECT_EQ(are_equal(back_propagation_lm.gradient, numerical_gradient), true);
*/
}
