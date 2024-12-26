#include "pch.h"
#include "../opennn/recurrent_layer.h"

TEST(RecurrentLayerTest, DefaultConstructor)
{
    RecurrentLayer recurrent_layer;

//    EXPECT_EQ(quasi_newton_method.has_loss_index(), false);
}


TEST(RecurrentLayerTest, GeneralConstructor)
{

    RecurrentLayer recurrent_layer;

    Index inputs_number;
    Index neurons_number;
    Index time_steps;
/*
    // Test

    inputs_number = 1;
    neurons_number = 1;
    time_steps = 1;

    recurrent_layer.set(inputs_number, neurons_number, time_steps);

    EXPECT_EQ(recurrent_layer.get_parameters_number() == 3);

    // Test

    inputs_number = 2;
    neurons_number = 3;

    recurrent_layer.set(inputs_number, neurons_number, time_steps);

    EXPECT_EQ(recurrent_layer.get_parameters_number() == 18);

    EXPECT_EQ(quasi_newton_method.has_loss_index(), true);
*/
}


TEST(RecurrentLayerTest, ForwardPropagate)
{

    const Index neurons_number = 4;
    const Index samples_number = 2;
    const Index inputs_number = 3;
    const Index time_steps = 1;
    bool is_training = false;

    Tensor<type, 2> outputs;

    Tensor<type, 1> parameters;
    Tensor<type, 2> new_weights;
    Tensor<type, 2> new_recurrent_weights;
    Tensor<type, 1> new_biases;

    pair<type*, dimensions> input_pairs;
    /*
    recurrent_layer.set(inputs_number, neurons_number, time_steps);

    recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction::HyperbolicTangent);

    Tensor<type, 2> inputs(samples_number, inputs_number);

    recurrent_layer.set_parameters_constant(type(1));
    inputs.setConstant(type(1));

    recurrent_layer_forward_propagation.set(samples_number, &recurrent_layer);

    Tensor<type*, 1> input_data(1);
    input_data(0) = inputs.data();

    input_pairs = { inputs.data(), {{samples_number, inputs_number}} };

    recurrent_layer.forward_propagate({ input_pairs }, &recurrent_layer_forward_propagation, is_training);

    outputs = recurrent_layer_forward_propagation.outputs;

    // Test

    samples_number = 3;
    inputs_number = 2;
    neurons_number = 2;
    time_steps = 1;

    recurrent_layer.set(inputs_number, neurons_number, time_steps);

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(1));

    recurrent_layer.set_activation_function("SoftPlus");
    recurrent_layer.set_timesteps(3);

    new_weights.resize(2, 2);
    new_weights.setConstant(type(1));
    new_recurrent_weights.resize(2, 2);
    new_recurrent_weights.setConstant(type(1));
    new_biases.resize(2);
    new_biases.setConstant(type(1));

    recurrent_layer.set_biases(new_biases);
    recurrent_layer.set_input_weights(new_weights);
    recurrent_layer.set_recurrent_weights(new_recurrent_weights);

    parameters = recurrent_layer.get_parameters();

    input_pairs = { inputs.data(), {{samples_number, inputs_number}} };

    recurrent_layer.forward_propagate({ input_pairs }, &recurrent_layer_forward_propagation, is_training);

    outputs = recurrent_layer_forward_propagation.outputs;
*/
}

/*

void RecurrentLayerTest::test_calculate_activations()
{
    Tensor<type, 1> combinations;
    Tensor<type, 1> activations;
    Tensor<type, 1> activation_derivatives;

    Tensor<type, 1> numerical_activation_derivative;
    Tensor<type, 0> maximum_difference;

}

*/
