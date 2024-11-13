#include "pch.h"

#include <iostream>

#include "../opennn/perceptron_layer.h"


TEST(PerceptronLayerTest, DefaultConstructor)
{
    PerceptronLayer perceptron_layer;

    EXPECT_EQ(perceptron_layer.get_type(), Layer::Type::Perceptron);
    EXPECT_EQ(perceptron_layer.get_input_dimensions(), dimensions{ 0 });
    EXPECT_EQ(perceptron_layer.get_output_dimensions(), dimensions{ 0 });
}


TEST(PerceptronLayerTest, GeneralConstructor)
{
    PerceptronLayer perceptron_layer({10}, {3}, PerceptronLayer::ActivationFunction::Linear);
    
    EXPECT_EQ(perceptron_layer.get_activation_function_string(), "Linear");
    EXPECT_EQ(perceptron_layer.get_input_dimensions(), dimensions{ 10 });
    EXPECT_EQ(perceptron_layer.get_output_dimensions(), dimensions{ 3 });
    EXPECT_EQ(perceptron_layer.get_parameters_number(), 33);
}


TEST(PerceptronLayerTest, CombinationsZero)
{
    PerceptronLayer perceptron_layer({1}, {1});
    perceptron_layer.set_parameters_constant(type(0));

    Tensor<type, 2> inputs(1, 1);
    inputs.setZero();

    Tensor<type, 2> combinations(1, 1);
    combinations.setConstant(type(1));

    perceptron_layer.calculate_combinations(inputs, combinations);

    EXPECT_EQ(combinations(0,0), 0);
}


TEST(PerceptronLayerTest, Activations)
{
    PerceptronLayer perceptron_layer({ 1 }, { 1 });
    perceptron_layer.set_parameters_constant(type(1));

    Tensor<type, 2> activations(1, 1);
    Tensor<type, 2> activation_derivatives(1, 1);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Logistic);
    activations.setConstant(type(1));
    perceptron_layer.calculate_activations(activations, activation_derivatives);

    EXPECT_NEAR(activations(0, 0), type(0.731), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(0.196), 0.001);
    
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::HyperbolicTangent);
    activations.setConstant(type(1));
    perceptron_layer.calculate_activations(activations, activation_derivatives);

    EXPECT_NEAR(activations(0, 0), type(0.761), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(0.41997), 0.001);
    
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);
    activations.setConstant(type(1));
    perceptron_layer.calculate_activations(activations, activation_derivatives);

    EXPECT_NEAR(activations(0, 0), type(1), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(1), 0.001);
    
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::RectifiedLinear);
    activations.setConstant(type(1));
    perceptron_layer.calculate_activations(activations, activation_derivatives);

    EXPECT_NEAR(activations(0, 0), type(1), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(1), 0.001);
    
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::ExponentialLinear);
    activations.setConstant(type(1));
    perceptron_layer.calculate_activations(activations, activation_derivatives);

    EXPECT_NEAR(activations(0, 0), type(1), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(1), 0.001);
    
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::ScaledExponentialLinear);
    activations.setConstant(type(1));
    perceptron_layer.calculate_activations(activations, activation_derivatives);

    EXPECT_NEAR(activations(0, 0), type(1.05), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(1.05), 0.001);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SoftPlus);
    activations.setConstant(type(1));
    perceptron_layer.calculate_activations(activations, activation_derivatives);

    EXPECT_NEAR(activations(0, 0), type(1.313), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(0.731), 0.001);
    /*
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SoftSign);
    activations.setConstant(type(1));
    perceptron_layer.calculate_activations(activations, activation_derivatives);

    EXPECT_NEAR(activations(0, 0), type(0.5), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(0.25), 0.001);
    */
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::HardSigmoid);
    activations.setConstant(type(1));
    perceptron_layer.calculate_activations(activations, activation_derivatives);

    EXPECT_NEAR(activations(0, 0), type(0.7), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(0.2), 0.001);

}


TEST(PerceptronLayerTest, ForwardPropagateZero)
{
    PerceptronLayer perceptron_layer({ 1 }, { 1 }, PerceptronLayer::ActivationFunction::Linear);
    perceptron_layer.set_parameters_constant(type(0));
/*
    unique_ptr<PerceptronLayerForwardPropagation> perceptron_layer_forward_propagation_x 
        = make_unique<PerceptronLayerForwardPropagation>(1, &perceptron_layer);

    perceptron_layer.forward_propagate({ make_pair(inputs.data(), dimensions{1, 1}) },
        perceptron_layer_forward_propagation,
        is_training);

    Tensor<type, 2> inputs(1, 1);
    inputs.setConstant(type(0));

    Tensor<type, 1> parameters;

    Tensor<type, 2> outputs;

    Tensor<type, 1> potential_parameters;

    pair<type*, dimensions> input_pairs;

    // Test

    bool is_training = true;

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    outputs = perceptron_layer_forward_propagation.outputs;

    assert_true(abs(outputs(0,0) - type(3)) < type(1e-3), LOG);
    assert_true(abs(outputs(0,1) - type(3)) < type(1e-3), LOG);

    assert_true(abs(perceptron_layer_forward_propagation.activation_derivatives(0,0) - type(1)) < type(1e-3), LOG);
    assert_true(abs(perceptron_layer_forward_propagation.activation_derivatives(0,1) - type(1)) < type(1e-3), LOG);

    EXPECT_EQ(perceptron_layer.get_type(), Layer::Type::Perceptron);
    EXPECT_EQ(perceptron_layer.get_input_dimensions(), dimensions{ 0 });
    EXPECT_EQ(perceptron_layer.get_output_dimensions(), dimensions{ 0 });
*/
}


/*

void PerceptronLayerTest::test_calculate_combinations()
{

    // Test

    perceptron_layer->set_parameters_constant(1);

    inputs.setConstant(type(3));

    perceptron_layer->calculate_combinations(inputs, combinations);

    assert_true(combinations.rank() == 2, LOG);
    assert_true(combinations.dimension(0) == 1, LOG);
    assert_true(abs(combinations(0,0) - type(7)) < type(1e-5), LOG);

    // Test

    inputs_number = 2;
    neurons_number = 2;
    samples_number = 1;

    perceptron_layer->set(inputs_number, neurons_number);
    perceptron_layer->set_parameters_constant(type(1));

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(1));

    combinations.resize(samples_number, neurons_number);
    combinations.setZero();

    perceptron_layer->calculate_combinations(inputs, combinations);

    assert_true(combinations.rank() == 2, LOG);
    assert_true(combinations.dimension(0) == samples_number, LOG);
    assert_true(combinations.dimension(1) == neurons_number, LOG);
    assert_true(abs(combinations(0,0) - type(3)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    combinations.resize(2, 4);
    combinations.setZero();

    perceptron_layer->set(3, 4);

    perceptron_layer->set_parameters_constant(1);

    inputs.resize(2,3);
    inputs.setConstant(type(0.5));

    perceptron_layer->set_parameters_constant(1);

    perceptron_layer->calculate_combinations(inputs, combinations);

    assert_true(combinations.rank() == 2, LOG);
    assert_true(combinations.dimension(0) == 2, LOG);
    assert_true(combinations.dimension(1) == 4, LOG);
    assert_true(abs(combinations(0,0) - type(3.5)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    inputs_number = 2;
    neurons_number = 4;
    samples_number = 1;

    perceptron_layer->set(inputs_number, neurons_number);

    perceptron_layer->set_parameters_constant(1);

    inputs.resize(samples_number, inputs_number);
    inputs.setValues({{type(0.5), type(0.5)}});

    combinations.resize(samples_number, neurons_number);
    combinations.setZero();

    perceptron_layer->calculate_combinations(inputs, combinations);

    assert_true(combinations.rank() == 2, LOG);
    assert_true(combinations.dimension(0) == samples_number, LOG);
    assert_true(combinations.dimension(1) == neurons_number, LOG);
    assert_true(Index(combinations(0,0)) == 2, LOG);

    // Test

    inputs_number = 1;
    neurons_number = 1;
    samples_number = 1;

    perceptron_layer->set(inputs_number, neurons_number, PerceptronLayer::ActivationFunction::HyperbolicTangent);

    perceptron_layer->set_parameters_constant(1);

    parameters_number = perceptron_layer->get_parameters_number();

    assert_true(parameters_number == 2, LOG);

    const Tensor<type, 1> parameters = perceptron_layer->get_parameters();

    assert_true(abs(parameters(0) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(parameters(1) - type(-0.5)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    inputs_number = 1;
    neurons_number = 1;
    samples_number = 1;

    perceptron_layer->set(inputs_number, neurons_number);

    inputs.resize(samples_number, inputs_number);
    inputs.setZero();

    perceptron_layer->set_parameters_constant(0);

    combinations.resize(samples_number, neurons_number);

    perceptron_layer->calculate_combinations(inputs, combinations);

    assert_true(combinations.rank() == 2, LOG);
    assert_true(combinations.dimension(0) == 1, LOG);
    assert_true(combinations.dimension(1) == 1, LOG);

    assert_true(abs(combinations(0,0)) < type(NUMERIC_LIMITS_MIN), LOG);

}


void PerceptronLayerTest::test_forward_propagate()
{
    Tensor<type, 1> parameters;
    Tensor<type, 2> inputs;
    Tensor<type, 2> outputs;

    Tensor<type, 1> potential_parameters;

    pair<type*, dimensions> input_pairs;

    // Test

    samples_number = 2;
    inputs_number = 2;
    neurons_number = 2;
    bool is_training = true;
    /*
    perceptron_layer.set(inputs_number, neurons_number, PerceptronLayer::ActivationFunction::Linear);
    perceptron_layer.set_parameters_constant(type(1));

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(1));

    perceptron_layer_forward_propagation->set(samples_number, perceptron_layer.get());

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    perceptron_layer->forward_propagate({input_pairs},
                                        perceptron_layer_forward_propagation,
                                        is_training);

    outputs = perceptron_layer_forward_propagation.outputs;

    assert_true(abs(outputs(0,0) - type(3)) < type(1e-3), LOG);
    assert_true(abs(outputs(0,1) - type(3)) < type(1e-3), LOG);

    assert_true(abs(perceptron_layer_forward_propagation.activation_derivatives(0,0) - type(1)) < type(1e-3), LOG);
    assert_true(abs(perceptron_layer_forward_propagation.activation_derivatives(0,1) - type(1)) < type(1e-3), LOG);

    // Test

    samples_number = 2;
    inputs_number = 2;
    neurons_number = 2;

    perceptron_layer.set(inputs_number, neurons_number, PerceptronLayer::ActivationFunction::HyperbolicTangent);
    perceptron_layer.set_parameters_constant(type(1));

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(1));    

    potential_parameters = perceptron_layer.get_parameters();

    perceptron_layer_forward_propagation.set(samples_number, &perceptron_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    perceptron_layer.forward_propagate({input_pairs}, &perceptron_layer_forward_propagation, is_training);

    outputs = perceptron_layer_forward_propagation.outputs;

    assert_true(abs(outputs(0,0) - type(0.99505)) < type(1e-3), LOG);
    assert_true(abs(outputs(0,1) - type(0.99505)) < type(1e-3), LOG);
    assert_true(abs(perceptron_layer_forward_propagation.activation_derivatives(0,0) - type(0.00986)) < type(1e-3), LOG);
    assert_true(abs(perceptron_layer_forward_propagation.activation_derivatives(0,1) - type(0.00986)) < type(1e-3), LOG);

    // Test

    samples_number = 1;
    inputs_number = 3;
    neurons_number = 4;

    perceptron_layer.set(inputs_number, neurons_number);

    synaptic_weights.resize(inputs_number, neurons_number);
    biases.resize( neurons_number);
    inputs.resize(samples_number, inputs_number);
    outputs.resize(1, neurons_number);

    inputs.setConstant(type(1));
    biases.setConstant(type(1));
    synaptic_weights.setValues({{type(1),type(-1),type(0),type(1)},
                                {type(2),type(-2),type(0),type(2)},
                                {type(3),type(-3),type(0),type(3)}});

    perceptron_layer.set_synaptic_weights(synaptic_weights);
    perceptron_layer.set_biases(biases);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);

    perceptron_layer_forward_propagation.set(samples_number, &perceptron_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    perceptron_layer.forward_propagate({input_pairs}, &perceptron_layer_forward_propagation, is_training);

    outputs = perceptron_layer_forward_propagation.outputs;

    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 4, LOG);
    assert_true(Index(outputs(0,0)) == 7, LOG);
    assert_true(Index(outputs(1,0)) == -5, LOG);
    assert_true(Index(outputs(2,0)) == 1, LOG);

    // Test

    inputs_number = 3;
    neurons_number = 2;

    perceptron_layer.set(inputs_number, neurons_number);
    perceptron_layer.set_parameters_constant(type(0));

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(0));

    outputs.resize(1, neurons_number);

    perceptron_layer_forward_propagation.set(samples_number, &perceptron_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    perceptron_layer.forward_propagate({input_pairs}, &perceptron_layer_forward_propagation, is_training);

    outputs = perceptron_layer_forward_propagation.outputs;

    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 2, LOG);
    assert_true(abs(outputs(0,0)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    inputs_number = 4;
    neurons_number = 2;

    perceptron_layer.set(4, 2);
    parameters.resize(10);

    parameters.setValues({type(-1),type(2),type(-3),type(4),type(-5),type(6),type(-7),type(8),type(-9),type(10) });

    perceptron_layer.set_parameters(parameters);

    inputs.resize(samples_number,inputs_number);
    inputs.setValues({{type(4),type(-3),type(2),type(-1)}});

    outputs.resize(1, neurons_number);

    perceptron_layer_forward_propagation.set(samples_number, &perceptron_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    perceptron_layer.forward_propagate({input_pairs}, &perceptron_layer_forward_propagation, is_training);

    outputs = perceptron_layer_forward_propagation.outputs;

    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 2, LOG);
    assert_true(abs(outputs(0,0) + type(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test 5

    inputs_number = 1;
    neurons_number = 2;

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(3.0));

    perceptron_layer.set(inputs_number, neurons_number);
    perceptron_layer.set_parameters_constant(type(-2.0));

    outputs.resize(1, neurons_number); 

    perceptron_layer_forward_propagation.set(samples_number, &perceptron_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    perceptron_layer.forward_propagate({input_pairs}, &perceptron_layer_forward_propagation, is_training);

    outputs = perceptron_layer_forward_propagation.outputs;
    parameters.resize(2);
    parameters.setConstant(type(1));

    // Test

    perceptron_layer.set(1, 1);

    inputs.resize(1,1);
    inputs.setRandom();

    parameters = perceptron_layer.get_parameters();

}

}
*/