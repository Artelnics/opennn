#include "pch.h"

#include <iostream>

#include "../opennn/tensors.h"
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


TEST(PerceptronLayerTest, Combinations)
{
    /*
    const Index samples_number = get_random_index(1, 10);
    const Index inputs_number = get_random_index(1, 10);
    const Index outputs_number = get_random_index(1, 10);

    PerceptronLayer perceptron_layer({inputs_number}, {outputs_number});
    perceptron_layer.set_parameters_constant(type(0));

    Tensor<type, 2> inputs(samples_number, inputs_number);
    inputs.setZero();

    Tensor<type, 2> combinations(samples_number, outputs_number);

    perceptron_layer.calculate_combinations(inputs, combinations);

    EXPECT_EQ(is_equal(combinations, type(0)), true);
    */
}


TEST(PerceptronLayerTest, Activations)
{
    /*
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
    
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::HardSigmoid);
    activations.setConstant(type(1));
    perceptron_layer.calculate_activations(activations, activation_derivatives);

    EXPECT_NEAR(activations(0, 0), type(0.7), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(0.2), 0.001);
    */
}


TEST(PerceptronLayerTest, ForwardPropagateZero)
{
    /*
    PerceptronLayer perceptron_layer({ 1 }, { 1 }, PerceptronLayer::ActivationFunction::Linear);
    perceptron_layer.set_parameters_constant(type(0));

    unique_ptr<LayerForwardPropagation> perceptron_layer_forward_propagation 
        = make_unique<PerceptronLayerForwardPropagation>(1, &perceptron_layer);

    Tensor<type, 2> inputs(1, 1);
    inputs.setConstant(type(0));

    perceptron_layer.forward_propagate({ make_pair(inputs.data(), dimensions{1, 1}) },
        perceptron_layer_forward_propagation,
        true);

    Tensor<type, 1> parameters;

    Tensor<type, 2> outputs;

    Tensor<type, 1> potential_parameters;

    pair<type*, dimensions> input_pairs;

    // Test
/*
    bool is_training = true;

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    outputs = perceptron_layer_forward_propagation.outputs;

    EXPECT_EQ(abs(outputs(0,0) - type(3)) < type(1e-3));
    EXPECT_EQ(abs(outputs(0,1) - type(3)) < type(1e-3));

    EXPECT_EQ(abs(perceptron_layer_forward_propagation.activation_derivatives(0,0) - type(1)) < type(1e-3));
    EXPECT_EQ(abs(perceptron_layer_forward_propagation.activation_derivatives(0,1) - type(1)) < type(1e-3));

    EXPECT_EQ(perceptron_layer.get_type(), Layer::Type::Perceptron);
    EXPECT_EQ(perceptron_layer.get_input_dimensions(), dimensions{ 0 });
    EXPECT_EQ(perceptron_layer.get_output_dimensions(), dimensions{ 0 });
*/
}


TEST(PerceptronLayerTest, ForwardPropagate)
{

    Tensor<type, 1> parameters;
    Tensor<type, 2> inputs;
    Tensor<type, 2> outputs;

    Tensor<type, 1> potential_parameters;

    ;

    const Index samples_number = 2;
    const Index inputs_number = 2;
    const Index neurons_number = 2;
    bool is_training = true;

    PerceptronLayer perceptron_layer({ inputs_number }, 
                                     { neurons_number }, 
                                     PerceptronLayer::ActivationFunction::Linear);
    perceptron_layer.set_parameters_constant(type(1));

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(1));

    PerceptronLayerForwardPropagation perceptron_layer_forward_propagation(samples_number, &perceptron_layer);

    const pair<type*, dimensions> input_pairs = {inputs.data(), {{samples_number, inputs_number}}};
/*
    perceptron_layer->forward_propagate({input_pairs},
                                        perceptron_layer_forward_propagation,
                                        is_training);

    outputs = perceptron_layer_forward_propagation.outputs;

    EXPECT_EQ(abs(outputs(0,0) - type(3)) < type(1e-3));
    EXPECT_EQ(abs(outputs(0,1) - type(3)) < type(1e-3));

    EXPECT_EQ(abs(perceptron_layer_forward_propagation.activation_derivatives(0,0) - type(1)) < type(1e-3));
    EXPECT_EQ(abs(perceptron_layer_forward_propagation.activation_derivatives(0,1) - type(1)) < type(1e-3));
*/
}
