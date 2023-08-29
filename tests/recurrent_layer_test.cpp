//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "recurrent_layer_test.h"

RecurrentLayerTest::RecurrentLayerTest() : UnitTesting()
{
}


RecurrentLayerTest::~RecurrentLayerTest()
{
}


void RecurrentLayerTest::test_constructor()
{
    cout << "test_constructor\n";

    Index inputs_number;
    Index neurons_number;

    Tensor<type, 2> synaptic_weights;
    Tensor<type, 2> recurrent_initializer;
    Tensor<type, 1> biases;

    // Test

    inputs_number = 1;
    neurons_number = 1;

    recurrent_layer.set(inputs_number, neurons_number);

    assert_true(recurrent_layer.get_parameters_number() == 3, LOG);
    assert_true(recurrent_layer.get_biases_number() == 1, LOG);

    // Test

    inputs_number = 2;
    neurons_number = 3;

    recurrent_layer.set(inputs_number, neurons_number);

    assert_true(recurrent_layer.get_parameters_number() == 18, LOG);
    assert_true(recurrent_layer.get_biases_number() == 3, LOG);
}


void RecurrentLayerTest::test_destructor()
{
    cout << "test_destructor\n";

    RecurrentLayer* recurrent_layer = new RecurrentLayer;
    delete recurrent_layer;
}


void RecurrentLayerTest::test_calculate_activations_derivatives()
{
    cout << "test_calculate_activation_derivative\n";

    Tensor<type, 1> parameters;
    Tensor<type, 2> inputs;
    Tensor<type, 2> combinations;
    Tensor<type, 2> activations;
    Tensor<type, 2> activations_derivatives;

    Tensor<Index, 1> combinations_dimensions;
    Tensor<Index, 1> activations_dimensions;
    Tensor<Index, 1> activations_derivatives_dimensions;

    Tensor<type, 1> numerical_activation_derivative;
    Tensor<type, 0> maximum_difference;

    // Test

    recurrent_layer.set(1, 1);
    combinations.resize(1,1);
    combinations.setZero();
    activations.resize(1,1);
    activations_derivatives.resize(1,1);

    combinations_dimensions = get_dimensions(combinations);
    activations_dimensions = get_dimensions(activations);
    activations_derivatives_dimensions = get_dimensions(activations_derivatives);

    recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction::Logistic);
    recurrent_layer.calculate_activations_derivatives(combinations.data(), combinations_dimensions, activations.data(), activations_dimensions, activations_derivatives.data(), activations_derivatives_dimensions);

    assert_true(activations_derivatives.rank() == 2, LOG);
    assert_true(activations_derivatives.dimension(0) == 1, LOG);
    assert_true(activations_derivatives.dimension(1) == 1, LOG);
    assert_true(activations_derivatives(0) - type(0.25) < type(NUMERIC_LIMITS_MIN), LOG);

    recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction::HyperbolicTangent);
    recurrent_layer.calculate_activations_derivatives(combinations.data(), combinations_dimensions, activations.data(), activations_dimensions, activations_derivatives.data(), activations_derivatives_dimensions);

    assert_true(activations_derivatives.rank() == 2, LOG);
    assert_true(activations_derivatives.dimension(0) == 1, LOG);
    assert_true(activations_derivatives.dimension(1) == 1, LOG);
    assert_true(activations_derivatives(0) - type(1.0) < type(NUMERIC_LIMITS_MIN), LOG);

    recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction::Linear);
    recurrent_layer.calculate_activations_derivatives(combinations.data(), combinations_dimensions, activations.data(), activations_dimensions, activations_derivatives.data(), activations_derivatives_dimensions);

    assert_true(activations_derivatives.rank() == 2, LOG);
    assert_true(activations_derivatives.dimension(0) == 1, LOG);
    assert_true(activations_derivatives.dimension(1) == 1, LOG);
    assert_true(activations_derivatives(0) - type(1.0) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    recurrent_layer.set(2, 4);

    combinations.resize(1,4);
    activations.resize(1,4);
    activations_derivatives.resize(1,4);

    combinations(0,0) = static_cast<type>(1.56);
    combinations(0,1) = static_cast<type>(-0.68);
    combinations(0,2) = static_cast<type>(0.91);
    combinations(0,3) = static_cast<type>(-1.99);

    combinations_dimensions = get_dimensions(combinations);
    activations_dimensions = get_dimensions(activations);
    activations_derivatives_dimensions = get_dimensions(activations_derivatives);

    recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction::Logistic);

    recurrent_layer.calculate_activations_derivatives(combinations.data(), combinations_dimensions,
                                                      activations.data(), activations_dimensions,
                                                      activations_derivatives.data(), activations_derivatives_dimensions);

    Tensor<type, 1> combinations_chip = combinations.chip(0,0);

    numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::get_activations, combinations_chip);

    for(Index i = 0; i < 4; i++)
    {
        assert_true(abs(activations_derivatives(i) - numerical_activation_derivative(i)) < 1.0e-3, LOG);
    }

    // Test

    recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction::HyperbolicTangent);

    recurrent_layer.calculate_activations_derivatives(combinations.data(), combinations_dimensions,
                                                      activations.data(), activations_dimensions,
                                                      activations_derivatives.data(), activations_derivatives_dimensions);

    combinations_chip = combinations.chip(0,0);

    numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::get_activations, combinations_chip);

    for(Index i = 0; i < 4; i++)
    {
        assert_true(abs(activations_derivatives(i) - numerical_activation_derivative(i)) < 1.0e-3, LOG);
    }

    // Test

    recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction::Linear);

    recurrent_layer.calculate_activations_derivatives(combinations.data(), combinations_dimensions,
                                                      activations.data(), activations_dimensions,
                                                      activations_derivatives.data(), activations_derivatives_dimensions);

    combinations_chip = combinations.chip(0,0);

    numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::get_activations, combinations_chip);

    for(Index i = 0; i < 4; i++)
    {
        assert_true(abs(activations_derivatives(i) - numerical_activation_derivative(i)) < 1.0e-3, LOG);
    }

    // Test

    recurrent_layer.set(4, 4);

    parameters.resize(14);
    parameters(0) = static_cast<type>(0.41);
    parameters(1) = static_cast<type>(-0.68);
    parameters(2) = static_cast<type>(0.14);
    parameters(3) = static_cast<type>(-0.50);
    parameters(4) = static_cast<type>(0.52);
    parameters(5) = static_cast<type>(-0.70);
    parameters(6) = static_cast<type>(0.85);
    parameters(7) = static_cast<type>(-0.18);
    parameters(8) = static_cast<type>(-0.65);
    parameters(9) = static_cast<type>(0.05);
    parameters(10) = static_cast<type>(0.85);
    parameters(11) = static_cast<type>(-0.18);
    parameters(12) = static_cast<type>(-0.65);
    parameters(13) = static_cast<type>(0.05);

    recurrent_layer.set_parameters(parameters);

    inputs.resize(1,4);
    inputs(0,0) = static_cast<type>(0.85);
    inputs(0,1) = static_cast<type>(-0.25);
    inputs(0,2) = static_cast<type>(0.29);
    inputs(0,3) = static_cast<type>(-0.77);

    recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction::Threshold);

    recurrent_layer.calculate_activations_derivatives(combinations.data(), combinations_dimensions,
                                                      activations.data(), activations_dimensions,
                                                      activations_derivatives.data(), activations_derivatives_dimensions);

    combinations_chip = combinations.chip(0,0);
    numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::get_activations, combinations_chip);

    for(Index i = 0; i < 4; i++)
        assert_true(abs(activations_derivatives(i) - numerical_activation_derivative(i)) < 1.0e-3, LOG);

    // Test

    recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction::SymmetricThreshold);

    recurrent_layer.calculate_activations_derivatives(combinations.data(), combinations_dimensions,
                                                      activations.data(), activations_dimensions,
                                                      activations_derivatives.data(), activations_derivatives_dimensions);

    combinations_chip = combinations.chip(0,0);
    numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::get_activations, combinations_chip);

    for(Index i = 0; i < 4; i++)
        assert_true(abs(activations_derivatives(i) - numerical_activation_derivative(i)) < 1.0e-3, LOG);

    // Test

    recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction::Logistic);

    recurrent_layer.calculate_activations_derivatives(combinations.data(), combinations_dimensions,
                                                      activations.data(), activations_dimensions,
                                                      activations_derivatives.data(), activations_derivatives_dimensions);

    combinations_chip = combinations.chip(0,0);
    numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::get_activations, combinations_chip);

    for(Index i = 0; i < 4; i++)
        assert_true(abs(activations_derivatives(i) - numerical_activation_derivative(i)) < 1.0e-3, LOG);


    // Test

    recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction::HyperbolicTangent);

    recurrent_layer.calculate_activations_derivatives(combinations.data(), combinations_dimensions,
                                                      activations.data(), activations_dimensions,
                                                      activations_derivatives.data(), activations_derivatives_dimensions);

    combinations_chip = combinations.chip(0,0);
    numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::get_activations, combinations_chip);

    for(Index i = 0; i < 4; i++)
        assert_true(abs(activations_derivatives(i) - numerical_activation_derivative(i)) < 1.0e-3, LOG);


    // Test

    recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction:: Linear);

    recurrent_layer.calculate_activations_derivatives(combinations.data(), combinations_dimensions,
                                                      activations.data(), activations_dimensions,
                                                      activations_derivatives.data(), activations_derivatives_dimensions);

    combinations_chip = combinations.chip(0,0);
    numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::get_activations, combinations_chip);

    for(Index i = 0; i < 4; i++)
        assert_true(abs(activations_derivatives(i) - numerical_activation_derivative(i)) < 1.0e-3, LOG);

}


void RecurrentLayerTest::test_forward_propagate()
{
    cout << "test_forward_propagate\n";
/*
    neurons_number = 4;
    samples_number = 2;
    inputs_number = 3;
    bool is_training = false;

    Tensor<type, 2> outputs;
    Tensor<Index, 1> outputs_dimensions;

    Tensor<type, 1> parameters;
    Tensor<type, 2> new_weights;
    Tensor<type, 2> new_recurrent_weights;
    Tensor<type, 1> new_biases;

    recurrent_layer.set(inputs_number, neurons_number);

    recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction::HyperbolicTangent);

    Tensor<type, 2> inputs(samples_number, inputs_number);
    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    recurrent_layer.set_parameters_constant(type(1));
    inputs.setConstant(type(1));

    recurrent_layer_forward_propagation.set(samples_number, &recurrent_layer);

    Tensor<type*, 1> inputs_data(1);
    inputs_data(0) = inputs.data();

    recurrent_layer.forward_propagate(inputs_data, inputs_dimensions, &recurrent_layer_forward_propagation, is_training);

    outputs = TensorMap<Tensor<type, 1>>(recurrent_layer_forward_propagation.outputs_data(0),
                                         recurrent_layer_forward_propagation.outputs_dimensions);

    assert_true(recurrent_layer_forward_propagation.combinations.rank() == 2, LOG);
    assert_true(recurrent_layer_forward_propagation.combinations.dimension(0) == samples_number, LOG);
    assert_true(recurrent_layer_forward_propagation.combinations.dimension(1) == neurons_number, LOG);

    // Test

    samples_number = 3;
    inputs_number = 2;
    neurons_number = 2;

    recurrent_layer.set(inputs_number, neurons_number);

    inputs.resize(samples_number,inputs_number);
    inputs.setConstant(type(1));
    inputs_dimensions = get_dimensions(inputs);

//    outputs.resize(samples_number, 2);
//    outputs_dimensions = get_dimensions(outputs);
    recurrent_layer.set_activation_function("SoftPlus");
    recurrent_layer.set_timesteps(3);

    new_weights.resize(2,2);
    new_weights.setConstant(type(1));
    new_recurrent_weights.resize(2,2);
    new_recurrent_weights.setConstant(type(1));
    new_biases.resize(2);
    new_biases.setConstant(type(1));

    recurrent_layer.set_biases(new_biases);
    recurrent_layer.set_input_weights(new_weights);
    recurrent_layer.set_recurrent_weights(new_recurrent_weights);

    parameters = recurrent_layer.get_parameters();

    recurrent_layer.forward_propagate(inputs.data(), inputs_dimensions, &recurrent_layer_forward_propagation, is_training);

    outputs = TensorMap<Tensor<type, 2>>(recurrent_layer_forward_propagation.outputs_data,
                                         recurrent_layer_forward_propagation.outputs_dimensions[0],
                                         recurrent_layer_forward_propagation.outputs_dimensions(1));
    */
}




void RecurrentLayerTest::run_test_case()
{
    cout << "Running recurrent layer test case...\n";

    // Constructor and destructor

    test_constructor();

    test_destructor();

    // Activation Derivatives

    test_calculate_activations_derivatives();

    // Forward propagate

    test_forward_propagate();

    cout << "End of recurrent layer test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
