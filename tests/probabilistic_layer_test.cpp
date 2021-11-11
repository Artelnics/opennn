//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   T E S T   C L A S S           
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "probabilistic_layer_test.h"

ProbabilisticLayerTest::ProbabilisticLayerTest() : UnitTesting()
{

}


ProbabilisticLayerTest::~ProbabilisticLayerTest()
{

}


void ProbabilisticLayerTest::test_constructor()
{
   cout << "test_constructor\n";

   // Default constructor

   ProbabilisticLayer probabilistic_layer_1;

   assert_true(probabilistic_layer_1.get_inputs_number() == 0, LOG);
   assert_true(probabilistic_layer_1.get_neurons_number() == 0, LOG);
   assert_true(probabilistic_layer_1.get_biases_number() == 0, LOG);
   assert_true(probabilistic_layer_1.get_synaptic_weights_number() == 0, LOG);
   assert_true(probabilistic_layer_1.get_parameters_number() == 0, LOG);

   // Probabilistic neurons number constructor

   ProbabilisticLayer probabilistic_layer_2;

   probabilistic_layer_2.set_neurons_number(0);

   assert_true(probabilistic_layer_2.get_inputs_number() == 0, LOG);
   assert_true(probabilistic_layer_2.get_neurons_number() == 0, LOG);
   assert_true(probabilistic_layer_2.get_biases_number() == 0, LOG);
   assert_true(probabilistic_layer_2.get_synaptic_weights_number() == 0, LOG);
   assert_true(probabilistic_layer_2.get_parameters_number() == 0, LOG);

   ProbabilisticLayer probabilistic_layer_3;

   probabilistic_layer_3.set_neurons_number(3);

   assert_true(probabilistic_layer_3.get_inputs_number() == 0, LOG);
   assert_true(probabilistic_layer_3.get_neurons_number() == 3, LOG);
   assert_true(probabilistic_layer_3.get_biases_number() == 3, LOG);
   assert_true(probabilistic_layer_3.get_synaptic_weights_number() == 0, LOG);
   assert_true(probabilistic_layer_3.get_parameters_number() == 3, LOG);
}


void ProbabilisticLayerTest::test_set()
{
   cout << "test_set\n";

   probabilistic_layer.set();

   assert_true(probabilistic_layer.get_biases_number() == 0, LOG);
   assert_true(probabilistic_layer.get_synaptic_weights_number() == 0, LOG);
}


void ProbabilisticLayerTest::test_set_default()
{
   cout << "test_set_default\n";

   probabilistic_layer.set_neurons_number(2);

   probabilistic_layer.set_default();

   assert_true(probabilistic_layer.get_activation_function() == OpenNN::ProbabilisticLayer::ActivationFunction::Softmax, LOG);
   assert_true(abs(probabilistic_layer.get_decision_threshold() - type(0.5)) < type(NUMERIC_LIMITS_MIN), LOG);
   assert_true(probabilistic_layer.get_display(), LOG);

   probabilistic_layer.set_neurons_number(1);

   probabilistic_layer.set_default();

   assert_true(probabilistic_layer.get_activation_function() == OpenNN::ProbabilisticLayer::ActivationFunction::Logistic, LOG);
}


void ProbabilisticLayerTest::test_set_biases()
{
   cout << "test_set_biases\n";

   Tensor<type, 2> biases;

   // Test

   biases.resize(1, 4);

    probabilistic_layer.set(1, 4);

    biases.setZero();

    probabilistic_layer.set_biases(biases);

    assert_true(probabilistic_layer.get_biases_number() == 4, LOG);

    assert_true(abs(probabilistic_layer.get_biases()(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(probabilistic_layer.get_biases()(3)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void ProbabilisticLayerTest::test_set_synaptic_weights()
{
   cout << "test_set_synaptic_weights\n";

   Tensor<type, 2> synaptic_weights;

   // Test

    probabilistic_layer.set(1, 2);

    synaptic_weights.resize(2, 1);

    synaptic_weights.setZero();

    probabilistic_layer.set_synaptic_weights(synaptic_weights);

    assert_true(probabilistic_layer.get_synaptic_weights().size() == 2, LOG);

    assert_true(abs(probabilistic_layer.get_synaptic_weights()(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(probabilistic_layer.get_synaptic_weights()(1)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void ProbabilisticLayerTest::test_set_parameters()
{
  cout << "test_set_parameters\n";

    probabilistic_layer.set(1, 2);

    Tensor<type, 1> parameters(4);

    parameters.setValues({ type(11),type(12),type(21),type(22)});

    probabilistic_layer.set_parameters(parameters);

    assert_true(probabilistic_layer.get_biases()(0) - parameters(0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(probabilistic_layer.get_synaptic_weights()(0) - parameters(2)  < type(NUMERIC_LIMITS_MIN), LOG);
}


void ProbabilisticLayerTest::test_set_decision_threshold()
{
   cout << "test_set_decision_threshold\n";

   // Test

   probabilistic_layer.set_decision_threshold(static_cast<type>(0.7));

   assert_true(abs(probabilistic_layer.get_decision_threshold() - static_cast<type>(0.7)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void ProbabilisticLayerTest::test_set_activation_function()
{
   cout << "test_set_activation_function\n";

   probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);
   assert_true(probabilistic_layer.get_activation_function() == ProbabilisticLayer::ActivationFunction::Softmax, LOG);

   probabilistic_layer.set_activation_function("Softmax");
   assert_true(probabilistic_layer.get_activation_function() == ProbabilisticLayer::ActivationFunction::Softmax, LOG);
}


void ProbabilisticLayerTest::test_calculate_combinations()
{
   cout << "test_calculate_combinations\n";

   Tensor<type, 2> biases(1,1);
   Tensor<type, 2> synaptic_weights(1,1);
   Tensor<type, 1> parameters(1);

   Tensor<type, 2> inputs(1,1);
   Tensor<type, 2> combinations(1,1);

   biases.setConstant(type(1));
   synaptic_weights.setConstant(type(2));

   probabilistic_layer.set(1,1);
   inputs.setConstant(type(3));

   probabilistic_layer.calculate_combinations(inputs, biases, synaptic_weights, combinations);

   assert_true(combinations.rank() == 2, LOG);
   assert_true(combinations.dimension(0) == 1, LOG);
   assert_true(combinations.dimension(1) == 1, LOG);
   assert_true(abs(combinations(0,0) - type(7)) < static_cast<type>(1e-5) , LOG);

}


void ProbabilisticLayerTest::test_calculate_activations()
{
   cout << "test_calculate_activations\n";

   Tensor<type, 2> biases;
   Tensor<type, 2> synaptic_weights;
   Tensor<type, 1> parameters;

   Tensor<type, 2> inputs;
   Tensor<type, 2> combinations;
   Tensor<type, 2> activations;

   // Test

   inputs_number = 1;
   neurons_number = 1;
   samples_number = 1;

   probabilistic_layer.set(inputs_number, neurons_number);

   probabilistic_layer.set_parameters_constant(type(1));

   inputs.resize(samples_number, inputs_number);
   inputs.setConstant(type(-1));

   combinations.resize(samples_number, neurons_number);
   probabilistic_layer.calculate_combinations(inputs, probabilistic_layer.get_biases(), probabilistic_layer.get_synaptic_weights(), combinations);

   probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Binary);

   activations.resize(samples_number, neurons_number);
   probabilistic_layer.calculate_activations(combinations, activations);

   assert_true(activations.rank() == 2, LOG);
   assert_true(activations.dimension(0) == 1, LOG);
   assert_true(activations.dimension(1) == 1, LOG);
   assert_true(static_cast<Index>(activations(0,0)) == 1 , LOG);

   // Test

   probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Logistic);

   probabilistic_layer.calculate_activations(combinations, activations);

   assert_true(activations.rank() == 2, LOG);
   assert_true(activations.dimension(0) == 1, LOG);
   assert_true(activations.dimension(1) == 1, LOG);
   assert_true(activations(0,0) - static_cast<type>(0.5) < type(NUMERIC_LIMITS_MIN), LOG);

   // Test

   probabilistic_layer.set(2, 2);
   probabilistic_layer.set_parameters_constant(2);

   combinations.resize(1,2);
   combinations.setZero();

   activations.resize(1,2);
   activations.setZero();

   inputs.resize(1,2);
   inputs.setConstant(2);

   probabilistic_layer.calculate_combinations(inputs, probabilistic_layer.get_biases(), probabilistic_layer.get_synaptic_weights(), combinations);

   probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Competitive);

   probabilistic_layer.calculate_activations(combinations, activations);

   assert_true(activations.rank() == 2, LOG);
   assert_true(activations.dimension(0) == 1, LOG);
   assert_true(activations.dimension(1) == 2, LOG);
   assert_true(static_cast<Index>(activations(0,0)) == 1, LOG);

   probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);
   probabilistic_layer.calculate_activations(combinations, activations);
   assert_true(abs(activations(0,0) - static_cast<type>(0.5)) < static_cast<type>(1e-3), LOG);

   // Test

   probabilistic_layer.set(3, 3);

   combinations.resize(1,3);
   combinations.setValues({{1,0,-1}});

   probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Competitive);

   activations.resize(1,3);
   probabilistic_layer.calculate_activations(combinations, activations);

   assert_true(activations.rank() == 2, LOG);
   assert_true(activations.dimension(0) == 1, LOG);
   assert_true(activations.dimension(1) == 3, LOG);
   assert_true(static_cast<Index>(activations(0,0)) == 1, LOG);
   assert_true(static_cast<Index>(activations(0,1)) == 0, LOG);
   assert_true(static_cast<Index>(activations(0,2)) == 0, LOG);

   probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);
   probabilistic_layer.calculate_activations(combinations, activations);
   assert_true(abs(activations(0,0) - static_cast<type>(0.6652)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(activations(0,1) - static_cast<type>(0.2447)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(activations(0,2) - static_cast<type>(0.09)) < static_cast<type>(1e-3), LOG);

}


void ProbabilisticLayerTest::test_calculate_activations_derivatives()
{
    cout << "test_calculate_activations_derivatives\n";

    Tensor<type, 2> combinations;
    Tensor<type, 2> activations;
    Tensor<type, 3> activations_derivatives;

    // Test

    inputs_number = 1;
    neurons_number = 1;

    probabilistic_layer.set(inputs_number, neurons_number);

    combinations.resize(inputs_number, neurons_number);
    combinations.setValues({{type(-1.55f)}});
    activations.resize(inputs_number, neurons_number);
    activations_derivatives.resize(neurons_number, neurons_number, inputs_number);

    probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Logistic);
    probabilistic_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);

    assert_true(abs(activations(0,0) - static_cast<type>(0.175)) < static_cast<type>(1e-2), LOG);

    assert_true(activations_derivatives.rank() == 3, LOG);
    assert_true(activations_derivatives.dimension(0) == neurons_number, LOG);
    assert_true(activations_derivatives.dimension(1) == neurons_number, LOG);
    assert_true(activations_derivatives.dimension(2) == inputs_number, LOG);

    assert_true(abs(activations_derivatives(0,0,0) - static_cast<type>(0.1444)) < static_cast<type>(1e-3), LOG);

    // Test

    inputs_number = 2;
    neurons_number = 3;

    probabilistic_layer.set(inputs_number, neurons_number);

    combinations.resize(inputs_number, neurons_number);
    combinations.setValues({{type(1), type(2), type(3)}});
    activations.resize(inputs_number, neurons_number);
    activations_derivatives.resize(neurons_number, neurons_number, inputs_number);

    probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);
    probabilistic_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);

    assert_true(activations_derivatives.rank() == 3, LOG);
    assert_true(activations_derivatives.dimension(0) == neurons_number, LOG);
    assert_true(activations_derivatives.dimension(1) == neurons_number, LOG);
    assert_true(activations_derivatives.dimension(2) == inputs_number, LOG);

    assert_true(abs(activations(0,0) - static_cast<type>(0.09)) < static_cast<type>(1e-3), LOG);

    assert_true(abs(activations_derivatives(0,0,0) - static_cast<type>(0.0819)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(1,1,0) - static_cast<type>(0.1848)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(2,2,0) - static_cast<type>(0.2227)) < static_cast<type>(1e-3), LOG);

    // Test

    inputs_number = 1;
    neurons_number = 4;

    probabilistic_layer.set(inputs_number, neurons_number);

    combinations.resize(inputs_number, neurons_number);
    combinations.setValues({{type(-1), type(2), type(-3), type(-4)}});
    activations.resize(inputs_number, neurons_number);
    activations_derivatives.resize(neurons_number, neurons_number, inputs_number);

    probabilistic_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);

    assert_true(activations_derivatives.rank() == 3, LOG);
    assert_true(activations_derivatives.dimension(0) == neurons_number, LOG);
    assert_true(activations_derivatives.dimension(1) == neurons_number, LOG);
    assert_true(activations_derivatives.dimension(2) == inputs_number, LOG);

    assert_true(abs(activations_derivatives(3,0,0) + static_cast<type>(0.00011)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(3,1,0) + static_cast<type>(0.00221)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(3,2,0) + static_cast<type>(0.00001)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(3,3,0) - static_cast<type>(0.00233)) < static_cast<type>(1e-3), LOG);

}

void ProbabilisticLayerTest::test_calculate_outputs()
{
    cout << "test_calculate_outputs\n";

    Tensor<type, 2> synaptic_weights;
    Tensor<type, 2> biases;

    Tensor<type, 2> inputs;
    Tensor<type, 1> parameters;

    Tensor<type,2> outputs;

    // Test

    probabilistic_layer.set(3, 4);

    synaptic_weights.resize(3, 4);
    biases.resize(1, 4);
    inputs.resize(1, 3);

    inputs.setConstant(type(1));
    biases.setConstant(type(1));
    synaptic_weights.setValues({
        {type(1),type(-1),type(0),type(1)},
        {type(2),type(-2),type(0),type(2)},
        {type(3),type(-3),type(0),type(3)}});

    probabilistic_layer.set_synaptic_weights(synaptic_weights);
    probabilistic_layer.set_biases(biases);

    probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);

    outputs = probabilistic_layer.calculate_outputs(inputs);

    Tensor<type, 1> perceptron_sol(4);
    perceptron_sol.setValues({ type(7),type(-5),type(1),type(7)});

    Tensor<type,0>div = perceptron_sol.exp().sum();
    Tensor<type, 1>sol_ = perceptron_sol.exp() / div(0);

    assert_true(outputs.rank() == 2, LOG);
    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 4, LOG);
    assert_true(static_cast<Index>(outputs(0,0)) == static_cast<Index >(sol_(0)), LOG);
    assert_true(static_cast<Index>(outputs(1,0)) == static_cast<Index >(sol_(1)), LOG);
    assert_true(static_cast<Index>(outputs(2,0)) == static_cast<Index >(sol_(2)), LOG);

    // Test 1_2

    probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Competitive);

    outputs = probabilistic_layer.calculate_outputs(inputs);

    assert_true(outputs.rank() == 2, LOG);
    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 4, LOG);
    assert_true(static_cast<Index>(outputs(0,0)) == 1, LOG);
    assert_true(static_cast<Index>(outputs(1,0)) == 0, LOG);
    assert_true(static_cast<Index>(outputs(2,0)) == 0, LOG);

    // Test

    biases.resize(1, 4);
    biases.setValues({{type(9)},{type(-8)},{type(7)},{type(-6)}});

    synaptic_weights.resize(2, 4);

    synaptic_weights.resize(2, 4);

    synaptic_weights.setValues({
        {type(-11), type(12), type(-13), type(14)},
        {type(21), type(-22), type(23), type(-24)}});

    probabilistic_layer.set_synaptic_weights(synaptic_weights);
    probabilistic_layer.set_biases(biases);

    inputs.resize(1, 2);
    inputs.setConstant(type(1));

    probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);

    outputs = probabilistic_layer.calculate_outputs(inputs);

    Tensor<type, 1>perceptron_sol_3(4);
    perceptron_sol.setValues({type(7),type(-5),type(1),type(7)});

    div = perceptron_sol.exp().sum();
    sol_ = perceptron_sol.exp() / div(0);

    assert_true(outputs.rank() == 2, LOG);
    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 4, LOG);
    assert_true(static_cast<Index >(outputs(0,0)) == static_cast<Index >(sol_(0)), LOG);
    assert_true(static_cast<Index >(outputs(1,0)) == static_cast<Index >(sol_(1)), LOG);
    assert_true(static_cast<Index >(outputs(2,0)) == static_cast<Index >(sol_(2)), LOG);

   // Test  3

   probabilistic_layer.set(3, 2);
   probabilistic_layer.set_parameters_constant(type(0));

   inputs.resize(1,3);
   inputs.setConstant(type(0));

   outputs = probabilistic_layer.calculate_outputs(inputs);

   assert_true(outputs.rank() == 2, LOG);
   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 2, LOG);
   assert_true(abs(outputs(0,0) - static_cast<type>(0.5)) < type(NUMERIC_LIMITS_MIN), LOG);

}


void ProbabilisticLayerTest::test_forward_propagate()
{
    cout << "test_forward_propagate\n";

    inputs_number = 4;
    neurons_number = 2;
    samples_number = 1;

    ProbabilisticLayer probabilistic_layer(inputs_number,neurons_number);

    probabilistic_layer.set_parameters_constant(type(1));

    Tensor<type, 2> inputs(samples_number,inputs_number);
    inputs.setConstant(type(1));

    probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);


    //Forward propagate

    ProbabilisticLayerForwardPropagation probabilistic_layer_forward_propagation(samples_number, &probabilistic_layer);

    probabilistic_layer.forward_propagate(inputs, &probabilistic_layer_forward_propagation);

    assert_true(probabilistic_layer_forward_propagation.combinations.rank() == 2, LOG);
    assert_true(probabilistic_layer_forward_propagation.combinations.dimension(0) == 1, LOG);
    assert_true(probabilistic_layer_forward_propagation.combinations.dimension(1) == 2, LOG);
    assert_true(abs(probabilistic_layer_forward_propagation.combinations(0,0) - static_cast<type>(3)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(probabilistic_layer_forward_propagation.combinations(0,1) - static_cast<type>(3)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(probabilistic_layer_forward_propagation.activations(0,0) - static_cast<type>(0.5)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(probabilistic_layer_forward_propagation.activations(0,1) - static_cast<type>(0.5)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(probabilistic_layer_forward_propagation.activations_derivatives(0,0,0) - static_cast<type>(0.25)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(probabilistic_layer_forward_propagation.activations_derivatives(0,1,0) + static_cast<type>(0.25)) < static_cast<type>(1e-3), LOG);
}


void ProbabilisticLayerTest::run_test_case()
{
   cout << "Running probabilistic layer test case...\n";

   // Constructor and destructor methods

   test_constructor();

   // Set methods

   test_set();
   test_set_default();
   test_set_biases();
   test_set_synaptic_weights();
   test_set_parameters();
   test_set_decision_threshold();

   // Activation function

   test_set_activation_function();

   // Probabilistic post-processing

   test_calculate_combinations();
   test_calculate_activations();
   test_calculate_activations_derivatives();
   test_calculate_outputs();

   // Forward propagate

   test_forward_propagate();

   cout << "End of probabilistic layer test case.\n\n";
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
