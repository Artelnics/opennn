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


void RecurrentLayerTest::test_calculate_activations_derivatives()
{
   cout << "test_calculate_activation_derivative\n";

   Tensor<type, 1> parameters;
   Tensor<type, 2> inputs;
   Tensor<type, 2> combinations;
   Tensor<type, 2> activations;

   Tensor<type, 2> activations_derivatives;
   Tensor<type, 2> numerical_activation_derivative;
   Tensor<type, 0> maximum_difference;

   // Test

   recurrent_layer.set(1, 1);
   combinations.resize(1,1);
   combinations.setZero();
   activations.resize(1,1);
   activations_derivatives.resize(1,1);

   recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction::Logistic);
//   recurrent_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);
   assert_true(activations_derivatives.rank() == 2, LOG);
   assert_true(activations_derivatives.dimension(0) == 1, LOG);
   assert_true(activations_derivatives.dimension(1) == 1, LOG);
   assert_true(activations_derivatives(0) == type(0.25), LOG);

   recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction::HyperbolicTangent);
//   recurrent_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);
   assert_true(activations_derivatives.rank() == 2, LOG);
   assert_true(activations_derivatives.dimension(0) == 1, LOG);
   assert_true(activations_derivatives.dimension(1) == 1, LOG);
   assert_true(activations_derivatives(0) == type(1.0), LOG);

   recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction::Linear);
//   recurrent_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);
   assert_true(activations_derivatives.rank() == 2, LOG);
   assert_true(activations_derivatives.dimension(0) == 1, LOG);
   assert_true(activations_derivatives.dimension(1) == 1, LOG);
   assert_true(activations_derivatives(0) == type(1.0), LOG);

   // Test

      recurrent_layer.set(2, 4);

      combinations.resize(1,4);
      combinations(0,0) = static_cast<type>(1.56);
      combinations(0,2) = static_cast<type>(-0.68);
      combinations(0,2) = static_cast<type>(0.91);
      combinations(0,3) = static_cast<type>(-1.99);

      recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction::Logistic);

//      recurrent_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);

//      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::calculate_activations, combinations);

//      assert_true((activations_derivatives - numerical_activation_derivative).abs() < 1.0e-3, LOG);

//      recurrent_layer.set_activation_function(RecurrentLayer::HyperbolicTangent);

//      activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations);

//      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::calculate_activations, combinations);

//      assert_true(absolute_value(activations_derivatives - numerical_activation_derivative) < 1.0e-3, LOG);

//      recurrent_layer.set_activation_function(RecurrentLayer::Linear);

//      activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations);

//      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::calculate_activations, combinations);

//      assert_true(absolute_value(activations_derivatives - numerical_activation_derivative) < 1.0e-3, LOG);


   // Test

      recurrent_layer.set(4, 2);

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

//      recurrent_layer.set_activation_function(RecurrentLayer::Threshold);
//      activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations);
//      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::calculate_activations, combinations);
//      assert_true(absolute_value((activations_derivatives - numerical_activation_derivative)) < 1.0e-3, LOG);

//      recurrent_layer.set_activation_function(RecurrentLayer::SymmetricThreshold);
//      activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations);
//      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::calculate_activations, combinations);
//      assert_true(absolute_value((activations_derivatives - numerical_activation_derivative)) < 1.0e-3, LOG);

//      recurrent_layer.set_activation_function(RecurrentLayer::Logistic);
//      activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations);
//      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::calculate_activations, combinations);
//      assert_true(absolute_value((activations_derivatives - numerical_activation_derivative)) < 1.0e-3, LOG);

//      recurrent_layer.set_activation_function(RecurrentLayer::HyperbolicTangent);
//      activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations);
//      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::calculate_activations, combinations);
//      assert_true(absolute_value((activations_derivatives - numerical_activation_derivative)) < 1.0e-3, LOG);

//      recurrent_layer.set_activation_function(RecurrentLayer::Linear);
//      activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations);
//      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::calculate_activations, combinations);
//      assert_true(absolute_value((activations_derivatives - numerical_activation_derivative)) < 1.0e-3, LOG);
}


void RecurrentLayerTest::test_calculate_outputs()
{
   cout << "test_calculate_outputs\n";

   Tensor<type, 2> inputs;
   Tensor<type, 2> outputs;

   Tensor<type, 1> parameters;

   Index samples;

   Tensor<type, 2> new_weights;
   Tensor<type, 2> new_recurrent_weights;
   Tensor<type, 1> new_biases;

   // Test

   samples = 3;

   recurrent_layer.set(2,2);

   inputs.resize(samples,2);
   inputs.setConstant(type(1));

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

   outputs = recurrent_layer.calculate_outputs(inputs);
}


void RecurrentLayerTest::run_test_case()
{
   cout << "Running recurrent layer test case...\n";

   // Constructor and destructor

   test_constructor();

   test_calculate_activations_derivatives();

   test_calculate_outputs();

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
