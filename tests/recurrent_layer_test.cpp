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

/*
void RecurrentLayerTest::test_constructor()
{
    cout << "test_constructor\n";

    RecurrentLayer recurrent_layer;
    Index inputs_number;
    Index neurons_number;

    Tensor<type, 2> synaptic_weights;
    Tensor<type, 2> recurrent_initializer;
    Tensor<type, 1> biases;

    // Test

    recurrent_layer.set();

    // Test

    inputs_number = 1;
    neurons_number = 1;

    recurrent_layer.set(inputs_number, neurons_number);

    assert_true(recurrent_layer.get_parameters_number() == 3, LOG);

    assert_true(recurrent_layer.get_biases().size() == 1, LOG);

    //Test

    inputs_number = 2;
    neurons_number = 3;

    recurrent_layer.set(inputs_number, neurons_number);

    assert_true(recurrent_layer.get_parameters_number() == 18, LOG);

    assert_true(recurrent_layer.get_biases().size() == 3, LOG);
}


void RecurrentLayerTest::test_destructor()
{
   cout << "test_destructor\n";

}

void RecurrentLayerTest::test_assignment_operator()
{
   cout << "test_assignment_operator\n";

   LongShortTermMemoryLayer long_short_term_memory_layer_1;

   long_short_term_memory_layer_1.set(4,3);

   LongShortTermMemoryLayer long_short_term_memory_layer_2 = long_short_term_memory_layer_1;

   assert_true(long_short_term_memory_layer_1.get_inputs_number() == 4, LOG);
   assert_true(long_short_term_memory_layer_1.get_neurons_number() == 3, LOG);
}


void RecurrentLayerTest::test_get_inputs_number()
{
   cout << "test_get_inputs_number\n";

   RecurrentLayer recurrent_layer;

   Index inputs_number;
   Index neurons_number;


   // Test

   recurrent_layer.set(0,0);
   assert_true(recurrent_layer.get_inputs_number() == 0, LOG);

   // Test

   inputs_number = 2;
   neurons_number = 3;

   recurrent_layer.set(inputs_number, neurons_number);
   assert_true(recurrent_layer.get_inputs_number() == inputs_number, LOG);
   assert_true(recurrent_layer.get_neurons_number() == neurons_number, LOG);
}


void RecurrentLayerTest::test_get_neurons_number()
{
   cout << "test_get_neurons_number\n";

   RecurrentLayer recurrent_layer;

   Index inputs_number;
   Index neurons_number;

   // Test

   recurrent_layer.set();
   assert_true(recurrent_layer.get_neurons_number() == 0, LOG);

   // Test

   inputs_number = 2;
   neurons_number = 3;

   recurrent_layer.set(inputs_number, neurons_number);
   assert_true(recurrent_layer.get_neurons_number() == 3, LOG);
}



void RecurrentLayerTest::test_get_biases()
{
   cout << "test_get_biases\n";

   RecurrentLayer recurrent_layer;

   Tensor<type, 1> biases;

   Index inputs_number;
   Index neurons_number;

   //Test

   neurons_number = 3;
   inputs_number = 2;

   recurrent_layer.set(inputs_number,neurons_number);

   biases.initialize_sequential();
   recurrent_layer.set_biases(biases);

   assert_true(biases == recurrent_layer.get_biases(), LOG);
}

void RecurrentLayerTest::test_get_weights()
{
   cout << "test_get_synaptic_weights\n";

   RecurrentLayer recurrent_layer;

   Tensor<type, 2> weights;

   //Test

   recurrent_layer.set(3,2);

   recurrent_layer.set_parameters_constant(4.0);

   assert_true(recurrent_layer.get_input_weights() == 4.0, LOG);


}

void RecurrentLayerTest::test_get_recurrent_initializer()
{
   cout << "test_get_recurrent_initializer\n";

   RecurrentLayer recurrent_layer;

   Tensor<type, 2> recurrent_weights;

   //Test

   recurrent_layer.set(1,2);

   recurrent_layer.set_parameters_constant(-1.0);

   recurrent_weights = recurrent_layer.get_recurrent_weights();

   assert_true(recurrent_weights.size() == 4, LOG);
   assert_true(recurrent_weights.dimension(0) == 2, LOG);
   assert_true(recurrent_weights.dimension(1) == 2, LOG);

   assert_true(recurrent_weights == -1.0, LOG);
}


void RecurrentLayerTest::test_get_parameters_number()
{
   cout << "test_get_parameters_number\n";

   RecurrentLayer recurrent_layer;

   // Test

   recurrent_layer.set(1, 1);

   assert_true(recurrent_layer.get_parameters_number() == 3, LOG);

   // Test

   recurrent_layer.set(3, 1);

   assert_true(recurrent_layer.get_parameters_number() == 5, LOG);

   // Test

   recurrent_layer.set(2, 4);

   assert_true(recurrent_layer.get_parameters_number() == 28, LOG);

   // Test

   recurrent_layer.set(4, 2);

   assert_true(recurrent_layer.get_parameters_number() == 14, LOG);
}


void RecurrentLayerTest::test_get_parameters()
{
   cout << "test_get_parameters\n";

   RecurrentLayer recurrent_layer;

   Tensor<type, 1> parameters;

   // Test

   recurrent_layer.set(1, 1);
   recurrent_layer.set_parameters_constant(1.0);

   parameters = recurrent_layer.get_parameters();

   assert_true(parameters.size() == 3, LOG);
   assert_true(parameters == 1.0, LOG);

   // Test

   recurrent_layer.set(2, 4);

   recurrent_layer.set_biases_constant(1.0);

   recurrent_layer.initialize_input_weights(0.5);

   recurrent_layer.initialize_recurrent_weights(-0.48);


   parameters = recurrent_layer.get_parameters();

   assert_true(parameters.size() == 28, LOG);

   assert_true(abs(parameters[0] - 0.5) < numeric_limits<type>::epsilon(), LOG);
   assert_true(abs(parameters[16] - -0.48) < numeric_limits<type>::epsilon(), LOG);
   assert_true(abs(parameters[27] - 1.0) < numeric_limits<type>::epsilon(), LOG);

   //Test

   Tensor<type, 1> biases;
   Tensor<type, 2> input_weights(3, 2);
   Tensor<type, 2> recurrent_weights(2,2);

   recurrent_layer.set(3,  2);
   biases.resize(2);
   biases[0] = 0.41;
   biases[1] = -0.70;
   recurrent_layer.set_biases(biases);

   input_weights.set_column(0,Tensor<type, 1> {0.5, -2.9, 0.2});
   input_weights.set_column(1,Tensor<type, 1> {7.2, 0.8, -1.2});

   recurrent_layer.set_input_weights(input_weights);

   recurrent_weights.set_column(0,Tensor<type, 1> {7.9, -2.3});
   recurrent_weights.set_column(1,Tensor<type, 1> {1.2, -1.5});

   recurrent_layer.set_recurrent_weights(recurrent_weights);
   cout<<"parameters:  "<< recurrent_layer.get_parameters()<<endl;
   cout<<"norm:  "<< recurrent_layer.calculate_parameters_norm()<<endl;

}



void RecurrentLayerTest::test_calculate_activations_derivatives()
{
   cout << "test_calculate_activation_derivative\n";

   NumericalDifferentiation numerical_differentiation;

   RecurrentLayer recurrent_layer;
   Tensor<type, 1> parameters;
   Tensor<type, 2> inputs;
   Tensor<type, 2> combinations_2d;
   Tensor<type, 2> activations_derivatives;
   Tensor<type, 2> numerical_activation_derivative;

    numerical_differentiation_tests = true;

   // Test

   recurrent_layer.set(1, 1);
   combinations_2d.resize({1,1}, 0.0);

   recurrent_layer.set_activation_function(RecurrentLayer::Logistic);
   activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations_2d);
   assert_true(activations_derivatives.rank() == 2, LOG);
   assert_true(activations_derivatives.dimension(0) == 1, LOG);
   assert_true(activations_derivatives.dimension(1) == 1, LOG);
   assert_true(activations_derivatives == 0.25, LOG);

   recurrent_layer.set_activation_function(RecurrentLayer::HyperbolicTangent);
   activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations_2d);
   assert_true(activations_derivatives.rank() == 2, LOG);
   assert_true(activations_derivatives.dimension(0) == 1, LOG);
   assert_true(activations_derivatives.dimension(1) == 1, LOG);
   assert_true(activations_derivatives == 1.0, LOG);

   recurrent_layer.set_activation_function(RecurrentLayer::Linear);
   activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations_2d);
   assert_true(activations_derivatives.rank() == 3, LOG);
   assert_true(activations_derivatives.dimension(0) == 1, LOG);
   assert_true(activations_derivatives.dimension(1) == 1, LOG);
   assert_true(activations_derivatives == 1.0, LOG);

   // Test

   if(numerical_differentiation_tests)
   {
      recurrent_layer.set(2, 4);

      combinations_2d.resize(({1,4}));
      combinations_2d(0,0) = 1.56;
      combinations_2d(0,2) = -0.68;
      combinations_2d(0,2)= 0.91;
      combinations_2d(0,3) = -1.99;

      recurrent_layer.set_activation_function(RecurrentLayer::Logistic);

      activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations_2d);

//      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::calculate_activations, combinations_2d);

//      assert_true((activations_derivatives - numerical_activation_derivative).abs() < 1.0e-3, LOG);


      recurrent_layer.set_activation_function(RecurrentLayer::HyperbolicTangent);

//      activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations_2d);

//      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::calculate_activations, combinations_2d);

//      assert_true(absolute_value(activations_derivatives - numerical_activation_derivative) < 1.0e-3, LOG);

      recurrent_layer.set_activation_function(RecurrentLayer::Linear);

      activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations_2d);

//      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::calculate_activations, combinations_2d);

//      assert_true(absolute_value(activations_derivatives - numerical_activation_derivative) < 1.0e-3, LOG);

   }

   // Test

   if(numerical_differentiation_tests)
   {
      recurrent_layer.set(4, 2);

      parameters.resize(14);
      parameters[0] = 0.41;
      parameters[1] = -0.68;
      parameters[2] = 0.14;
      parameters[3] = -0.50;
      parameters[4] = 0.52;
      parameters[5] = -0.70;
      parameters[6] = 0.85;
      parameters[7] = -0.18;
      parameters[8] = -0.65;
      parameters[9] = 0.05;
      parameters[10] = 0.85;
      parameters[11] = -0.18;
      parameters[12] = -0.65;
      parameters[13] = 0.05;

      recurrent_layer.set_parameters(parameters);

      inputs.resize(({1,4}));
      inputs[0] = 0.85;
      inputs[1] = -0.25;
      inputs[2] = 0.29;
      inputs[3] = -0.77;

//      recurrent_layer.set_activation_function(RecurrentLayer::Threshold);
//      activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations_2d);
//      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::calculate_activations, combinations_2d);
//      assert_true(absolute_value((activations_derivatives - numerical_activation_derivative)) < 1.0e-3, LOG);

//      recurrent_layer.set_activation_function(RecurrentLayer::SymmetricThreshold);
//      activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations_2d);
//      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::calculate_activations, combinations_2d);
//      assert_true(absolute_value((activations_derivatives - numerical_activation_derivative)) < 1.0e-3, LOG);

//      recurrent_layer.set_activation_function(RecurrentLayer::Logistic);
//      activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations_2d);
//      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::calculate_activations, combinations_2d);
//      assert_true(absolute_value((activations_derivatives - numerical_activation_derivative)) < 1.0e-3, LOG);

//      recurrent_layer.set_activation_function(RecurrentLayer::HyperbolicTangent);
//      activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations_2d);
//      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::calculate_activations, combinations_2d);
//      assert_true(absolute_value((activations_derivatives - numerical_activation_derivative)) < 1.0e-3, LOG);

//      recurrent_layer.set_activation_function(RecurrentLayer::Linear);
//      activations_derivatives = recurrent_layer.calculate_activations_derivatives(combinations_2d);
//      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(recurrent_layer, &RecurrentLayer::calculate_activations, combinations_2d);
//      assert_true(absolute_value((activations_derivatives - numerical_activation_derivative)) < 1.0e-3, LOG);

   }
}


void RecurrentLayerTest::test_calculate_combinations()
{
   cout << "test_calculate_combinations\n";
}


void RecurrentLayerTest::test_calculate_outputs()
{
   cout << "test_calculate_outputs\n";

   RecurrentLayer recurrent_layer;
   Tensor<type, 2> inputs;
   Tensor<type, 2> outputs;

   Tensor<type, 1> parameters;

   Index instances = 3;

   Tensor<type, 2> new_weights;
   Tensor<type, 2> new_recurrent_weights;
   Tensor<type, 1> new_biases;

   //Test

   recurrent_layer.set(2,2);

   inputs.resize({instances,2},1.0);

   recurrent_layer.set_activation_function("SoftPlus");

   recurrent_layer.set_timesteps(3);

   new_weights.set(recurrent_layer.get_inputs_number(),recurrent_layer.get_neurons_number(),1.0);
   new_recurrent_weights.set(recurrent_layer.get_neurons_number(),recurrent_layer.get_neurons_number(),1.0);
   new_biases.resize(recurrent_layer.get_neurons_number(), 1.0);

   recurrent_layer.set_biases(new_biases);
   recurrent_layer.set_input_weights(new_weights);
   recurrent_layer.set_recurrent_weights(new_recurrent_weights);

   parameters = recurrent_layer.get_parameters();

   assert_true(recurrent_layer.calculate_outputs(inputs) == recurrent_layer.calculate_outputs(inputs, parameters), LOG);

   assert_true(recurrent_layer.calculate_outputs(inputs, new_biases, new_weights, new_recurrent_weights) == recurrent_layer.calculate_outputs(inputs, parameters), LOG);

   assert_true(recurrent_layer.calculate_outputs(inputs) == recurrent_layer.calculate_outputs(inputs,new_biases, new_weights, new_recurrent_weights), LOG);

   cout<<"outputs:"<<recurrent_layer.calculate_outputs(inputs) <<endl;
}
*/

void RecurrentLayerTest::run_test_case()
{
   cout << "Running recurrent layer test case...\n";
/*
   // Constructor and destructor

   test_constructor();

   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Inputs and perceptrons

   test_get_inputs_number();
   test_get_neurons_number();

   // Parameters

   test_get_biases();
   test_get_weights();
   test_get_recurrent_initializer();

   test_get_parameters_number();
   test_get_parameters();

//   test_calculate_activations_derivatives();

//   test_calculate_combinations();
//   test_calculate_outputs();
*/
   cout << "End of recurrent layer test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
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
