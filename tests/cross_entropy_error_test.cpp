//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   T E S T   C L A S S           
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cross_entropy_error_test.h"

CrossEntropyErrorTest::CrossEntropyErrorTest() : UnitTesting() 
{
}


CrossEntropyErrorTest::~CrossEntropyErrorTest()
{
}


void CrossEntropyErrorTest::test_calculate_training_error()
{
   cout << "test_calculate_training_error\n";

   DataSet data_set;

   NeuralNetwork neural_network;
   Vector<double> parameters;

   PerceptronLayer perceptron_layer;
   ScalingLayer scaling_layer;

   CrossEntropyError cee(&neural_network, &data_set);

   double cross_entropy_error;

   // Test

   data_set.generate_sum_data(10,2);

   Vector<Descriptives> inputs = data_set.scale_inputs_minimum_maximum();

   scaling_layer.set_neurons_number(1);
   scaling_layer.set_inputs_number(1);
   scaling_layer.set_descriptives(inputs);

//   data_set.set(10,1,1);

//   Matrix<double> data(10,2);
//   data.initialize_identity();

//   data_set.set_data(data);

   data_set.set_training();

   neural_network.set(NeuralNetwork::Approximation, {1,1});

   perceptron_layer.set(1,1);
   perceptron_layer.set_activation_function("Logistic");
/*
   neural_network.add_layer(&scaling_layer);
   neural_network.add_layer(&perceptron_layer);
*/
   neural_network.initialize_parameters(0.0);

   parameters = neural_network.get_parameters();

   cross_entropy_error = cee.calculate_training_error();

   assert_true(abs(cross_entropy_error - cee.calculate_training_error(parameters)) < 1.0e-03, LOG);

}


void CrossEntropyErrorTest::test_calculate_selection_error()
{
   cout << "test_calculate_selection_error\n";

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1,1});
   DataSet data_set;
   
   CrossEntropyError cee(&neural_network, &data_set);

   double selection_error;

   // Test

//   PerceptronLayer perceptron_layer(1,1);

//   neural_network.add_layer(&perceptron_layer);

   neural_network.initialize_parameters(0.0);

   const Vector<double> parameters = neural_network.get_parameters();
   
   data_set.set(1,1,1);
   data_set.initialize_data(0.0);
   data_set.set_selection();

   selection_error = cee.calculate_selection_error();

   assert_true(selection_error == 0.0, LOG);
}


void CrossEntropyErrorTest::test_calculate_training_error_gradient()
{
   cout << "test_calculate_training_error_gradient\n";

   NeuralNetwork neural_network;

   DataSet data_set;

   CrossEntropyError cee(&neural_network, &data_set);

   Vector<double> error_gradient;
   Vector<double> numerical_error_gradient;

   size_t instances_number;
   size_t inputs_number;
   size_t outputs_number;
   size_t hidden_neurons;

   ScalingLayer scaling_layer;

   RecurrentLayer recurrent_layer;

   LongShortTermMemoryLayer long_short_term_memory_layer;

   PerceptronLayer hidden_perceptron_layer;
   PerceptronLayer output_perceptron_layer;

   ProbabilisticLayer probabilistic_layer;

   neural_network.set();

   // Test perceptron and probabilistic
{
   instances_number = 10;
   inputs_number = 3;
   outputs_number = 2;
   hidden_neurons = 2;

   data_set.set(instances_number, inputs_number, outputs_number);

   data_set.randomize_data_normal();

   data_set.set_training();

   neural_network.set(NeuralNetwork::Classification,{inputs_number, hidden_neurons, outputs_number});
/*
   hidden_perceptron_layer.set(inputs_number, hidden_neurons);
   output_perceptron_layer.set(hidden_neurons, outputs_number);
   probabilistic_layer.set(outputs_number, outputs_number);

   neural_network.add_layer(&hidden_perceptron_layer);
   neural_network.add_layer(&output_perceptron_layer);
   neural_network.add_layer(&probabilistic_layer);
*/
   neural_network.randomize_parameters_normal();

   error_gradient = cee.calculate_training_error_gradient();

   numerical_error_gradient = cee.calculate_training_error_gradient_numerical_differentiation();

   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);

}

   neural_network.set();

   // Test lstm
{
   instances_number = 10;
   inputs_number = 3;
   outputs_number = 2;
   hidden_neurons = 2;

   data_set.set(instances_number, inputs_number, outputs_number);

   data_set.randomize_data_normal();

   data_set.set_training();

   neural_network.set(NeuralNetwork::Forecasting,{inputs_number, hidden_neurons, outputs_number});
/*
   long_short_term_memory_layer.set(inputs_number, hidden_neurons);
   output_perceptron_layer.set(hidden_neurons, outputs_number);

   neural_network.add_layer(&long_short_term_memory_layer);
   neural_network.add_layer(&output_perceptron_layer);
*/
   neural_network.randomize_parameters_normal();

   error_gradient = cee.calculate_training_error_gradient();

   numerical_error_gradient = cee.calculate_training_error_gradient_numerical_differentiation();

   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
}

   neural_network.set();

   // Test recurrent
   /*
{
   instances_number = 10;
   inputs_number = 3;
   outputs_number = 2;
   hidden_neurons = 2;

   data_set.set(instances_number, inputs_number, outputs_number);

   data_set.randomize_data_normal();

   data_set.set_training();

   recurrent_layer.set(inputs_number, hidden_neurons);
   output_perceptron_layer.set(hidden_neurons, outputs_number);

   neural_network.add_layer(&scaling_layer);
   neural_network.add_layer(&recurrent_layer);
   neural_network.add_layer(&output_perceptron_layer);

   neural_network.randomize_parameters_normal();

   error_gradient = cee.calculate_training_error_gradient();

   numerical_error_gradient = cee.calculate_training_error_gradient_numerical_differentiation();

   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
}
   */
}


void CrossEntropyErrorTest::test_to_XML()   
{
	cout << "test_to_XML\n"; 
}


void CrossEntropyErrorTest::test_from_XML()
{
	cout << "test_from_XML\n"; 
}


void CrossEntropyErrorTest::run_test_case()
{
    cout << "Running cross entropy error test case...\n";

   // Get methods

   // Set methods

   // Error methods

   test_calculate_training_error();

   test_calculate_selection_error();

   test_calculate_training_error_gradient();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   cout << "End of cross entropy error test case.\n";
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
