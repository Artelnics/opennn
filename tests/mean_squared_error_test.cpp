//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E A N   S Q U A R E D   E R R O R   T E S T   C L A S S             
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "mean_squared_error_test.h"


MeanSquaredErrorTest::MeanSquaredErrorTest() : UnitTesting() 
{
}


MeanSquaredErrorTest::~MeanSquaredErrorTest()
{
}


void MeanSquaredErrorTest::test_constructor()
{
   cout << "test_constructor\n";

   // Default

   MeanSquaredError mse1;

   assert_true(mse1.has_neural_network() == false, LOG);
   assert_true(mse1.has_data_set() == false, LOG);

   // Neural network

   NeuralNetwork nn2;
   MeanSquaredError mse2(&nn2);

   assert_true(mse2.has_neural_network() == true, LOG);
   assert_true(mse2.has_data_set() == false, LOG);

   // Neural network and data set

   NeuralNetwork nn3;
   DataSet ds3;
   MeanSquaredError mse3(&nn3, &ds3);

   assert_true(mse3.has_neural_network() == true, LOG);
   assert_true(mse3.has_data_set() == true, LOG);

}


void MeanSquaredErrorTest::test_destructor()
{
    cout << "test_destructor\n";
}


void MeanSquaredErrorTest::test_calculate_training_error()
{
   cout << "test_calculate_training_error\n";

   Vector<double> parameters;

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1, 1, 1});
   neural_network.initialize_parameters(0.0);

   DataSet data_set(1, 1, 1);
   data_set.initialize_data(0.0);

   MeanSquaredError mean_squared_error(&neural_network, &data_set);

//   assert_true(mean_squared_error.calculate_training_error() == 0.0, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1});
   neural_network.randomize_parameters_normal();

   parameters = neural_network.get_parameters();

   data_set.set(1, 1, 1);
   data_set.randomize_data_normal();

//   assert_true(abs(mean_squared_error.calculate_training_error() - mean_squared_error.calculate_error(parameters)) < numeric_limits<double>::min(), LOG);

   neural_network.set(NeuralNetwork::Approximation, {1, 1, 1});
   neural_network.initialize_parameters(0.0);

   data_set.set(1, 1, 1);
   data_set.initialize_data(0.0);
   data_set.set_training();

   assert_true(mean_squared_error.calculate_training_error() == 0.0, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1});
   neural_network.randomize_parameters_uniform();

   parameters = neural_network.get_parameters();

   data_set.set(1, 1, 1);
   data_set.randomize_data_normal();
   data_set.set_training();

   assert_true(abs(mean_squared_error.calculate_training_error() - mean_squared_error.calculate_training_error(parameters)) < numeric_limits<double>::min(), LOG);
}


void MeanSquaredErrorTest::test_calculate_training_error_gradient()
{
   cout << "test_calculate_training_error_gradient\n";

   NeuralNetwork neural_network;

   DataSet data_set;

   MeanSquaredError mean_squared_error(&neural_network, &data_set);

   Vector<double> error_gradient;
   Vector<double> numerical_error_gradient;

   size_t instances_number;
   size_t inputs_number;
   size_t outputs_number;
   size_t hidden_neurons;

   ScalingLayer* scaling_layer = new ScalingLayer();

   RecurrentLayer* recurrent_layer = new RecurrentLayer();

   LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer();

   PerceptronLayer* hidden_perceptron_layer = new PerceptronLayer();
   PerceptronLayer* output_perceptron_layer = new PerceptronLayer();

   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer();

   // Test trivial
{
   instances_number = 100;
   inputs_number = 1;
   outputs_number = 1;

   data_set.set(instances_number, inputs_number, outputs_number);

   data_set.initialize_data(0.0);

   hidden_perceptron_layer->set(inputs_number, outputs_number);
   neural_network.add_layer(hidden_perceptron_layer);

   neural_network.initialize_parameters(0.0);

   numerical_error_gradient = mean_squared_error.calculate_training_error_gradient_numerical_differentiation();

   error_gradient = mean_squared_error.calculate_training_error_gradient();

   assert_true(error_gradient.size() == neural_network.get_parameters_number(), LOG);
   assert_true(error_gradient == 0.0, LOG);
}

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

   hidden_perceptron_layer->set(inputs_number, hidden_neurons);
   output_perceptron_layer->set(hidden_neurons, outputs_number);
   probabilistic_layer->set(outputs_number, outputs_number);

   neural_network.add_layer(hidden_perceptron_layer);
   neural_network.add_layer(output_perceptron_layer);
   neural_network.add_layer(probabilistic_layer);

   neural_network.randomize_parameters_normal();

   error_gradient = mean_squared_error.calculate_training_error_gradient();

   numerical_error_gradient = mean_squared_error.calculate_training_error_gradient_numerical_differentiation();

   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
}

   neural_network.set();

   // Test lstm
{
   instances_number = 5;
   inputs_number = 4;
   outputs_number = 2;
   hidden_neurons = 3;

   data_set.set(instances_number, inputs_number, outputs_number);

   data_set.randomize_data_normal();

   data_set.set_training();

   long_short_term_memory_layer->set(inputs_number, hidden_neurons);
   output_perceptron_layer->set(hidden_neurons, outputs_number);

   neural_network.add_layer(long_short_term_memory_layer);
   neural_network.add_layer(output_perceptron_layer);

   neural_network.randomize_parameters_normal();

   error_gradient = mean_squared_error.calculate_training_error_gradient();

   numerical_error_gradient = mean_squared_error.calculate_training_error_gradient_numerical_differentiation();

   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
}

   neural_network.set();

   // Test recurrent
{
   instances_number = 92;
   inputs_number = 3;
   outputs_number = 1;
   hidden_neurons = 4;

   data_set.set(instances_number, inputs_number, outputs_number);

   data_set.randomize_data_normal();

   data_set.set_training();

   recurrent_layer->set(inputs_number, hidden_neurons);
   recurrent_layer->set_timesteps(1);

   output_perceptron_layer->set(hidden_neurons, outputs_number);

   neural_network.add_layer(recurrent_layer);
   neural_network.add_layer(output_perceptron_layer);

   neural_network.randomize_parameters_normal();

   error_gradient = mean_squared_error.calculate_training_error_gradient();

   numerical_error_gradient = mean_squared_error.calculate_training_error_gradient_numerical_differentiation();

   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
}
}


void MeanSquaredErrorTest::test_calculate_selection_error()
{
   cout << "test_calculate_selection_error\n";

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1, 1, 1});

   neural_network.initialize_parameters(0.0);

   DataSet data_set(1, 1, 1);

   data_set.set_selection();

   data_set.initialize_data(0.0);

   MeanSquaredError mean_squared_error(&neural_network, &data_set);

   double selection_error = mean_squared_error.calculate_selection_error();

   assert_true(selection_error == 0.0, LOG);

}


void MeanSquaredErrorTest::test_calculate_training_error_terms()
{
   cout << "test_calculate_training_error_terms\n";

   NeuralNetwork neural_network;
   Vector<size_t> hidden_layers_size;
   Vector<double> parameters;

   DataSet data_set;
   
   MeanSquaredError mean_squared_error(&neural_network, &data_set);

   double error;

   Vector<double> error_terms;

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1});
   neural_network.randomize_parameters_normal();

   data_set.set(1, 1, 1);
   data_set.randomize_data_normal();

   //const Matrix<double> inputs = data_set.get_training_input_data();
   //const Matrix<double> targets = data_set.get_training_target_data();
   //const Matrix<double> outputs = nn.calculate_outputs(inputs);

   error = mean_squared_error.calculate_training_error();

   //error_terms = mean_squared_error.calculate_training_error_terms(outputs, targets);

   assert_true(abs(dot(error_terms, error_terms) - error) < 1.0e-3, LOG);
}


void MeanSquaredErrorTest::test_calculate_training_error_terms_Jacobian()
{
   cout << "test_calculate_training_error_terms_Jacobian\n";

   NumericalDifferentiation nd;

   NeuralNetwork neural_network;
   Vector<size_t> architecture;
   Vector<double> parameters;

   DataSet data_set;

   MeanSquaredError mean_squared_error(&neural_network, &data_set);

   Vector<double> error_gradient;

   Vector<double> error_terms;
   Matrix<double> terms_Jacobian;
   Matrix<double> numerical_Jacobian_terms;

   Tensor<double> inputs;
   Tensor<double> targets;
   Tensor<double> outputs;

   Tensor<double> output_gradient;
   Vector<Tensor<double>> layers_delta;

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1});

   neural_network.initialize_parameters(0.0);

   data_set.set(1, 1, 1);

   data_set.initialize_data(0.0);

   inputs = data_set.get_training_input_data();
   targets = data_set.get_training_target_data();
   outputs = neural_network.calculate_outputs(inputs);

   Vector<Layer::FirstOrderActivations> forward_propagation = neural_network.calculate_trainable_forward_propagation(inputs);

   output_gradient = mean_squared_error.calculate_output_gradient(outputs, targets);

   layers_delta = mean_squared_error.calculate_layers_delta(forward_propagation, output_gradient);

   terms_Jacobian = mean_squared_error.calculate_error_terms_Jacobian(inputs, forward_propagation, layers_delta);

   assert_true(terms_Jacobian.get_rows_number() == data_set.get_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == neural_network.get_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);

   // Test 

   neural_network.set(NeuralNetwork::Approximation, {3, 4, 2});
   neural_network.initialize_parameters(0.0);

   data_set.set(3, 2, 5);
   mean_squared_error.set(&neural_network, &data_set);
   data_set.initialize_data(0.0);

   inputs = data_set.get_training_input_data();
   targets = data_set.get_training_target_data();
   outputs = neural_network.calculate_outputs(inputs);

   //forward_propagation = nn.calculate_forward_propagation(inputs);

   output_gradient = mean_squared_error.calculate_output_gradient(outputs, targets);

   layers_delta = mean_squared_error.calculate_layers_delta(forward_propagation, output_gradient);

   terms_Jacobian = mean_squared_error.calculate_error_terms_Jacobian(inputs, forward_propagation, layers_delta);

   assert_true(terms_Jacobian.get_rows_number() == data_set.get_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == neural_network.get_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

   architecture.set(3);
   architecture[0] = 2;
   architecture[1] = 1;
   architecture[2] = 2;

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.initialize_parameters(0.0);

   data_set.set(2, 2, 5);
   mean_squared_error.set(&neural_network, &data_set);
   data_set.initialize_data(0.0);

   inputs = data_set.get_training_input_data();
   targets = data_set.get_training_target_data();
   outputs = neural_network.calculate_outputs(inputs);

   forward_propagation = neural_network.calculate_trainable_forward_propagation(inputs);

   output_gradient = mean_squared_error.calculate_output_gradient(outputs, targets);

   layers_delta = mean_squared_error.calculate_layers_delta(forward_propagation, output_gradient);

   terms_Jacobian = mean_squared_error.calculate_error_terms_Jacobian(inputs, forward_propagation, layers_delta);

   assert_true(terms_Jacobian.get_rows_number() == data_set.get_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == neural_network.get_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);
   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1});
   neural_network.initialize_parameters(0.0);
   //nn.set_layer_activation_function(0, PerceptronLayer::Linear);
//   nn.randomize_parameters_normal();
   parameters = neural_network.get_parameters();

   data_set.set(1, 1, 1);
//   data_set.randomize_data_normal();
   data_set.initialize_data(1.0);

   inputs = data_set.get_training_input_data();
   targets = data_set.get_training_target_data();
   outputs = neural_network.calculate_outputs(inputs);

//   forward_propagation = nn.calculate_forward_propagation(inputs);

   output_gradient = mean_squared_error.calculate_output_gradient(outputs, targets);

   layers_delta = mean_squared_error.calculate_layers_delta(forward_propagation, output_gradient);

   cout << "layers delta: " << layers_delta << endl;

   terms_Jacobian = mean_squared_error.calculate_error_terms_Jacobian(inputs, forward_propagation, layers_delta);

   numerical_Jacobian_terms = nd.calculate_Jacobian(mean_squared_error, &MeanSquaredError::calculate_training_error_terms, parameters);

   cout << "Terms Jacobian: " << terms_Jacobian << endl;
   cout << "Numerical: " << numerical_Jacobian_terms << endl;

   assert_true(absolute_value(terms_Jacobian-numerical_Jacobian_terms) < 1.0e-3, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 2, 2});
   neural_network.randomize_parameters_normal();
   parameters = neural_network.get_parameters();

   data_set.set(2, 2, 2);
   data_set.randomize_data_normal();

//   terms_Jacobian = mean_squared_error.calculate_error_terms_Jacobian();
   numerical_Jacobian_terms = nd.calculate_Jacobian(mean_squared_error, &MeanSquaredError::calculate_training_error_terms, parameters);

   assert_true(absolute_value(terms_Jacobian-numerical_Jacobian_terms) < 1.0e-3, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 2, 2});
   neural_network.randomize_parameters_normal();

   data_set.set(2, 2, 2);
   data_set.randomize_data_normal();
   
//   error_gradient = mean_squared_error.calculate_error_gradient({0, 1});

//   error_terms = mean_squared_error.calculate_training_error_terms();
//   terms_Jacobian = mean_squared_error.calculate_error_terms_Jacobian();

//   assert_true(absolute_value((terms_Jacobian.calculate_transpose()).dot(error_terms)*2.0 - error_gradient) < 1.0e-3, LOG);
}


void MeanSquaredErrorTest::test_to_XML()
{
   cout << "test_to_XML\n";
}


void MeanSquaredErrorTest::test_from_XML()
{
   cout << "test_from_XML\n";
}


void MeanSquaredErrorTest::run_test_case()
{
   cout << "Running mean squared error test case...\n";

   // Constructor and destructor methods

//   test_constructor();
//   test_destructor();

   // Get methods

   // Set methods

   // Error methods

   test_calculate_training_error();

   test_calculate_selection_error();

   test_calculate_training_error_gradient();

   // Error terms methods

//   test_calculate_training_error_terms();
//   test_calculate_training_error_terms_Jacobian();

   // Serialization methods

//   test_to_XML();
//   test_from_XML();

   cout << "End of mean squared error test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lemser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lemser General Public License for more details.

// You should have received a copy of the GNU Lemser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
