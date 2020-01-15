//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M I N K O W S K I   E R R O R   T E S T   C L A S S                   
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "minkowski_error_test.h"


MinkowskiErrorTest::MinkowskiErrorTest() : UnitTesting() 
{
}


MinkowskiErrorTest::~MinkowskiErrorTest() 
{
}


void MinkowskiErrorTest::test_constructor()
{
   cout << "test_constructor\n";

   // Default

   MinkowskiError me1;

   assert_true(me1.has_neural_network() == false, LOG);
   assert_true(me1.has_data_set() == false, LOG);

   // Neural network

   NeuralNetwork nn2;
   MinkowskiError me2(&nn2);

   assert_true(me2.has_neural_network() == true, LOG);
   assert_true(me2.has_data_set() == false, LOG);

   // Neural network and data set

   NeuralNetwork nn3;
   DataSet ds3;
   MinkowskiError me3(&nn3, &ds3);

   assert_true(me3.has_neural_network() == true, LOG);
   assert_true(me3.has_data_set() == true, LOG);
}


void MinkowskiErrorTest::test_destructor()
{
   cout << "test_destructor\n";
}


void MinkowskiErrorTest::test_get_Minkowski_parameter()
{
   cout << "test_get_Minkowski_parameter\n";

   MinkowskiError me;

   me.set_Minkowski_parameter(1.0);
   
   assert_true(me.get_Minkowski_parameter() == 1.0, LOG);
}


void MinkowskiErrorTest::test_set_Minkowski_parameter()
{
   cout << "test_set_Minkowski_parameter\n";
}


void MinkowskiErrorTest::test_calculate_training_error()
{
   cout << "test_calculate_training_error\n";

   Vector<double> parameters;

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1,1,1});
   neural_network.initialize_parameters(0.0);

   DataSet data_set(1,1,1);
   data_set.initialize_data(0.0);

   MinkowskiError minkowski_error(&neural_network, &data_set);

   assert_true(minkowski_error.calculate_training_loss() == 0.0, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 2});
   neural_network.randomize_parameters_normal();

   parameters = neural_network.get_parameters();

   data_set.set(1, 1, 2);
   data_set.randomize_data_normal();

   assert_true(abs(minkowski_error.calculate_training_error() - minkowski_error.calculate_training_error(parameters)) < numeric_limits<double>::min(), LOG);
}


void MinkowskiErrorTest::test_calculate_selection_error()
{
   cout << "test_calculate_selection_error\n";  
}


void MinkowskiErrorTest::test_calculate_training_error_gradient()
{
   cout << "test_calculate_training_error_gradient\n";

   NeuralNetwork neural_network;

   DataSet data_set;

   MinkowskiError me(&neural_network, &data_set);

   Vector<double> error_gradient;
   Vector<double> numerical_error_gradient;

   size_t instances_number;
   size_t inputs_number;
   size_t outputs_number;
   size_t hidden_neurons;

   RecurrentLayer* recurrent_layer = new RecurrentLayer;

   LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer;

   PerceptronLayer* hidden_perceptron_layer = new PerceptronLayer;
   PerceptronLayer* output_perceptron_layer = new PerceptronLayer;

   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer;

   // Test trivial
{
   instances_number = 10;
   inputs_number = 1;
   outputs_number = 1;

   data_set.set(instances_number, inputs_number, outputs_number);

   data_set.initialize_data(0.0);

   hidden_perceptron_layer->set(inputs_number, outputs_number);
   neural_network.add_layer(hidden_perceptron_layer);

   neural_network.initialize_parameters(0.0);

   numerical_error_gradient = me.calculate_training_error_gradient_numerical_differentiation();

   error_gradient = me.calculate_training_error_gradient();

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

   error_gradient = me.calculate_training_error_gradient();

   numerical_error_gradient = me.calculate_training_error_gradient_numerical_differentiation();

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

   long_short_term_memory_layer->set(inputs_number, hidden_neurons);
   output_perceptron_layer->set(hidden_neurons, outputs_number);

   neural_network.add_layer(long_short_term_memory_layer);
   neural_network.add_layer(output_perceptron_layer);

   neural_network.randomize_parameters_normal();

   error_gradient = me.calculate_training_error_gradient();

   numerical_error_gradient = me.calculate_training_error_gradient_numerical_differentiation();

   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
}

   neural_network.set();

   // Test recurrent
{
   instances_number = 10;
   inputs_number = 3;
   outputs_number = 2;
   hidden_neurons = 2;

   data_set.set(instances_number, inputs_number, outputs_number);

   data_set.randomize_data_normal();

   data_set.set_training();

   recurrent_layer->set(inputs_number, hidden_neurons);
   output_perceptron_layer->set(hidden_neurons, outputs_number);

   neural_network.add_layer(recurrent_layer);
   neural_network.add_layer(output_perceptron_layer);

   neural_network.randomize_parameters_normal();

   error_gradient = me.calculate_training_error_gradient();

   numerical_error_gradient = me.calculate_training_error_gradient_numerical_differentiation();

   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
}

   // Test convolutional
{
   instances_number = 5;
   inputs_number = 147;
   outputs_number = 1;

   data_set.set(instances_number, inputs_number, outputs_number);
   data_set.set_input_variables_dimensions(Vector<size_t>({3,7,7}));
   data_set.set_target_variables_dimensions(Vector<size_t>({1}));
   data_set.randomize_data_normal();
   data_set.set_training();

   const double parameters_minimum = -100.0;
   const double parameters_maximum = 100.0;

   ConvolutionalLayer* convolutional_layer_1 = new ConvolutionalLayer({3,7,7}, {2,2,2});
   Tensor<double> filters_1({2,3,2,2}, 0);
   filters_1.randomize_uniform(parameters_minimum,parameters_maximum);
   convolutional_layer_1->set_synaptic_weights(filters_1);
   Vector<double> biases_1(2, 0);
   biases_1.randomize_uniform(parameters_minimum, parameters_maximum);
   convolutional_layer_1->set_biases(biases_1);

   ConvolutionalLayer* convolutional_layer_2 = new ConvolutionalLayer(convolutional_layer_1->get_outputs_dimensions(), {2,2,2});
   convolutional_layer_2->set_padding_option(OpenNN::ConvolutionalLayer::Same);
   Tensor<double> filters_2({2,2,2,2}, 0);
   filters_2.randomize_uniform(parameters_minimum, parameters_maximum);
   convolutional_layer_2->set_synaptic_weights(filters_2);
   Vector<double> biases_2(2, 0);
   biases_2.randomize_uniform(parameters_minimum, parameters_maximum);
   convolutional_layer_2->set_biases(biases_2);

   PoolingLayer* pooling_layer_1 = new PoolingLayer(convolutional_layer_2->get_outputs_dimensions(), {2,2});

   ConvolutionalLayer* convolutional_layer_3 = new ConvolutionalLayer(pooling_layer_1->get_outputs_dimensions(), {1,2,2});
   convolutional_layer_3->set_padding_option(OpenNN::ConvolutionalLayer::Same);
   Tensor<double> filters_3({1,2,2,2}, 0);
   filters_3.randomize_uniform(parameters_minimum, parameters_maximum);
   convolutional_layer_3->set_synaptic_weights(filters_3);
   Vector<double> biases_3(1, 0);
   biases_3.randomize_uniform(parameters_minimum, parameters_maximum);
   convolutional_layer_3->set_biases(biases_3);

   PoolingLayer* pooling_layer_2 = new PoolingLayer(convolutional_layer_3->get_outputs_dimensions(), {2,2});
   pooling_layer_2->set_pooling_method(PoolingLayer::MaxPooling);

   PoolingLayer* pooling_layer_3 = new PoolingLayer(pooling_layer_2->get_outputs_dimensions(), {2,2});
   pooling_layer_3->set_pooling_method(PoolingLayer::MaxPooling);

   PerceptronLayer* perceptron_layer = new PerceptronLayer(pooling_layer_3->get_outputs_dimensions().calculate_product(), 3, OpenNN::PerceptronLayer::ActivationFunction::Linear);
   perceptron_layer->randomize_parameters_uniform(parameters_minimum, parameters_maximum);

   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(perceptron_layer->get_neurons_number(), outputs_number);
   probabilistic_layer->randomize_parameters_uniform(parameters_minimum, parameters_maximum);

   neural_network.set();
   neural_network.add_layer(convolutional_layer_1);
   neural_network.add_layer(convolutional_layer_2);
   neural_network.add_layer(pooling_layer_1);
   neural_network.add_layer(convolutional_layer_3);
   neural_network.add_layer(pooling_layer_2);
   neural_network.add_layer(pooling_layer_3);
   neural_network.add_layer(perceptron_layer);
   neural_network.add_layer(probabilistic_layer);

   numerical_error_gradient = me.calculate_training_error_gradient_numerical_differentiation();

   error_gradient = me.calculate_training_error_gradient();

   assert_true(absolute_value(numerical_error_gradient - error_gradient) < 1e-3, LOG);
}
}


void MinkowskiErrorTest::test_to_XML()   
{
   cout << "test_to_XML\n";  

   MinkowskiError me;

   tinyxml2::XMLDocument* document;

   // Test

   document = me.to_XML();

   assert_true(document != nullptr, LOG);

   delete document;

}


/// @todo

void MinkowskiErrorTest::test_from_XML()   
{
   cout << "test_from_XML\n";
/*
   MinkowskiError me1;
   MinkowskiError me2;

  tinyxml2::XMLDocument* document;

  // Test

  me1.set_Minkowski_parameter(1.33);
  me1.set_display(false);

  document = me1.to_XML();

  me2.from_XML(*document);

  delete document;

  assert_true(me2.get_Minkowski_parameter() == 1.33, LOG);
*/
}


void MinkowskiErrorTest::run_test_case()
{
   cout << "Running Minkowski error test case...\n";  

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   test_get_Minkowski_parameter();

   // Set methods

   test_set_Minkowski_parameter();

   // Error methods

   test_calculate_training_error();
   test_calculate_selection_error();
   test_calculate_training_error_gradient();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   cout << "End of Minkowski error test case.\n";
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
