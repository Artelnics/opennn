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


void CrossEntropyErrorTest::test_calculate_error()
{
   cout << "test_calculate_error\n";

   DataSet data_set;

   NeuralNetwork neural_network;
   Tensor<type, 1> parameters;
   Tensor<Index, 1> architecture(2);
/*
   PerceptronLayer perceptron_layer;
   ScalingLayer scaling_layer;

   CrossEntropyError cee(&neural_network, &data_set);

   type cross_entropy_error;

   // Test

   data_set.generate_sum_data(10,2);

//   Tensor<Descriptives, 1> inputs = data_set.scale_input_variables_minimum_maximum();

   scaling_layer.set_neurons_number(1);
   scaling_layer.set_inputs_number(1);
//   scaling_layer.set_descriptives(inputs);

//   data_set.set(10,1,1);

//   Tensor<type, 2> data(10,2);
//   data.initialize_identity();

//   data_set.set_data(data);

   data_set.set_training();

   architecture.setValues({1,1});

   neural_network.set(NeuralNetwork::Approximation, architecture);

   neural_network.set_parameters_constant(0.0);

   parameters = neural_network.get_parameters();

//   cross_entropy_error = cee.calculate_error();

//   assert_true(abs(cross_entropy_error - cee.calculate_training_error(parameters)) < 1.0e-3, LOG);
*/

}


void CrossEntropyErrorTest::test_calculate_error_gradient()
{
   cout << "test_calculate_error_gradient\n";

   NeuralNetwork neural_network;

   DataSet data_set;

   CrossEntropyError cee(&neural_network, &data_set);

   Tensor<type, 1> error_gradient;
   Tensor<type, 1> numerical_error_gradient;

   Index samples_number;
   Index inputs_number;
   Index outputs_number;
   Index hidden_neurons;

//   ScalingLayer scaling_layer;

//   RecurrentLayer recurrent_layer;

//   LongShortTermMemoryLayer long_short_term_memory_layer;

   neural_network.set();

   // Test perceptron and probabilistic
{
   samples_number = 10;
   inputs_number = 3;
   outputs_number = 2;
   hidden_neurons = 2;

   data_set.set(samples_number, inputs_number, outputs_number);

   data_set.set_data_random();

   data_set.set_training();

   Tensor<Index, 1> architecture(3);
   architecture.setValues({inputs_number, hidden_neurons, outputs_number});

//   neural_network.set(NeuralNetwork::Classification, architecture);

//   neural_network.set_parameters_random();

//   error_gradient = cee.calculate_error_gradient();

//   numerical_error_gradient = cee.calculate_error_gradient_numerical_differentiation();

//   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);

}

   neural_network.set();

   // Test lstm
{
   samples_number = 10;
   inputs_number = 3;
   outputs_number = 2;
   hidden_neurons = 2;

   data_set.set(samples_number, inputs_number, outputs_number);

   data_set.set_data_random();

   data_set.set_training();

//   Tensor<Index, 1> architecture;

//   architecture.setValues({inputs_number, hidden_neurons, outputs_number});

//   neural_network.set(NeuralNetwork::Forecasting, architecture);

//   long_short_term_memory_layer.set(inputs_number, hidden_neurons);
//   output_perceptron_layer.set(hidden_neurons, outputs_number);

//   neural_network.add_layer(&long_short_term_memory_layer);
//   neural_network.add_layer(&output_perceptron_layer);

//   neural_network.set_parameters_random();

//   error_gradient = cee.calculate_error_gradient();

//   numerical_error_gradient = cee.calculate_error_gradient_numerical_differentiation();

//   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
}

   neural_network.set();

   // Test recurrent

{
   samples_number = 10;
   inputs_number = 3;
   outputs_number = 2;
   hidden_neurons = 2;

   data_set.set(samples_number, inputs_number, outputs_number);

   data_set.set_data_random();

   data_set.set_training();

//   recurrent_layer.set(inputs_number, hidden_neurons);
//   output_perceptron_layer.set(hidden_neurons, outputs_number);

//   neural_network.add_layer(&scaling_layer);
//   neural_network.add_layer(&recurrent_layer);
//   neural_network.add_layer(&output_perceptron_layer);

   neural_network.set_parameters_random();

//   error_gradient = cee.calculate_error_gradient();

//   numerical_error_gradient = cee.calculate_error_gradient_numerical_differentiation();

//   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
}

   // Test convolutional
{
   samples_number = 5;
   inputs_number = 147;
   outputs_number = 1;

   data_set.set(samples_number, inputs_number, outputs_number);
//   data_set.set_input_variables_dimensions(Tensor<Index, 1>({3,7,7}));
//   data_set.set_target_variables_dimensions(Tensor<Index, 1>({1}));
//   data_set.set_data_random();
//   data_set.set_training();

   const type parameters_minimum = -100.0;
   const type parameters_maximum = 100.0;

//   ConvolutionalLayer* convolutional_layer_1 = new ConvolutionalLayer({3,7,7}, {2,2,2});
//   Tensor<type, 2> filters_1({2,3,2,2}, 0);
//   filters_1.setRandom(parameters_minimum,parameters_maximum);
//   convolutional_layer_1->set_synaptic_weights(filters_1);
//   Tensor<type, 1> biases_1(2, 0);
//   biases_1.setRandom(parameters_minimum, parameters_maximum);
//   convolutional_layer_1->set_biases(biases_1);

//   ConvolutionalLayer* convolutional_layer_2 = new ConvolutionalLayer(convolutional_layer_1->get_outputs_dimensions(), {2,2,2});
//   convolutional_layer_2->set_padding_option(OpenNN::ConvolutionalLayer::Same);
//   Tensor<type, 2> filters_2({2,2,2,2}, 0);
//   filters_2.setRandom(parameters_minimum, parameters_maximum);
//   convolutional_layer_2->set_synaptic_weights(filters_2);
//   Tensor<type, 1> biases_2(2, 0);
//   biases_2.setRandom(parameters_minimum, parameters_maximum);
//   convolutional_layer_2->set_biases(biases_2);

//   PoolingLayer* pooling_layer_1 = new PoolingLayer(convolutional_layer_2->get_outputs_dimensions(), {2,2});

//   ConvolutionalLayer* convolutional_layer_3 = new ConvolutionalLayer(pooling_layer_1->get_outputs_dimensions(), {1,2,2});
//   convolutional_layer_3->set_padding_option(OpenNN::ConvolutionalLayer::Same);
//   Tensor<type, 2> filters_3({1,2,2,2}, 0);
//   filters_3.setRandom(parameters_minimum, parameters_maximum);
//   convolutional_layer_3->set_synaptic_weights(filters_3);
//   Tensor<type, 1> biases_3(1);
//   biases_3.setRandom(parameters_minimum, parameters_maximum);
//   convolutional_layer_3->set_biases(biases_3);

//   PoolingLayer* pooling_layer_2 = new PoolingLayer(convolutional_layer_3->get_outputs_dimensions(), {2,2});
//   pooling_layer_2->set_pooling_method(PoolingLayer::MaxPooling);

//   PoolingLayer* pooling_layer_3 = new PoolingLayer(pooling_layer_2->get_outputs_dimensions(), {2,2});
//   pooling_layer_3->set_pooling_method(PoolingLayer::MaxPooling);

//   PerceptronLayer* perceptron_layer = new PerceptronLayer(pooling_layer_3->get_outputs_dimensions().calculate_product(), 3, OpenNN::PerceptronLayer::ActivationFunction::Linear);
//   perceptron_layer->set_parameters_random(parameters_minimum, parameters_maximum);

//   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(perceptron_layer->get_neurons_number(), outputs_number);
//   probabilistic_layer->set_parameters_random(parameters_minimum, parameters_maximum);

   neural_network.set();
//   neural_network.add_layer(convolutional_layer_1);
//   neural_network.add_layer(convolutional_layer_2);
//   neural_network.add_layer(pooling_layer_1);
//   neural_network.add_layer(convolutional_layer_3);
//   neural_network.add_layer(pooling_layer_2);
//   neural_network.add_layer(pooling_layer_3);
//   neural_network.add_layer(perceptron_layer);
//   neural_network.add_layer(probabilistic_layer);

//   numerical_error_gradient = cee.calculate_error_gradient_numerical_differentiation();

//   error_gradient = cee.calculate_error_gradient();

//   assert_true(absolute_value(numerical_error_gradient - error_gradient) < 1e-3, LOG);
}

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

   test_calculate_error();
   test_calculate_error_gradient();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   cout << "End of cross entropy error test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2020 Artificial Intelligence Techniques, SL.
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
