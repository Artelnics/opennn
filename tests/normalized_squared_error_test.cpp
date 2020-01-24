//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z E D   S Q U A R E D   E R R O R   T E S T   C L A S S 
//
//   Artificial Intelligence Techniques, S.L. (Artelnics)                            
//   artelnics@artelnics.com

#include "normalized_squared_error_test.h"

NormalizedSquaredErrorTest::NormalizedSquaredErrorTest(void) : UnitTesting() 
{
}


NormalizedSquaredErrorTest::~NormalizedSquaredErrorTest(void)
{
}

/*
void NormalizedSquaredErrorTest::test_constructor(void)
{
   cout << "test_constructor\n";

   // Default

   NormalizedSquaredError normalized_squared_error_1;

   assert_true(normalized_squared_error_1.has_neural_network() == false, LOG);
   assert_true(normalized_squared_error_1.has_data_set() == false, LOG);

   // Neural network

   NeuralNetwork neural_network_2;
   NormalizedSquaredError normalized_squared_error_2(&neural_network_2);

   assert_true(normalized_squared_error_2.has_neural_network() == true, LOG);
   assert_true(normalized_squared_error_2.has_data_set() == false, LOG);

   // Neural network and data set

   NeuralNetwork neural_network_3;
   DataSet data_set_3;
   NormalizedSquaredError nse3(&neural_network_3, &data_set_3);

   assert_true(nse3.has_neural_network() == true, LOG);
   assert_true(nse3.has_data_set() == true, LOG);
}


void NormalizedSquaredErrorTest::test_destructor(void)
{
   cout << "test_destructor\n";
}


void NormalizedSquaredErrorTest::test_calculate_training_error(void)
{
   cout << "test_calculate_training_error\n";

   Tensor<type, 1> parameters;

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1,1});

   DataSet data_set(1,1,1);

   Index instances_number;
   Index inputs_number;
   Index outputs_number;
   Index hidden_neurons;

   Tensor<double, 2> new_data(2, 2);
   new_data(0,0) = -1.0;
   new_data(0,1) = -1.0;
   new_data(1,0) = 1.0;
   new_data(1,1) = 1.0;

   data_set.set_data(new_data);
   data_set.set_training();

   NormalizedSquaredError normalized_squared_error(&neural_network, &data_set);

//   assert_true(normalized_squared_error.calculate_training_error() == 0.0, LOG);

   // Test

   instances_number = 7;
   inputs_number = 8;
   outputs_number = 5;
   hidden_neurons = 3;

   neural_network.set(NeuralNetwork::Approximation, {inputs_number, hidden_neurons, outputs_number});
   neural_network.set_parameters_random();

   parameters = neural_network.get_parameters();

   data_set.set(instances_number, inputs_number, outputs_number);
   data_set.set_data_random();

   normalized_squared_error.set_normalization_coefficient();

   assert_true(abs(normalized_squared_error.calculate_training_error() - normalized_squared_error.calculate_training_error(parameters)) < 1.0e-03, LOG);
}


void NormalizedSquaredErrorTest::test_calculate_training_error_gradient(void)
{
   cout << "test_calculate_training_error_gradient\n";

   NeuralNetwork neural_network;

   DataSet data_set;

   NormalizedSquaredError nse(&neural_network, &data_set);

   Tensor<type, 1> error_gradient;
   Tensor<type, 1> numerical_error_gradient;

   Index instances_number;
   Index inputs_number;
   Index outputs_number;
   Index hidden_neurons;

//   ScalingLayer* scaling_layer = new ScalingLayer();

   RecurrentLayer* recurrent_layer = new RecurrentLayer();

   LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer();

   PerceptronLayer* hidden_perceptron_layer = new PerceptronLayer();
   PerceptronLayer* output_perceptron_layer = new PerceptronLayer();

   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer();

   // Test trivial
{
   instances_number = 10;
   inputs_number = 1;
   outputs_number = 1;

   data_set.set(instances_number, inputs_number, outputs_number);

   data_set.initialize_data(0.0);

   hidden_perceptron_layer->set(inputs_number, outputs_number);
   neural_network.add_layer(hidden_perceptron_layer);

   neural_network.set_parameters_constant(0.0);

   nse.set_normalization_coefficient(1.0);

   numerical_error_gradient = nse.calculate_training_error_gradient_numerical_differentiation();

   error_gradient = nse.calculate_training_error_gradient();

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

   data_set.set_data_random();

   data_set.set_training();

   hidden_perceptron_layer->set(inputs_number, hidden_neurons);
   output_perceptron_layer->set(hidden_neurons, outputs_number);
   probabilistic_layer->set(outputs_number, outputs_number);

   neural_network.add_layer(hidden_perceptron_layer);
   neural_network.add_layer(output_perceptron_layer);
   neural_network.add_layer(probabilistic_layer);

   neural_network.set_parameters_random();

   nse.set_normalization_coefficient();

   error_gradient = nse.calculate_training_error_gradient();

   numerical_error_gradient = nse.calculate_training_error_gradient_numerical_differentiation();

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

   data_set.set_data_random();

   data_set.set_training();

   long_short_term_memory_layer->set(inputs_number, hidden_neurons);
   output_perceptron_layer->set(hidden_neurons, outputs_number);

   neural_network.add_layer(long_short_term_memory_layer);
   neural_network.add_layer(output_perceptron_layer);

   neural_network.set_parameters_random();

   nse.set_normalization_coefficient();

   error_gradient = nse.calculate_training_error_gradient();

   numerical_error_gradient = nse.calculate_training_error_gradient_numerical_differentiation();

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

   data_set.set_data_random();

   data_set.set_training();

   recurrent_layer->set(inputs_number, hidden_neurons);
   output_perceptron_layer->set(hidden_neurons, outputs_number);

   neural_network.add_layer(recurrent_layer);
   neural_network.add_layer(output_perceptron_layer);

   neural_network.set_parameters_random();

   nse.set_normalization_coefficient();

   error_gradient = nse.calculate_training_error_gradient();

   numerical_error_gradient = nse.calculate_training_error_gradient_numerical_differentiation();

   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
}

   // Test convolutional
{
   instances_number = 5;
   inputs_number = 147;
   outputs_number = 1;

   data_set.set(instances_number, inputs_number, outputs_number);
   data_set.set_input_variables_dimensions(Tensor<Index, 1>({3,7,7}));
   data_set.set_target_variables_dimensions(Tensor<Index, 1>({1}));
   data_set.set_data_random();
   data_set.set_training();

   const double parameters_minimum = -100.0;
   const double parameters_maximum = 100.0;

   ConvolutionalLayer* convolutional_layer_1 = new ConvolutionalLayer({3,7,7}, {2,2,2});
   Tensor<double, 2> filters_1({2,3,2,2}, 0);
   filters_1.setRandom(parameters_minimum,parameters_maximum);
   convolutional_layer_1->set_synaptic_weights(filters_1);
   Tensor<type, 1> biases_1(2, 0);
   biases_1.setRandom(parameters_minimum, parameters_maximum);
   convolutional_layer_1->set_biases(biases_1);

   ConvolutionalLayer* convolutional_layer_2 = new ConvolutionalLayer(convolutional_layer_1->get_outputs_dimensions(), {2,2,2});
   convolutional_layer_2->set_padding_option(OpenNN::ConvolutionalLayer::Same);
   Tensor<double, 2> filters_2({2,2,2,2}, 0);
   filters_2.setRandom(parameters_minimum, parameters_maximum);
   convolutional_layer_2->set_synaptic_weights(filters_2);
   Tensor<type, 1> biases_2(2, 0);
   biases_2.setRandom(parameters_minimum, parameters_maximum);
   convolutional_layer_2->set_biases(biases_2);

   PoolingLayer* pooling_layer_1 = new PoolingLayer(convolutional_layer_2->get_outputs_dimensions(), {2,2});

   ConvolutionalLayer* convolutional_layer_3 = new ConvolutionalLayer(pooling_layer_1->get_outputs_dimensions(), {1,2,2});
   convolutional_layer_3->set_padding_option(OpenNN::ConvolutionalLayer::Same);
   Tensor<double, 2> filters_3({1,2,2,2}, 0);
   filters_3.setRandom(parameters_minimum, parameters_maximum);
   convolutional_layer_3->set_synaptic_weights(filters_3);
   Tensor<type, 1> biases_3(1, 0);
   biases_3.setRandom(parameters_minimum, parameters_maximum);
   convolutional_layer_3->set_biases(biases_3);

   PoolingLayer* pooling_layer_2 = new PoolingLayer(convolutional_layer_3->get_outputs_dimensions(), {2,2});
   pooling_layer_2->set_pooling_method(PoolingLayer::MaxPooling);

   PoolingLayer* pooling_layer_3 = new PoolingLayer(pooling_layer_2->get_outputs_dimensions(), {2,2});
   pooling_layer_3->set_pooling_method(PoolingLayer::MaxPooling);

   PerceptronLayer* perceptron_layer = new PerceptronLayer(pooling_layer_3->get_outputs_dimensions().calculate_product(), 3, OpenNN::PerceptronLayer::ActivationFunction::Linear);
   perceptron_layer->set_parameters_random(parameters_minimum, parameters_maximum);

   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(perceptron_layer->get_neurons_number(), outputs_number);
   probabilistic_layer->set_parameters_random(parameters_minimum, parameters_maximum);

   neural_network.set();
   neural_network.add_layer(convolutional_layer_1);
   neural_network.add_layer(convolutional_layer_2);
   neural_network.add_layer(pooling_layer_1);
   neural_network.add_layer(convolutional_layer_3);
   neural_network.add_layer(pooling_layer_2);
   neural_network.add_layer(pooling_layer_3);
   neural_network.add_layer(perceptron_layer);
   neural_network.add_layer(probabilistic_layer);

   numerical_error_gradient = nse.calculate_training_error_gradient_numerical_differentiation();

   error_gradient = nse.calculate_training_error_gradient();

   assert_true(absolute_value(numerical_error_gradient - error_gradient) < 1e-3, LOG);
}
}


void NormalizedSquaredErrorTest::test_calculate_training_error_terms(void)
{
   cout << "test_calculate_training_error_terms\n";

   NeuralNetwork neural_network;
   Tensor<Index, 1> architecture;
   Tensor<type, 1> network_parameters;

   DataSet data_set;

   NormalizedSquaredError nse(&neural_network, &data_set);

//   double error;

   Tensor<type, 1> error_terms;

   Index instances_number;
   Index inputs_number;
   Index outputs_number;

   // Test

   instances_number = 7;
   inputs_number = 6;
   outputs_number = 7;

   neural_network.set(NeuralNetwork::Approximation, {inputs_number, outputs_number});
   neural_network.set_parameters_random();

   data_set.set(instances_number, inputs_number, outputs_number);
   data_set.set_data_random();

   nse.set_normalization_coefficient();

//   error = nse.calculate_training_error();

//   error_terms = nse.calculate_training_error_terms();

//   assert_true(abs((error_terms*error_terms).sum() - error) < 1.0e-3, LOG);

}


void NormalizedSquaredErrorTest::test_calculate_training_error_terms_Jacobian(void)
{
   cout << "test_calculate_training_error_terms_Jacobian\n";

   NumericalDifferentiation nd;

   NeuralNetwork neural_network;
   Tensor<Index, 1> hidden_layers_size;
   Tensor<type, 1> network_parameters;

   DataSet data_set;

   NormalizedSquaredError nse(&neural_network, &data_set);

   Tensor<type, 1> error_gradient;

   Tensor<type, 1> error_terms;
   Tensor<double, 2> terms_Jacobian;
   Tensor<double, 2> numerical_Jacobian_terms;

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1});
   neural_network.set_parameters_random();
   network_parameters = neural_network.get_parameters();

   data_set.set(2, 1, 1);
   data_set.set_data_random();

//   terms_Jacobian = nse.calculate_error_terms_Jacobian();
   numerical_Jacobian_terms = nd.calculate_Jacobian(nse, &NormalizedSquaredError::calculate_training_error_terms, network_parameters);

   assert_true(absolute_value(terms_Jacobian-numerical_Jacobian_terms) < 1.0e-3, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 2, 2});
   neural_network.set_parameters_random();
   network_parameters = neural_network.get_parameters();

   data_set.set(2, 2, 2);
   data_set.set_data_random();

//   terms_Jacobian = nse.calculate_error_terms_Jacobian();
   numerical_Jacobian_terms = nd.calculate_Jacobian(nse, &NormalizedSquaredError::calculate_training_error_terms, network_parameters);

   assert_true(absolute_value(terms_Jacobian-numerical_Jacobian_terms) < 1.0e-3, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 2, 2});
   neural_network.set_parameters_random();

   data_set.set(2,2,2);
   data_set.set_data_random();
   
//   error_gradient = nse.calculate_error_gradient();

//   error_terms = nse.calculate_training_error_terms();
//   terms_Jacobian = nse.calculate_error_terms_Jacobian();

//   assert_true(absolute_value((terms_Jacobian.calculate_transpose()).dot(error_terms)*2.0 - error_gradient) < 1.0e-3, LOG);

}


void NormalizedSquaredErrorTest::test_calculate_squared_errors(void)
{
    cout << "test_calculate_squared_errors\n";

    NeuralNetwork neural_network;

    DataSet data_set;

    NormalizedSquaredError nse(&neural_network, &data_set);

    Tensor<type, 1> squared_errors;

    // Test

    neural_network.set(NeuralNetwork::Approximation, {1, 1});
    neural_network.set_parameters_random();

    data_set.set(2, 1, 1);
    data_set.set_data_random();

    squared_errors = nse.calculate_squared_errors();

    assert_true(squared_errors.size() == 2, LOG);
}


void NormalizedSquaredErrorTest::test_calculate_maximal_errors(void)
{
    cout << "test_calculate_maximal_errors\n";

    NeuralNetwork neural_network;

    DataSet data_set;

    NormalizedSquaredError nse(&neural_network, &data_set);

    Tensor<type, 1> squared_errors;
    Tensor<Index, 1> maximal_errors;

    // Test

    neural_network.set(NeuralNetwork::Approximation, {1, 1});
    neural_network.set_parameters_random();

    data_set.set(3, 1, 1);
    data_set.set_data_random();

    squared_errors = nse.calculate_squared_errors();
    maximal_errors = nse.calculate_maximal_errors(3);

    assert_true(maximal_errors.size() == 3, LOG);

    assert_true(squared_errors.get_subvector(maximal_errors).is_decrescent(), LOG);
}


void NormalizedSquaredErrorTest::test_to_XML(void)
{
   cout << "test_to_XML\n";
}


void NormalizedSquaredErrorTest::test_from_XML(void)
{
   cout << "test_from_XML\n";
}
*/

void NormalizedSquaredErrorTest::run_test_case(void)
{
   cout << "Running normalized squared error test case...\n";
/*
   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   // Set methods

   // Error methods

   test_calculate_training_error();

   test_calculate_training_error_gradient();

   // Error terms methods

   test_calculate_training_error_terms();

   test_calculate_training_error_terms_Jacobian();

   // Squared errors methods

   test_calculate_squared_errors();

   test_calculate_maximal_errors();

   // Serialization methods

   test_to_XML();
   test_from_XML();
*/
   cout << "End of normalized squared error test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lenser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lenser General Public License for more details.

// You should have received a copy of the GNU Lenser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
