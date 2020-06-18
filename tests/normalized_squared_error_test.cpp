//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z E D   S Q U A R E D   E R R O R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques, S.L. (Artelnics)
//   artelnics@artelnics.com

#include "normalized_squared_error_test.h"
#include <omp.h>

NormalizedSquaredErrorTest::NormalizedSquaredErrorTest(void) : UnitTesting()
{
}


NormalizedSquaredErrorTest::~NormalizedSquaredErrorTest(void)
{
}

void NormalizedSquaredErrorTest::test_constructor(void) // @todo
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


void NormalizedSquaredErrorTest::test_destructor(void) // @todo
{
   cout << "test_destructor\n";
}

void NormalizedSquaredErrorTest::test_calculate_normalization_coefficient(void) // @todo
{
   cout << "test_calculate_normalization_coefficient\n";

   NeuralNetwork neural_network;
   DataSet data_set;
   NormalizedSquaredError nse(&neural_network, &data_set);

   Index instances_number = 4;
   Index inputs_number = 4;
   Index outputs_number = 4;    //targets_number or means_number

   Tensor<type, 1> targets_mean(outputs_number);
   Tensor<type, 2> targets(instances_number, outputs_number);

   // Test

   data_set.generate_random_data(instances_number, inputs_number+outputs_number);

   Tensor<string, 1> uses(8);
   uses.setValues({"Input", "Input", "Input", "Input", "Target", "Target", "Target", "Target"});

   data_set.set_columns_uses(uses);

   targets = data_set.get_target_data();
   //targets_mean = data_set.calculate_training_targets_mean();

   Tensor<Index, 1> architecture(2);
   architecture.setValues({inputs_number, outputs_number});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_random();

//   data_set.set(instances_number, inputs_number, outputs_number);
//   data_set.set_data_random();

   type normalization_coefficient = nse.calculate_normalization_coefficient(targets, targets_mean);

   assert_true(normalization_coefficient > 0, LOG);
}


void NormalizedSquaredErrorTest::test_calculate_error(void) // @todo
{
   cout << "test_calculate_error\n";

   Tensor<Index, 1> architecture;
   Tensor<type, 1> parameters;

   NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);

   DataSet data_set(1,1,1);

   Index instances_number;
   Index inputs_number;
   Index outputs_number;
   Index hidden_neurons;

   Tensor<type, 2> new_data(2, 2);
   new_data(0,0) = -1.0;
   new_data(0,1) = -1.0;
   new_data(1,0) = 1.0;
   new_data(1,1) = 1.0;

   data_set.set_data(new_data);
   data_set.set_training();

   NormalizedSquaredError normalized_squared_error(&neural_network, &data_set);

//   assert_true(normalized_squared_error.calculate_error() == 0.0, LOG);

   // Test

   instances_number = 7;
   inputs_number = 8;
   outputs_number = 5;
   hidden_neurons = 3;

   architecture.setValues({inputs_number, hidden_neurons, outputs_number});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_random();

   parameters = neural_network.get_parameters();

   data_set.set(instances_number, inputs_number, outputs_number);
   data_set.set_data_random();

   normalized_squared_error.set_normalization_coefficient();

//   assert_true(abs(normalized_squared_error.calculate_error() - normalized_squared_error.calculate_training_error(parameters)) < 1.0e-3, LOG);
}


void NormalizedSquaredErrorTest::test_calculate_error_gradient(void) // @todo
{
   cout << "test_calculate_error_gradient\n";

//   NeuralNetwork neural_network;

//   DataSet data_set;

//   NormalizedSquaredError nse(&neural_network, &data_set);

//   Tensor<type, 1> error_gradient;
//   Tensor<type, 1> numerical_error_gradient;

//   Index instances_number;
//   Index inputs_number;
//   Index outputs_number;
//   Index hidden_neurons;

//   ScalingLayer* scaling_layer = new ScalingLayer();

//   RecurrentLayer* recurrent_layer = new RecurrentLayer();

//   LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer();

//   PerceptronLayer* hidden_perceptron_layer = new PerceptronLayer();
//   PerceptronLayer* output_perceptron_layer = new PerceptronLayer();

//   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer();

//   // Test trivial
//{
//   data_set.set_thread_pool_device(thread_pool_device);

//   instances_number = 10;
//   inputs_number = 1;
//   outputs_number = 1;

//   data_set.set(instances_number, inputs_number, outputs_number);
//   data_set.initialize_data(0.0);
//   data_set.set_training();

//   DataSet::Batch batch(instances_number, &data_set);

//   Tensor<Index, 1> instances_indices = data_set.get_training_instances_indices();
//   const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
//   const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

//   batch.fill(instances_indices, input_indices, target_indices);

//   hidden_perceptron_layer->set(inputs_number, outputs_number);
//   neural_network.add_layer(hidden_perceptron_layer);

//   neural_network.set_thread_pool_device(thread_pool_device);

//   neural_network.set_parameters_constant(0.0);

//   nse.set_normalization_coefficient(1.0);

//   nse.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

//   NeuralNetwork::ForwardPropagation forward_propagation(instances_number, &neural_network);
//   LossIndex::BackPropagation training_back_propagation(instances_number, &nse);

//   neural_network.forward_propagate(batch, forward_propagation);

//   nse.set_thread_pool_device(thread_pool_device);

//   nse.back_propagate(batch, forward_propagation, training_back_propagation);
//   error_gradient = training_back_propagation.gradient;

//   numerical_error_gradient = nse.calculate_error_gradient_numerical_differentiation(&nse);

//   assert_true((error_gradient.dimension(0) == neural_network.get_parameters_number()) , LOG);
//   assert_true(std::all_of(error_gradient.data(), error_gradient.data()+error_gradient.size(), [](type i) { return (i-static_cast<type>(0))<std::numeric_limits<type>::min(); }), LOG);
//}

//   neural_network.set();

//   // Test perceptron and probabilistic
//{

//   instances_number = 3;
//   inputs_number = 1;
//   hidden_neurons = 1;
//   outputs_number = 3;

//   data_set.set(instances_number, inputs_number, outputs_number);

//   Tensor<type,2> data(3,4);

//   data(0,0) = static_cast<type>(0.2);
//   data(0,1) = 1;
//   data(0,2) = 0;
//   data(0,3) = 0;
//   data(1,0) = static_cast<type>(0.3);
//   data(1,1) = 0;
//   data(1,2) = 1;
//   data(1,3) = 0;
//   data(2,0) = static_cast<type>(0.5);
//   data(2,1) = 0;
//   data(2,2) = 0;
//   data(2,3) = 1;

//   data_set.set_data(data);

//   Tensor<string, 1> columns_uses(4);
//   columns_uses(0) = "Input";
//   columns_uses(1) = "Target";
//   columns_uses(2) = "Target";
//   columns_uses(3) = "Target";

//   data_set.set_columns_uses(columns_uses);

//   cout << "Data: " << data_set.get_data() << endl;

//   data_set.set_training();

//   DataSet::Batch batch(instances_number, &data_set);

//   Tensor<Index, 1> instances_indices = data_set.get_training_instances_indices();
//   const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
//   const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

//   batch.fill(instances_indices, input_indices, target_indices);

//   hidden_perceptron_layer->set(inputs_number, outputs_number);
//   output_perceptron_layer->set(hidden_neurons, outputs_number);
//   probabilistic_layer->set(outputs_number, outputs_number);

//   neural_network.add_layer(hidden_perceptron_layer);
//   neural_network.add_layer(output_perceptron_layer);
//  neural_network.add_layer(probabilistic_layer);

//   neural_network.set_thread_pool_device(thread_pool_device);

//   Tensor<type,2> perceptron_weights(1,3);
//   perceptron_weights(0,0) = static_cast<type>(-0.318137);
//   perceptron_weights(0,1) = static_cast<type>(-0.0745666);
//   perceptron_weights(0,2) = static_cast<type>(-0.0732468);

//   Tensor<type,2> perceptron_biases(1,3);
//   perceptron_biases(0,0) = static_cast<type>(1.88443);
//   perceptron_biases(0,1) = static_cast<type>(0.776795);
//   perceptron_biases(0,2) = static_cast<type>(0.0126074);

//   Tensor<type,2> probabilistic_weights(3,3);
//   probabilistic_weights(0,0) = static_cast<type>(-0.290916);
//   probabilistic_weights(0,1) = static_cast<type>(2.1764);
//   probabilistic_weights(0,2) = static_cast<type>(-1.71237);
//   probabilistic_weights(1,0) = static_cast<type>(-0.147688);
//   probabilistic_weights(1,1) = static_cast<type>(1.71663);
//   probabilistic_weights(1,2) = static_cast<type>(0.349156);
//   probabilistic_weights(2,0) = static_cast<type>(-0.302181);
//   probabilistic_weights(2,1) = static_cast<type>(1.18804);
//   probabilistic_weights(2,2) = static_cast<type>(0.754033);

//   Tensor<type,2> probabilistic_biases(1,3);
//   probabilistic_biases(0,0) = static_cast<type>(1.95245);
//   probabilistic_biases(0,1) = static_cast<type>(0.68821);
//   probabilistic_biases(0,2) = static_cast<type>(1.75451);

//   neural_network.set_parameters_random();

//   hidden_perceptron_layer->set_synaptic_weights(perceptron_weights);
//   hidden_perceptron_layer->set_biases(perceptron_biases);

//   probabilistic_layer->set_synaptic_weights(probabilistic_weights);
//   probabilistic_layer->set_biases(probabilistic_biases);

//   cout << "perceptron w: " << hidden_perceptron_layer->get_synaptic_weights() << endl;
//   cout << "perceptron b: " << hidden_perceptron_layer->get_biases() << endl;

//   cout << "probabilistic w: " << probabilistic_layer->get_synaptic_weights() << endl;
//   cout << "probabilistic b: " << probabilistic_layer->get_biases() << endl;

//   nse.set_normalization_coefficient();

//   nse.set_thread_pool_device(thread_pool_device);

//   NeuralNetwork::ForwardPropagation forward_propagation(instances_number, &neural_network);
//   LossIndex::BackPropagation training_back_propagation(instances_number, &nse);

//   neural_network.forward_propagate(batch, forward_propagation);

//   nse.back_propagate(batch, forward_propagation, training_back_propagation);

//   error_gradient = training_back_propagation.gradient;

//   numerical_error_gradient = nse.calculate_error_gradient_numerical_differentiation(&nse);

//   const Tensor<type, 1> difference = error_gradient-numerical_error_gradient;

//   assert_true(std::all_of(difference.data(), difference.data()+difference.size(), [](type i) { return (i)<static_cast<type>(1.0e-3); }), LOG);
//}


//   neural_network.set();

//   // Test lstm
//{
//   instances_number = 10;
//   inputs_number = 3;
//   outputs_number = 2;
//   hidden_neurons = 2;

//   data_set.set(instances_number, inputs_number, outputs_number);

//   data_set.set_data_random();

//   data_set.set_training();

//   long_short_term_memory_layer->set(inputs_number, hidden_neurons);
//   output_perceptron_layer->set(hidden_neurons, outputs_number);

//   neural_network.add_layer(long_short_term_memory_layer);
//   neural_network.add_layer(output_perceptron_layer);

//   neural_network.set_parameters_random();

//   nse.set_normalization_coefficient();

//   error_gradient = nse.calculate_error_gradient();

//   numerical_error_gradient = nse.calculate_error_gradient_numerical_differentiation();

//   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
//}

//   neural_network.set();

//   // Test recurrent
//{
//   instances_number = 10;
//   inputs_number = 3;
//   outputs_number = 2;
//   hidden_neurons = 2;

//   data_set.set(instances_number, inputs_number, outputs_number);

//   data_set.set_data_random();

//   data_set.set_training();

//   recurrent_layer->set(inputs_number, hidden_neurons);
//   output_perceptron_layer->set(hidden_neurons, outputs_number);

//   neural_network.add_layer(recurrent_layer);
//   neural_network.add_layer(output_perceptron_layer);

//   neural_network.set_parameters_random();

//   nse.set_normalization_coefficient();

//   error_gradient = nse.calculate_error_gradient();

//   numerical_error_gradient = nse.calculate_error_gradient_numerical_differentiation();

//   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
//}

//   // Test convolutional
//{
//   instances_number = 5;
//   inputs_number = 147;
//   outputs_number = 1;

//   data_set.set(instances_number, inputs_number, outputs_number);
//   data_set.set_input_variables_dimensions(Tensor<Index, 1>({3,7,7}));
//   data_set.set_target_variables_dimensions(Tensor<Index, 1>({1}));
//   data_set.set_data_random();
//   data_set.set_training();

//   const type parameters_minimum = -100.0;
//   const type parameters_maximum = 100.0;

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
//   Tensor<type, 1> biases_3(1, 0);
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

//   neural_network.set();
//   neural_network.add_layer(convolutional_layer_1);
//   neural_network.add_layer(convolutional_layer_2);
//   neural_network.add_layer(pooling_layer_1);
//   neural_network.add_layer(convolutional_layer_3);
//   neural_network.add_layer(pooling_layer_2);
//   neural_network.add_layer(pooling_layer_3);
//   neural_network.add_layer(perceptron_layer);
//   neural_network.add_layer(probabilistic_layer);

//   numerical_error_gradient = nse.calculate_error_gradient_numerical_differentiation();

//   error_gradient = nse.calculate_error_gradient();

//   assert_true(absolute_value(numerical_error_gradient - error_gradient) < 1e-3, LOG);
//}
}


void NormalizedSquaredErrorTest::test_calculate_error_terms(void) // @todo
{
   cout << "test_calculate_error_terms\n";

//   NeuralNetwork neural_network;
//   Tensor<Index, 1> architecture;
//   Tensor<type, 1> network_parameters;

//   DataSet data_set;

//   NormalizedSquaredError nse(&neural_network, &data_set);

//   type error;

//   Tensor<type, 1> error_terms;

//   Index instances_number;
//   Index inputs_number;
//   Index outputs_number;

//   // Test

//   instances_number = 7;
//   inputs_number = 6;
//   outputs_number = 7;

//   neural_network.set(NeuralNetwork::Approximation, {inputs_number, outputs_number});
//   neural_network.set_parameters_random();

//   data_set.set(instances_number, inputs_number, outputs_number);
//   data_set.set_data_random();

//   nse.set_normalization_coefficient();

//   error = nse.calculate_error();

//   error_terms = nse.calculate_training_error_terms();

//   assert_true(abs((error_terms*error_terms).sum() - error) < 1.0e-3, LOG);

}


void NormalizedSquaredErrorTest::test_calculate_error_terms_Jacobian(void) // @todo
{
   cout << "test_calculate_error_terms_Jacobian\n";

//   NumericalDifferentiation nd;

//   NeuralNetwork neural_network;
//   Tensor<Index, 1> hidden_layers_size;
//   Tensor<type, 1> network_parameters;

//   DataSet data_set;

//   NormalizedSquaredError nse(&neural_network, &data_set);

//   Tensor<type, 1> error_gradient;

//   Tensor<type, 1> error_terms;
//   Tensor<type, 2> terms_Jacobian;
//   Tensor<type, 2> numerical_Jacobian_terms;

//   // Test

//   architecture.setValues({1,1});

//   neural_network.set(NeuralNetwork::Approximation, architecture);
//   neural_network.set_parameters_random();
//   network_parameters = neural_network.get_parameters();

//   data_set.set(2, 1, 1);
//   data_set.set_data_random();

//   terms_Jacobian = nse.calculate_error_terms_Jacobian();
//   numerical_Jacobian_terms = nd.calculate_Jacobian(nse, &NormalizedSquaredError::calculate_training_error_terms, network_parameters);

//   assert_true(absolute_value(terms_Jacobian-numerical_Jacobian_terms) < 1.0e-3, LOG);

//   // Test

//   neural_network.set(NeuralNetwork::Approximation, {2, 2, 2});
//   neural_network.set_parameters_random();
//   network_parameters = neural_network.get_parameters();

//   data_set.set(2, 2, 2);
//   data_set.set_data_random();

//   terms_Jacobian = nse.calculate_error_terms_Jacobian();
//   numerical_Jacobian_terms = nd.calculate_Jacobian(nse, &NormalizedSquaredError::calculate_training_error_terms, network_parameters);

//   assert_true(absolute_value(terms_Jacobian-numerical_Jacobian_terms) < 1.0e-3, LOG);

//   // Test

//   neural_network.set(NeuralNetwork::Approximation, {2, 2, 2});
//   neural_network.set_parameters_random();

//   data_set.set(2,2,2);
//   data_set.set_data_random();

//   error_gradient = nse.calculate_error_gradient();

//   error_terms = nse.calculate_training_error_terms();
//   terms_Jacobian = nse.calculate_error_terms_Jacobian();

//   assert_true(absolute_value((terms_Jacobian.calculate_transpose()).dot(error_terms)*2.0 - error_gradient) < 1.0e-3, LOG);

}


void NormalizedSquaredErrorTest::test_calculate_squared_errors(void) // @todo
{
    cout << "test_calculate_squared_errors\n";

//    NeuralNetwork neural_network;

//    DataSet data_set;

//    NormalizedSquaredError nse(&neural_network, &data_set);

//    Tensor<type, 1> squared_errors;

//    // Test

//    architecture.setValues({1,1});

//    neural_network.set(NeuralNetwork::Approximation, architecture);
//    neural_network.set_parameters_random();

//    data_set.set(2, 1, 1);
//    data_set.set_data_random();

//    squared_errors = nse.calculate_squared_errors();

//    assert_true(squared_errors.size() == 2, LOG);
}


void NormalizedSquaredErrorTest::test_calculate_maximal_errors(void) // @todo
{
//    cout << "test_calculate_maximal_errors\n";

//    NeuralNetwork neural_network;

//    DataSet data_set;

//    NormalizedSquaredError nse(&neural_network, &data_set);

//    Tensor<type, 1> squared_errors;
//    Tensor<Index, 1> maximal_errors;

//    // Test

//    architecture.setValues({1,1});

//    neural_network.set(NeuralNetwork::Approximation, architecture);
//    neural_network.set_parameters_random();

//    data_set.set(3, 1, 1);
//    data_set.set_data_random();

//    squared_errors = nse.calculate_squared_errors();
//    maximal_errors = nse.calculate_maximal_errors(3);

//    assert_true(maximal_errors.size() == 3, LOG);

//    assert_true(squared_errors.get_subvector(maximal_errors).is_decrescent(), LOG);
}


void NormalizedSquaredErrorTest::test_to_XML(void) // @todo
{
   cout << "test_to_XML\n";
}


void NormalizedSquaredErrorTest::test_from_XML(void) // @todo
{
   cout << "test_from_XML\n";
}


void NormalizedSquaredErrorTest::run_test_case(void) // @todo
{
   cout << "Running normalized squared error test case...\n";

   // Constructor and destructor methods

//   test_constructor();
//   test_destructor();

//   test_calculate_normalization_coefficient();


//   // Get methods

//   // Set methods

//   // Error methods

//   test_calculate_error();

//   test_calculate_error_gradient();

//   // Error terms methods

//   test_calculate_error_terms();

//   test_calculate_error_terms_Jacobian();

//   // Squared errors methods

//   test_calculate_squared_errors();

//   test_calculate_maximal_errors();

//   // Serialization methods

//   test_to_XML();
//   test_from_XML();

   cout << "End of normalized squared error test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2020 Artificial Intelligence Techniques SL.
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
