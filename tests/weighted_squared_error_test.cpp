//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   W E I G H T E D   S Q U A R E D   E R R O R   T E S T   C L A S S     
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "weighted_squared_error_test.h"


WeightedSquaredErrorTest::WeightedSquaredErrorTest() : UnitTesting()
{
    weighted_squared_error.set(&neural_network, &data_set);

    weighted_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
}


WeightedSquaredErrorTest::~WeightedSquaredErrorTest()
{
}


void WeightedSquaredErrorTest::test_constructor()
{
   cout << "test_constructor\n";

   // Default

   WeightedSquaredError weighted_squared_error_1;

   assert_true(!weighted_squared_error_1.has_neural_network(), LOG);
   assert_true(!weighted_squared_error_1.has_data_set(), LOG);

   // Neural network and data set

   WeightedSquaredError weighted_squared_error_2(&neural_network, &data_set);

   assert_true(weighted_squared_error_2.has_neural_network(), LOG);
   assert_true(weighted_squared_error_2.has_data_set(), LOG);
}


void WeightedSquaredErrorTest::test_calculate_error()
{
   cout << "test_calculate_error\n";

   Tensor<type, 2> data;

   Index samples_number;
   Index inputs_number;
   Index outputs_number;
   Index neurons_number;

   Tensor<type, 1> parameters; 

   Tensor<Index,1> samples_indices;
   Tensor<Index,1> inputs_indices;
   Tensor<Index,1> targets_indices;

   // Test

   samples_number = 1;
   inputs_number = 1;
   outputs_number = 1;
   neurons_number = 1;

   neural_network.set(NeuralNetwork::Classification, {1, 2});

   neural_network.set_parameters_constant(1);
   data_set.set(1, 1, 1);

   data.resize(2, 2);
   data(0,0) = 0.0;
   data(0,1) = 0.0;
   data(1,0) = 1.0;
   data(1,1) = 1.0;

   data_set.set_data(data);
   data_set.set_training();

   weighted_squared_error.set_weights();

   samples_indices = data_set.get_used_samples_indices();
   inputs_indices = data_set.get_input_variables_indices();
   targets_indices = data_set.get_target_variables_indices();

   batch.set(samples_number, &data_set);
   batch.fill(samples_indices, inputs_indices, targets_indices);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &weighted_squared_error);
   weighted_squared_error.calculate_error(batch, forward_propagation, back_propagation);

   assert_true(back_propagation.error == 1, LOG);

   // Test

   data_set.set(3, 3, 1);

   data.resize(3, 3);
   data(0,0) = 0.0;
   data(0,1) = 0.0;
   data(0,2) = 0.0;
   data(1,0) = 1.0;
   data(1,1) = 1.0;
   data(1,2) = 1.0;
   data(2,0) = 1.0;
   data(2,1) = 0.0;
   data(2,2) = 0.0;
   data_set.set_data(data);

  neural_network.set(NeuralNetwork::Approximation, {3, 1});

  neural_network.set_parameters_constant(0.0);

  weighted_squared_error.set_weights();

  assert_true(weighted_squared_error.get_positives_weight() != weighted_squared_error.get_negatives_weight(), LOG);
}


void WeightedSquaredErrorTest::test_calculate_error_gradient()
{
   cout << "test_calculate_error_gradient\n";

   Tensor<type, 2> data;

   Tensor<Index,1> samples_indices;
   Tensor<Index,1> inputs_indices;
   Tensor<Index,1> targets_indices;

   Tensor<type, 1> error_gradient;
   Tensor<type, 1> numerical_error_gradient;
   Tensor<type, 0> maximum_difference;

   Index samples_number;
   Index inputs_number;
   Index outputs_number;
   Index neurons_number;

   ScalingLayer* scaling_layer = new ScalingLayer;

   RecurrentLayer* recurrent_layer = new RecurrentLayer;

   LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer;

   PerceptronLayer* perceptron_layer_1 = new PerceptronLayer;
   PerceptronLayer* perceptron_layer_2 = new PerceptronLayer;

   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer;

   Tensor<type, 1> parameters;


   // Test trivial
{
       neural_network.set(NeuralNetwork::Classification, {1, 1});

       neural_network.set_parameters_constant(1);
       data_set.set(1, 1, 1);

       samples_number = 1;
       inputs_number = 1;
       outputs_number = 1;
       neurons_number = 1;

       data.resize(2, 2);
       data(0,0) = 0.0;
       data(0,1) = 0.0;
       data(1,0) = 1.0;
       data(1,1) = 1.0;

       data_set.set_data(data);
       data_set.set_training();

       weighted_squared_error.set_weights();

       samples_indices = data_set.get_used_samples_indices();
       inputs_indices = data_set.get_input_variables_indices();
       targets_indices = data_set.get_target_variables_indices();

       batch.set(samples_number, &data_set);
       batch.fill(samples_indices, inputs_indices, targets_indices);

       forward_propagation.set(samples_number, &neural_network);
       neural_network.forward_propagate(batch, forward_propagation);

       back_propagation.set(samples_number, &weighted_squared_error);
       weighted_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//       weighted_squared_error.calculate_error(batch, forward_propagation, back_propagation);

       numerical_error_gradient = weighted_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(back_propagation.gradient(0)-1.1499 < 1e-3, LOG); // @todo 1e-2 precission
       assert_true(back_propagation.gradient(1)-0 < 1e-3, LOG);
}

   neural_network.set();

   // Test perceptron and probabilistic
{
   samples_number = 10;
   inputs_number = 3;
   outputs_number = 1;
   neurons_number = 2;

   data_set.set(samples_number, inputs_number, outputs_number);

   Tensor<type, 2> inputs(samples_number,inputs_number);

   inputs.setRandom();

   Tensor<type, 2> outputs(samples_number, outputs_number);
//   outputs[0] = 1.0;
//   outputs[1] = 0.0;

//   for(Index i = 2; i < samples_number; i++)
//   {
//        if((static_cast<Index>(inputs.calculate_row_sum(i))%2)) < numeric_limits<type>::min())
//        {
//            outputs[i] = 0.0;
//        }
//        else
//        {
//            outputs[i] = 1.0;
//        }
//   }

//   const Tensor<type, 2> data = inputs.append_column(outputs);

//   data_set.set_data(data);

   data_set.set_training();

   perceptron_layer_1->set(inputs_number, neurons_number);
   perceptron_layer_2->set(neurons_number, outputs_number);
   probabilistic_layer->set(outputs_number, outputs_number);

   neural_network.add_layer(perceptron_layer_1);
   neural_network.add_layer(perceptron_layer_2);
   neural_network.add_layer(probabilistic_layer);

   neural_network.set_parameters_random();

//   error_gradient = weighted_squared_error.calculate_error_gradient();

   numerical_error_gradient = weighted_squared_error.calculate_gradient_numerical_differentiation();

   maximum_difference = (error_gradient - numerical_error_gradient).abs().maximum();

   assert_true(maximum_difference(0) < 1.0e-3, LOG);
}

   neural_network.set();

   // Test lstm
{
   samples_number = 10;
   inputs_number = 3;
   outputs_number = 1;
   neurons_number = 2;

   data_set.set(samples_number, inputs_number, outputs_number);

   Tensor<type, 2> inputs(samples_number,inputs_number);

   inputs.setRandom();

   Tensor<type, 2> outputs(samples_number, outputs_number);

//   outputs[0] = 1.0;
//   outputs[1] = 0.0;

//   for(Index i = 2; i < samples_number; i++)
//   {
//        if((static_cast<Index>(inputs.calculate_row_sum(i))%2)) < numeric_limits<type>::min())
//        {
//            outputs[i] = 0.0;
//        }
//        else
//        {
//            outputs[i] = 1.0;
//        }
//   }

//   const Tensor<type, 2> data = inputs.append_column(outputs);

//   data_set.set_data(data);

//   data_set.set_training();

//   long_short_term_memory_layer->set(inputs_number, hidden_neurons);
//   perceptron_layer_2->set(hidden_neurons, outputs_number);

//   neural_network.add_layer(long_short_term_memory_layer);
//   neural_network.add_layer(perceptron_layer_2);

   neural_network.set_parameters_random();

//   error_gradient = weighted_squared_error.calculate_error_gradient();

//   numerical_error_gradient = weighted_squared_error.calculate_gradient_numerical_differentiation();

   maximum_difference = (error_gradient - numerical_error_gradient).abs().sum();

   assert_true(maximum_difference(0) < 1.0e-3, LOG);
}

//   neural_network.set();

   // Test recurrent
{
//   samples_number = 10;
//   inputs_number = 3;
//   outputs_number = 1;
//   hidden_neurons = 2;

//   data_set.set(samples_number, inputs_number, outputs_number);

//   Tensor<type, 2> inputs(samples_number,inputs_number);

//   inputs.setRandom();

//   Tensor<type, 1> outputs(samples_number, outputs_number);
//   outputs[0] = 1.0;
//   outputs[1] = 0.0;

//   for(Index i = 2; i < samples_number; i++)
//   {
//        if((static_cast<Index>(inputs.calculate_row_sum(i))%2)) < numeric_limits<type>::min())
//        {
//            outputs[i] = 0.0;
//        }
//        else
//        {
//            outputs[i] = 1.0;
//        }
//   }

//   const Tensor<type, 2> data = inputs.append_column(outputs);

//   data_set.set_data(data);

//   data_set.set_training();

//   recurrent_layer->set(inputs_number, hidden_neurons);
//   perceptron_layer_2->set(hidden_neurons, outputs_number);

//   neural_network.add_layer(recurrent_layer);
//   neural_network.add_layer(perceptron_layer_2);

   neural_network.set_parameters_random();

//   error_gradient = weighted_squared_error.calculate_error_gradient();

//   numerical_error_gradient = weighted_squared_error.calculate_gradient_numerical_differentiation();

   maximum_difference = (error_gradient - numerical_error_gradient).abs().sum();

   assert_true(maximum_difference(0) < 1.0e-3, LOG);
}

   // Test convolutional
{
//   samples_number = 5;
//   inputs_number = 147;
//   outputs_number = 1;

//   data_set.set(samples_number, inputs_number, outputs_number);

//   Tensor<type, 2> inputs(samples_number,inputs_number);
//   inputs.setRandom();

//   Tensor<type, 1> outputs(samples_number, outputs_number);
//   outputs[0] = 1.0;
//   outputs[1] = 0.0;

//   for(Index i = 2; i < samples_number; i++)
//   {
//        if((static_cast<Index>(inputs.calculate_row_sum(i))%2)) < numeric_limits<type>::min())
//        {
//            outputs[i] = 0.0;
//        }
//        else
//        {
//            outputs[i] = 1.0;
//        }
//   }

//   const Tensor<type, 2> data = inputs.append_column(outputs);

//   data_set.set_data(data);
//   data_set.set_input_variables_dimensions(Tensor<Index, 1>({3,7,7}));
//   data_set.set_target_variables_dimensions(Tensor<Index, 1>({1}));
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
//   Tensor<type, 1> biases(2, 0);
//   biases.setRandom(parameters_minimum, parameters_maximum);
//   convolutional_layer_2->set_biases(biases);

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

//   numerical_error_gradient = weighted_squared_error.calculate_gradient_numerical_differentiation();

//   error_gradient = weighted_squared_error.calculate_error_gradient();

       maximum_difference = (error_gradient - numerical_error_gradient).abs().sum();

       assert_true(maximum_difference(0) < 1.0e-3, LOG);
}
}


void WeightedSquaredErrorTest::test_calculate_squared_errors()
{
   cout << "test_calculate_squared_errors\n";

   Tensor<type, 1> parameters;
   
   type error;

   Tensor<type, 1> squared_errors;

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 1});
   neural_network.set_parameters_random();

   data_set.set(3, 2, 2);

//   data_set.generate_data_binary_classification(3, 2);

//   error = weighted_squared_error.calculate_error();

//   squared_errors = weighted_squared_error.calculate_training_error_terms();

//   assert_true(abs((squared_errors*squared_errors).sum() - error) < 1.0e-3, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {3, 1});
   neural_network.set_parameters_random();

   data_set.set(9, 3, 1);

//   data_set.generate_data_binary_classification(9, 3);

//   error = weighted_squared_error.calculate_error();

//   squared_errors = weighted_squared_error.calculate_training_error_terms();

//   assert_true(abs((squared_errors*squared_errors).sum() - error) < 1.0e-3, LOG);
}


void WeightedSquaredErrorTest::test_calculate_squared_errors_jacobian()
{
   cout << "test_calculate_squared_errors_jacobian\n";

   Tensor<Index, 1> samples_indices;
   Tensor<Index, 1> input_indices;
   Tensor<Index, 1> target_indices;

   Index samples_number;
   Index inputs_number;
   Index hidden_neurons_number;
   Index outputs_number;

   Tensor<type, 2> numerical_squared_errors_jacobian;

   // Test probabilistic (binary)
   {
       samples_number = 2;
       inputs_number = 2;
       hidden_neurons_number = 3;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       neural_network.set(NeuralNetwork::Classification, {inputs_number, hidden_neurons_number, outputs_number});

       neural_network.set_parameters_random();

       weighted_squared_error.set_normalization_coefficient();

       batch.set(samples_number, &data_set);
       batch.fill(samples_indices, input_indices, target_indices);

       forward_propagation.set(samples_number, &neural_network);
       neural_network.forward_propagate(batch, forward_propagation);

       back_propagation.set(samples_number, &weighted_squared_error);
       weighted_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       back_propagation_lm.set(samples_number, &weighted_squared_error);
//       weighted_squared_error.back_propagate(batch, forward_propagation, weighted_squared_error);

       numerical_squared_errors_jacobian = weighted_squared_error.calculate_Jacobian_numerical_differentiation();

       assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_squared_errors_jacobian, static_cast<type>(1e-3)), LOG);
   }
}


void WeightedSquaredErrorTest::run_test_case()
{
   cout << "Running weighted squared error test case...\n";

   // Constructor and destructor methods

   test_constructor();

   // Error methods

   test_calculate_error();

   test_calculate_error_gradient();

   // Squared errors methods

   test_calculate_squared_errors();
   test_calculate_squared_errors_jacobian();

   // Loss hessian methods

   cout << "End of weighted squared error test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lewser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lewser General Public License for more details.

// You should have received a copy of the GNU Lewser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
