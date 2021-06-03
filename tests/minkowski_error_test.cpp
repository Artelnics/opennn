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
    minkowski_error.set(&neural_network, &data_set);

    minkowski_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
}


MinkowskiErrorTest::~MinkowskiErrorTest() 
{
}


void MinkowskiErrorTest::test_constructor()
{
   cout << "test_constructor\n";

   // Default

   MinkowskiError minkowski_error_1;

   assert_true(!minkowski_error_1.has_neural_network(), LOG);
   assert_true(!minkowski_error_1.has_data_set(), LOG);

   // Neural network and data set

   MinkowskiError minkowski_error_2(&neural_network, &data_set);

   assert_true(minkowski_error_2.has_neural_network(), LOG);
   assert_true(minkowski_error_2.has_data_set(), LOG);
}


void MinkowskiErrorTest::test_get_Minkowski_parameter()
{
   cout << "test_get_Minkowski_parameter\n"; 

   minkowski_error.set_Minkowski_parameter(1.0);
   
   assert_true(minkowski_error.get_Minkowski_parameter() == 1.0, LOG);
}


void MinkowskiErrorTest::test_calculate_error()
{
   cout << "test_calculate_error\n";

   Index samples_number;
   Index inputs_number;
   Index targets_number;

   Tensor<type, 1> parameters;

   Tensor<type, 2> data;

   Tensor<Index,1> training_samples_indices;
   Tensor<Index,1> inputs_indices;
   Tensor<Index,1> targets_indices;

   // Test

   data_set.set(1, 1, 1);
   data_set.set_data_constant(0.0);

   minkowski_error.set_Minkowski_parameter(1.5);

   // Test trivial

   samples_number = 1;
   inputs_number = 1;
   targets_number = 1;

   training_samples_indices = data_set.get_training_samples_indices();
   inputs_indices = data_set.get_input_variables_indices();
   targets_indices = data_set.get_target_variables_indices();

   neural_network.set(NeuralNetwork::Approximation, {inputs_number, targets_number});
   neural_network.set_parameters_constant(0);

   batch.set(samples_number, &data_set);
   batch.fill(training_samples_indices, inputs_indices, targets_indices);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &minkowski_error);
   minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

   minkowski_error.calculate_error(batch, forward_propagation, back_propagation);

   assert_true(back_propagation.error == 0.0, LOG);

   // Test

   neural_network.set_parameters_constant(1);

   forward_propagation.set(samples_number, &neural_network);
   back_propagation.set(samples_number, &minkowski_error);

   neural_network.forward_propagate(batch, forward_propagation);
   minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

   minkowski_error.calculate_error(batch, forward_propagation, back_propagation);

   assert_true(back_propagation.error - 0.761 < 1.0e-3, LOG);
}


void MinkowskiErrorTest::test_calculate_error_gradient()
{
   cout << "test_calculate_error_gradient\n";

   Index samples_number;

   Tensor<Index, 1> samples_indices;
   Tensor<Index, 1> input_indices;
   Tensor<Index, 1> target_indices;

   Index inputs_number;
   Index outputs_number;
   Index neurons_number;

   RecurrentLayer* recurrent_layer = new RecurrentLayer;

   LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer;

   PerceptronLayer* perceptron_layer_1 = new PerceptronLayer;
   PerceptronLayer* perceptron_layer_2 = new PerceptronLayer;

   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer;

   Tensor<type, 1> error_gradient;
   Tensor<type, 1> numerical_error_gradient;

   // Trivial test
   {
       samples_number = 10;
       inputs_number = 1;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);
       data_set.set_data_constant(0.0);
       data_set.set_training();

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       perceptron_layer_1->set(inputs_number, outputs_number);
       neural_network.add_layer(perceptron_layer_1);

       neural_network.set_parameters_constant(0.0);

       batch.set(samples_number, &data_set);
       batch.fill(samples_indices, input_indices, target_indices);

       forward_propagation.set(samples_number, &neural_network);
       neural_network.forward_propagate(batch, forward_propagation);

       back_propagation.set(samples_number, &minkowski_error);
       minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = minkowski_error.calculate_gradient_numerical_differentiation();

       assert_true((error_gradient.dimension(0) == neural_network.get_parameters_number()) , LOG);
       assert_true(all_of(error_gradient.data(), error_gradient.data()+error_gradient.size(),
                          [](type i) { return (i-static_cast<type>(0))<numeric_limits<type>::min(); }), LOG);
   }

   neural_network.set();

   // Test perceptron

   {
       samples_number = 10;
       inputs_number = 3;
       outputs_number = 5;

       const Index neurons_number = 6;

       data_set.set(samples_number, inputs_number, outputs_number);
       data_set.set_data_random();
       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       perceptron_layer_1->set(inputs_number, neurons_number);
       perceptron_layer_2->set(neurons_number, outputs_number);

       neural_network.add_layer(perceptron_layer_1);
       neural_network.add_layer(perceptron_layer_2);

       neural_network.set_parameters_random();

       forward_propagation.set(samples_number, &neural_network);
       back_propagation.set(samples_number, &minkowski_error);

       neural_network.forward_propagate(batch, forward_propagation);

       minkowski_error.back_propagate(batch, forward_propagation, back_propagation);
       error_gradient = back_propagation.gradient;

       numerical_error_gradient = minkowski_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   // Test perceptron and binary probabilistic
   {
       samples_number = 3;
       inputs_number = 3;
       neurons_number = 4;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       neural_network.set(NeuralNetwork::Classification, {inputs_number, neurons_number, outputs_number});

       neural_network.set_parameters_random();

       batch.set(samples_number, &data_set);
       batch.fill(samples_indices, input_indices, target_indices);

       forward_propagation.set(samples_number, &neural_network);
       neural_network.forward_propagate(batch, forward_propagation);

       back_propagation.set(samples_number, &minkowski_error);
       minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = minkowski_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test perceptron and multiple probabilistic
   {
       samples_number = 3;
       inputs_number = 3;
       neurons_number = 2;
       outputs_number = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.set(samples_number, &data_set);
       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Classification, {inputs_number, neurons_number, outputs_number});

       neural_network.set_parameters_random();

       forward_propagation.set(samples_number, &neural_network);
       back_propagation.set(samples_number, &minkowski_error);

       neural_network.forward_propagate(batch, forward_propagation);

       minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = minkowski_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test lstm

   {
       samples_number = 4;
       inputs_number = 3;
       outputs_number = 2;
       neurons_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_random();

       data_set.set_training();

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       long_short_term_memory_layer->set(inputs_number, neurons_number);

       neural_network.add_layer(long_short_term_memory_layer);

       neural_network.set_parameters_random();

       long_short_term_memory_layer->set_timesteps(2);

       batch.set(samples_number, &data_set);
       batch.fill(samples_indices, input_indices, target_indices);

       forward_propagation.set(samples_number, &neural_network);
       neural_network.forward_propagate(batch, forward_propagation);

       back_propagation.set(samples_number, &minkowski_error);
       minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = minkowski_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test recurrent
   {
       samples_number = 4;
       inputs_number = 2;
       outputs_number = 1;
       neurons_number = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_random();

       data_set.set_training();

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       recurrent_layer->set(inputs_number, neurons_number);

       neural_network.add_layer(recurrent_layer);

       neural_network.set_parameters_random();

       recurrent_layer->set_timesteps(2);

       batch.set(samples_number, &data_set);
       batch.fill(samples_indices, input_indices, target_indices);

       forward_propagation.set(samples_number, &neural_network);
       neural_network.forward_propagate(batch, forward_propagation);

       back_propagation.set(samples_number, &minkowski_error);
       minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = minkowski_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test recurrent and perceptron
   {
       samples_number = 4;
       inputs_number = 2;
       outputs_number = 1;
       neurons_number = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_random();

       data_set.set_training();

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       recurrent_layer->set(inputs_number, neurons_number);
       perceptron_layer_1->set(neurons_number, outputs_number);

       neural_network.add_layer(recurrent_layer);
       neural_network.add_layer(perceptron_layer_1);

       neural_network.set_parameters_random();

       recurrent_layer->set_timesteps(2);

       batch.set(samples_number, &data_set);
       batch.fill(samples_indices, input_indices, target_indices);

       forward_propagation.set(samples_number, &neural_network);
       neural_network.forward_propagate(batch, forward_propagation);

       back_propagation.set(samples_number, &minkowski_error);
       minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = minkowski_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test recurrent and binary probabilistic
   {
       samples_number = 4;
       inputs_number = 3;
       neurons_number = 4;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       recurrent_layer->set(inputs_number, neurons_number);
       probabilistic_layer->set(neurons_number, outputs_number);

       neural_network.add_layer(recurrent_layer);
       neural_network.add_layer(probabilistic_layer);

       neural_network.set_parameters_random();

       recurrent_layer->set_timesteps(2);

       batch.set(samples_number, &data_set);
       batch.fill(samples_indices, input_indices, target_indices);

       forward_propagation.set(samples_number, &neural_network);
       neural_network.forward_propagate(batch, forward_propagation);

       back_propagation.set(samples_number, &minkowski_error);
       minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = minkowski_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test recurrent and multiple probabilistic
   {
       samples_number = 3;
       inputs_number = 3;
       neurons_number = 2;
       outputs_number = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       recurrent_layer->set(inputs_number, neurons_number);
       probabilistic_layer->set(neurons_number, outputs_number);

       neural_network.add_layer(recurrent_layer);
       neural_network.add_layer(probabilistic_layer);

       neural_network.set_parameters_random();

       recurrent_layer->set_timesteps(2);

       batch.set(samples_number, &data_set);
       batch.fill(samples_indices, input_indices, target_indices);

       forward_propagation.set(samples_number, &neural_network);
       neural_network.forward_propagate(batch, forward_propagation);

       back_propagation.set(samples_number, &minkowski_error);
       minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = minkowski_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   // Test Perceptron LM
   {
       samples_number = 2;
       inputs_number = 2;
       neurons_number = 3;
       outputs_number = 4;

       data_set.set(samples_number, inputs_number, outputs_number);
       data_set.set_data_random();
       data_set.set_training();

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       neural_network.set(NeuralNetwork::Approximation, {inputs_number, neurons_number, outputs_number});

       neural_network.set_parameters_random();

       batch.set(samples_number, &data_set);
       batch.fill(samples_indices, input_indices, target_indices);

       forward_propagation.set(samples_number, &neural_network);
       neural_network.forward_propagate(batch, forward_propagation);

       back_propagation.set(samples_number, &minkowski_error);
       minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

       back_propagation_lm.set(samples_number, &minkowski_error);
       minkowski_error.back_propagate(batch, forward_propagation, back_propagation_lm);

       numerical_error_gradient = minkowski_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(back_propagation.gradient, back_propagation_lm.gradient, static_cast<type>(1.0e-3)), LOG);
       assert_true(are_equal(back_propagation_lm.gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test probabilistic (binary) LM
   {
       samples_number = 2;
       inputs_number = 2;
       neurons_number = 3;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Classification, {inputs_number, neurons_number, outputs_number});

       neural_network.set_parameters_random();

       forward_propagation.set(samples_number, &neural_network);
       back_propagation.set(samples_number, &minkowski_error);
       back_propagation_lm.set(samples_number, &minkowski_error);

       neural_network.forward_propagate(batch, forward_propagation);
       minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

       minkowski_error.back_propagate(batch, forward_propagation, back_propagation_lm);

       numerical_error_gradient = minkowski_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(back_propagation.gradient, back_propagation_lm.gradient, static_cast<type>(1.0e-3)), LOG);
       assert_true(are_equal(back_propagation_lm.gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test probabilistic (multiple) LM
   {
       samples_number = 2;
       inputs_number = 2;
       neurons_number = 3;
       outputs_number = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       neural_network.set(NeuralNetwork::Classification, {inputs_number, neurons_number, outputs_number});

       neural_network.set_parameters_random();

       batch.set(samples_number, &data_set);
       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.forward_propagate(batch, forward_propagation);
       forward_propagation.set(samples_number, &neural_network);

       back_propagation.set(samples_number, &minkowski_error);
       minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

       back_propagation_lm.set(samples_number, &minkowski_error);
       minkowski_error.back_propagate(batch, forward_propagation, back_propagation_lm);

       numerical_error_gradient = minkowski_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(back_propagation.gradient, back_propagation_lm.gradient, static_cast<type>(1.0e-3)), LOG);
       assert_true(are_equal(back_propagation_lm.gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   // Test convolutional
   {
       // @todo

       neural_network.set();

       samples_number = 2;

       Index channels_number = 1;
       Index rows_number = 3;
       Index columns_number = 3;

       Index kernels_number = 2;
       Index kernels_rows_number = 2;
       Index kernels_columns_number = 2;

       inputs_number = channels_number*rows_number*columns_number;
       outputs_number = kernels_number*kernels_rows_number*kernels_columns_number;

       Tensor<Index, 1> input_variables_dimensions(4);
       input_variables_dimensions[0] = samples_number;
       input_variables_dimensions[1] = channels_number;
       input_variables_dimensions[2] = rows_number;
       input_variables_dimensions[3] = columns_number;

       data_set.set(samples_number, inputs_number, outputs_number);
       data_set.set_input_variables_dimensions(input_variables_dimensions);
       data_set.set_data_constant(0.5);
       data_set.set_training();

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.set(samples_number, &data_set);
       batch.fill(samples_indices, input_indices, target_indices);

       Tensor<Index, 1> kernels_dimensions(4);
       kernels_dimensions(0) = kernels_number;
       kernels_dimensions(1) = channels_number;
       kernels_dimensions(2) = kernels_rows_number;
       kernels_dimensions(3) = kernels_columns_number;

       ConvolutionalLayer* convolutional_layer_1 = new ConvolutionalLayer(input_variables_dimensions, kernels_dimensions);
       convolutional_layer_1->set_parameters_constant(static_cast<type>(0.7));
       convolutional_layer_1->set_activation_function(ConvolutionalLayer::ActivationFunction::HyperbolicTangent);

       neural_network.add_layer(convolutional_layer_1);

       forward_propagation.set(samples_number, &neural_network);

       back_propagation.set(samples_number, &minkowski_error);

       neural_network.forward_propagate(batch, forward_propagation);

       minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

       numerical_error_gradient = minkowski_error.calculate_gradient_numerical_differentiation();
   }
}


void MinkowskiErrorTest::run_test_case()
{
   cout << "Running Minkowski error test case...\n";  

   // Constructor and destructor methods

   test_constructor();

   // Get methods

   test_get_Minkowski_parameter();

   // Set methods

//   test_set_Minkowski_parameter();

   // Error methods

   test_calculate_error();
//   test_calculate_selection_error();
   test_calculate_error_gradient();

   cout << "End of Minkowski error test case.\n\n";
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
