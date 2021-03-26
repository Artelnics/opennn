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

   assert_true(!mse1.has_neural_network(), LOG);
   assert_true(!mse1.has_data_set(), LOG);

   // Neural network and data set

   NeuralNetwork nn3;
   DataSet ds3;
   MeanSquaredError mse3(&nn3, &ds3);

   assert_true(mse3.has_neural_network(), LOG);
   assert_true(mse3.has_data_set(), LOG);
}


void MeanSquaredErrorTest::test_destructor()
{
    cout << "test_destructor\n";
}


void MeanSquaredErrorTest::test_calculate_error()
{
   cout << "test_calculate_error\n";

   //Case1

   Tensor<type, 1> parameters;

   Tensor<Index, 1> architecture(3);

   architecture.setValues({1,1,1});

   NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(0.0);

   DataSet data_set(1, 1, 1);
   data_set.initialize_data(0.0);

   MeanSquaredError mean_squared_error(&neural_network, &data_set);
   DataSetBatch batch(1, &data_set);


   Index batch_samples_number = batch.get_samples_number();

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(0.0);

   data_set.set(1, 1, 1);
   data_set.initialize_data(0.0);
   data_set.set_training();

   NeuralNetworkForwardPropagation forward_propagation(batch_samples_number, &neural_network);

   LossIndexBackPropagation back_propagation(batch_samples_number, &mean_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);

   mean_squared_error.calculate_error(batch, forward_propagation, back_propagation);

   assert_true(back_propagation.error == 0.0, LOG);

   //Case2

   Tensor<type, 1> parameters_2;

   Tensor<Index, 1> architecture2(2);
   architecture2.setValues({1,1});

   neural_network.set(NeuralNetwork::Approximation, architecture2);
   neural_network.set_parameters_random();

   parameters_2 = neural_network.get_parameters();

   data_set.set(1, 2, 1);

   Tensor<type, 2> data(1, 3);
   data.setValues({{1, 2, 3}});
   data_set.set_data(data);

   neural_network.set_parameters_constant(1);

   NeuralNetworkForwardPropagation forward_propagation_2(batch_samples_number, &neural_network);

   LossIndexBackPropagation back_propagation_2(batch_samples_number, &mean_squared_error);

   neural_network.forward_propagate(batch, forward_propagation_2);

   mean_squared_error.calculate_error(batch, forward_propagation_2, back_propagation_2);

   assert_true(abs(back_propagation_2.error - 1) < 1.0e-3, LOG);

   assert_true(back_propagation_2.error == 1.0, LOG);

}


void MeanSquaredErrorTest::test_calculate_error_gradient()
{
   cout << "test_calculate_error_gradient\n";

   DataSet data_set;

   Index samples_number;

   Tensor<Index, 1> samples_indices;
   Tensor<Index, 1> input_indices;
   Tensor<Index, 1> target_indices;

   DataSetBatch batch;

   NeuralNetwork neural_network;

   Index inputs_number;
   Index hidden_neurons;
   Index outputs_number;

   NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);

   Tensor<type, 1> error_gradient;
   Tensor<type, 1> numerical_error_gradient;

   PerceptronLayer* perceptron_layer_1;
   PerceptronLayer* perceptron_layer_2;
   ProbabilisticLayer* probabilistic_layer;
   RecurrentLayer* recurrent_layer;
   LongShortTermMemoryLayer* long_short_term_memory_layer;

   MeanSquaredError mean_squared_error(&neural_network, &data_set);

   LossIndexBackPropagation training_back_propagation(samples_number, &mean_squared_error);

   // Test trivial

   data_set.generate_Rosenbrock_data(100,2);
   data_set.set_training();

      samples_number = 10;
      inputs_number = 1;
      outputs_number = 1;

      data_set.set(samples_number, inputs_number, outputs_number);
      data_set.initialize_data(0.0);
      data_set.set_training();

      batch.set(samples_number, &data_set);

      samples_indices = data_set.get_training_samples_indices();
      input_indices = data_set.get_input_variables_indices();
      target_indices = data_set.get_target_variables_indices();

      batch.fill(samples_indices, input_indices, target_indices);

      perceptron_layer_1->set(inputs_number, outputs_number);
      neural_network.add_layer(perceptron_layer_1);

      neural_network.set_parameters_constant(0.0);

      mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

      forward_propagation.set(samples_number, &neural_network);
      training_back_propagation.set(samples_number, &mean_squared_error);

      neural_network.forward_propagate(batch, forward_propagation);

      mean_squared_error.back_propagate(batch, forward_propagation, training_back_propagation);
      error_gradient = training_back_propagation.gradient;

      numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, 1.0e-3), LOG);

      assert_true((error_gradient.dimension(0) == neural_network.get_parameters_number()) , LOG);
      assert_true(std::all_of(error_gradient.data(), error_gradient.data()+error_gradient.size(), [](type i) { return (i-static_cast<type>(0))<std::numeric_limits<type>::min(); }), LOG);

   // Test perceptron and probabilistic
{
        samples_number = 10;
        inputs_number = 3;
        outputs_number = 3;
        hidden_neurons = 2;

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_training();
        data_set.set_data_binary_random();

        batch.set(samples_number, &data_set);

        samples_indices = data_set.get_training_samples_indices();
        input_indices = data_set.get_input_variables_indices();
        target_indices = data_set.get_target_variables_indices();

        batch.fill(samples_indices, input_indices, target_indices);

        Tensor<Index, 1> architecture(3);
        architecture[0] = inputs_number;
        architecture[1] = hidden_neurons;
        architecture[2] = outputs_number;

        neural_network.set(NeuralNetwork::Classification, architecture);

        mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

        forward_propagation.set(samples_number, &neural_network);
        training_back_propagation.set(samples_number, &mean_squared_error);

        neural_network.forward_propagate(batch, forward_propagation);

        mean_squared_error.back_propagate(batch, forward_propagation, training_back_propagation);
        error_gradient = training_back_propagation.gradient;

        numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, 1.0e-3), LOG);
  }

   // Test lstm

{
       samples_number = 5;
       inputs_number = 4;
       outputs_number = 2;
       hidden_neurons = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_random();
       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       long_short_term_memory_layer->set(inputs_number, hidden_neurons);
       perceptron_layer_2->set(hidden_neurons, outputs_number);

       neural_network.add_layer(long_short_term_memory_layer);
       neural_network.add_layer(perceptron_layer_2);

       neural_network.set_parameters_random();

       numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

//       error_gradient = mean_squared_error.calculate_error_gradient();
//       assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
}

   // Test recurrent
{
       samples_number = 92;
       inputs_number = 3;
       outputs_number = 1;
       hidden_neurons = 4;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_random();

       data_set.set_training();

       recurrent_layer->set(inputs_number, hidden_neurons);
       recurrent_layer->set_timesteps(1);

       perceptron_layer_2->set(hidden_neurons, outputs_number);

       neural_network.set();

       neural_network.add_layer(recurrent_layer);
       neural_network.add_layer(perceptron_layer_2);

       neural_network.set_parameters_random();

       //error_gradient = mean_squared_error.calculate_error_gradient();

       numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

       //assert_true(are_equal(error_gradient - numerical_error_gradient, 1.0e-3), LOG);
}

   // Test convolutional
{
//   samples_number = 5;
//   inputs_number = 147;
//   outputs_number = 1;

//   data_set.set(samples_number, inputs_number, outputs_number);
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

//   numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

//   error_gradient = mean_squared_error.calculate_error_gradient();

//   assert_true(absolute_value(numerical_error_gradient - error_gradient) < 1e-3, LOG);
}

}


void MeanSquaredErrorTest::test_calculate_squared_errors()
{
   cout << "test_calculate_squared_errors\n";

   NeuralNetwork neural_network;
   Tensor<Index, 1> hidden_layers_size;

   Index parameters;
   DataSet data_set;
   
   MeanSquaredError mean_squared_error(&neural_network, &data_set);

   DataSetBatch batch(1, &data_set);


   Index batch_samples_number = batch.get_samples_number();

   Tensor<type, 1> squared_errors;

   // Test

   Tensor<Index, 1> architecture(3);
   architecture.setValues({1,1});


   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_random();

   data_set.set(1, 1, 1);
   data_set.set_data_random();

   NeuralNetworkForwardPropagation forward_propagation(batch_samples_number, &neural_network);
   LossIndexBackPropagationLM loss_index_back_propagation_lm(batch_samples_number, &mean_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);

   mean_squared_error.calculate_squared_errors(batch, forward_propagation, loss_index_back_propagation_lm);
   squared_errors=loss_index_back_propagation_lm.squared_errors;

//   Eigen::array<int, 2> vector_times_vector = {Eigen::array<int, 2> ({1,1})};

//   const Tensor<type, 0> product_result = squared_errors.contract(squared_errors, vector_times_vector);

//   assert_true(abs(product_result(0) - error) < 1.0e-3, LOG);
}


void MeanSquaredErrorTest::test_calculate_squared_errors_jacobian()
{
   cout << "test_calculate_squared_errors_jacobian\n";

  NumericalDifferentiation nd;

  NeuralNetwork neural_network;
  Tensor<Index, 1> architecture;
  Tensor<type, 1> parameters;

  DataSet data_set;

  MeanSquaredError mean_squared_error(&neural_network, &data_set);

  Tensor<type, 1> error_gradient;

  Tensor<type, 1> squared_errors;
  Tensor<type, 2> terms_Jacobian;
  Tensor<type, 2> numerical_squared_errors_jacobian;

  Tensor<type, 2> inputs;
  Tensor<type, 2> targets;
  Tensor<type, 2> outputs;

  Tensor<type, 2> output_delta;
  Tensor<Tensor<type, 2>, 1> layers_delta;

   // Test

    architecture.setValues({1,1});

    neural_network.set(NeuralNetwork::Approximation, architecture);

    neural_network.set_parameters_constant(0.0);

    data_set.set(1, 1, 1);
    data_set.initialize_data(0.0);

    inputs = data_set.get_training_input_data();
    targets = data_set.get_training_target_data();
    outputs = neural_network.calculate_outputs(inputs);

//   Tensor<LayerForwardPropagation, 1> forward_propagation = neural_network.forward_propagate(inputs);

//   output_delta = mean_squared_error.calculate_output_delta(outputs, targets);

//   layers_delta = mean_squared_error.calculate_layers_delta(forward_propagation, output_delta);

//   terms_Jacobian = mean_squared_error.calculate_squared_errors_jacobian(inputs, forward_propagation, layers_delta);

   assert_true(terms_Jacobian.dimension(0) == data_set.get_training_samples_number(), LOG);
   assert_true(terms_Jacobian.dimension(1) == neural_network.get_parameters_number(), LOG);
//   assert_true(terms_Jacobian == 0.0, LOG);

   // Test 

   architecture.resize(3);
   architecture.setValues({3, 4, 2});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(0.0);

   data_set.set(3, 2, 5);
   mean_squared_error.set(&neural_network, &data_set);
   data_set.initialize_data(0.0);

   inputs = data_set.get_training_input_data();
   targets = data_set.get_training_target_data();
   outputs = neural_network.calculate_outputs(inputs);

//   forward_propagation = nn.forward_propagate(inputs);

//   output_delta = mean_squared_error.calculate_output_delta(outputs, targets);

//   layers_delta = mean_squared_error.calculate_layers_delta(forward_propagation, output_delta);

//   terms_Jacobian = mean_squared_error.calculate_squared_errors_jacobian(inputs, forward_propagation, layers_delta);

   assert_true(terms_Jacobian.dimension(0) == data_set.get_training_samples_number(), LOG);
   assert_true(terms_Jacobian.dimension(1) == neural_network.get_parameters_number(), LOG);
//   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

   architecture.resize(3);
   architecture[0] = 2;
   architecture[1] = 1;
   architecture[2] = 2;

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(0.0);

   data_set.set(2, 2, 5);
   mean_squared_error.set(&neural_network, &data_set);
   data_set.initialize_data(0.0);

   inputs = data_set.get_training_input_data();
   targets = data_set.get_training_target_data();
   outputs = neural_network.calculate_outputs(inputs);

//   forward_propagation = neural_network.forward_propagate(inputs);

//   output_delta = mean_squared_error.calculate_output_delta(outputs, targets);

//   layers_delta = mean_squared_error.calculate_layers_delta(forward_propagation, output_delta);

//   terms_Jacobian = mean_squared_error.calculate_squared_errors_jacobian(inputs, forward_propagation, layers_delta);

   assert_true(terms_Jacobian.dimension(0) == data_set.get_training_samples_number(), LOG);
   assert_true(terms_Jacobian.dimension(1) == neural_network.get_parameters_number(), LOG);
//   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

   architecture.setValues({1,1});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(0.0);
   //nn.set_layer_activation_function(0, PerceptronLayer::Linear);
//   nn.set_parameters_random();
   parameters = neural_network.get_parameters();

   data_set.set(1, 1, 1);
   data_set.set_data_random();
   data_set.initialize_data(1.0);

   inputs = data_set.get_training_input_data();
   targets = data_set.get_training_target_data();
   outputs = neural_network.calculate_outputs(inputs);

//   forward_propagation = nn.forward_propagate(inputs);

//   output_delta = mean_squared_error.calculate_output_delta(outputs, targets);

//   layers_delta = mean_squared_error.calculate_layers_delta(forward_propagation, output_delta);

//   terms_Jacobian = mean_squared_error.calculate_squared_errors_jacobian(inputs, forward_propagation, layers_delta);

//   numerical_squared_errors_jacobian = nd.calculate_Jacobian(mean_squared_error, &MeanSquaredError::calculate_training_error_terms, parameters);

//   assert_true(absolute_value(terms_Jacobian-numerical_squared_errors_jacobian) < 1.0e-3, LOG);

   // Test

//   architecture.setValues({1,1});

//   neural_network.set(NeuralNetwork::Approximation, architecture);
//   neural_network.set_parameters_random();
//   parameters = neural_network.get_parameters();

//   data_set.set(2, 2, 2);
//   data_set.set_data_random();

//   terms_Jacobian = mean_squared_error.calculate_squared_errors_jacobian();
//   numerical_squared_errors_jacobian = nd.calculate_Jacobian(mean_squared_error, &MeanSquaredError::calculate_training_error_terms, parameters);

//   assert_true(absolute_value(terms_Jacobian-numerical_squared_errors_jacobian) < 1.0e-3, LOG);

   // Test

//   architecture.setValues({2,2,2});

//   neural_network.set(NeuralNetwork::Approximation, architecture);
//   neural_network.set_parameters_random();

//   data_set.set(2, 2, 2);
//   data_set.set_data_random();
   
//   error_gradient = mean_squared_error.calculate_error_gradient({0, 1});

//   squared_errors = mean_squared_error.calculate_training_error_terms();
//   terms_Jacobian = mean_squared_error.calculate_squared_errors_jacobian();

//   assert_true(absolute_value((terms_Jacobian.calculate_transpose()).dot(squared_errors)*2.0 - error_gradient) < 1.0e-3, LOG);
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

   test_constructor();
   test_destructor();

   // Get methods

   // Set methods

   // Error methods

   test_calculate_error();

   test_calculate_error_gradient();

   // Squared errors methods

   //test_calculate_squared_errors();
   //test_calculate_squared_errors_jacobian();

   // Serialization methods

//   test_to_XML();
//   test_from_XML();

   cout << "End of mean squared error test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
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
