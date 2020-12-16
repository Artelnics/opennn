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
   DataSet::Batch batch(1, &data_set);


   Index batch_samples_number = batch.get_samples_number();

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(0.0);

   data_set.set(1, 1, 1);
   data_set.initialize_data(0.0);
   data_set.set_training();

   NeuralNetwork::ForwardPropagation forward_propagation(batch_samples_number, &neural_network);

   LossIndex::BackPropagation back_propagation(batch_samples_number, &mean_squared_error);

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

   NeuralNetwork::ForwardPropagation forward_propagation_2(batch_samples_number, &neural_network);

   LossIndex::BackPropagation back_propagation_2(batch_samples_number, &mean_squared_error);

   neural_network.forward_propagate(batch, forward_propagation_2);

   mean_squared_error.calculate_error(batch, forward_propagation_2, back_propagation_2);

   assert_true(abs(back_propagation_2.error - 1) < 1.0e-3, LOG);

   assert_true(back_propagation_2.error == 1.0, LOG);

}


void MeanSquaredErrorTest::test_calculate_error_gradient()
{
   cout << "test_calculate_error_gradient\n";

   NeuralNetwork neural_network;

   DataSet data_set;

   data_set.generate_Rosenbrock_data(100,2);
   data_set.set_training();

   MeanSquaredError mean_squared_error(&neural_network, &data_set);

   Tensor<type, 1> error_gradient;
   Tensor<type, 1> numerical_error_gradient;

   Index samples_number;
   Index inputs_number;
   Index hidden_neurons;
   Index outputs_number;


   PerceptronLayer* hidden_perceptron_layer = new PerceptronLayer();
   PerceptronLayer* output_perceptron_layer = new PerceptronLayer();
   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer();
   LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer();


   // Test trivial


      samples_number = 10;
      inputs_number = 1;
      outputs_number = 1;

      data_set.set(samples_number, inputs_number, outputs_number);
      data_set.initialize_data(0.0);
      data_set.set_training();

      DataSet::Batch batch(samples_number, &data_set);

      Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
      const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
      const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

      batch.fill(samples_indices, input_indices, target_indices);

      hidden_perceptron_layer->set(inputs_number, outputs_number);
      neural_network.add_layer(hidden_perceptron_layer);

      neural_network.set_parameters_constant(0.0);

      MeanSquaredError mse(&neural_network, &data_set);

      mse.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

      NeuralNetwork::ForwardPropagation forward_propagation(samples_number, &neural_network);
      LossIndex::BackPropagation training_back_propagation(samples_number, &mse);

      neural_network.forward_propagate(batch, forward_propagation);

      mse.back_propagate(batch, forward_propagation, training_back_propagation);
      error_gradient = training_back_propagation.gradient;

      numerical_error_gradient = mse.calculate_error_gradient_numerical_differentiation(&mse);

      const Tensor<type, 1> difference = error_gradient-numerical_error_gradient;

      assert_true((error_gradient.dimension(0) == neural_network.get_parameters_number()) , LOG);
      assert_true(std::all_of(error_gradient.data(), error_gradient.data()+error_gradient.size(), [](type i) { return (i-static_cast<type>(0))<std::numeric_limits<type>::min(); }), LOG);

   // Test perceptron and probabilistic
{
        samples_number = 10;
        inputs_number = 3;
        outputs_number = 3;
        hidden_neurons = 2;

        DataSet data_set_2;

        data_set_2.set(samples_number, inputs_number, outputs_number);
        data_set_2.set_training();
        data_set_2.set_data_binary_random();

        DataSet::Batch batch_1(samples_number, &data_set_2);

        Tensor<Index, 1> samples_indices_1 = data_set_2.get_training_samples_indices();
        const Tensor<Index, 1> input_indices_1 = data_set_2.get_input_variables_indices();
        const Tensor<Index, 1> target_indices_1 = data_set_2.get_target_variables_indices();

        batch_1.fill(samples_indices_1, input_indices_1, target_indices_1);

        Tensor<Index, 1> architecture(3);
        architecture[0] = inputs_number;
        architecture[1] = hidden_neurons;
        architecture[2] = outputs_number;

        NeuralNetwork neural_network_1(NeuralNetwork::Classification, architecture);

        MeanSquaredError mse_1(&neural_network_1, &data_set_2);

        mse_1.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

        NeuralNetwork::ForwardPropagation forward_propagation(samples_number, &neural_network_1);
        LossIndex::BackPropagation training_back_propagation(samples_number, &mse_1);

        neural_network_1.forward_propagate(batch_1, forward_propagation);

        mse_1.back_propagate(batch_1, forward_propagation, training_back_propagation);
        error_gradient = training_back_propagation.gradient;

        numerical_error_gradient = mse_1.calculate_error_gradient_numerical_differentiation(&mse_1);

        const Tensor<type, 1> difference1 = error_gradient-numerical_error_gradient;

        assert_true(std::all_of(difference.data(), difference.data()+difference.size(), [](type i) { return (i)<static_cast<type>(1.0e-3); }), LOG);
  }

  /* // Test lstm

{
       samples_number = 5;
       inputs_number = 4;
       outputs_number = 2;
       hidden_neurons = 3;

       DataSet data_set_3;

       data_set_3.set(samples_number, inputs_number, outputs_number);

       data_set_3.set_data_random();
       data_set_3.set_training();

       DataSet::Batch batch(samples_number, &data_set_3);

       Tensor<Index, 1> samples_indices = data_set_3.get_training_samples_indices();
       const Tensor<Index, 1> input_indices = data_set_3.get_input_variables_indices();
       const Tensor<Index, 1> target_indices = data_set_3.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       long_short_term_memory_layer->set(inputs_number, hidden_neurons);
       output_perceptron_layer->set(hidden_neurons, outputs_number);

       cout << "hola" <<endl;

       neural_network.add_layer(long_short_term_memory_layer);
       neural_network.add_layer(output_perceptron_layer);

       neural_network.set_parameters_random();

      // error_gradient = mean_squared_error.calculate_error_gradient();

      //numerical_error_gradient = mean_squared_error.calculate_error_gradient_numerical_differentiation();

      // assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
}
*/
//   neural_network.set();

   // Test recurrent
{
//   samples_number = 92;
//   inputs_number = 3;
//   outputs_number = 1;
//   hidden_neurons = 4;

//   data_set.set(samples_number, inputs_number, outputs_number);

//   data_set.set_data_random();

//   data_set.set_training();

//   recurrent_layer->set(inputs_number, hidden_neurons);
//   recurrent_layer->set_timesteps(1);

//   output_perceptron_layer->set(hidden_neurons, outputs_number);

//   neural_network.add_layer(recurrent_layer);
//   neural_network.add_layer(output_perceptron_layer);

//   neural_network.set_parameters_random();

//   error_gradient = mean_squared_error.calculate_error_gradient();

//   numerical_error_gradient = mean_squared_error.calculate_error_gradient_numerical_differentiation();

//   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
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

//   numerical_error_gradient = mean_squared_error.calculate_error_gradient_numerical_differentiation();

//   error_gradient = mean_squared_error.calculate_error_gradient();

//   assert_true(absolute_value(numerical_error_gradient - error_gradient) < 1e-3, LOG);
}

}

/*
void MeanSquaredErrorTest::test_calculate_error_terms()
{
   cout << "test_calculate_error_terms\n";

   NeuralNetwork neural_network;
   Tensor<Index, 1> hidden_layers_size;

   Index parameters;
   DataSet data_set;
   
   MeanSquaredError mean_squared_error(&neural_network, &data_set);

   DataSet::Batch batch(1, &data_set);


   Index batch_samples_number = batch.get_samples_number();

   Tensor<type, 1> error_terms;

   // Test

   Tensor<Index, 1> architecture(3);
   architecture.setValues({1,1});


   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_random();

   data_set.set(1, 1, 1);
   data_set.set_data_random();

   NeuralNetwork::ForwardPropagation forward_propagation(batch_samples_number, &neural_network);
   LossIndex::SecondOrderLoss second_order_loss(parameters,batch_samples_number);

   neural_network.forward_propagate(batch, forward_propagation);

   mean_squared_error.calculate_error_terms(batch, forward_propagation, second_order_loss);
   error_terms=second_order_loss.error_terms;

//   Eigen::array<int, 2> vector_times_vector = {Eigen::array<int, 2> ({1,1})};

//   const Tensor<type, 0> product_result = error_terms.contract(error_terms, vector_times_vector);

//   assert_true(abs(product_result(0) - error) < 1.0e-3, LOG);
}
*/

void MeanSquaredErrorTest::test_calculate_error_terms_Jacobian()
{
   cout << "test_calculate_error_terms_Jacobian\n";

//  NumericalDifferentiation nd;

//  NeuralNetwork neural_network;
//  Tensor<Index, 1> architecture;
//  Tensor<type, 1> parameters;

//  DataSet data_set;

//  MeanSquaredError mean_squared_error(&neural_network, &data_set);

//  Tensor<type, 1> error_gradient;

//  Tensor<type, 1> error_terms;
//  Tensor<type, 2> terms_Jacobian;
//  Tensor<type, 2> numerical_Jacobian_terms;

//  Tensor<type, 2> inputs;
//  Tensor<type, 2> targets;
//  Tensor<type, 2> outputs;

//  Tensor<type, 2> output_gradient;
//  Tensor<Tensor<type, 2>, 1> layers_delta;

   // Test

//   architecture.setValues({1,1});

//   neural_network.set(NeuralNetwork::Approximation, architecture);

//   neural_network.set_parameters_constant(0.0);

//   data_set.set(1, 1, 1);
//  data_set.initialize_data(0.0);

//  inputs = data_set.get_training_input_data();
// targets = data_set.get_training_target_data();
//   outputs = neural_network.calculate_outputs(inputs);

//   Tensor<Layer::ForwardPropagation, 1> forward_propagation = neural_network.forward_propagate(inputs);

//   output_gradient = mean_squared_error.calculate_output_gradient(outputs, targets);

//   layers_delta = mean_squared_error.calculate_layers_delta(forward_propagation, output_gradient);

//   terms_Jacobian = mean_squared_error.calculate_error_terms_Jacobian(inputs, forward_propagation, layers_delta);

//   assert_true(terms_Jacobian.dimension(0) == data_set.get_training_samples_number(), LOG);
//   assert_true(terms_Jacobian.dimension(1) == neural_network.get_parameters_number(), LOG);
//   assert_true(terms_Jacobian == 0.0, LOG);

   // Test 

//   neural_network.set(NeuralNetwork::Approximation, {3, 4, 2});
//   neural_network.set_parameters_constant(0.0);

//   data_set.set(3, 2, 5);
//   mean_squared_error.set(&neural_network, &data_set);
//   data_set.initialize_data(0.0);

//   inputs = data_set.get_training_input_data();
//   targets = data_set.get_training_target_data();
//   outputs = neural_network.calculate_outputs(inputs);

   //forward_propagation = nn.forward_propagate(inputs);

//   output_gradient = mean_squared_error.calculate_output_gradient(outputs, targets);

//   layers_delta = mean_squared_error.calculate_layers_delta(forward_propagation, output_gradient);

//   terms_Jacobian = mean_squared_error.calculate_error_terms_Jacobian(inputs, forward_propagation, layers_delta);

//   assert_true(terms_Jacobian.dimension(0) == data_set.get_training_samples_number(), LOG);
//   assert_true(terms_Jacobian.dimension(1) == neural_network.get_parameters_number(), LOG);
//   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

//   architecture.resize(3);
//   architecture[0] = 2;
//   architecture[1] = 1;
//   architecture[2] = 2;

//   neural_network.set(NeuralNetwork::Approximation, architecture);
//   neural_network.set_parameters_constant(0.0);

//   data_set.set(2, 2, 5);
//   mean_squared_error.set(&neural_network, &data_set);
//   data_set.initialize_data(0.0);

//   inputs = data_set.get_training_input_data();
//   targets = data_set.get_training_target_data();
//   outputs = neural_network.calculate_outputs(inputs);

//   forward_propagation = neural_network.forward_propagate(inputs);

//   output_gradient = mean_squared_error.calculate_output_gradient(outputs, targets);

//   layers_delta = mean_squared_error.calculate_layers_delta(forward_propagation, output_gradient);

//   terms_Jacobian = mean_squared_error.calculate_error_terms_Jacobian(inputs, forward_propagation, layers_delta);

//   assert_true(terms_Jacobian.dimension(0) == data_set.get_training_samples_number(), LOG);
//   assert_true(terms_Jacobian.dimension(1) == neural_network.get_parameters_number(), LOG);
//   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

//   architecture.setValues({1,1});

//   neural_network.set(NeuralNetwork::Approximation, architecture);
//   neural_network.set_parameters_constant(0.0);
//   //nn.set_layer_activation_function(0, PerceptronLayer::Linear);
////   nn.set_parameters_random();
//   parameters = neural_network.get_parameters();

//   data_set.set(1, 1, 1);
////   data_set.set_data_random();
//   data_set.initialize_data(1.0);

//   inputs = data_set.get_training_input_data();
//   targets = data_set.get_training_target_data();
//   outputs = neural_network.calculate_outputs(inputs);

//   forward_propagation = nn.forward_propagate(inputs);

//   output_gradient = mean_squared_error.calculate_output_gradient(outputs, targets);

//   layers_delta = mean_squared_error.calculate_layers_delta(forward_propagation, output_gradient);

//   cout << "layers delta: " << layers_delta << endl;

//   terms_Jacobian = mean_squared_error.calculate_error_terms_Jacobian(inputs, forward_propagation, layers_delta);

//   numerical_Jacobian_terms = nd.calculate_Jacobian(mean_squared_error, &MeanSquaredError::calculate_training_error_terms, parameters);

//   cout << "Terms Jacobian: " << terms_Jacobian << endl;
//   cout << "Numerical: " << numerical_Jacobian_terms << endl;

//   assert_true(absolute_value(terms_Jacobian-numerical_Jacobian_terms) < 1.0e-3, LOG);

   // Test

//   architecture.setValues({1,1});

//   neural_network.set(NeuralNetwork::Approximation, architecture);
//   neural_network.set_parameters_random();
//   parameters = neural_network.get_parameters();

//   data_set.set(2, 2, 2);
//   data_set.set_data_random();

//   terms_Jacobian = mean_squared_error.calculate_error_terms_Jacobian();
//   numerical_Jacobian_terms = nd.calculate_Jacobian(mean_squared_error, &MeanSquaredError::calculate_training_error_terms, parameters);

//   assert_true(absolute_value(terms_Jacobian-numerical_Jacobian_terms) < 1.0e-3, LOG);

   // Test

//   architecture.setValues({2,2,2});

//   neural_network.set(NeuralNetwork::Approximation, architecture);
//   neural_network.set_parameters_random();

//   data_set.set(2, 2, 2);
//   data_set.set_data_random();
   
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

   test_constructor();
   test_destructor();

   // Get methods

   // Set methods

   // Error methods

   test_calculate_error();

   test_calculate_error_gradient();

//   // Error terms methods

   //test_calculate_error_terms();
   //test_calculate_error_terms_Jacobian();

//   // Serialization methods

//   test_to_XML();
//   test_from_XML();

   cout << "End of mean squared error test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2020 Artificial Intelligence Techniques, SL.
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
