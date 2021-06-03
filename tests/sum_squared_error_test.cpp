//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S U M   S Q U A R E D   E R R O R   T E S T   C L A S S               
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "sum_squared_error_test.h"

SumSquaredErrorTest::SumSquaredErrorTest() : UnitTesting() 
{
    sum_squared_error.set(&neural_network, &data_set);

    sum_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
}


SumSquaredErrorTest::~SumSquaredErrorTest() 
{
}


void SumSquaredErrorTest::test_constructor()
{
   cout << "test_constructor\n";

   // Default

   SumSquaredError sum_squared_error_1;

   assert_true(!sum_squared_error_1.has_neural_network(), LOG);
   assert_true(!sum_squared_error_1.has_data_set(), LOG);

   // Neural network and data set

   SumSquaredError sum_squared_error_4(&neural_network, &data_set);

   assert_true(sum_squared_error_4.has_neural_network(), LOG);
   assert_true(sum_squared_error_4.has_data_set(), LOG);
}


void SumSquaredErrorTest::test_calculate_error()
{
   cout << "test_calculate_error\n";

   Tensor<type, 2> data;

   Tensor<Index,1> training_samples_indices;
   Tensor<Index,1> input_variables_indices;
   Tensor<Index,1> target_variables_indices;

   Index inputs_number;
   Index targets_number;
   Index samples_number;

   Index neurons_number;

   RecurrentLayer* recurrent_layer_pointer = new RecurrentLayer;
   PerceptronLayer* perceptron_layer_pointer = new PerceptronLayer;

   Tensor<type, 1> parameters;

   // Test

   samples_number = 1;
   inputs_number = 1;
   targets_number = 1;

   data_set.set(samples_number, inputs_number, targets_number);
   data_set.set_data_constant(0.0);
   data_set.set_training();

   training_samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   neural_network.set(NeuralNetwork::Approximation, {inputs_number, targets_number});
   neural_network.set_parameters_constant(0.0);

   batch.set(samples_number, &data_set);
   batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &sum_squared_error);
   sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   sum_squared_error.calculate_error(batch, forward_propagation, back_propagation);

   assert_true(back_propagation.error == 0.0, LOG);

   // Test

   samples_number = 2;
   inputs_number = 2;
   targets_number = 2;

   data_set.set(samples_number, inputs_number, targets_number);
   data_set.set_data_constant(1.0);
   data_set.set_training();

   training_samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   neural_network.set_parameters_constant(0.0);

   batch.set(samples_number, &data_set);
   batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &sum_squared_error);
   sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   sum_squared_error.calculate_error(batch, forward_propagation, back_propagation);

   assert_true(back_propagation.error == 2.0, LOG);

   // Test

   samples_number = 2;
   inputs_number = 2;
   targets_number = 2;
   neurons_number = 2;

   data_set.set(samples_number, inputs_number, targets_number);
   data_set.set_data_constant(0.0);
   data_set.set_training();

   training_samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   neural_network.set();

   recurrent_layer_pointer->set(inputs_number, neurons_number);
   neural_network.add_layer(recurrent_layer_pointer);

   perceptron_layer_pointer->set(neurons_number, targets_number);
   neural_network.add_layer(perceptron_layer_pointer);

   neural_network.set_parameters_constant(0.0);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   batch.set(samples_number, &data_set);
   batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

   back_propagation.set(samples_number, &sum_squared_error);
   sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   sum_squared_error.calculate_error(batch, forward_propagation, back_propagation);

   assert_true(back_propagation.error == 0.0, LOG);

   // Test

   samples_number = 9;
   inputs_number = 3;
   targets_number = 2;
   neurons_number = 2;

   data.resize(samples_number,inputs_number+targets_number);

   data.setValues({
    {-1,-1,1,-1,3},
    {-1,0,1,0,2},
    {-1,1,1,1,2},
    {0,-1,1,0,2},
    {0,0,1,1,1},
    {0,1,1,2,2},
    {1,-2,1,1,3},
    {1,0,1,2,2},
    {1,1,1,3,3}});

   data_set.set(data);

   data_set.set_training();

   training_samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   neural_network.set(NeuralNetwork::Approximation, {inputs_number, neurons_number, targets_number});

   neural_network.set_parameters_constant(1.0);

   batch.set(samples_number, &data_set);
   batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

   forward_propagation.set(samples_number, &neural_network);
   back_propagation.set(samples_number, &sum_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);
   sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   sum_squared_error.calculate_error(batch, forward_propagation, back_propagation);

   assert_true(back_propagation.error - 8.241 < 1e-3, LOG);
}


void SumSquaredErrorTest::test_calculate_output_delta()
{
   cout << "test_calculate_output_delta\n";

   Index samples_number;
   Index neurons_number;
   Index outputs_number;

   Tensor<type, 2> data;

   Tensor<Index,1> training_samples_indices;
   Tensor<Index,1> input_variables_indices;
   Tensor<Index,1> target_variables_indices;

   PerceptronLayer* perceptron_layer_pointer = new PerceptronLayer;
   RecurrentLayer* recurrent_layer_pointer = new RecurrentLayer;

   Tensor<type, 1> parameters;

   Index inputs_number;
   Index targets_number;
   
   Tensor<type, 1> numerical_gradient;

   // Test

   data_set.set(samples_number, inputs_number, targets_number);
   data_set.set_data_constant(0.0);
   data_set.set_training();

   training_samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   inputs_number = 2;
   targets_number = 2;

   neural_network.set(NeuralNetwork::Approximation, {inputs_number, targets_number});

   neural_network.set_parameters_constant(0.0);

   batch.set(samples_number, &data_set);
   batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

   forward_propagation.set(samples_number, &neural_network);
   back_propagation.set(samples_number, &sum_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);
   sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   sum_squared_error.calculate_output_delta(batch, forward_propagation, back_propagation);

//   assert_true(back_propagation.output_delta(0) == 0.0, LOG);
//   assert_true(back_propagation.output_delta(1) == 0.0, LOG);

   // Test

   samples_number = 1;
   inputs_number = 2;
   targets_number = 2;

   data_set.set(samples_number, inputs_number, targets_number);
   data_set.set_data_constant(1.0);
   data_set.set_training();

   training_samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   neural_network.set_parameters_constant(0.0);

   batch.set(samples_number, &data_set);
   batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &sum_squared_error);
   sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   sum_squared_error.calculate_output_delta(batch, forward_propagation, back_propagation);

   numerical_gradient = sum_squared_error.calculate_gradient_numerical_differentiation();

//   assert_true(abs(training_back_propagation_1.output_delta(0)-numerical_gradient(4)) < static_cast<type>(1e-3), LOG);
//   assert_true(abs(training_back_propagation_1.output_delta(1)-numerical_gradient(5)) < static_cast<type>(1e-3), LOG);

   // Test 2_1 / Perceptron

   samples_number = 3;
   inputs_number = 1;
   neurons_number = 1;
   outputs_number = 2;

   data.resize(samples_number,inputs_number+outputs_number);
   data.setValues({{-1,-1,3},{-1,0,2},{-1,1,2},});

   data_set.set(data);
   data_set.set(samples_number, inputs_number, outputs_number);
   data_set.set_training();

   training_samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   neural_network.set();

   perceptron_layer_pointer->set(neurons_number,outputs_number);
   neural_network.add_layer(perceptron_layer_pointer);

   parameters.resize(16);
   parameters.setValues({1,1,2,0, 1,2,1,1, 1,2,1,0, 1,1,2,1});
   neural_network.set_parameters(parameters);

   batch.set(3, &data_set);
   batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &sum_squared_error);
   sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   sum_squared_error.calculate_output_delta(batch, forward_propagation, back_propagation);

   numerical_gradient.resize(neural_network.get_parameters_number());
   numerical_gradient = sum_squared_error.calculate_gradient_numerical_differentiation();

//   assert_true(abs(back_propagation.output_delta(0,1) + static_cast<type>(4.476)) < static_cast<type>(1e-3), LOG);
//   assert_true(abs(back_propagation.output_delta(1,0) + static_cast<type>(1.523)) < static_cast<type>(1e-3), LOG);
//   assert_true(abs(back_propagation.output_delta(2,1) + static_cast<type>(2.476)) < static_cast<type>(1e-3), LOG);

   // Test Recurrent

   neural_network.set();

   RecurrentLayer* recurrent_layer = new RecurrentLayer(inputs_number, outputs_number);
   recurrent_layer->initialize_hidden_states(0.0);
   recurrent_layer->set_timesteps(10);
   neural_network.add_layer(recurrent_layer);

   neural_network.set_parameters(parameters);

   forward_propagation.set(samples_number, &neural_network);
   back_propagation.set(samples_number, &sum_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);
   sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   sum_squared_error.calculate_output_delta(batch, forward_propagation, back_propagation);

//   assert_true(abs(back_propagation.output_delta(0,1) + 6) < static_cast<type>(1e-3), LOG);
//   assert_true(abs(back_propagation.output_delta(1,0) + 0) < static_cast<type>(1e-3), LOG);
//   assert_true(abs(back_propagation.output_delta(2,1) + 4) < static_cast<type>(1e-3), LOG);

}


void SumSquaredErrorTest::test_calculate_Jacobian_gradient()
{
   cout << "test_calculate_Jacobian_gradient\n";

   Tensor<type, 2> data;

   Tensor<type, 1> parameters;

   Index samples_number;

   Index inputs_number;
   Index targets_number;

   Tensor<Index,1> training_samples_indices;
   Tensor<Index,1> input_variables_indices;
   Tensor<Index,1> target_variables_indices;

   // Test

   inputs_number = 2;
   targets_number = 3;

   data_set.set(1, inputs_number, targets_number);
   data_set.set_data_constant(0.0);
   data_set.set_training();

   batch.set(samples_number, &data_set);

   training_samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

   // Neural network

   neural_network.set(NeuralNetwork::Approximation, {inputs_number, targets_number});
   neural_network.set_parameters_constant(0.0);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &sum_squared_error);
   sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   back_propagation_lm.set(training_samples_indices.size(), &sum_squared_error);

//   sum_squared_error.calculate_squared_errors_jacobian(batch, forward_propagation, training_back_propagation, loss_index_back_propagation_lm);
//   sum_squared_error.calculate_gradient(batch, forward_propagation, loss_index_back_propagation_lm);

   assert_true(back_propagation_lm.gradient(0) == 0.0, LOG);
   assert_true(back_propagation_lm.gradient(1) == 0.0, LOG);
}


void SumSquaredErrorTest::test_calculate_error_gradient()
{
   cout << "test_calculate_error_gradient\n";

   Tensor<type, 1> parameters;

   Tensor<type, 1> gradient;
   Tensor<type, 1> numerical_gradient;

   Tensor<type, 0> maximum_difference;

   Index inputs_number;
   Index outputs_number;
   Index samples_number;
   Index neurons_number;

   LongShortTermMemoryLayer* long_short_term_memory_layer_pointer = new LongShortTermMemoryLayer;
   RecurrentLayer* recurrent_layer_pointer = new RecurrentLayer;
   PerceptronLayer* perceptron_layer_pointer = new PerceptronLayer;

   // Test lstm

    samples_number = 10;
    inputs_number = 3;
    neurons_number = 2;
    outputs_number = 4;

    data_set.set(samples_number, inputs_number, outputs_number);

    data_set.set_data_random();

    data_set.set_training();

    neural_network.set();

    long_short_term_memory_layer_pointer->set(inputs_number, neurons_number);

    long_short_term_memory_layer_pointer->set_timesteps(8);

    neural_network.add_layer(long_short_term_memory_layer_pointer);

    perceptron_layer_pointer->set(neurons_number,outputs_number);

    neural_network.add_layer(perceptron_layer_pointer);

    neural_network.set_parameters_random();

    parameters = neural_network.get_parameters();

    numerical_gradient = sum_squared_error.calculate_gradient_numerical_differentiation();

//    gradient = sum_squared_error.calculate_error_gradient();

//    assert_true(numerical_gradient-gradient < 1.0e-3, LOG);

   neural_network.set();

   // Test recurrent
    samples_number = 5;
    inputs_number = 3;
    neurons_number = 7;
    outputs_number = 3;

    data_set.set(samples_number, inputs_number, outputs_number);

    data_set.set_data_random();

    data_set.set_training();

    neural_network.set();

    RecurrentLayer* recurrent_layer = new RecurrentLayer(inputs_number, neurons_number);

    recurrent_layer->initialize_hidden_states(0.0);
    recurrent_layer->set_timesteps(10);

    neural_network.add_layer(recurrent_layer);

    perceptron_layer_pointer->set(neurons_number, outputs_number);

    neural_network.add_layer(perceptron_layer_pointer);

    neural_network.set_parameters_random();

    parameters = neural_network.get_parameters();

    numerical_gradient = sum_squared_error.calculate_gradient_numerical_differentiation();

//    gradient = sum_squared_error.calculate_error_gradient();

//    assert_true(numerical_gradient-gradient < 1.0e-3, LOG);

   // Test perceptron

   neural_network.set();
    samples_number = 5;
    inputs_number = 2;
    neurons_number = 7;
    outputs_number = 4;

    data_set.set(samples_number, inputs_number, outputs_number);

    data_set.set_data_random();

    data_set.set_training();

    PerceptronLayer* perceptron_layer_1 = new PerceptronLayer(inputs_number, neurons_number);

    neural_network.add_layer(perceptron_layer_1);

    PerceptronLayer* perceptron_layer_2 = new PerceptronLayer(neurons_number, outputs_number);

    neural_network.add_layer(perceptron_layer_2);

    numerical_gradient = sum_squared_error.calculate_gradient_numerical_differentiation();

//    gradient = sum_squared_error.calculate_error_gradient();

//    assert_true(numerical_gradient-gradient < 1.0e-3, LOG);

   // Test convolutional
   samples_number = 5;
   inputs_number = 147;
   outputs_number = 1;

   data_set.set(samples_number, inputs_number, outputs_number);
//   data_set.set_input_variables_dimensions(Tensor<Index, 1>({3,7,7}));
//   data_set.set_target_variables_dimensions(Tensor<Index, 1>({1}));
   data_set.set_data_random();
   data_set.set_training();

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
//   filters_2.setRandom();
//   convolutional_layer_2->set_synaptic_weights(filters_2);
//   Tensor<type, 1> biases(2, 0);
//   biases.setRandom();
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

   neural_network.set();
//   neural_network.add_layer(convolutional_layer_1);
//   neural_network.add_layer(convolutional_layer_2);
//   neural_network.add_layer(pooling_layer_1);
//   neural_network.add_layer(convolutional_layer_3);
//   neural_network.add_layer(pooling_layer_2);
//   neural_network.add_layer(pooling_layer_3);
//   neural_network.add_layer(perceptron_layer);
//   neural_network.add_layer(probabilistic_layer);

   numerical_gradient = sum_squared_error.calculate_gradient_numerical_differentiation();

//   gradient = sum_squared_error.calculate_error_gradient();

//   maximum_difference = (error_gradient - numerical_error_gradient).abs().sum();

   assert_true(maximum_difference(0) < 1.0e-3, LOG);
}


void SumSquaredErrorTest::test_calculate_squared_errors_jacobian()
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

   // Test Perceptron

   samples_number = 2;
   inputs_number = 2;
   hidden_neurons_number = 3;
   outputs_number = 4;

   data_set.set(samples_number, inputs_number, outputs_number);
   data_set.set_data_random();
   data_set.set_training();

   samples_indices = data_set.get_training_samples_indices();
   input_indices = data_set.get_input_variables_indices();
   target_indices = data_set.get_target_variables_indices();

   batch.set(samples_number, &data_set);
   batch.fill(samples_indices, input_indices, target_indices);

   neural_network.set(NeuralNetwork::Approximation, {inputs_number, hidden_neurons_number, outputs_number});

   neural_network.set_parameters_random();

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &sum_squared_error);
   sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   back_propagation_lm.set(samples_number, &sum_squared_error);
   sum_squared_error.back_propagate(batch, forward_propagation, back_propagation_lm);

   numerical_squared_errors_jacobian = sum_squared_error.calculate_Jacobian_numerical_differentiation();

   assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_squared_errors_jacobian, 1.0e-3), LOG);

   // Test probabilistic (binary)

       samples_number = 2;
       inputs_number = 2;
       hidden_neurons_number = 3;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Classification, {inputs_number, hidden_neurons_number, outputs_number});

       neural_network.set_parameters_random();

       forward_propagation.set(samples_number, &neural_network);
       neural_network.forward_propagate(batch, forward_propagation);

       back_propagation.set(samples_number, &sum_squared_error);
       sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       back_propagation_lm.set(samples_number, &sum_squared_error);
       sum_squared_error.back_propagate(batch, forward_propagation, back_propagation_lm);

       numerical_squared_errors_jacobian = sum_squared_error.calculate_Jacobian_numerical_differentiation();

       assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_squared_errors_jacobian, static_cast<type>(1e-3)), LOG);

   // Test probabilistic (multiple)

       samples_number = 2;
       inputs_number = 2;
       hidden_neurons_number = 3;
       outputs_number = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       neural_network.set(NeuralNetwork::Classification, {inputs_number, hidden_neurons_number, outputs_number});

       neural_network.set_parameters_random();

       batch.set(samples_number, &data_set);
       batch.fill(samples_indices, input_indices, target_indices);

       forward_propagation.set(samples_number, &neural_network);
       neural_network.forward_propagate(batch, forward_propagation);

       back_propagation.set(samples_number, &sum_squared_error);
       sum_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       back_propagation_lm.set(samples_number, &sum_squared_error);
       sum_squared_error.back_propagate(batch, forward_propagation, back_propagation_lm);

       numerical_squared_errors_jacobian = sum_squared_error.calculate_Jacobian_numerical_differentiation();

       assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_squared_errors_jacobian, static_cast<type>(1e-3)), LOG);

}


void SumSquaredErrorTest::run_test_case()
{
   cout << "Running sum squared error test case...\n";

   // Constructor and destructor methods

   test_constructor();

   // Error methods

   test_calculate_error();

   test_calculate_output_delta();

   test_calculate_error_gradient();

   test_calculate_Jacobian_gradient();

   // Squared errors methods

   test_calculate_squared_errors_jacobian();

   cout << "End of sum squared error test case.\n\n";
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
