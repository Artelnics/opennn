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
    cross_entropy_error.set(&neural_network, &data_set);

    cross_entropy_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
}


CrossEntropyErrorTest::~CrossEntropyErrorTest()
{
}


void CrossEntropyErrorTest::test_calculate_error_binary_classification()
{
    Tensor<Index,1> training_samples_indices;
    Tensor<Index,1> input_variables_indices;
    Tensor<Index,1> target_variables_indices;

    Tensor<type, 1> parameters;

    Index inputs_number;
    Index targets_number;
    Index samples_number;

    // Test binary

//    cross_entropy_error.back_propagate(batch, forward_propagation, back_propagation);

    // Test

    inputs_number = 1;
    targets_number = 1;
    samples_number = 1;

    data_set.set(1, 1, 1);
    data_set.set_data_constant(0);
    data_set.set_training();

    training_samples_indices = data_set.get_training_samples_indices();
    input_variables_indices = data_set.get_input_variables_indices();
    target_variables_indices = data_set.get_target_variables_indices();

    batch.set(samples_number, &data_set);
    batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

    neural_network.set(NeuralNetwork::Classification, {1, 1});
    neural_network.set_parameters_constant(1);

    forward_propagation.set(1, &neural_network);
    neural_network.forward_propagate(batch, forward_propagation);

    back_propagation.set(1, &cross_entropy_error);
    cross_entropy_error.back_propagate(batch, forward_propagation, back_propagation);

    cross_entropy_error.calculate_error(batch, forward_propagation, back_propagation);

    assert_true(back_propagation.error < 1e-3, LOG);

    // Test

    data_set.set_data_constant(1);

    neural_network.set_parameters_constant(1);

    batch.set(samples_number, &data_set);
    batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

    neural_network.forward_propagate(batch, forward_propagation);

    cross_entropy_error.back_propagate(batch, forward_propagation, back_propagation);

    cross_entropy_error.calculate_error(batch, forward_propagation, back_propagation);

    assert_true(back_propagation.error < 1e-3, LOG);

}


void CrossEntropyErrorTest::test_calculate_error_multiple_classification()
{
    Tensor<type, 2> data;

    Tensor<Index,1> training_samples_indices;
    Tensor<Index,1> input_variables_indices;
    Tensor<Index,1> target_variables_indices;

    Index samples_number;
    Index inputs_number;
    Index targets_number;

    Tensor<type, 1> parameters;

    // Data set

    inputs_number = 2;
    targets_number = 1;

    data_set.set(1, 2, 1);
    data_set.set_data_constant(0);
    data_set.set_training();

    training_samples_indices = data_set.get_training_samples_indices();
    input_variables_indices = data_set.get_input_variables_indices();
    target_variables_indices = data_set.get_target_variables_indices();

    neural_network.set(NeuralNetwork::Classification, {inputs_number, targets_number});
    neural_network.set_parameters_constant(0);

    batch.set(samples_number, &data_set);
    batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

    forward_propagation.set(samples_number, &neural_network);
    neural_network.forward_propagate(batch, forward_propagation);

    back_propagation.set(samples_number, &cross_entropy_error);
    cross_entropy_error.back_propagate(batch, forward_propagation, back_propagation);

    cross_entropy_error.calculate_error(batch, forward_propagation, back_propagation);

    assert_true(abs(back_propagation.error - 0.693) < 1e-3, LOG);

    // Test multiple

    inputs_number = 2;
    targets_number = 2;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(0);
    data_set.set_training();

    training_samples_indices = data_set.get_training_samples_indices();
    input_variables_indices = data_set.get_input_variables_indices();
    target_variables_indices = data_set.get_target_variables_indices();

    neural_network.set(NeuralNetwork::Classification, {inputs_number, targets_number});
    neural_network.set_parameters_constant(0);

    batch.set(samples_number, &data_set);
    batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

    forward_propagation.set(samples_number, &neural_network);
    neural_network.forward_propagate(batch, forward_propagation);

    back_propagation.set(samples_number, &cross_entropy_error);
    cross_entropy_error.calculate_error(batch, forward_propagation, back_propagation);

    assert_true(back_propagation.error < 1e-3, LOG);
}


void CrossEntropyErrorTest::test_calculate_error()
{
   cout << "test_calculate_error\n";
/*
   test_calculate_error_binary_classification();

   test_calculate_error_multiple_classification();
*/
}


void CrossEntropyErrorTest::test_calculate_error_gradient_binary_classification()
{
    Tensor<type, 2> data;

    Tensor<Index,1> training_samples_indices;
    Tensor<Index,1> input_variables_indices;
    Tensor<Index,1> target_variables_indices;

    Tensor<type, 1> error_gradient;
    Tensor<type, 1> numerical_error_gradient;

    Index samples_number;
    Index inputs_number;
    Index outputs_number;
    Index neurons_number;

    // Test

    samples_number = 2;
    inputs_number = 10;
    outputs_number = 1;
    neurons_number = 3;

    data_set.set(samples_number, inputs_number, outputs_number);

    data.resize(samples_number, inputs_number+outputs_number);
    data.setRandom();

    data(0, 1) = 1;
    data(1, 1) = 0;

    data_set.set_data(data);

    data_set.set_data_binary_random();

    data_set.set_training();

    training_samples_indices = data_set.get_training_samples_indices();

    input_variables_indices = data_set.get_input_variables_indices();
    target_variables_indices = data_set.get_target_variables_indices();

    neural_network.set(NeuralNetwork::Classification, {inputs_number, neurons_number, outputs_number});

    neural_network.set_parameters_random();

    batch.set(samples_number, &data_set);
    batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

    forward_propagation.set(samples_number, &neural_network);
    neural_network.forward_propagate(batch, forward_propagation);

    back_propagation.set(samples_number, &cross_entropy_error);
    cross_entropy_error.back_propagate(batch, forward_propagation, back_propagation);

    numerical_error_gradient = cross_entropy_error.calculate_gradient_numerical_differentiation();

    assert_true(are_equal(back_propagation.gradient, numerical_error_gradient, 1.0e-3), LOG);
}


void CrossEntropyErrorTest::test_calculate_error_gradient_multiple_classification()
{
    Tensor<type, 2> data;

    Tensor<Index,1> training_samples_indices;
    Tensor<Index,1> input_variables_indices;
    Tensor<Index,1> target_variables_indices;

    Tensor<type, 1> error_gradient;
    Tensor<type, 1> numerical_error_gradient;

    Index samples_number;
    Index inputs_number;
    Index outputs_number;
    Index neurons_number;

}


void CrossEntropyErrorTest::test_calculate_error_gradient_recurrent()
{
    Tensor<type, 2> data;

    Tensor<Index,1> training_samples_indices;
    Tensor<Index,1> input_variables_indices;
    Tensor<Index,1> target_variables_indices;

    Index samples_number;
    Index inputs_number;
    Index outputs_number;
    Index neurons_number;

    Tensor<type, 1> error_gradient;
    Tensor<type, 1> numerical_error_gradient;

    Tensor<type, 0> maximum_difference;

    RecurrentLayer* recurrent_layer_pointer = new RecurrentLayer;
    PerceptronLayer* perceptron_layer_pointer = new PerceptronLayer;

    // Test

    samples_number = 10;
    inputs_number = 3;
    outputs_number = 2;
    neurons_number = 2;

   data_set.set(samples_number, inputs_number, outputs_number);
   data_set.set_training();

   data_set.set_data_random();

   recurrent_layer_pointer->set(inputs_number, neurons_number);
   neural_network.add_layer(recurrent_layer_pointer);

   perceptron_layer_pointer->set(neurons_number, outputs_number);

   neural_network.add_layer(perceptron_layer_pointer);

   neural_network.set_parameters_random();

//   error_gradient = cross_entropy_error.calculate_error_gradient();

   numerical_error_gradient = cross_entropy_error.calculate_gradient_numerical_differentiation();

   maximum_difference = (error_gradient - numerical_error_gradient).abs().maximum();

   assert_true(maximum_difference(0) < 1.0e-3, LOG);
}


void CrossEntropyErrorTest::test_calculate_error_gradient_long_short_term_memory()
{
    Tensor<type, 2> data;

    Tensor<Index,1> training_samples_indices;
    Tensor<Index,1> input_variables_indices;
    Tensor<Index,1> target_variables_indices;

    Tensor<type, 1> error_gradient;
    Tensor<type, 1> numerical_error_gradient;

    Tensor<type, 0> maximum_difference;

    Index samples_number;
    Index inputs_number;
    Index outputs_number;
    Index neurons_number;

    LongShortTermMemoryLayer* long_short_term_memory_layer_pointer = new LongShortTermMemoryLayer;
    PerceptronLayer* perceptron_layer_pointer = new PerceptronLayer;

    // Test

    samples_number = 10;
    inputs_number = 3;
    outputs_number = 2;
    neurons_number = 2;

    data_set.set(samples_number, inputs_number, outputs_number);

    data_set.set_training();

    data_set.set_data_random();

    neural_network.set(NeuralNetwork::Forecasting, {inputs_number, neurons_number, outputs_number});

    long_short_term_memory_layer_pointer->set(inputs_number, neurons_number);
    neural_network.add_layer(long_short_term_memory_layer_pointer);

    perceptron_layer_pointer->set(neurons_number, outputs_number);
    neural_network.add_layer(perceptron_layer_pointer);

    neural_network.set_parameters_random();

 //   error_gradient = cross_entropy_error.calculate_error_gradient();

    numerical_error_gradient = cross_entropy_error.calculate_gradient_numerical_differentiation();

    maximum_difference = (error_gradient - numerical_error_gradient).abs().maximum();

    assert_true(maximum_difference(0) < 1.0e-3, LOG);
}


void CrossEntropyErrorTest::test_calculate_error_gradient_convolutional()
{
    Tensor<type, 2> data;

    Tensor<Index,1> training_samples_indices;
    Tensor<Index,1> input_variables_indices;
    Tensor<Index,1> target_variables_indices;

    Tensor<type, 1> error_gradient;
    Tensor<type, 1> numerical_error_gradient;

    Tensor<type, 0> maximum_diference;

    Index samples_number;
    Index inputs_number;
    Index outputs_number;
    Index neurons_number;

    // Test

    samples_number = 5;
    inputs_number = 147;
    outputs_number = 1;

 //   data_set.set(samples_number, inputs_number, outputs_number);
 //   data_set.set_input_variables_dimensions(Tensor<Index, 1>({3,7,7}));
 //   data_set.set_target_variables_dimensions(Tensor<Index, 1>({1}));
    data_set.set_data_random();
    data_set.set_training();

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
 //   Tensor<type, 1> biases_3(1);
 //   biases_3.setRandom(parameters_minimum, parameters_maximum);
 //   convolutional_layer_3->set_biases(biases_3);

 //   PoolingLayer* pooling_layer_2 = new PoolingLayer(convolutional_layer_3->get_outputs_dimensions(), {2,2});
 //   pooling_layer_2->set_pooling_method(PoolingLayer::MaxPooling);

 //   PoolingLayer* pooling_layer_3 = new PoolingLayer(pooling_layer_2->get_outputs_dimensions(), {2,2});
 //   pooling_layer_3->set_pooling_method(PoolingLayer::MaxPooling);

 //   PerceptronLayer* perceptron_layer = new PerceptronLayer(pooling_layer_3->get_outputs_dimensions().calculate_product(), 3, OpenNN::PerceptronLayer::ActivationFunction::Linear);
 //   perceptron_layer->set_parameters_random();

 //   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(perceptron_layer->get_neurons_number(), outputs_number);
 //   probabilistic_layer->set_parameters_random();

 //   neural_network.set();
 //   neural_network.add_layer(convolutional_layer_1);
 //   neural_network.add_layer(convolutional_layer_2);
 //   neural_network.add_layer(pooling_layer_1);
 //   neural_network.add_layer(convolutional_layer_3);
 //   neural_network.add_layer(pooling_layer_2);
 //   neural_network.add_layer(pooling_layer_3);
 //   neural_network.add_layer(perceptron_layer);
 //   neural_network.add_layer(probabilistic_layer);

 //   numerical_error_gradient = cross_entropy_error.calculate_gradient_numerical_differentiation();

 //   error_gradient = cross_entropy_error.calculate_error_gradient();

    maximum_diference = (numerical_error_gradient - error_gradient).abs().maximum();

    assert_true(maximum_diference(0) < 1e-3, LOG);

}


void CrossEntropyErrorTest::test_calculate_error_gradient()
{
   cout << "test_calculate_error_gradient\n";

//   test_calculate_error_gradient_binary_classification();
//   test_calculate_error_gradient_multiple_classification();

//   test_calculate_error_gradient_recurrent();
//   test_calculate_error_gradient_long_short_term_memory();
//   test_calculate_error_gradient_convolutional();

}


void CrossEntropyErrorTest::run_test_case()
{
    cout << "Running cross entropy error test case...\n";

   test_calculate_error();

   test_calculate_error_gradient();

   cout << "End of cross entropy error test case.\n\n";
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
