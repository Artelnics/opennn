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
}


SumSquaredErrorTest::~SumSquaredErrorTest() 
{
}

/*
void SumSquaredErrorTest::test_constructor()
{
   cout << "test_constructor\n";

   // Default

   SumSquaredError sum_squared_error_1;

   assert_true(sum_squared_error_1.has_neural_network() == false, LOG);
   assert_true(sum_squared_error_1.has_data_set() == false, LOG);

   // Neural network

   NeuralNetwork neural_network_1;
   SumSquaredError sum_squared_error_2(&neural_network_1);

   assert_true(sum_squared_error_2.has_neural_network() == true, LOG);
   assert_true(sum_squared_error_2.has_data_set() == false, LOG);

   // Neural network and data set

   NeuralNetwork neural_network_2;
   DataSet data_set;
   SumSquaredError sum_squared_error_3(&neural_network_2, &data_set);

   assert_true(sum_squared_error_3.has_neural_network() == true, LOG);
   assert_true(sum_squared_error_3.has_data_set() == true, LOG);
}


void SumSquaredErrorTest::test_destructor()
{
   cout << "test_destructor\n";
}


void SumSquaredErrorTest::test_calculate_training_error()
{
   cout << "test_calculate_training_error\n";

   NeuralNetwork neural_network;
   Tensor<type, 1> parameters;

   DataSet data_set;
   Tensor<type, 2> data;

   SumSquaredError sum_squared_error(&neural_network, &data_set);

   type training_error;

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 2});
   neural_network.set_parameters_constant(0.0);

   data_set.set(1, 2, 2);
   data_set.initialize_data(0.0);

   training_error = 0.0;

   training_error = sum_squared_error.calculate_training_error();

   assert_true(training_error == 0.0, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 2});
   neural_network.set_parameters_constant(0.0);

   data_set.set(1, 2, 2);
   data_set.initialize_data(1.0);

   training_error = sum_squared_error.calculate_training_error();

   assert_true(training_error == 2.0, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 2});
   neural_network.set_parameters_random();

   parameters = neural_network.get_parameters();

   data_set.set(10, 2, 2);
   data_set.set_data_random();

   assert_true(abs(sum_squared_error.calculate_training_error() - sum_squared_error.calculate_training_error(parameters)) < numeric_limits<type>::min(), LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1});
   neural_network.set_parameters_random();

   parameters = neural_network.get_parameters();

   data_set.set(1, 1, 1);
   data_set.set_data_random();

   assert_true(abs(sum_squared_error.calculate_training_error() - sum_squared_error.calculate_training_error(parameters*2.0)) > numeric_limits<type>::min(), LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1});
   neural_network.set_parameters_random();

   parameters = neural_network.get_parameters();

   data_set.set(1, 1, 1);
   data_set.set_data_random();

   assert_true(sum_squared_error.calculate_training_error() != 0.0, LOG);

    // Test

   data_set.set(1, 1, 1);
   data_set.initialize_data(0.0);
   data_set.set_training();

   neural_network.set();

   RecurrentLayer* recurrent_layer = new RecurrentLayer(1, 1);

   neural_network.add_layer(recurrent_layer);

   PerceptronLayer* perceptron_layer = new PerceptronLayer(1,1);

   neural_network.add_layer(perceptron_layer);

   neural_network.set_parameters_constant(0.0);

   training_error = sum_squared_error.calculate_training_error();

   assert_true(abs(training_error) < numeric_limits<type>::min(), LOG);
}


void SumSquaredErrorTest::test_calculate_layers_delta()
{
   cout << "test_calculate_layers_delta\n";

   DataSet data_set;
   NeuralNetwork neural_network;
   NumericalDifferentiation numerical_differentation;

   Tensor<type, 1> parameters;

   SumSquaredError sum_squared_error(&neural_network, &data_set);

   // Test

    Index inputs_number = 2;
    Index outputs_number = 2;
    Index instances_number = 10;
    Index hidden_neurons = 1;

    Tensor<Index, 1> instances;
    instances.set(instances_number);
    instances.initialize_sequential();

    neural_network.set(NeuralNetwork::Approximation, {inputs_number,hidden_neurons,outputs_number});
    neural_network.set_parameters_random();

    data_set.set(instances_number,inputs_number,outputs_number);
    data_set.set_data_random();

    Tensor<type, 2> inputs = data_set.get_input_data(instances);
    Tensor<type, 2> targets = data_set.get_target_data(instances);

    Tensor<type, 2> outputs = neural_network.calculate_outputs(inputs);
    Tensor<type, 2> output_gradient = sum_squared_error.calculate_output_gradient(outputs, targets);

    Tensor<Layer::ForwardPropagation, 1> forward_propagation = neural_network.forward_propagate(inputs);

    Tensor<Tensor<type, 2>, 1> layers_delta = sum_squared_error.calculate_layers_delta(forward_propagation, output_gradient);

    assert_true(layers_delta[0].dimension(0) == instances_number, LOG);
    assert_true(layers_delta[0].dimension(1) == hidden_neurons, LOG);
    assert_true(layers_delta[1].dimension(0) == instances_number, LOG);
    assert_true(layers_delta[1].dimension(1) == outputs_number, LOG);
}


void SumSquaredErrorTest::test_calculate_training_error_gradient()
{
   cout << "test_calculate_training_error_gradient\n";

   DataSet data_set;
   NeuralNetwork neural_network;
   SumSquaredError sum_squared_error(&neural_network, &data_set);

   Tensor<Index, 1> architecture;

   Tensor<type, 1> parameters;
   Tensor<type, 1> gradient;
   Tensor<type, 1> numerical_gradient;
   Tensor<type, 1> error;

   Index inputs_number;
   Index outputs_number;
   Index instances_number;
   Index hidden_neurons;

   // Test lstm
{
    instances_number = 10;
    inputs_number = 3;
    hidden_neurons = 2;
    outputs_number = 4;

    data_set.set(instances_number, inputs_number, outputs_number);

    data_set.set_data_random();

    data_set.set_training();

    neural_network.set();

    LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer(inputs_number, hidden_neurons);

    long_short_term_memory_layer->set_timesteps(8);

    neural_network.add_layer(long_short_term_memory_layer);

    PerceptronLayer* perceptron_layer = new PerceptronLayer(hidden_neurons,outputs_number);

    neural_network.add_layer(perceptron_layer);

    neural_network.set_parameters_random();

    parameters = neural_network.get_parameters();

    numerical_gradient = sum_squared_error.calculate_training_error_gradient_numerical_differentiation();

    gradient = sum_squared_error.calculate_training_error_gradient();

    assert_true(numerical_gradient-gradient < 1.0e-3, LOG);
}

   neural_network.set();

   // Test recurrent
{
    instances_number = 5;
    inputs_number = 3;
    hidden_neurons = 7;
    outputs_number = 3;

    data_set.set(instances_number, inputs_number, outputs_number);

    data_set.set_data_random();

    data_set.set_training();

    neural_network.set();

    RecurrentLayer* recurrent_layer = new RecurrentLayer(inputs_number, hidden_neurons);

    recurrent_layer->initialize_hidden_states(0.0);
    recurrent_layer->set_timesteps(10);

    neural_network.add_layer(recurrent_layer);

    PerceptronLayer* perceptron_layer = new PerceptronLayer(hidden_neurons,outputs_number);

    neural_network.add_layer(perceptron_layer);

    neural_network.set_parameters_random();

    parameters = neural_network.get_parameters();

    numerical_gradient = sum_squared_error.calculate_training_error_gradient_numerical_differentiation();

    gradient = sum_squared_error.calculate_training_error_gradient();

    assert_true(numerical_gradient-gradient < 1.0e-3, LOG);
}

   // Test perceptron

   neural_network.set();
{
    instances_number = 5;
    inputs_number = 2;
    hidden_neurons = 7;
    outputs_number = 4;

    data_set.set(instances_number, inputs_number, outputs_number);

    data_set.set_data_random();

    data_set.set_training();

    PerceptronLayer* hidden_perceptron_layer = new PerceptronLayer(inputs_number, hidden_neurons);

    neural_network.add_layer(hidden_perceptron_layer);

    PerceptronLayer* output_perceptron_layer = new PerceptronLayer(hidden_neurons, outputs_number);

    neural_network.add_layer(output_perceptron_layer);

    numerical_gradient = sum_squared_error.calculate_training_error_gradient_numerical_differentiation();

    gradient = sum_squared_error.calculate_training_error_gradient();

    assert_true(numerical_gradient-gradient < 1.0e-3, LOG);
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

   const type parameters_minimum = -100.0;
   const type parameters_maximum = 100.0;

   ConvolutionalLayer* convolutional_layer_1 = new ConvolutionalLayer({3,7,7}, {2,2,2});
   Tensor<type, 2> filters_1({2,3,2,2}, 0);
   filters_1.setRandom(parameters_minimum,parameters_maximum);
   convolutional_layer_1->set_synaptic_weights(filters_1);
   Tensor<type, 1> biases_1(2, 0);
   biases_1.setRandom(parameters_minimum, parameters_maximum);
   convolutional_layer_1->set_biases(biases_1);

   ConvolutionalLayer* convolutional_layer_2 = new ConvolutionalLayer(convolutional_layer_1->get_outputs_dimensions(), {2,2,2});
   convolutional_layer_2->set_padding_option(OpenNN::ConvolutionalLayer::Same);
   Tensor<type, 2> filters_2({2,2,2,2}, 0);
   filters_2.setRandom(parameters_minimum, parameters_maximum);
   convolutional_layer_2->set_synaptic_weights(filters_2);
   Tensor<type, 1> biases_2(2, 0);
   biases_2.setRandom(parameters_minimum, parameters_maximum);
   convolutional_layer_2->set_biases(biases_2);

   PoolingLayer* pooling_layer_1 = new PoolingLayer(convolutional_layer_2->get_outputs_dimensions(), {2,2});

   ConvolutionalLayer* convolutional_layer_3 = new ConvolutionalLayer(pooling_layer_1->get_outputs_dimensions(), {1,2,2});
   convolutional_layer_3->set_padding_option(OpenNN::ConvolutionalLayer::Same);
   Tensor<type, 2> filters_3({1,2,2,2}, 0);
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

   numerical_gradient = sum_squared_error.calculate_training_error_gradient_numerical_differentiation();

   gradient = sum_squared_error.calculate_training_error_gradient();

   assert_true(absolute_value(numerical_gradient - gradient) < 1e-3, LOG);
}
}


void SumSquaredErrorTest::test_calculate_training_error_terms()
{
   cout << "test_calculate_training_error_terms\n";
}


void SumSquaredErrorTest::test_calculate_training_error_terms_Jacobian()
{   
   cout << "test_calculate_training_error_terms_Jacobian\n";

   NumericalDifferentiation nd;

   NeuralNetwork neural_network;
   Tensor<Index, 1> architecture;
   Tensor<type, 1> parameters;

   DataSet data_set;

   SumSquaredError sum_squared_error(&neural_network, &data_set);

   Tensor<type, 1> gradient;

   Tensor<type, 1> terms;
   Tensor<type, 2> terms_Jacobian;
   Tensor<type, 2> numerical_Jacobian_terms;

   Tensor<Index, 1> instances;

   Tensor<type, 2> inputs;
   Tensor<type, 2> targets;

   Tensor<type, 2> outputs;
   Tensor<type, 2> output_gradient;

   Tensor<Tensor<type, 2>, 1> layers_activations;

   Tensor<Tensor<type, 2>, 1> layers_activations_derivatives;

   Tensor<Tensor<type, 2>, 1> layers_delta;

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1, 1});

   neural_network.set_parameters_constant(0.0);

   data_set.set(1, 1, 1);

   data_set.initialize_data(0.0);

   instances.set(1,0);
   //instances.initialize_sequential();

   inputs = data_set.get_input_data(instances);
   targets = data_set.get_target_data(instances);

   outputs = neural_network.calculate_outputs(inputs);
   output_gradient = sum_squared_error.calculate_output_gradient(outputs, targets);

   Tensor<Layer::ForwardPropagation, 1> forward_propagation = neural_network.forward_propagate(inputs);

   layers_delta = sum_squared_error.calculate_layers_delta(forward_propagation, output_gradient);

   terms_Jacobian = sum_squared_error.calculate_error_terms_Jacobian(inputs, forward_propagation, layers_delta);

   assert_true(terms_Jacobian.dimension(0) == data_set.get_instances_number(), LOG);
   assert_true(terms_Jacobian.dimension(1) == neural_network.get_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);

   // Test 

   neural_network.set(NeuralNetwork::Approximation, {3, 4, 2});
   neural_network.set_parameters_constant(0.0);

   data_set.set(3, 2, 5);
   sum_squared_error.set(&neural_network, &data_set);
   data_set.initialize_data(0.0);

//   terms_Jacobian = sum_squared_error.calculate_error_terms_Jacobian();

   assert_true(terms_Jacobian.dimension(0) == data_set.get_training_instances_number(), LOG);
   assert_true(terms_Jacobian.dimension(1) == neural_network.get_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

   architecture.resize(3);
   architecture[0] = 5;
   architecture[1] = 1;
   architecture[2] = 2;

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(0.0);

   data_set.set(5, 2, 3);
   sum_squared_error.set(&neural_network, &data_set);
   data_set.initialize_data(0.0);

//   terms_Jacobian = sum_squared_error.calculate_error_terms_Jacobian();

   assert_true(terms_Jacobian.dimension(0) == data_set.get_training_instances_number(), LOG);
   assert_true(terms_Jacobian.dimension(1) == neural_network.get_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1, 1});
   neural_network.set_parameters_random();
   parameters = neural_network.get_parameters();

   data_set.set(1, 1, 1);
   data_set.set_data_random();

//   terms_Jacobian = sum_squared_error.calculate_error_terms_Jacobian();
//   numerical_Jacobian_terms = nd.calculate_Jacobian(sse, &SumSquaredError::calculate_training_terms, parameters);

   assert_true(absolute_value(terms_Jacobian-numerical_Jacobian_terms) < 1.0e-3, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 2, 2});
   neural_network.set_parameters_random();
   parameters = neural_network.get_parameters();

   data_set.set(2, 2, 2);
   data_set.set_data_random();

//   terms_Jacobian = sum_squared_error.calculate_error_terms_Jacobian();
//   numerical_Jacobian_terms = nd.calculate_Jacobian(sse, &SumSquaredError::calculate_training_terms, parameters);

   assert_true(absolute_value(terms_Jacobian-numerical_Jacobian_terms) < 1.0e-3, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 2, 2});
   neural_network.set_parameters_random();

   data_set.set(2, 2, 2);
   data_set.set_data_random();

//   gradient = sum_squared_error.calculate_gradient();

//   terms = sum_squared_error.calculate_training_error_terms();
//   terms_Jacobian = sum_squared_error.calculate_error_terms_Jacobian();

//   assert_true(absolute_value((terms_Jacobian.calculate_transpose()).dot(terms)*2.0 - gradient) < 1.0e-3, LOG);

}

void SumSquaredErrorTest::test_calculate_selection_error()
{
   cout << "test_calculate_selection_error\n";

//   NeuralNetwork neural_network;
//   DataSet data_set;
//   SumSquaredError sum_squared_error(&neural_network, &data_set);

//   type selection_error;

   // Test

//   nn.set();

//   nn.construct_multilayer_perceptron();

//   data_set.set();

//   Tensor<Index, 1> indices;

//   selection_error = sum_squared_error.calculate_indices_error(indices);
   
//   assert_true(selection_error == 0.0, LOG);
}


void SumSquaredErrorTest::test_calculate_squared_errors()
{
   cout << "test_calculate_squared_errors\n";

   NeuralNetwork neural_network;

   DataSet data_set;

   SumSquaredError sum_squared_error(&neural_network, &data_set);

   Tensor<type, 1> squared_errors;

//   type error;

   // Test 

   neural_network.set(NeuralNetwork::Approximation, {1,1,1});

   neural_network.set_parameters_constant(0.0);

   data_set.set(1,1,1);

   data_set.initialize_data(0.0);

//   squared_errors = sum_squared_error.calculate_squared_errors();

   assert_true(squared_errors.size() == 1, LOG);
   assert_true(squared_errors == 0.0, LOG);   

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2,2,2});

   neural_network.set_parameters_random();

   data_set.set(2,2,2);

   data_set.set_data_random();

//   squared_errors = sum_squared_error.calculate_squared_errors();

//   error = sum_squared_error.calculate_training_error();

//   assert_true(abs(squared_errors.sum() - error) < 1.0e-12, LOG);
}


void SumSquaredErrorTest::test_to_XML()   
{
    cout << "test_to_XML\n";

    SumSquaredError sum_squared_error;

//    tinyxml2::XMLDocument* document;

    // Test

//    document = sum_squared_error.to_XML();

//    assert_true(document != nullptr, LOG);

//    delete document;
}


void SumSquaredErrorTest::test_from_XML()
{
    cout << "test_from_XML\n";

    SumSquaredError sum_squared_error1;
    SumSquaredError sum_squared_error2;

   tinyxml2::XMLDocument* document;

   // Test

//   sum_squared_error1.set_display(false);

//   document = sum_squared_error1.to_XML();

//   sum_squared_error2.from_XML(*document);

//   delete document;

//   assert_true(sum_squared_error2.get_display() == false, LOG);
}
*/

void SumSquaredErrorTest::run_test_case()
{
   cout << "Running sum squared error test case...\n";
/*
   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   // Set methods

   // Error methods

   test_calculate_training_error();

   test_calculate_layers_delta();

   test_calculate_training_error_gradient();

   // Error terms methods

//   test_calculate_training_error_terms();

//   test_calculate_training_error_terms_Jacobian();

   //Serialization methods

    test_to_XML();

    test_from_XML();
*/
   cout << "End of sum squared error test case.\n";
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
