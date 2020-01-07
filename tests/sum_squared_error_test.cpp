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
   Vector<double> parameters;

   DataSet data_set;
   Matrix<double> data;

   SumSquaredError sum_squared_error(&neural_network, &data_set);

   double training_error;

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 2});
   neural_network.initialize_parameters(0.0);

   data_set.set(1, 2, 2);
   data_set.initialize_data(0.0);

   training_error = 0.0;

   training_error = sum_squared_error.calculate_training_error();

   assert_true(training_error == 0.0, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 2});
   neural_network.initialize_parameters(0.0);

   data_set.set(1, 2, 2);
   data_set.initialize_data(1.0);

   training_error = sum_squared_error.calculate_training_error();

   assert_true(training_error == 2.0, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 2});
   neural_network.randomize_parameters_normal();

   parameters = neural_network.get_parameters();

   data_set.set(10, 2, 2);
   data_set.randomize_data_normal();


   ///@todo
//   assert_true(abs(sum_squared_error.calculate_training_error() - sum_squared_error.calculate_training_error(parameters)) < numeric_limits<double>::min(), LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1});
   neural_network.randomize_parameters_normal();

   parameters = neural_network.get_parameters();

   data_set.set(1, 1, 1);
   data_set.randomize_data_normal();

   assert_true(abs(sum_squared_error.calculate_training_error() - sum_squared_error.calculate_training_error(parameters*2.0)) > numeric_limits<double>::min(), LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1});
   neural_network.randomize_parameters_normal();

   parameters = neural_network.get_parameters();

   data_set.set(1, 1, 1);
   data_set.randomize_data_normal();

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

   neural_network.initialize_parameters(0.0);

   training_error = sum_squared_error.calculate_training_error();

   assert_true(abs(training_error) < numeric_limits<double>::min(), LOG);
}


void SumSquaredErrorTest::test_calculate_layers_delta()
{
   cout << "test_calculate_layers_delta\n";

   DataSet data_set;
   NeuralNetwork neural_network;
   NumericalDifferentiation numerical_differentation;

   Vector<double> parameters;

   SumSquaredError sum_squared_error(&neural_network, &data_set);

   // Test

    size_t inputs_number = 2;
    size_t outputs_number = 2;
    size_t instances_number = 10;
    size_t hidden_neurons = 1;

    Vector<size_t> instances;
    instances.set(instances_number);
    instances.initialize_sequential();

    neural_network.set(NeuralNetwork::Approximation, {inputs_number,hidden_neurons,outputs_number});
    neural_network.randomize_parameters_normal();

    data_set.set(instances_number,inputs_number,outputs_number);
    data_set.randomize_data_normal();

    Tensor<double> inputs = data_set.get_input_data(instances);
    Tensor<double> targets = data_set.get_target_data(instances);

    Tensor<double> outputs = neural_network.calculate_outputs(inputs);
    Tensor<double> output_gradient = sum_squared_error.calculate_output_gradient(outputs, targets);

    Vector<Layer::FirstOrderActivations> forward_propagation = neural_network.calculate_trainable_forward_propagation(inputs);

    Vector<Tensor<double>> layers_delta = sum_squared_error.calculate_layers_delta(forward_propagation, output_gradient);

    assert_true(layers_delta[0].get_dimension(0) == instances_number, LOG);
    assert_true(layers_delta[0].get_dimension(1) == hidden_neurons, LOG);
    assert_true(layers_delta[1].get_dimension(0) == instances_number, LOG);
    assert_true(layers_delta[1].get_dimension(1) == outputs_number, LOG);
}


void SumSquaredErrorTest::test_calculate_training_error_gradient()
{
   cout << "test_calculate_training_error_gradient\n";

   DataSet data_set;
   NeuralNetwork neural_network;
   SumSquaredError sum_squared_error(&neural_network, &data_set);

   Vector<size_t> architecture;

   Vector<double> parameters;
   Vector<double> gradient;
   Vector<double> numerical_gradient;
   Vector<double> error;

   size_t inputs_number;
   size_t outputs_number;
   size_t instances_number;
   size_t hidden_neurons;

   // Test lstm
{
    instances_number = 10;
    inputs_number = 3;
    hidden_neurons = 2;
    outputs_number = 4;

    data_set.set(instances_number, inputs_number, outputs_number);

    data_set.randomize_data_normal();

    data_set.set_training();

    neural_network.set();

    LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer(inputs_number, hidden_neurons);

    long_short_term_memory_layer->set_timesteps(8);

    neural_network.add_layer(long_short_term_memory_layer);

    PerceptronLayer* perceptron_layer = new PerceptronLayer(hidden_neurons,outputs_number);

    neural_network.add_layer(perceptron_layer);

    neural_network.randomize_parameters_normal();

    parameters = neural_network.get_parameters();

    numerical_gradient = sum_squared_error.calculate_training_error_gradient_numerical_differentiation();

    gradient = sum_squared_error.calculate_training_error_gradient();

    ///@todo sometimes fail

//    assert_true(numerical_gradient-gradient < 1.0e-3, LOG);
}

   neural_network.set();

   // Test recurrent
{
    instances_number = 5;
    inputs_number = 3;
    hidden_neurons = 7;
    outputs_number = 3;

    data_set.set(instances_number, inputs_number, outputs_number);

    data_set.randomize_data_normal();

    data_set.set_training();

    neural_network.set();

    RecurrentLayer* recurrent_layer = new RecurrentLayer(inputs_number, hidden_neurons);

    recurrent_layer->initialize_hidden_states(0.0);
    recurrent_layer->set_timesteps(10);

    neural_network.add_layer(recurrent_layer);

    PerceptronLayer* perceptron_layer = new PerceptronLayer(hidden_neurons,outputs_number);

    neural_network.add_layer(perceptron_layer);

    neural_network.randomize_parameters_normal();

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

    data_set.randomize_data_normal();

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

   instances_number = 5;
   inputs_number = 75;
   outputs_number = 1;

   data_set.set(instances_number, inputs_number, outputs_number);
   data_set.set_input_variables_dimensions(Vector<size_t>({3,5,5}));
   data_set.set_target_variables_dimensions(Vector<size_t>({1}));
   data_set.randomize_data_normal();
   data_set.set_training();

   ConvolutionalLayer* convolutional_layer_1 = new ConvolutionalLayer({3,5,5}, {2,2,2});
   convolutional_layer_1->set_padding_option(OpenNN::ConvolutionalLayer::Same);
   Tensor<double> filters_1({2,3,2,2}, 0);
   filters_1.randomize_normal();
   convolutional_layer_1->set_synaptic_weights(filters_1);
   Vector<double> biases_1(2, 0);
   biases_1.randomize_normal();
   convolutional_layer_1->set_biases(biases_1);

   PoolingLayer* pooling_layer_1 = new PoolingLayer(convolutional_layer_1->get_outputs_dimensions(), {2,2});

   ConvolutionalLayer* convolutional_layer_2 = new ConvolutionalLayer(pooling_layer_1->get_outputs_dimensions(), {1,2,2});
   convolutional_layer_2->set_padding_option(OpenNN::ConvolutionalLayer::Same);
   Tensor<double> filters_2({1,2,2,2}, 0);
   filters_2.randomize_normal();
   convolutional_layer_2->set_synaptic_weights(filters_2);
   Vector<double> biases_2(1, 0);
   biases_2.randomize_normal();
   convolutional_layer_2->set_biases(biases_2);

   PoolingLayer* pooling_layer_2 = new PoolingLayer(convolutional_layer_2->get_outputs_dimensions(), {2,2});

   PerceptronLayer* perceptron_layer = new PerceptronLayer(pooling_layer_2->get_outputs_dimensions().calculate_product(), 3, OpenNN::PerceptronLayer::ActivationFunction::Linear);

   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(perceptron_layer->get_neurons_number(), outputs_number);

   neural_network.set();
   neural_network.add_layer(convolutional_layer_1);
   neural_network.add_layer(pooling_layer_1);
   neural_network.add_layer(convolutional_layer_2);
   neural_network.add_layer(pooling_layer_2);
   neural_network.add_layer(perceptron_layer);
   neural_network.add_layer(probabilistic_layer);

   numerical_gradient = sum_squared_error.calculate_training_error_gradient_numerical_differentiation();

   gradient = sum_squared_error.calculate_training_error_gradient();

   assert_true(numerical_gradient - gradient < 1e-3 &&
               gradient - numerical_gradient < 1e-3, LOG);
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
   Vector<size_t> architecture;
   Vector<double> parameters;

   DataSet data_set;

   SumSquaredError sum_squared_error(&neural_network, &data_set);

   Vector<double> gradient;

   Vector<double> terms;
   Matrix<double> terms_Jacobian;
   Matrix<double> numerical_Jacobian_terms;

   Vector<size_t> instances;

   Tensor<double> inputs;
   Tensor<double> targets;

   Tensor<double> outputs;
   Tensor<double> output_gradient;

   Vector<Tensor<double>> layers_activations;

   Vector<Tensor<double>> layers_activations_derivatives;

   Vector<Tensor<double>> layers_delta;

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1, 1});

   neural_network.initialize_parameters(0.0);

   data_set.set(1, 1, 1);

   data_set.initialize_data(0.0);

   instances.set(1,0);
   //instances.initialize_sequential();

   inputs = data_set.get_input_data(instances);
   targets = data_set.get_target_data(instances);

   outputs = neural_network.calculate_outputs(inputs);
   output_gradient = sum_squared_error.calculate_output_gradient(outputs, targets);

   Vector<Layer::FirstOrderActivations> forward_propagation = neural_network.calculate_trainable_forward_propagation(inputs);

   layers_delta = sum_squared_error.calculate_layers_delta(forward_propagation, output_gradient);

   terms_Jacobian = sum_squared_error.calculate_error_terms_Jacobian(inputs, forward_propagation, layers_delta);

   assert_true(terms_Jacobian.get_rows_number() == data_set.get_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == neural_network.get_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);

   // Test 

   neural_network.set(NeuralNetwork::Approximation, {3, 4, 2});
   neural_network.initialize_parameters(0.0);

   data_set.set(3, 2, 5);
   sum_squared_error.set(&neural_network, &data_set);
   data_set.initialize_data(0.0);

//   terms_Jacobian = sum_squared_error.calculate_error_terms_Jacobian();

   assert_true(terms_Jacobian.get_rows_number() == data_set.get_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == neural_network.get_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

   architecture.set(3);
   architecture[0] = 5;
   architecture[1] = 1;
   architecture[2] = 2;

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.initialize_parameters(0.0);

   data_set.set(5, 2, 3);
   sum_squared_error.set(&neural_network, &data_set);
   data_set.initialize_data(0.0);

//   terms_Jacobian = sum_squared_error.calculate_error_terms_Jacobian();

   assert_true(terms_Jacobian.get_rows_number() == data_set.get_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == neural_network.get_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1, 1});
   neural_network.randomize_parameters_normal();
   parameters = neural_network.get_parameters();

   data_set.set(1, 1, 1);
   data_set.randomize_data_normal();

//   terms_Jacobian = sum_squared_error.calculate_error_terms_Jacobian();
//   numerical_Jacobian_terms = nd.calculate_Jacobian(sse, &SumSquaredError::calculate_training_terms, parameters);

   assert_true(absolute_value(terms_Jacobian-numerical_Jacobian_terms) < 1.0e-3, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 2, 2});
   neural_network.randomize_parameters_normal();
   parameters = neural_network.get_parameters();

   data_set.set(2, 2, 2);
   data_set.randomize_data_normal();

//   terms_Jacobian = sum_squared_error.calculate_error_terms_Jacobian();
//   numerical_Jacobian_terms = nd.calculate_Jacobian(sse, &SumSquaredError::calculate_training_terms, parameters);

   assert_true(absolute_value(terms_Jacobian-numerical_Jacobian_terms) < 1.0e-3, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 2, 2});
   neural_network.randomize_parameters_normal();

   data_set.set(2, 2, 2);
   data_set.randomize_data_normal();

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

//   double selection_error;

   // Test

//   nn.set();

//   nn.construct_multilayer_perceptron();

//   data_set.set();

//   Vector<size_t> indices;

//   selection_error = sum_squared_error.calculate_indices_error(indices);
   
//   assert_true(selection_error == 0.0, LOG);
}


void SumSquaredErrorTest::test_calculate_squared_errors()
{
   cout << "test_calculate_squared_errors\n";

   NeuralNetwork neural_network;

   DataSet data_set;

   SumSquaredError sum_squared_error(&neural_network, &data_set);

   Vector<double> squared_errors;

   double error;

   // Test 

   neural_network.set(NeuralNetwork::Approximation, {1,1,1});

   neural_network.initialize_parameters(0.0);

   data_set.set(1,1,1);

   data_set.initialize_data(0.0);

//   squared_errors = sum_squared_error.calculate_squared_errors();

   assert_true(squared_errors.size() == 1, LOG);
   assert_true(squared_errors == 0.0, LOG);   

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2,2,2});

   neural_network.randomize_parameters_normal();

   data_set.set(2,2,2);

   data_set.randomize_data_normal();

//   squared_errors = sum_squared_error.calculate_squared_errors();

//   error = sum_squared_error.calculate_training_error();

//   assert_true(abs(squared_errors.calculate_sum() - error) < 1.0e-12, LOG);
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


void SumSquaredErrorTest::run_test_case()
{
   cout << "Running sum squared error test case...\n";

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
