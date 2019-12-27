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

   Vector<double> parameters;

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1,1});

   DataSet data_set(1,1,1);

   size_t instances_number;
   size_t inputs_number;
   size_t outputs_number;
   size_t hidden_neurons;

   Matrix<double> new_data(2, 2);
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
   neural_network.randomize_parameters_normal();

   parameters = neural_network.get_parameters();

   data_set.set(instances_number, inputs_number, outputs_number);
   data_set.randomize_data_normal();

   normalized_squared_error.set_normalization_coefficient();

   assert_true(abs(normalized_squared_error.calculate_training_error() - normalized_squared_error.calculate_training_error(parameters)) < 1.0e-03, LOG);
}


void NormalizedSquaredErrorTest::test_calculate_training_error_gradient(void)
{
   cout << "test_calculate_training_error_gradient\n";

   NeuralNetwork neural_network;

   DataSet data_set;

   NormalizedSquaredError nse(&neural_network, &data_set);

   Vector<double> error_gradient;
   Vector<double> numerical_error_gradient;

   size_t instances_number;
   size_t inputs_number;
   size_t outputs_number;
   size_t hidden_neurons;

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

   neural_network.initialize_parameters(0.0);

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

   data_set.randomize_data_normal();

   data_set.set_training();

   hidden_perceptron_layer->set(inputs_number, hidden_neurons);
   output_perceptron_layer->set(hidden_neurons, outputs_number);
   probabilistic_layer->set(outputs_number, outputs_number);

   neural_network.add_layer(hidden_perceptron_layer);
   neural_network.add_layer(output_perceptron_layer);
   neural_network.add_layer(probabilistic_layer);

   neural_network.randomize_parameters_normal();

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

   data_set.randomize_data_normal();

   data_set.set_training();

   long_short_term_memory_layer->set(inputs_number, hidden_neurons);
   output_perceptron_layer->set(hidden_neurons, outputs_number);

   neural_network.add_layer(long_short_term_memory_layer);
   neural_network.add_layer(output_perceptron_layer);

   neural_network.randomize_parameters_normal();

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

   data_set.randomize_data_normal();

   data_set.set_training();

   recurrent_layer->set(inputs_number, hidden_neurons);
   output_perceptron_layer->set(hidden_neurons, outputs_number);

   neural_network.add_layer(recurrent_layer);
   neural_network.add_layer(output_perceptron_layer);

   neural_network.randomize_parameters_normal();

   nse.set_normalization_coefficient();

   error_gradient = nse.calculate_training_error_gradient();

   numerical_error_gradient = nse.calculate_training_error_gradient_numerical_differentiation();

   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
}
}


void NormalizedSquaredErrorTest::test_calculate_training_error_terms(void)
{
   cout << "test_calculate_training_error_terms\n";

   NeuralNetwork neural_network;
   Vector<size_t> architecture;
   Vector<double> network_parameters;

   DataSet data_set;

   NormalizedSquaredError nse(&neural_network, &data_set);

//   double error;

   Vector<double> error_terms;

   size_t instances_number;
   size_t inputs_number;
   size_t outputs_number;

   // Test

   instances_number = 7;
   inputs_number = 6;
   outputs_number = 7;

   neural_network.set(NeuralNetwork::Approximation, {inputs_number, outputs_number});
   neural_network.randomize_parameters_normal();

   data_set.set(instances_number, inputs_number, outputs_number);
   data_set.randomize_data_normal();

   nse.set_normalization_coefficient();

//   error = nse.calculate_training_error();

//   error_terms = nse.calculate_training_error_terms();

//   assert_true(abs((error_terms*error_terms).calculate_sum() - error) < 1.0e-3, LOG);

}


void NormalizedSquaredErrorTest::test_calculate_training_error_terms_Jacobian(void)
{
   cout << "test_calculate_training_error_terms_Jacobian\n";

   NumericalDifferentiation nd;

   NeuralNetwork neural_network;
   Vector<int> hidden_layers_size;
   Vector<double> network_parameters;

   DataSet data_set;

   NormalizedSquaredError nse(&neural_network, &data_set);

   Vector<double> error_gradient;

   Vector<double> error_terms;
   Matrix<double> terms_Jacobian;
   Matrix<double> numerical_Jacobian_terms;

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1});
   neural_network.randomize_parameters_normal();
   network_parameters = neural_network.get_parameters();

   data_set.set(2, 1, 1);
   data_set.randomize_data_normal();

//   terms_Jacobian = nse.calculate_error_terms_Jacobian();
   numerical_Jacobian_terms = nd.calculate_Jacobian(nse, &NormalizedSquaredError::calculate_training_error_terms, network_parameters);

   assert_true(absolute_value(terms_Jacobian-numerical_Jacobian_terms) < 1.0e-3, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 2, 2});
   neural_network.randomize_parameters_normal();
   network_parameters = neural_network.get_parameters();

   data_set.set(2, 2, 2);
   data_set.randomize_data_normal();

//   terms_Jacobian = nse.calculate_error_terms_Jacobian();
   numerical_Jacobian_terms = nd.calculate_Jacobian(nse, &NormalizedSquaredError::calculate_training_error_terms, network_parameters);

   assert_true(absolute_value(terms_Jacobian-numerical_Jacobian_terms) < 1.0e-3, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 2, 2});
   neural_network.randomize_parameters_normal();

   data_set.set(2,2,2);
   data_set.randomize_data_normal();
   
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

    Vector<double> squared_errors;

    // Test

    neural_network.set(NeuralNetwork::Approximation, {1, 1});
    neural_network.randomize_parameters_normal();

    data_set.set(2, 1, 1);
    data_set.randomize_data_normal();

    squared_errors = nse.calculate_squared_errors();

    assert_true(squared_errors.size() == 2, LOG);
}


void NormalizedSquaredErrorTest::test_calculate_maximal_errors(void)
{
    cout << "test_calculate_maximal_errors\n";

    NeuralNetwork neural_network;

    DataSet data_set;

    NormalizedSquaredError nse(&neural_network, &data_set);

    Vector<double> squared_errors;
    Vector<size_t> maximal_errors;

    // Test

    neural_network.set(NeuralNetwork::Approximation, {1, 1});
    neural_network.randomize_parameters_normal();

    data_set.set(3, 1, 1);
    data_set.randomize_data_normal();

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


void NormalizedSquaredErrorTest::run_test_case(void)
{
   cout << "Running normalized squared error test case...\n";

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
