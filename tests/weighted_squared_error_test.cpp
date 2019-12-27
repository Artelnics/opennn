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
}


WeightedSquaredErrorTest::~WeightedSquaredErrorTest()
{
}


void WeightedSquaredErrorTest::test_constructor()
{
   cout << "test_constructor\n";

   // Default

   WeightedSquaredError wse1;

   assert_true(wse1.has_neural_network() == false, LOG);
   assert_true(wse1.has_data_set() == false, LOG);

   // Neural network

   NeuralNetwork nn2;
   WeightedSquaredError wse2(&nn2);

   assert_true(wse2.has_neural_network() == true, LOG);
   assert_true(wse2.has_data_set() == false, LOG);

   // Neural network and data set

   NeuralNetwork nn3;
   DataSet ds3;
   WeightedSquaredError wse3(&nn3, &ds3);

   assert_true(wse3.has_neural_network() == true, LOG);
   assert_true(wse3.has_data_set() == true, LOG);
}


void WeightedSquaredErrorTest::test_destructor()
{
}


void WeightedSquaredErrorTest::test_calculate_training_error()
{
   cout << "test_calculate_training_error\n";

   Vector<double> parameters;

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1, 1, 1});
   neural_network.initialize_parameters(0.0);

   DataSet data_set(1, 1, 1);
   data_set.initialize_data(0.0);

   WeightedSquaredError wse(&neural_network, &data_set);

   assert_true(wse.calculate_training_error() == 0.0, LOG);

   // Test

   size_t instances_number = 1000;
   size_t inputs_number = 90;
   size_t outputs_number = 1;
   size_t hidden_neurons_number = 180;

   data_set.set(instances_number, inputs_number, outputs_number);
   data_set.generate_data_binary_classification(instances_number, inputs_number);
   data_set.set_training();

   neural_network.set(NeuralNetwork::Approximation, {inputs_number, hidden_neurons_number, outputs_number});
   neural_network.randomize_parameters_normal();

//   nn.set_layer_activation_function(0, PerceptronLayer::Logistic);
//   nn.set_layer_activation_function(1, PerceptronLayer::Logistic);

   parameters = neural_network.get_parameters();

   wse.set_negatives_weight(1.0);
   wse.set_positives_weight(2.0);

   assert_true(abs(wse.calculate_training_error() - wse.calculate_training_error(parameters)) < 1.0e-3, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1});
   neural_network.randomize_parameters_normal();

   parameters = neural_network.get_parameters();

   data_set.set(2, 1, 1);
   data_set.generate_data_binary_classification(2, 1);
   data_set.set_training();

   assert_true(wse.calculate_training_error() == wse.calculate_training_error(parameters), LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, Vector<size_t>({3, 1}));

   data_set.set(2, 3, 1);
   data_set.generate_data_binary_classification(2, 3);
   data_set.set_training();

   wse.set_weights();

   parameters = neural_network.get_parameters();

   assert_true(wse.calculate_training_error() == wse.calculate_training_error(parameters), LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {3, 1});

   neural_network.initialize_parameters(0.0);

   data_set.set(2, 3, 1);
   data_set.generate_data_binary_classification(2, 3);

   wse.set_weights();

   assert_true(wse.get_positives_weight() == wse.get_negatives_weight(), LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {3, 1});

   neural_network.initialize_parameters(0.0);

   data_set.set(3, 3, 1);
   data_set.generate_data_binary_classification(3, 3);

   wse.set_weights();

   assert_true(wse.get_positives_weight() != wse.get_negatives_weight(), LOG);
}


void WeightedSquaredErrorTest::test_calculate_training_error_gradient()
{
   cout << "test_calculate_training_error_gradient\n";

   NeuralNetwork neural_network;

   DataSet data_set;

   WeightedSquaredError wse(&neural_network, &data_set);

   Vector<double> error_gradient;
   Vector<double> numerical_error_gradient;

   size_t instances_number;
   size_t inputs_number;
   size_t outputs_number;
   size_t hidden_neurons;

   ScalingLayer scaling_layer;

   RecurrentLayer recurrent_layer;

   LongShortTermMemoryLayer long_short_term_memory_layer;

   PerceptronLayer hidden_perceptron_layer;
   PerceptronLayer output_perceptron_layer;

   ProbabilisticLayer probabilistic_layer;

   // Test trivial
{
   instances_number = 10;
   inputs_number = 1;
   outputs_number = 1;

   data_set.set(instances_number, inputs_number, outputs_number);

   data_set.initialize_data(0.0);

   hidden_perceptron_layer.set(inputs_number, outputs_number);
   neural_network.add_layer(&hidden_perceptron_layer);

   neural_network.initialize_parameters(0.0);

   numerical_error_gradient = wse.calculate_training_error_gradient_numerical_differentiation();

   error_gradient = wse.calculate_training_error_gradient();

   assert_true(error_gradient.size() == neural_network.get_parameters_number(), LOG);
   assert_true(error_gradient == 0.0, LOG);
}

   neural_network.set();

   // Test perceptron and probabilistic
{
   instances_number = 10;
   inputs_number = 3;
   outputs_number = 1;
   hidden_neurons = 2;

   data_set.set(instances_number, inputs_number, outputs_number);

   Matrix<double> inputs(instances_number,inputs_number);

   inputs.randomize_normal();

   Vector<double> outputs(instances_number, outputs_number);
   outputs[0] = 1.0;
   outputs[1] = 0.0;

   for(size_t i = 2; i < instances_number; i++)
   {
        if((static_cast<int>(inputs.calculate_row_sum(i))%2) == 0.0)
        {
            outputs[i] = 0.0;
        }
        else
        {
            outputs[i] = 1.0;
        }
   }

   const Matrix<double> data = inputs.append_column(outputs);

   data_set.set_data(data);

   data_set.set_training();

   hidden_perceptron_layer.set(inputs_number, hidden_neurons);
   output_perceptron_layer.set(hidden_neurons, outputs_number);
   probabilistic_layer.set(outputs_number, outputs_number);

   neural_network.add_layer(&hidden_perceptron_layer);
   neural_network.add_layer(&output_perceptron_layer);
   neural_network.add_layer(&probabilistic_layer);

   neural_network.randomize_parameters_normal();

   error_gradient = wse.calculate_training_error_gradient();

   numerical_error_gradient = wse.calculate_training_error_gradient_numerical_differentiation();

   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
}

   neural_network.set();

   // Test lstm
{
   instances_number = 10;
   inputs_number = 3;
   outputs_number = 1;
   hidden_neurons = 2;

   data_set.set(instances_number, inputs_number, outputs_number);

   Matrix<double> inputs(instances_number,inputs_number);

   inputs.randomize_normal();

   Vector<double> outputs(instances_number, outputs_number);
   outputs[0] = 1.0;
   outputs[1] = 0.0;

   for(size_t i = 2; i < instances_number; i++)
   {
        if((static_cast<int>(inputs.calculate_row_sum(i))%2) == 0.0)
        {
            outputs[i] = 0.0;
        }
        else
        {
            outputs[i] = 1.0;
        }
   }

   const Matrix<double> data = inputs.append_column(outputs);

   data_set.set_data(data);

   data_set.set_training();

   long_short_term_memory_layer.set(inputs_number, hidden_neurons);
   output_perceptron_layer.set(hidden_neurons, outputs_number);

   neural_network.add_layer(&long_short_term_memory_layer);
   neural_network.add_layer(&output_perceptron_layer);

   neural_network.randomize_parameters_normal();

   error_gradient = wse.calculate_training_error_gradient();

   numerical_error_gradient = wse.calculate_training_error_gradient_numerical_differentiation();

   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
}

   neural_network.set();

   // Test recurrent
{
   instances_number = 10;
   inputs_number = 3;
   outputs_number = 1;
   hidden_neurons = 2;

   data_set.set(instances_number, inputs_number, outputs_number);

   Matrix<double> inputs(instances_number,inputs_number);

   inputs.randomize_normal();

   Vector<double> outputs(instances_number, outputs_number);
   outputs[0] = 1.0;
   outputs[1] = 0.0;

   for(size_t i = 2; i < instances_number; i++)
   {
        if((static_cast<int>(inputs.calculate_row_sum(i))%2) == 0.0)
        {
            outputs[i] = 0.0;
        }
        else
        {
            outputs[i] = 1.0;
        }
   }

   const Matrix<double> data = inputs.append_column(outputs);

   data_set.set_data(data);

   data_set.set_training();

   recurrent_layer.set(inputs_number, hidden_neurons);
   output_perceptron_layer.set(hidden_neurons, outputs_number);

   neural_network.add_layer(&recurrent_layer);
   neural_network.add_layer(&output_perceptron_layer);

   neural_network.randomize_parameters_normal();

   error_gradient = wse.calculate_training_error_gradient();

   numerical_error_gradient = wse.calculate_training_error_gradient_numerical_differentiation();

   assert_true(absolute_value(error_gradient - numerical_error_gradient) < 1.0e-3, LOG);
}
}


void WeightedSquaredErrorTest::test_calculate_selection_error()
{
   cout << "test_calculate_selection_error\n";

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1, 1, 1});

   neural_network.initialize_parameters(0.0);

   DataSet data_set(1, 1, 1);

   data_set.set_selection();

   data_set.initialize_data(0.0);

   WeightedSquaredError wse(&neural_network, &data_set);

   double selection_error = wse.calculate_selection_error();

   assert_true(selection_error == 0.0, LOG);

}


void WeightedSquaredErrorTest::test_calculate_training_error_terms()
{
   cout << "test_calculate_training_error_terms\n";

   NeuralNetwork neural_network;
   Vector<size_t> hidden_layers_size;
   Vector<double> parameters;

   DataSet data_set;
   
   WeightedSquaredError wse(&neural_network, &data_set);

   double error;

   Vector<double> error_terms;

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 1});
   neural_network.randomize_parameters_normal();

   data_set.set(3, 2, 2);
   data_set.generate_data_binary_classification(3, 2);

   error = wse.calculate_training_error();

//   error_terms = wse.calculate_training_error_terms();

   assert_true(abs((error_terms*error_terms).calculate_sum() - error) < 1.0e-3, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {3, 1});
   neural_network.randomize_parameters_normal();

   data_set.set(9, 3, 1);
   data_set.generate_data_binary_classification(9, 3);

   error = wse.calculate_training_error();

//   error_terms = wse.calculate_training_error_terms();

   assert_true(abs((error_terms*error_terms).calculate_sum() - error) < 1.0e-3, LOG);
}


void WeightedSquaredErrorTest::test_calculate_training_error_terms_Jacobian()
{
   cout << "test_calculate_training_error_terms_Jacobian\n";

   NumericalDifferentiation nd;

   NeuralNetwork neural_network;
   Vector<size_t> architecture;
   Vector<double> parameters;

   DataSet data_set;

   WeightedSquaredError wse(&neural_network, &data_set);

   Vector<double> error_gradient;

   Vector<double> error_terms;
   Matrix<double> terms_Jacobian;
   Matrix<double> numerical_Jacobian_terms;

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1});

   neural_network.initialize_parameters(0.0);

   data_set.set(1, 1, 1);

   data_set.generate_data_binary_classification(3, 1);

//   terms_Jacobian = wse.calculate_error_terms_Jacobian();

   assert_true(terms_Jacobian.get_rows_number() == data_set.get_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == neural_network.get_parameters_number(), LOG);
//   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {3, 4, 2});
   neural_network.initialize_parameters(0.0);

   data_set.set(3, 2, 5);
   wse.set(&neural_network, &data_set);
   data_set.generate_data_binary_classification(3, 3);

//   terms_Jacobian = wse.calculate_error_terms_Jacobian();

   assert_true(terms_Jacobian.get_rows_number() == data_set.get_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == neural_network.get_parameters_number(), LOG);
//   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

   architecture.set(3);
   architecture[0] = 2;
   architecture[1] = 1;
   architecture[2] = 2;

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.initialize_parameters(0.0);

   data_set.set(2, 2, 5);
   wse.set(&neural_network, &data_set);
   data_set.generate_data_binary_classification(3, 2);

//   terms_Jacobian = wse.calculate_error_terms_Jacobian();

   assert_true(terms_Jacobian.get_rows_number() == data_set.get_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == neural_network.get_parameters_number(), LOG);
//   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1, 1});
   neural_network.randomize_parameters_normal();
   parameters = neural_network.get_parameters();

   data_set.set(3, 1, 1);
   data_set.generate_data_binary_classification(3, 1);

//   terms_Jacobian = wse.calculate_error_terms_Jacobian();
//   numerical_Jacobian_terms = nd.calculate_Jacobian(wse, &WeightedSquaredError::calculate_training_error_terms, parameters);

   //assert_true(absolute_value(terms_Jacobian-numerical_Jacobian_terms) < 1.0e-3, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {2, 2, 1});
   neural_network.randomize_parameters_normal();
   parameters = neural_network.get_parameters();

   data_set.set(2, 2, 1);
   data_set.generate_data_binary_classification(2, 2);

//   terms_Jacobian = wse.calculate_error_terms_Jacobian();
//   numerical_Jacobian_terms = nd.calculate_Jacobian(wse, &WeightedSquaredError::calculate_training_error_terms, parameters);

   assert_true(absolute_value(terms_Jacobian-numerical_Jacobian_terms) < 1.0e-3, LOG);

   // Test

//   nn.set(2, 2, 2);
//   nn.randomize_parameters_normal();

//   data_set.set(2, 2, 2);
//   data_set.generate_data_binary_classification(4, 2);
   
//   error_gradient = wse.calculate_gradient();

//   error_terms = wse.calculate_training_error_terms();
//   terms_Jacobian = wse.calculate_error_terms_Jacobian();

//   cout << (terms_Jacobian.calculate_transpose()).dot(error_terms)*2.0 << endl;
//   cout << error_gradient << endl;

//   assert_true(absolute_value((terms_Jacobian.calculate_transpose()).dot(error_terms)*2.0 - error_gradient) < 1.0e-3, LOG);
}


void WeightedSquaredErrorTest::test_to_XML()
{
   cout << "test_to_XML\n";
}


void WeightedSquaredErrorTest::test_from_XML()
{
   cout << "test_from_XML\n";
}


void WeightedSquaredErrorTest::run_test_case()
{
   cout << "Running weighted squared error test case...\n";

   // Constructor and destructor methods

//   test_constructor();
//   test_destructor();

   // Get methods

   // Set methods

   // Error methods

//   test_calculate_training_loss();

//   test_calculate_selection_error();

   test_calculate_training_error_gradient();

   // Error terms methods

//   test_calculate_training_error_terms();

//   test_calculate_training_error_terms_Jacobian();

   // Loss hessian methods

   // Serialization methods

//   test_to_XML();
//   test_from_XML();

   cout << "End of weighted squared error test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
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
