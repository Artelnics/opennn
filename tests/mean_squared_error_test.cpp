/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M E A N   S Q U A R E D   E R R O R   T E S T   C L A S S                                                  */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "mean_squared_error_test.h"

using namespace OpenNN;

// GENERAL CONSTRUCTOR

MeanSquaredErrorTest::MeanSquaredErrorTest() : UnitTesting() 
{
}


// DESTRUCTOR

MeanSquaredErrorTest::~MeanSquaredErrorTest()
{
}


// METHODS


void MeanSquaredErrorTest::test_constructor()
{
   message += "test_constructor\n";

   // Default

   MeanSquaredError mse1;

   assert_true(mse1.has_neural_network() == false, LOG);
   assert_true(mse1.has_data_set() == false, LOG);

   // Neural network

   NeuralNetwork nn2;
   MeanSquaredError mse2(&nn2);

   assert_true(mse2.has_neural_network() == true, LOG);
   assert_true(mse2.has_data_set() == false, LOG);

   // Neural network and data set

   NeuralNetwork nn3;
   DataSet ds3;
   MeanSquaredError mse3(&nn3, &ds3);

   assert_true(mse3.has_neural_network() == true, LOG);
   assert_true(mse3.has_data_set() == true, LOG);

}


void MeanSquaredErrorTest::test_destructor()
{
    message += "test_destructor\n";

}


void MeanSquaredErrorTest::test_calculate_error()
{
   message += "test_calculate_error\n";
/*
   Vector<double> parameters;

   NeuralNetwork nn(1, 1, 1);
   nn.initialize_parameters(0.0);

   DataSet ds(1, 1, 1);
   ds.initialize_data(0.0);

   MeanSquaredError mse(&nn, &ds);

   assert_true(mse.calculate_all_instances_error() == 0.0, LOG);

   // Test

   nn.set(1, 1);
   nn.randomize_parameters_normal();

   parameters = nn.get_parameters();

   ds.set(1, 1, 1);
   ds.randomize_data_normal();

   assert_true((mse.calculate_all_instances_error() - mse.calculate_error(parameters)) < std::numeric_limits<double>::min(), LOG);
*/
}


void MeanSquaredErrorTest::test_calculate_error_gradient()
{
   message += "test_calculate_error_gradient\n";

   NumericalDifferentiation nd;

   NeuralNetwork nn;
   Vector<size_t> multilayer_perceptron_architecture;

   Vector<double> parameters;

   DataSet ds;

   MeanSquaredError mse(&nn, &ds);

   Vector<double> gradient;
   Vector<double> numerical_gradient;
   Vector<double> error;

   // Test

   nn.set(1, 1, 1);

   nn.initialize_parameters(0.0);

   ds.set(1, 1, 1);

   ds.initialize_data(0.0);

   gradient = mse.calculate_training_error_gradient();

   assert_true(gradient.size() == nn.get_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test 

   nn.set(3, 4, 2);
   nn.initialize_parameters(0.0);

   ds.set(3, 2, 5);
   mse.set(&nn, &ds);
   ds.initialize_data(0.0);

   gradient = mse.calculate_training_error_gradient();

   assert_true(gradient.size() == nn.get_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test

   multilayer_perceptron_architecture.set(3);
   multilayer_perceptron_architecture[0] = 2;
   multilayer_perceptron_architecture[1] = 1;
   multilayer_perceptron_architecture[2] = 3;

   nn.set(multilayer_perceptron_architecture);
   nn.initialize_parameters(0.0);

   ds.set(2, 3, 5);
   mse.set(&nn, &ds);
   ds.initialize_data(0.0);

   gradient = mse.calculate_training_error_gradient();

   assert_true(gradient.size() == nn.get_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test

   nn.set(1, 1, 1);

   nn.initialize_parameters(0.0);

   ds.set(1, 1, 1);

   ds.initialize_data(0.0);

   gradient = mse.calculate_training_error_gradient();

   assert_true(gradient.size() == nn.get_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test 

   nn.set(3, 4, 2);
   nn.initialize_parameters(0.0);

   ds.set(3, 2, 5);
   mse.set(&nn, &ds);
   ds.initialize_data(0.0);

   gradient = mse.calculate_training_error_gradient();

   assert_true(gradient.size() == nn.get_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test

   nn.set(2, 10, 3);
   //nn.initialize_parameters(1.0);
   nn.randomize_parameters_normal();
   parameters = nn.get_parameters();

   ds.set(10, 2, 3);
   ds.initialize_data(1.0);

   const Vector<size_t> indices(0,1,9);

   gradient = mse.calculate_training_error_gradient();
   numerical_gradient = nd.calculate_gradient(mse, &MeanSquaredError::calculate_training_error, parameters);
   assert_true((gradient - numerical_gradient).calculate_absolute_value() < 1.0e-3, LOG);

   // Test

   ds.initialize_data(1.0);

   nn.randomize_parameters_normal();
   parameters = nn.get_parameters();

   gradient = mse.calculate_training_error_gradient();
   numerical_gradient = nd.calculate_gradient(mse, &MeanSquaredError::calculate_training_error, parameters);
   error = (gradient - numerical_gradient).calculate_absolute_value();

}

/*
void MeanSquaredErrorTest::test_calculate_selection_error()
{
   message += "test_calculate_selection_error\n";

   NeuralNetwork nn(1, 1, 1);

   nn.initialize_parameters(0.0);

   DataSet ds(1, 1, 1);

   ds.get_instances_pointer()->set_selection();

   ds.initialize_data(0.0);

   MeanSquaredError mse(&nn, &ds);  

   double selection_error = mse.calculate_error({0});

   assert_true(selection_error == 0.0, LOG);
}
*/

void MeanSquaredErrorTest::test_calculate_error_terms()
{
   message += "test_calculate_error_terms\n";

   NeuralNetwork nn;
   Vector<size_t> hidden_layers_size;
   Vector<double> parameters;

   DataSet ds;
   
   MeanSquaredError mse(&nn, &ds);

   double error;

   Vector<double> error_terms;

   // Test

   nn.set(1, 1);
   nn.randomize_parameters_normal();

   ds.set(1, 1, 1);
   ds.randomize_data_normal();

   const Matrix<double> inputs = ds.get_training_inputs();
   const Matrix<double> targets = ds.get_training_targets();
   const Matrix<double> outputs = nn.calculate_outputs(inputs);

   error = mse.calculate_training_error();

   error_terms = mse.calculate_error_terms(outputs, targets);

   assert_true(fabs((error_terms.dot(error_terms)) - error) < 1.0e-3, LOG);
}


void MeanSquaredErrorTest::test_calculate_error_terms_Jacobian()
{
   message += "test_calculate_error_terms_Jacobian\n";

   NumericalDifferentiation nd;

   NeuralNetwork nn;
   Vector<size_t> multilayer_perceptron_architecture;
   Vector<double> parameters;

   DataSet ds;

   MeanSquaredError mse(&nn, &ds);

   Vector<double> error_gradient;

   Vector<double> error_terms;
   Matrix<double> terms_Jacobian;
   Matrix<double> numerical_Jacobian_terms;

   Matrix<double> inputs;
   Matrix<double> targets;
   Matrix<double> outputs;

   Matrix<double> output_gradient;
   Vector<Matrix<double>> layers_delta;

   MultilayerPerceptron::FirstOrderForwardPropagation first_order_propagation(0);

   // Test
/*
   nn.set(1, 1);

   nn.initialize_parameters(0.0);

   ds.set(1, 1, 1);

   ds.initialize_data(0.0);

   inputs = ds.get_inputs();
   targets = ds.get_targets();
   outputs = nn.calculate_outputs(inputs);

   first_order_propagation = nn.get_multilayer_perceptron_pointer()->calculate_first_order_forward_propagation(inputs);

   output_gradient = mse.calculate_output_gradient(outputs, targets);

   layers_delta = mse.calculate_layers_delta(first_order_propagation.layers_activations, output_gradient);

   terms_Jacobian = mse.calculate_error_terms_Jacobian(inputs, first_order_propagation.layers_activation_derivatives, layers_delta);

   assert_true(terms_Jacobian.get_rows_number() == ds.get_instances().get_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == nn.get_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);

   // Test 

   nn.set(3, 4, 2);
   nn.initialize_parameters(0.0);

   ds.set(3, 2, 5);
   mse.set(&nn, &ds);
   ds.initialize_data(0.0);

   inputs = ds.get_inputs();
   targets = ds.get_targets();
   outputs = nn.calculate_outputs(inputs);

   first_order_propagation = nn.get_multilayer_perceptron_pointer()->calculate_first_order_forward_propagation(inputs);

   output_gradient = mse.calculate_output_gradient(outputs, targets);

   layers_delta = mse.calculate_layers_delta(first_order_propagation.layers_activations, output_gradient);

   terms_Jacobian = mse.calculate_error_terms_Jacobian(inputs, first_order_propagation.layers_activation_derivatives, layers_delta);

   assert_true(terms_Jacobian.get_rows_number() == ds.get_instances().get_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == nn.get_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);


   // Test

   multilayer_perceptron_architecture.set(3);
   multilayer_perceptron_architecture[0] = 2;
   multilayer_perceptron_architecture[1] = 1;
   multilayer_perceptron_architecture[2] = 2;

   nn.set(multilayer_perceptron_architecture);
   nn.initialize_parameters(0.0);

   ds.set(2, 2, 5);
   mse.set(&nn, &ds);
   ds.initialize_data(0.0);

   inputs = ds.get_inputs();
   targets = ds.get_targets();
   outputs = nn.calculate_outputs(inputs);

   first_order_propagation = nn.get_multilayer_perceptron_pointer()->calculate_first_order_forward_propagation(inputs);

   output_gradient = mse.calculate_output_gradient(outputs, targets);

   layers_delta = mse.calculate_layers_delta(first_order_propagation.layers_activations, output_gradient);

   terms_Jacobian = mse.calculate_error_terms_Jacobian(inputs, first_order_propagation.layers_activation_derivatives, layers_delta);

   assert_true(terms_Jacobian.get_rows_number() == ds.get_instances().get_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == nn.get_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);
*/
   // Test

   nn.set(1, 1);
   nn.initialize_parameters(0.0);
   nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, PerceptronLayer::Linear);
//   nn.randomize_parameters_normal();
   parameters = nn.get_parameters();

   ds.set(1, 1, 1);
//   ds.randomize_data_normal();
   ds.initialize_data(1.0);

   inputs = ds.get_inputs();
   targets = ds.get_targets();
   outputs = nn.calculate_outputs(inputs);

   first_order_propagation = nn.get_multilayer_perceptron_pointer()->calculate_first_order_forward_propagation(inputs);

   output_gradient = mse.calculate_output_gradient(outputs, targets);

   layers_delta = mse.calculate_layers_delta(first_order_propagation.layers_activations, output_gradient);

   cout << "layers delta: " << layers_delta << endl;

   terms_Jacobian = mse.calculate_error_terms_Jacobian(inputs, first_order_propagation.layers_activation_derivatives, layers_delta);

   numerical_Jacobian_terms = nd.calculate_Jacobian(mse, &MeanSquaredError::calculate_error_terms, parameters);

   cout << "Terms Jacobian: " << terms_Jacobian << endl;
   cout << "Numerical: " << numerical_Jacobian_terms << endl;

   assert_true((terms_Jacobian-numerical_Jacobian_terms).calculate_absolute_value() < 1.0e-3, LOG);

   // Test
/*
   nn.set(2, 2, 2);
   nn.randomize_parameters_normal();
   parameters = nn.get_parameters();

   ds.set(2, 2, 2);
   ds.randomize_data_normal();

   terms_Jacobian = mse.calculate_error_terms_Jacobian();
   numerical_Jacobian_terms = nd.calculate_Jacobian(mse, &MeanSquaredError::calculate_error_terms, parameters);

   assert_true((terms_Jacobian-numerical_Jacobian_terms).calculate_absolute_value() < 1.0e-3, LOG);

   // Test

   nn.set(2, 2, 2);
   nn.randomize_parameters_normal();

   ds.set(2, 2, 2);
   ds.randomize_data_normal();
   
   error_gradient = mse.calculate_error_gradient({0, 1});

   error_terms = mse.calculate_error_terms();
   terms_Jacobian = mse.calculate_error_terms_Jacobian();

   assert_true(((terms_Jacobian.calculate_transpose()).dot(error_terms)*2.0 - error_gradient).calculate_absolute_value() < 1.0e-3, LOG);
*/
}


void MeanSquaredErrorTest::test_calculate_Hessian()
{
    message += "test_calculate_Hessian\n";

    NumericalDifferentiation nd;
    DataSet ds;
    NeuralNetwork nn;
    MeanSquaredError mse(&nn, &ds);

    Vector<double> parameters;
    Matrix<double> Hessian;
    Matrix<double> numerical_Hessian;

    Vector<size_t> architecture;

    // Test activation linear

//    {
//        nn.set();
//        nn.construct_multilayer_perceptron();

//        ds.set();

//        Hessian = mse.calculate_second_order_loss().Hessian;

//        assert_true(Hessian.get_rows_number() == 0, LOG);
//        assert_true(Hessian.get_columns_number() == 0, LOG);
//    }

    // Test activation linear

    {
        ds.set(20, 1, 1);
        ds.randomize_data_normal();
        ds.get_instances_pointer()->set_training();

        nn.set(1,1);
        nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, PerceptronLayer::Linear);

        nn.randomize_parameters_normal();
        parameters = nn.get_parameters();
/*
        LossIndex::SecondOrderErrorTerms results = mse.calculate_terms_second_order_loss();

        Hessian = results.Hessian_approximation;
        nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
        numerical_Hessian = nd.calculate_Hessian(mse, &MeanSquaredError::calculate_training_error, parameters);

        assert_true((Hessian - numerical_Hessian).calculate_absolute_value() < 1.0e-3, LOG);
*/
    }
/*
    // Test activation logistic

    {
        ds.set(1, 2, 2);
        ds.randomize_data_normal();

        nn.set(2,2);

        nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Logistic);

        nn.randomize_parameters_normal();
        parameters = nn.get_parameters();

        Hessian = sse.calculate_Hessian();
        numerical_Hessian = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss, parameters);

        assert_true((Hessian - numerical_Hessian).calculate_absolute_value() < 1.0e-3, LOG);
    }

    // Test activation hyperbolic tangent

    {
        ds.set(3, 2, 4);
        ds.randomize_data_normal();

        nn.set(2,4);

        nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::HyperbolicTangent);

        nn.randomize_parameters_normal();
        parameters = nn.get_parameters();

        Hessian = sse.calculate_Hessian();
        numerical_Hessian = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss, parameters);

        assert_true((Hessian - numerical_Hessian).calculate_absolute_value() < 1.0e-3, LOG);
    }

    // Test activation linear

    {
        ds.set(1,2,5);
        ds.randomize_data_normal();

        nn.set(2, 5);

        nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Linear);

        nn.randomize_parameters_normal();
        parameters = nn.get_parameters();

        Hessian = sse.calculate_Hessian();
        numerical_Hessian = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss, parameters);

        assert_true((Hessian - numerical_Hessian).calculate_absolute_value() < 1.0e-3, LOG);
    }

    // Test activation logistic

    {
        ds.set(1,2,4);
        ds.randomize_data_normal();

        nn.set(2,4);

        nn.randomize_parameters_normal();

        nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Logistic);

        parameters = nn.get_parameters();

        Hessian.clear();
        numerical_Hessian.clear();

        Hessian = sse.calculate_Hessian();
        numerical_Hessian = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss, parameters);

        assert_true((Hessian - numerical_Hessian).calculate_absolute_value() < 1.0e-3, LOG);
    }

    // Test activation logistic

    {
        ds.set(1,1,1);
        ds.randomize_data_normal();

        nn.set(1,1);

        nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Logistic);

        parameters = nn.get_parameters();

        Hessian.clear();
        numerical_Hessian.clear();

        Hessian = sse.calculate_Hessian();
        numerical_Hessian = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss, parameters);

        assert_true((Hessian - numerical_Hessian).calculate_absolute_value() < 1.0e-3, LOG);
    }


    // Test activation hyperbolic tangent

    {
        ds.set(1,1,1);
        ds.randomize_data_normal();

        nn.set(1,1);

        nn.randomize_parameters_normal();

        nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::HyperbolicTangent);

        parameters = nn.get_parameters();

        Hessian.clear();
        numerical_Hessian.clear();

        Hessian = sse.calculate_Hessian();
        numerical_Hessian = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss, parameters);

        assert_true((Hessian - numerical_Hessian).calculate_absolute_value() < 1.0e-3, LOG);
    }

    // Test activation hyperbolic tangent

    {
        ds.set(1,5,5);
        ds.randomize_data_normal();

        nn.set(5,5);

        nn.randomize_parameters_normal();

        nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::HyperbolicTangent);

        parameters = nn.get_parameters();

        Hessian.clear();
        numerical_Hessian.clear();

        Hessian = sse.calculate_Hessian();
        numerical_Hessian = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss, parameters);

        assert_true((Hessian - numerical_Hessian).calculate_absolute_value() < 1.0e-3, LOG);
    }


    // Test activation linear (single hidden layer)

 {
    ds.set(1, 2, 2);
    ds.randomize_data_normal();

    nn.set(2, 2, 2);

    nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Linear);
    nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(1, Perceptron::Linear);

    parameters = nn.get_parameters();

    Hessian = sse.calculate_single_hidden_layer_Hessian();
    numerical_Hessian = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss, parameters);

    assert_true((Hessian - numerical_Hessian).calculate_absolute_value() < 1.0e-3, LOG);
 }

    // Test activation linear (single hidden layer)

 {
    ds.set(1, 1, 2);
    ds.randomize_data_normal();

    nn.set(1, 2, 2);

    nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Linear);
    nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(1, Perceptron::Linear);

    parameters = nn.get_parameters();

    Hessian = sse.calculate_single_hidden_layer_Hessian();

    numerical_Hessian = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss, parameters);

    assert_true((Hessian - numerical_Hessian).calculate_absolute_value() < 1.0e-3, LOG);
 }
 */
/*    // Test activation logistic (single hidden layer)
 {
    ds.set(1,1,1);
    ds.randomize_data_normal();
    //ds.initialize_data(1.0);

    nn.set(1,1,1);
 //   nn.initialize_parameters(1.0);

    nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Logistic);
    nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(1, Perceptron::Logistic);

    parameters = nn.get_parameters();

    Hessian = sse.calculate_single_hidden_layer_Hessian();
    Matrix<double> complete_Hessian = sse.calculate_Hessian();

    cout << "Single hidden layer Hessian: \n" << Hessian << endl;
    cout << "Complete Hessian: \n" << complete_Hessian << endl;

    numerical_Hessian = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss, parameters);

    assert_true((Hessian - numerical_Hessian).calculate_absolute_value() < 1.0e-3, LOG);
    assert_true((Hessian - complete_Hessian).calculate_absolute_value() < 1.0e-3, LOG);
 }

    // Test
 {
    ds.set(1,1,1);
    //ds.randomize_data_normal();
    ds.initialize_data(1.0);

    nn.set(1,1,1);

    architecture.set(4);

    architecture[0] = 1;
    architecture[1] = 1;
    architecture[2] = 1;
    architecture[3] = 1;

    Vector< Matrix<double> > weights(3);

    for(size_t i = 0; i < 3; i++)
    {
        Matrix<double> layer_weights(1,1,(double)i+1.0);
        weights[i] = layer_weights;
    }

    nn.set(architecture);
    nn.get_multilayer_perceptron_pointer()->initialize_biases(0.0);
    nn.get_multilayer_perceptron_pointer()->set_layers_synaptic_weights(weights);

    nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Linear);
    nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(1, Perceptron::Linear);
    nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(2, Perceptron::Linear);

    parameters = nn.get_parameters();

    Hessian = sse.calculate_Hessian();

    numerical_Hessian = nd.calculate_Hessian(sse, &SumSquaredError::calculate_error, parameters);

    cout << "Hessian: \n" << Hessian << endl;
    cout << "Numerical Hessian: \n" << numerical_Hessian << endl;

 //   Vector<size_t> columns(4,1,5);
 //   Vector<size_t> rows(0,1,1);

 //   assert_true((Hessian.get_submatrix(rows,columns)-numerical_Hessian.get_submatrix(rows,columns)).calculate_absolute_value() < 1.0e-3, LOG);

    assert_true((Hessian - numerical_Hessian).calculate_absolute_value() < 1.0e-3, LOG);
*/
}


void MeanSquaredErrorTest::test_to_XML()
{
   message += "test_to_XML\n";
}


void MeanSquaredErrorTest::test_from_XML()
{
   message += "test_from_XML\n";
}


void MeanSquaredErrorTest::run_test_case()
{
   message += "Running mean squared error test case...\n";

   // Constructor and destructor methods

//   test_constructor();
//   test_destructor();

   // Get methods

   // Set methods

   // Error methods

//   test_calculate_error();
//   test_calculate_selection_error();

//   test_calculate_error_gradient();

   // Error terms methods

//   test_calculate_error_terms();
//   test_calculate_error_terms_Jacobian();

   // Loss Hessian methods

   test_calculate_Hessian();

   // Serialization methods

//   test_to_XML();
//   test_from_XML();

   message += "End of mean squared error test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques, SL.
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
