/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   E R R O R   T E R M   T E S T   C L A S S                                                                  */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "mock_error_term.h"
#include "error_term_test.h"

using namespace OpenNN;


// GENERAL CONSTRUCTOR

ErrorTermTest::ErrorTermTest() : UnitTesting() 
{
}


// DESTRUCTOR

ErrorTermTest::~ErrorTermTest()
{
}


// METHODS

void ErrorTermTest::test_constructor()
{
   message += "test_constructor\n";
}


void ErrorTermTest::test_destructor()
{
   message += "test_destructor\n";
}


void ErrorTermTest::test_assingment_operator()
{
   message += "test_assingment_operator\n";

   MockErrorTerm mpt1;
   MockErrorTerm mpt2;

   mpt2 = mpt1;

}


void ErrorTermTest::test_equal_to_operator()
{
   message += "test_equal_to_operator\n";

   MockErrorTerm mpt1;
   MockErrorTerm mpt2;

   assert_true(mpt2 == mpt1, LOG);
}


void ErrorTermTest::test_get_neural_network_pointer()
{
   message += "test_get_neural_network_pointer\n";

   MockErrorTerm of;
   NeuralNetwork nn;

   // Test

   of.set_neural_network_pointer(&nn);
   assert_true(of.get_neural_network_pointer() != NULL,	LOG);
}


void ErrorTermTest::test_get_numerical_differentiation_pointer()
{
   message += "test_get_numerical_differentiation_pointer\n";

   MockErrorTerm mpt;
  
   // Test

   mpt.construct_numerical_differentiation();

   assert_true(mpt.get_numerical_differentiation_pointer() != NULL,	LOG);
}


void ErrorTermTest::test_get_display()
{
   message += "test_get_display\n";

   MockErrorTerm mpt;

   // Test

   mpt.set_display(true);
   assert_true(mpt.get_display() == true, LOG);

   mpt.set_display(false);
   assert_true(mpt.get_display() == false, LOG);

}


void ErrorTermTest::test_set_neural_network_pointer()
{
   message += "test_set_neural_network_pointer\n";

   MockErrorTerm mpt;
   NeuralNetwork nn;

   // Test

   mpt.set_neural_network_pointer(&nn);
   assert_true(mpt.get_neural_network_pointer() != NULL, LOG);

}


void ErrorTermTest::test_set_numerical_differentiation_pointer()
{
   message += "test_set_numerical_differentiation_pointer\n";

//   MockErrorTerm mpt;

//   NeuralNetwork nn;

}


void ErrorTermTest::test_set_default()
{
   message += "test_set_default\n";

   MockErrorTerm mpt;

   // Test

   mpt.set_default();

}


void ErrorTermTest::test_set_display()
{
   message += "test_set_display\n";
}


void ErrorTermTest::test_calculate_layers_delta()
{
   message += "test_calculate_layers_delta\n";

   NeuralNetwork nn;

   Vector<double> inputs;

   Vector< Vector<double> > layers_activation_derivative; 
   Vector<double> output_gradient;

   MockErrorTerm mpt(&nn);

   Vector< Vector<double> > layers_delta;

   // Test 

   nn.construct_multilayer_perceptron();

   layers_delta = mpt.calculate_layers_delta(layers_activation_derivative, output_gradient);

   assert_true(layers_delta.size() == 0, LOG);

   // Test

   nn.set(1, 1);

   inputs.set(1, 0.0);

   layers_activation_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_derivative(inputs); 

   output_gradient.set(1, 0.0);

   layers_delta = mpt.calculate_layers_delta(layers_activation_derivative, output_gradient);

   assert_true(layers_delta.size() == 1, LOG);
   assert_true(layers_delta[0].size() == 1, LOG);
   assert_true(layers_delta[0] == 0.0, LOG);

   // Test

   nn.set(1, 1, 1);

   inputs.set(1, 0.0);

   layers_activation_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_derivative(inputs); 

   output_gradient.set(1, 0.0);

   layers_delta = mpt.calculate_layers_delta(layers_activation_derivative, output_gradient);

   assert_true(layers_delta.size() == 2, LOG);
   assert_true(layers_delta[0].size() == 1, LOG);
   assert_true(layers_delta[0] == 0.0, LOG);
   assert_true(layers_delta[1].size() == 1, LOG);
   assert_true(layers_delta[1] == 0.0, LOG);

   // Test

   nn.set(4, 3, 5);

   inputs.set(4, 0.0);

   layers_activation_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_derivative(inputs); 

   output_gradient.set(5, 0.0);

   layers_delta = mpt.calculate_layers_delta(layers_activation_derivative, output_gradient);

   assert_true(layers_delta.size() == 2, LOG);
   assert_true(layers_delta[0].size() == 3, LOG);
   assert_true(layers_delta[0] == 0.0, LOG);
   assert_true(layers_delta[1].size() == 5, LOG);
   assert_true(layers_delta[1] == 0.0, LOG);

}


// @todo

void ErrorTermTest::test_calculate_interlayers_Delta()
{
   message += "test_calculate_interlayers_Delta\n";
/*
   NeuralNetwork nn;
   DataSet ds;

   Vector<size_t> layers_size;

   Vector<double> inputs;
   Vector<double> targets;
   Vector<double> outputs;

   Vector< Vector<double> > layers_activation_derivative;
   Vector< Vector<double> > layers_activation_second_derivative;

   Matrix< Matrix<double> > interlayers_combination_combination_Jacobian;

   Vector<double> output_gradient;
   Matrix<double> output_Hessian;

   Vector< Vector<double> > layers_delta;

   Vector<double> parameters;
   Vector<double> combinations;

   MockErrorTerm mpt(&nn, &ds);

   Matrix< Matrix<double> > interlayers_Delta;
   Matrix<double> output_interlayers_Delta;
   Matrix<double> numerical_interlayers_Delta;

   NumericalDifferentiation nd;

   SumSquaredError sse(&nn, &ds);

   // Test 

   nn.construct_multilayer_perceptron();

   interlayers_combination_combination_Jacobian = nn.get_multilayer_perceptron_pointer()->calculate_interlayers_combination_combination_Jacobian(inputs);

   layers_delta = mpt.calculate_layers_delta(layers_activation_derivative, output_gradient);
   interlayers_Delta = mpt.calculate_interlayers_Delta(layers_activation_derivative, layers_activation_second_derivative, interlayers_combination_combination_Jacobian, output_gradient, output_Hessian, layers_delta);

   assert_true(interlayers_Delta.get_rows_number() == 0, LOG);
   assert_true(interlayers_Delta.get_columns_number() == 0, LOG);

   // Test

   nn.set(1, 1);

   inputs.set(1, 0.0);

   layers_activation_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_derivative(inputs); 
   layers_activation_second_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_second_derivative(inputs); 

   interlayers_combination_combination_Jacobian = nn.get_multilayer_perceptron_pointer()->calculate_interlayers_combination_combination_Jacobian(inputs);

   output_gradient.set(1,0.0);
   output_Hessian.set(1,1,0.0);

   layers_delta = mpt.calculate_layers_delta(layers_activation_derivative, output_gradient);
   interlayers_Delta = mpt.calculate_interlayers_Delta(layers_activation_derivative, layers_activation_second_derivative, interlayers_combination_combination_Jacobian, output_gradient, output_Hessian, layers_delta);

   assert_true(interlayers_Delta.get_rows_number() == 1, LOG);
   assert_true(interlayers_Delta.get_columns_number() == 1, LOG);
   assert_true(interlayers_Delta(0,0).get_rows_number() == 1, LOG);
   assert_true(interlayers_Delta(0,0).get_columns_number() == 1, LOG);
   assert_true(interlayers_Delta(0,0) == 0.0, LOG);

   // Test

   nn.set(1, 1);

   nn.initialize_parameters(0.0);

   inputs.set(1, 0.0);

   outputs = nn.calculate_outputs(inputs);

   layers_activation_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_derivative(inputs); 
   layers_activation_second_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_second_derivative(inputs); 

   interlayers_combination_combination_Jacobian = nn.get_multilayer_perceptron_pointer()->calculate_interlayers_combination_combination_Jacobian(inputs);

   output_gradient.set(1, 2.0*outputs[0]);
   
   output_Hessian.set(1, 1);
   output_Hessian.randomize_normal();

   layers_delta = mpt.calculate_layers_delta(layers_activation_derivative, output_gradient);
   interlayers_Delta = mpt.calculate_interlayers_Delta(layers_activation_derivative, layers_activation_second_derivative, interlayers_combination_combination_Jacobian, output_gradient, output_Hessian, layers_delta);

   assert_true(interlayers_Delta.get_rows_number() == 1, LOG);
   assert_true(interlayers_Delta.get_columns_number() == 1, LOG);
   assert_true(interlayers_Delta(0,0).get_rows_number() == 1, LOG);
   assert_true(interlayers_Delta(0,0).get_columns_number() == 1, LOG);
   assert_true(interlayers_Delta(0,0) == output_Hessian(0,0), LOG);

   // Test

   layers_size.set(4, 1);

   nn.set(layers_size);

   inputs.set(1, 1.0);

   outputs = nn.calculate_outputs(inputs);
   layers_activation_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_derivative(inputs); 
   layers_activation_second_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_second_derivative(inputs); 

   interlayers_combination_combination_Jacobian = nn.get_multilayer_perceptron_pointer()->calculate_interlayers_combination_combination_Jacobian(inputs);

   output_gradient.set(1, 2.0*outputs[0]);
   
   output_Hessian.set(1, 1, 2.0);

   layers_delta = mpt.calculate_layers_delta(layers_activation_derivative, output_gradient);
   interlayers_Delta = mpt.calculate_interlayers_Delta(layers_activation_derivative, layers_activation_second_derivative, interlayers_combination_combination_Jacobian, output_gradient, output_Hessian, layers_delta);

   assert_true(interlayers_Delta.get_rows_number() == 3, LOG);
   assert_true(interlayers_Delta.get_columns_number() == 3, LOG);
   assert_true(interlayers_Delta(0,0).get_rows_number() == 1, LOG);
   assert_true(interlayers_Delta(0,0).get_columns_number() == 1, LOG);

   // Test
/*
   ds.set(1,10,5);
   ds.initialize_data(1.0);

   inputs = ds.arrange_input_data().get_column(0);

   nn.set(10, 5);
   nn.get_multilayer_perceptron_pointer()->initialize_biases(0.0);
   nn.get_multilayer_perceptron_pointer()->initialize_synaptic_weights(1.0);

   parameters = nn.arrange_parameters();

   combinations = nn.get_multilayer_perceptron_pointer()->calculate_layers_combination(inputs)[0];

   nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Linear);

   interlayers_Delta = mpt.calculate_output_interlayers_Delta(layers_activation_derivative,
                                                              layers_activation_second_derivative,
                                                              interlayers_combination_combination_Jacobian,
                                                              output_gradient,
                                                              output_Hessian,
                                                              layers_delta);

   numerical_interlayers_Delta = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss_combinations, 1, combinations);

   assert_true((interlayers_Delta[0] - numerical_interlayers_Delta).calculate_absolute_value() < 1.0e-3, LOG);

   // Test
/*
   ds.set(1,10,5);
   ds.randomize_data_normal();

   inputs = ds.arrange_input_data().get_column(0);

   nn.set(10, 5);
//   nn.get_multilayer_perceptron_pointer()->initialize_biases(0.0);
//   nn.get_multilayer_perceptron_pointer()->initialize_synaptic_weights(1.0);

   parameters = nn.arrange_parameters();

   combinations = nn.get_multilayer_perceptron_pointer()->calculate_layers_combination(inputs)[0];

   nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Linear);

   interlayers_Delta = mpt.calculate_interlayers_Delta();
   numerical_interlayers_Delta = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss_combinations, combinations);

   assert_true((interlayers_Delta[0] - numerical_interlayers_Delta).calculate_absolute_value() < 1.0e-3, LOG);
*/

   // Test
/*
   ds.set(1,1,1);
   ds.randomize_data_normal();

   inputs = ds.arrange_input_data().get_column(0);

   nn.set(1,1);

   parameters = nn.arrange_parameters();

   combinations = nn.get_multilayer_perceptron_pointer()->calculate_layers_combination(inputs)[0];

   nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Linear);

   interlayers_Delta = mpt.calculate_interlayers_Delta();
   numerical_interlayers_Delta = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss_output_combinations, combinations);

   assert_true((interlayers_Delta[0] - numerical_interlayers_Delta).calculate_absolute_value() < 1.0e-3, LOG);
*/
   // Test
/*
   ds.set(1,1,1);
   ds.randomize_data_uniform(0,1);

   inputs = ds.arrange_input_data().get_column(0);

   nn.set(1,1);

   parameters = nn.arrange_parameters();

   combinations = nn.get_multilayer_perceptron_pointer()->calculate_layers_combination(inputs)[0];

   nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Logistic);

   interlayers_Delta = mpt.calculate_interlayers_Delta();
   numerical_interlayers_Delta = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss_output_combinations, combinations);

   assert_true((interlayers_Delta[0] - numerical_interlayers_Delta).calculate_absolute_value() < 1.0e-3, LOG);
*/
   // Test
/*
   ds.set(1,1,1);
   ds.randomize_data_normal();

   inputs = ds.arrange_input_data().get_column(0);

   nn.set(1,1);

   parameters = nn.arrange_parameters();

   combinations = nn.get_multilayer_perceptron_pointer()->calculate_layers_combination(inputs)[0];

   nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::HyperbolicTangent);

   interlayers_Delta = mpt.calculate_interlayers_Delta();
   numerical_interlayers_Delta = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss_output_combinations, combinations);

   assert_true((interlayers_Delta[0] - numerical_interlayers_Delta).calculate_absolute_value() < 1.0e-3, LOG);

    // Test output interlayers

   ds.set(1,1,1);
   ds.randomize_data_normal();

   inputs = ds.arrange_input_data().get_column(0);
   targets = ds.arrange_target_data().get_column(0);

   nn.set(1,1,1);
   nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Logistic);
   nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(1, Perceptron::Logistic);

   combinations = nn.get_multilayer_perceptron_pointer()->calculate_layers_combination(inputs)[1];

   outputs = nn.calculate_outputs(inputs);

   interlayers_combination_combination_Jacobian = nn.get_multilayer_perceptron_pointer()->calculate_interlayers_combination_combination_Jacobian(inputs);

   layers_activation_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_derivative(inputs);
   layers_activation_second_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_second_derivative(inputs);

   output_gradient = sse.calculate_output_gradient(outputs, targets);
   output_Hessian = sse.calculate_output_Hessian(outputs, targets);

   output_interlayers_Delta = mpt.calculate_output_interlayers_Delta(
                                                 layers_activation_derivative[1],
                                                 layers_activation_second_derivative[1],
                                                 output_gradient,
                                                 output_Hessian);

   const size_t index_1 = 1;

   numerical_interlayers_Delta = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss_combination, index_1, combinations);

   assert_true((output_interlayers_Delta - numerical_interlayers_Delta).calculate_absolute_value() < 1.0e-3, LOG);
   */
}


void ErrorTermTest::test_calculate_point_gradient()
{
   message += "test_calculate_point_gradient\n";

   NeuralNetwork nn;

   size_t network_parameters_number;

   Vector<double> inputs;

   Vector< Vector<double> > layers_activation; 
   Vector< Vector<double> > layers_activation_derivative; 
   Vector<double> output_gradient;

   MockErrorTerm mpt(&nn);

   Vector< Vector<double> > layers_delta;

   Vector<double> point_gradient;

   // Test 
   
   nn.set();
   nn.construct_multilayer_perceptron();

   inputs.set();

   layers_activation_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_derivative(inputs); 

   layers_delta = mpt.calculate_layers_delta(layers_activation_derivative, output_gradient);

   point_gradient = mpt.calculate_point_gradient(inputs, layers_activation, layers_delta);

   assert_true(point_gradient.size() == 0, LOG);

   // Test

   nn.set(1, 1);
   
   network_parameters_number = nn.count_parameters_number();

   inputs.set(1, 0.0);

   layers_activation = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation(inputs); 
   layers_activation_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_derivative(inputs); 

   output_gradient.set(1, 0.0);

   layers_delta = mpt.calculate_layers_delta(layers_activation_derivative, output_gradient);

   point_gradient = mpt.calculate_point_gradient(inputs, layers_activation, layers_delta);

   assert_true(point_gradient.size() == network_parameters_number, LOG);
   assert_true(point_gradient[0] == 0.0, LOG);

   // Test

   nn.set(1, 1, 1);

   network_parameters_number = nn.count_parameters_number();

   inputs.set(1, 0.0);

   layers_activation = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation(inputs); 
   layers_activation_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_derivative(inputs); 

   output_gradient.set(1, 0.0);

   layers_delta = mpt.calculate_layers_delta(layers_activation_derivative, output_gradient);

   point_gradient = mpt.calculate_point_gradient(inputs, layers_activation, layers_delta);

   assert_true(point_gradient.size() == network_parameters_number, LOG);
   assert_true(point_gradient[0] == 0.0, LOG);

}


// @todo

void ErrorTermTest::test_calculate_point_Hessian()
{
   message += "test_calculate_point_Hessian\n";
/*
   NeuralNetwork nn;

//   size_t parameters_number;

   Vector<double> inputs;
   Vector<double> targets;

   Vector< Vector<double> > layers_activation; 
   Vector< Vector<double> > layers_activation_derivative; 
   Vector< Vector<double> > layers_activation_second_derivative; 

   Matrix< Matrix<double> > interlayers_combination_combination_Jacobian;

   Vector< Vector< Vector<double> > > perceptrons_combination_parameters_gradient;

   Vector<double> output_gradient;
   Matrix<double> output_Hessian;

   MockErrorTerm mpt(&nn);

   Vector< Vector<double> > layers_delta;
   Matrix< Matrix<double> > interlayers_Delta;

   Matrix<double> point_Hessian;
   
   // Test 

   nn.set();
   nn.construct_multilayer_perceptron();

   layers_delta = mpt.calculate_layers_delta(layers_activation_derivative, output_gradient);
   interlayers_Delta = mpt.calculate_interlayers_Delta(layers_activation_derivative, layers_activation_second_derivative, interlayers_combination_combination_Jacobian, output_gradient, output_Hessian, layers_delta);

   point_Hessian = mpt.calculate_point_Hessian(layers_activation_derivative, perceptrons_combination_parameters_gradient, interlayers_combination_combination_Jacobian, layers_delta, interlayers_Delta);

   assert_true(point_Hessian.get_rows_number() == 0, LOG);
   assert_true(point_Hessian.get_columns_number() == 0, LOG);

   // Test

   nn.set(1, 1);

   nn.get_multilayer_perceptron_pointer()->initialize_synaptic_weights(1.0);
   nn.get_multilayer_perceptron_pointer()->initialize_biases(1.0);
   nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Linear);

   inputs.set(1, 1.0);
   targets.set(1, 1.0);

   Vector< Vector< Vector<double> > > second_order_forward_propagation(3);
   second_order_forward_propagation = nn.get_multilayer_perceptron_pointer()->calculate_second_order_forward_propagation(inputs);

   Vector< Vector<double> > layers_inputs(nn.get_layers_number());

   layers_inputs[0] = inputs;

   for(size_t j = 1; j < nn.get_layers_number(); j++)
   {
      layers_inputs[j] = layers_activation[j-1];
   }

   perceptrons_combination_parameters_gradient = nn.get_multilayer_perceptron_pointer()->calculate_perceptrons_combination_parameters_gradient(layers_inputs);

   interlayers_combination_combination_Jacobian = nn.get_multilayer_perceptron_pointer()->calculate_interlayers_combination_combination_Jacobian(inputs);
   layers_activation = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation(inputs);
   layers_activation_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_derivative(inputs);
   layers_activation_second_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_second_derivative(inputs);

   output_gradient.set(1, 2*(inputs-targets)[0]);
   output_Hessian.set(1, 1);
   output_Hessian.initialize_diagonal(2.0);
   output_Hessian.set(1, 1);
   output_Hessian.initialize_diagonal(2.0);

   layers_delta = mpt.calculate_layers_delta(layers_activation_derivative, output_gradient);
   interlayers_Delta = mpt.calculate_interlayers_Delta(layers_activation_derivative, layers_activation_second_derivative, interlayers_combination_combination_Jacobian, output_gradient, output_Hessian, layers_delta);

   point_Hessian = mpt.calculate_point_Hessian(layers_activation_derivative, perceptrons_combination_parameters_gradient, interlayers_combination_combination_Jacobian, layers_delta, interlayers_Delta);

//   assert_true(point_Hessian.get_rows_number() == network_parameters_number, LOG);
//   assert_true(point_Hessian.get_columns_number() == network_parameters_number, LOG);
//   assert_true(point_Hessian(0,0) == 0.0, LOG);

   // Test

//   nn.set(1, 1, 1);
 
//   parameters_number = nn.count_parameters_number();

//   inputs.set(1, 0.0);

//   layers_activation = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation(inputs);
//   layers_activation_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_derivative(inputs);
//   layers_activation_second_derivative = nn.get_multilayer_perceptron_pointer()->calculate_layers_activation_second_derivative(inputs);

//   output_gradient.set(1, 0.0);
//   output_Hessian.set(1, 1, 0.0);

//   interlayers_combination_combination_Jacobian = nn.get_multilayer_perceptron_pointer()->calculate_interlayers_combination_combination_Jacobian(inputs);
//   layers_delta = mpt.calculate_layers_delta(layers_activation_derivative, output_gradient);
//   interlayers_Delta = mpt.calculate_interlayers_Delta(layers_activation_derivative, layers_activation_second_derivative, interlayers_combination_combination_Jacobian, output_gradient, output_Hessian, layers_delta);

//   point_Hessian = mpt.calculate_point_Hessian(layers_activation_derivative, perceptrons_combination_parameters_gradient, interlayers_combination_combination_Jacobian, layers_delta, interlayers_Delta);

//   assert_true(point_Hessian.get_rows_number() == network_parameters_number, LOG);
//   assert_true(point_Hessian.get_columns_number() == network_parameters_number, LOG);
//   assert_true(point_Hessian(0,0) == 0.0, LOG);
*/
}


void ErrorTermTest::test_calculate_error()
{
   message += "test_calculate_error\n";

   NeuralNetwork nn;
   MockErrorTerm mpt(&nn);

   // Test

   nn.set(1,1,1);

   nn.initialize_parameters(0.0);

   assert_true(mpt.calculate_error() == 0.0, LOG);

}


void ErrorTermTest::test_calculate_selection_error()
{
   message += "test_calculate_selection_error\n";

   MockErrorTerm mpt;

   double selection_error;

   // Test

   selection_error = mpt.calculate_selection_error();

   assert_true(selection_error == 0.0, LOG);

}


void ErrorTermTest::test_calculate_gradient()
{
   message += "test_calculate_gradient\n";

   NumericalDifferentiation nd;

   NeuralNetwork nn;

   size_t parameters_number;

   Vector<double> parameters;

   MockErrorTerm mpt(&nn);

   Vector<double> gradient;
   Vector<double> numerical_gradient;

   // Test 

   nn.set(1, 1);

   nn.initialize_parameters(0.0);

   gradient = mpt.calculate_gradient();

   parameters_number = nn.count_parameters_number();

   assert_true(gradient.size() == parameters_number, LOG);
   assert_true(gradient == 0.0, LOG);

   // Test

   nn.set(1, 1);

   nn.randomize_parameters_normal();

   parameters = nn.arrange_parameters();

   gradient = mpt.calculate_gradient();

   numerical_gradient = nd.calculate_gradient(mpt, &MockErrorTerm::calculate_error, parameters);

   assert_true((gradient-numerical_gradient).calculate_absolute_value() < 1.0e-3, LOG);

}


void ErrorTermTest::test_calculate_Hessian()
{
   message += "test_calculate_Hessian\n";

   NeuralNetwork nn;
   size_t parameters_number;
   Vector<double> parameters;
   
   MockErrorTerm mpt(&nn);
   Matrix<double> Hessian;

   // Test

   nn.set(1, 1, 1);

   nn.initialize_parameters(0.0);

   parameters_number = nn.count_parameters_number();
   parameters = nn.arrange_parameters();

   Hessian = mpt.calculate_Hessian();

   assert_true(Hessian.get_rows_number() == parameters_number, LOG);
   assert_true(Hessian.get_columns_number() == parameters_number, LOG);

   // Test

   nn.set();

   parameters_number = nn.count_parameters_number();

   Hessian = mpt.calculate_Hessian();

   assert_true(Hessian.get_rows_number() == parameters_number, LOG);
   assert_true(Hessian.get_columns_number() == parameters_number, LOG);

   // Test

   nn.set(1, 1, 1);

   nn.initialize_parameters(0.0);

   parameters_number = nn.count_parameters_number();
   parameters = nn.arrange_parameters();

   Hessian = mpt.calculate_Hessian();

   assert_true(Hessian.get_rows_number() == parameters_number, LOG);
   assert_true(Hessian.get_columns_number() == parameters_number, LOG);

}


void ErrorTermTest::test_calculate_terms()
{
    message += "test_calculate_terms\n";

    NeuralNetwork nn;

    MockErrorTerm mpt(&nn);

    Vector<double> terms;

    // Test

    nn.set(1, 1);

    terms = mpt.calculate_terms();

    assert_true(terms.size() == 2, LOG);

}


void ErrorTermTest::test_calculate_terms_Jacobian()
{
    message += "test_calculate_terms_Jacobian\n";

    NeuralNetwork nn;

    MockErrorTerm mpt(&nn);

    Matrix<double> terms_Jacobian;

    // Test

    nn.set(1, 1);

    terms_Jacobian = mpt.calculate_terms_Jacobian();

    assert_true(terms_Jacobian.get_rows_number() == 2, LOG);
    assert_true(terms_Jacobian.get_columns_number() == 2, LOG);

}


void ErrorTermTest::test_to_XML()
{
   message += "test_to_XML\n";

   MockErrorTerm mpt;

   tinyxml2::XMLDocument* document;

   document = mpt.to_XML();

   assert_true(document != NULL, LOG);

   delete document;
}


void ErrorTermTest::test_write_information()
{
   message += "test_write_information\n";

   MockErrorTerm mpt;

   string information;

   // Test

   information = mpt.write_information();

   assert_true(information.empty(), LOG);
}


void ErrorTermTest::run_test_case()
{
   message += "Running error term test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Operators

   test_assingment_operator();
   test_equal_to_operator();

   // Get methods

   test_get_neural_network_pointer();

   test_get_numerical_differentiation_pointer();

   // Serialization methods

   test_get_display();

   // Set methods

   test_set_neural_network_pointer();

   test_set_numerical_differentiation_pointer();

   test_set_default();

   // Serialization methods

   test_set_display();

   // delta methods

   test_calculate_layers_delta();
   test_calculate_interlayers_Delta();

   // Point objective function methods

   test_calculate_point_gradient();
   test_calculate_point_Hessian();

   // Objective methods

   test_calculate_error();

   test_calculate_selection_error();

   test_calculate_gradient();
   test_calculate_Hessian();

   test_calculate_terms();
   test_calculate_terms_Jacobian();

   // Serialization methods

   test_to_XML();

   test_write_information();

   message += "End of error term test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques, SL.
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
