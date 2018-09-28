/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S U M   S Q U A R E D   E R R O R   T E S T   C L A S S                                                    */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/


// Unit testing includes

#include "sum_squared_error_test.h"

using namespace OpenNN;


// GENERAL CONSTRUCTOR

SumSquaredErrorTest::SumSquaredErrorTest() : UnitTesting() 
{
}


// DESTRUCTOR

SumSquaredErrorTest::~SumSquaredErrorTest() 
{
}


// METHODS

void SumSquaredErrorTest::test_constructor()
{
   message += "test_constructor\n";

   // Default

   SumSquaredError sse1;

   assert_true(sse1.has_neural_network() == false, LOG);
   assert_true(sse1.has_data_set() == false, LOG);

   // Neural network

   NeuralNetwork nn2;
   SumSquaredError sse2(&nn2);

   assert_true(sse2.has_neural_network() == true, LOG);
   assert_true(sse2.has_data_set() == false, LOG);

   // Neural network and data set

   NeuralNetwork nn3;
   DataSet ds3;
   SumSquaredError sse3(&nn3, &ds3);

   assert_true(sse3.has_neural_network() == true, LOG);
   assert_true(sse3.has_data_set() == true, LOG);
}


void SumSquaredErrorTest::test_destructor()
{
   message += "test_destructor\n";
}


void SumSquaredErrorTest::test_calculate_loss()   
{
   message += "test_calculate_loss\n";

   NeuralNetwork nn;
   Vector<double> parameters;

   DataSet ds;
   Matrix<double> data;
   MissingValues* missing_values_pointer = ds.get_missing_values_pointer();

   SumSquaredError sse(&nn, &ds);

   double loss;

   // Test

   nn.set(1, 1);
   nn.initialize_parameters(0.0);

   ds.set(1, 1, 1);
   ds.initialize_data(0.0);

   loss = sse.calculate_error();

   assert_true(loss == 0.0, LOG);

   // Test

   nn.set(1, 1, 1);
   nn.initialize_parameters(0.0);

   ds.set(1, 1, 1);
   ds.initialize_data(1.0);

   loss = sse.calculate_error();

   assert_true(loss == 1.0, LOG);

   // Test

   nn.set(1, 1);
   nn.randomize_parameters_normal();

   parameters = nn.arrange_parameters();

   ds.set(1, 1, 1);
   ds.randomize_data_normal();

   assert_true(sse.calculate_error() == sse.calculate_error(parameters), LOG);

   // Test

   nn.set(1, 1);
   nn.randomize_parameters_normal();

   parameters = nn.arrange_parameters();

   ds.set(1, 1, 1);
   ds.randomize_data_normal();

   assert_true(sse.calculate_error() != sse.calculate_error(parameters*2.0), LOG);

   // Test

   nn.set(1, 1);
   nn.randomize_parameters_normal();

   parameters = nn.arrange_parameters();

   ds.set(1, 1, 1);
   ds.randomize_data_normal();

   missing_values_pointer->append(0, 0);

//   assert_true(sse.calculate_loss() == 0.0, LOG);
}


void SumSquaredErrorTest::test_calculate_gradient()
{
   message += "test_calculate_gradient\n";

   NumericalDifferentiation nd;
   DataSet ds;
   NeuralNetwork nn;
   SumSquaredError sse(&nn, &ds);

   Vector<size_t> architecture;

   Vector<double> parameters;
   Vector<double> gradient;
   Vector<double> numerical_gradient;
   Vector<double> error;

   // Test 

   nn.set(1, 1, 1);
   nn.initialize_parameters(0.0);

   ds.set(1, 1, 1);
   ds.initialize_data(0.0);

   gradient = sse.calculate_gradient();

   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test 

   nn.set(3, 4, 2);
   nn.initialize_parameters(0.0);

   ds.set(3, 2, 5);
   sse.set(&nn, &ds);
   ds.initialize_data(0.0);

   gradient.clear();

   gradient = sse.calculate_gradient();

   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test

   architecture.set(3);
   architecture[0] = 5;
   architecture[1] = 1;
   architecture[2] = 2;

   nn.set(architecture);
   nn.initialize_parameters(0.0);

   ds.set(5, 5, 2);
   sse.set(&nn, &ds);
   ds.initialize_data(0.0);

   gradient.clear();

   gradient = sse.calculate_gradient();

   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test

   nn.set(1, 1, 1);

   nn.initialize_parameters(0.0);

   ds.set(1, 1, 1);

   ds.initialize_data(0.0);

   gradient.clear();

   gradient = sse.calculate_gradient();

   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test 

   nn.set(3, 4, 2);
   nn.initialize_parameters(0.0);

   ds.set(3, 3, 2);
   sse.set(&nn, &ds);
   ds.initialize_data(0.0);

   gradient.clear();

   gradient = sse.calculate_gradient();

   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test

   nn.set(2, 3, 4);
   nn.initialize_parameters(0.0);

   ds.set(2, 4, 5);
   sse.set(&nn, &ds);
   ds.initialize_data(0.0);

   gradient.clear();

   gradient = sse.calculate_gradient();

   assert_true(gradient.size() == nn.count_parameters_number(), LOG);
   assert_true(gradient == 0.0, LOG);

   // Test

   for(unsigned i = 0; i < 100; i++)
   {

   ds.initialize_data(1.0);

   nn.randomize_parameters_normal();
   parameters = nn.arrange_parameters();

   gradient.clear();

   gradient = sse.calculate_gradient();
   numerical_gradient = nd.calculate_gradient(sse, &SumSquaredError::calculate_error, parameters);
   error = (gradient - numerical_gradient).calculate_absolute_value();

   assert_true(error < 1.0e-3, LOG);
   }

   // Test

   nn.set(1, 1, 1);
   nn.initialize_parameters(1.0);
   parameters = nn.arrange_parameters();

   ds.set(1, 1, 1);

   ds.initialize_data(1.0);

   gradient.clear();

   gradient = sse.calculate_gradient();
   numerical_gradient = nd.calculate_gradient(sse, &SumSquaredError::calculate_error, parameters);
   assert_true((gradient - numerical_gradient).calculate_absolute_value() < 1.0e-3, LOG);

   // Test

   architecture.set(1000, 1);

   nn.set(architecture);
   nn.randomize_parameters_normal();

   ds.set(10, 1, 1);
   ds.randomize_data_normal();

   sse.set(&nn, &ds);

   gradient.clear();

   gradient = sse.calculate_gradient();
}


// @todo

void SumSquaredErrorTest::test_calculate_Hessian()
{
   message += "test_calculate_Hessian\n";

   NumericalDifferentiation nd;
   DataSet ds;
   NeuralNetwork nn;
   SumSquaredError sse(&nn, &ds);

   Vector<double> parameters;
   Matrix<double> Hessian;
   Matrix<double> numerical_Hessian;

   Vector<size_t> architecture;

   // Test activation linear
/*
   {
       nn.set();
       nn.construct_multilayer_perceptron();

       ds.set();

       Hessian = sse.calculate_Hessian();

       assert_true(Hessian.get_rows_number() == 0, LOG);
       assert_true(Hessian.get_columns_number() == 0, LOG);
   }

   // Test activation linear

   {
       ds.set(1, 2, 2);
       ds.randomize_data_normal();

       nn.set(2,2);

       nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Linear);

       nn.randomize_parameters_normal();
       parameters = nn.arrange_parameters();

       Hessian = sse.calculate_Hessian();
       numerical_Hessian = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss, parameters);

       assert_true((Hessian - numerical_Hessian).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test activation logistic

   {
       ds.set(1, 2, 2);
       ds.randomize_data_normal();

       nn.set(2,2);

       nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Logistic);

       nn.randomize_parameters_normal();
       parameters = nn.arrange_parameters();

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
       parameters = nn.arrange_parameters();

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
       parameters = nn.arrange_parameters();

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

       parameters = nn.arrange_parameters();

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

       parameters = nn.arrange_parameters();

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

       parameters = nn.arrange_parameters();

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

       parameters = nn.arrange_parameters();

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

   parameters = nn.arrange_parameters();

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

   parameters = nn.arrange_parameters();

   Hessian = sse.calculate_single_hidden_layer_Hessian();

   numerical_Hessian = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss, parameters);

   assert_true((Hessian - numerical_Hessian).calculate_absolute_value() < 1.0e-3, LOG);
}
*/
  /* // Test activation logistic (single hidden layer)
{
   ds.set(1,1,1);
   ds.randomize_data_normal();
   //ds.initialize_data(1.0);

   nn.set(1,1,1);
//   nn.initialize_parameters(1.0);

   nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Logistic);
   nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(1, Perceptron::Logistic);

   parameters = nn.arrange_parameters();

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

   parameters = nn.arrange_parameters();

   Hessian = sse.calculate_Hessian();

   numerical_Hessian = nd.calculate_Hessian(sse, &SumSquaredError::calculate_error, parameters);

   cout << "Hessian: \n" << Hessian << endl;
   cout << "Numerical Hessian: \n" << numerical_Hessian << endl;

//   Vector<size_t> columns(4,1,5);
//   Vector<size_t> rows(0,1,1);

//   assert_true((Hessian.arrange_submatrix(rows,columns)-numerical_Hessian.arrange_submatrix(rows,columns)).calculate_absolute_value() < 1.0e-3, LOG);

   assert_true((Hessian - numerical_Hessian).calculate_absolute_value() < 1.0e-3, LOG);
}

 /*  // Test activation hyperbolic tangent (single hidden layer)
{
   ds.set(1, 2, 2);
   ds.randomize_data_normal();

   nn.set(2,2,2);

   nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::HyperbolicTangent);
   nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(1, Perceptron::HyperbolicTangent);

   parameters = nn.arrange_parameters();

   Hessian = sse.calculate_single_hidden_layer_Hessian();

   numerical_Hessian = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss, parameters);

   assert_true((Hessian - numerical_Hessian).calculate_absolute_value() < 1.0e-3, LOG);
}

   // Test
/*
{
       ds.set(1,1,1);
       ds.randomize_data_normal();
       //ds.initialize_data(1.0);

       Vector<size_t> architecture(4,1);

       Vector<size_t> rows1(0,1,1);
       Vector<size_t> columns1(0,1,1);
       Vector<size_t> rows2(0,1,1);
       Vector<size_t> columns2(2,1,3);
       Vector<size_t> rows3(0,1,1);
       Vector<size_t> columns3(4,1,5);
       Vector<size_t> rows4(2,1,3);
       Vector<size_t> columns4(2,1,3);
       Vector<size_t> rows5(2,1,3);
       Vector<size_t> columns5(4,1,5);
       Vector<size_t> rows6(4,1,5);
       Vector<size_t> columns6(4,1,5);


       Vector<Matrix<double>> weights(3);

       for(size_t i = 0; i < 3; i++)
       {
           Matrix<double> layer_weights(1,1,(double)i);
           weights[i] = layer_weights;
       }

       nn.set(architecture);
       nn.randomize_parameters_uniform();
       nn.get_multilayer_perceptron_pointer()->initialize_biases(1.0);
       nn.get_multilayer_perceptron_pointer()->set_layers_synaptic_weights(weights);

       nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(0, Perceptron::Linear);
       nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(1, Perceptron::Linear);
       nn.get_multilayer_perceptron_pointer()->set_layer_activation_function(2, Perceptron::Linear);

       parameters = nn.arrange_parameters();

       Hessian = sse.calculate_Hessian();
       numerical_Hessian = nd.calculate_Hessian(sse, &SumSquaredError::calculate_loss, parameters);

       assert_true((Hessian.arrange_submatrix(rows1, columns1) - numerical_Hessian.arrange_submatrix(rows1, columns1)).calculate_absolute_value() < 1.0e-3, LOG);
       assert_true((Hessian.arrange_submatrix(rows2, columns2) - numerical_Hessian.arrange_submatrix(rows2, columns2)).calculate_absolute_value() < 1.0e-3, LOG);
       assert_true((Hessian.arrange_submatrix(rows3, columns3) - numerical_Hessian.arrange_submatrix(rows3, columns3)).calculate_absolute_value() < 1.0e-3, LOG);
       assert_true((Hessian.arrange_submatrix(rows4, columns4) - numerical_Hessian.arrange_submatrix(rows4, columns4)).calculate_absolute_value() < 1.0e-3, LOG);
       assert_true((Hessian.arrange_submatrix(rows5, columns5) - numerical_Hessian.arrange_submatrix(rows5, columns5)).calculate_absolute_value() < 1.0e-3, LOG);
       assert_true((Hessian.arrange_submatrix(rows6, columns6) - numerical_Hessian.arrange_submatrix(rows6, columns6)).calculate_absolute_value() < 1.0e-3, LOG);}
}
*/
}


void SumSquaredErrorTest::test_calculate_terms()
{
   message += "test_calculate_terms\n";
}


void SumSquaredErrorTest::test_calculate_terms_Jacobian()
{   
   message += "test_calculate_terms_Jacobian\n";

   NumericalDifferentiation nd;

   NeuralNetwork nn;
   Vector<size_t> architecture;
   Vector<double> parameters;

   DataSet ds;

   SumSquaredError sse(&nn, &ds);

   Vector<double> gradient;

   Vector<double> terms;
   Matrix<double> terms_Jacobian;
   Matrix<double> numerical_Jacobian_terms;

   // Test

   nn.set(1, 1);

   nn.initialize_parameters(0.0);

   ds.set(1, 1, 1);

   ds.initialize_data(0.0);

   terms_Jacobian = sse.calculate_terms_Jacobian();

   assert_true(terms_Jacobian.get_rows_number() == ds.get_instances().get_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == nn.count_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);

   // Test 

   nn.set(3, 4, 2);
   nn.initialize_parameters(0.0);

   ds.set(3, 2, 5);
   sse.set(&nn, &ds);
   ds.initialize_data(0.0);

   terms_Jacobian = sse.calculate_terms_Jacobian();

   assert_true(terms_Jacobian.get_rows_number() == ds.get_instances().count_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == nn.count_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

   architecture.set(3);
   architecture[0] = 5;
   architecture[1] = 1;
   architecture[2] = 2;

   nn.set(architecture);
   nn.initialize_parameters(0.0);

   ds.set(5, 2, 3);
   sse.set(&nn, &ds);
   ds.initialize_data(0.0);

   terms_Jacobian = sse.calculate_terms_Jacobian();

   assert_true(terms_Jacobian.get_rows_number() == ds.get_instances().count_training_instances_number(), LOG);
   assert_true(terms_Jacobian.get_columns_number() == nn.count_parameters_number(), LOG);
   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

   nn.set(1, 1, 1);
   nn.randomize_parameters_normal();
   parameters = nn.arrange_parameters();

   ds.set(1, 1, 1);
   ds.randomize_data_normal();

   terms_Jacobian = sse.calculate_terms_Jacobian();
   numerical_Jacobian_terms = nd.calculate_Jacobian(sse, &SumSquaredError::calculate_terms, parameters);

   assert_true((terms_Jacobian-numerical_Jacobian_terms).calculate_absolute_value() < 1.0e-3, LOG);

   // Test

   nn.set(2, 2, 2);
   nn.randomize_parameters_normal();
   parameters = nn.arrange_parameters();

   ds.set(2, 2, 2);
   ds.randomize_data_normal();

   terms_Jacobian = sse.calculate_terms_Jacobian();
   numerical_Jacobian_terms = nd.calculate_Jacobian(sse, &SumSquaredError::calculate_terms, parameters);

   assert_true((terms_Jacobian-numerical_Jacobian_terms).calculate_absolute_value() < 1.0e-3, LOG);

   // Test

   nn.set(2, 2, 2);
   nn.randomize_parameters_normal();

   ds.set(2, 2, 2);
   ds.randomize_data_normal();
   
   gradient = sse.calculate_gradient();

   terms = sse.calculate_terms();
   terms_Jacobian = sse.calculate_terms_Jacobian();

   assert_true(((terms_Jacobian.calculate_transpose()).dot(terms)*2.0 - gradient).calculate_absolute_value() < 1.0e-3, LOG);
}


void SumSquaredErrorTest::test_calculate_selection_loss()
{
   message += "test_calculate_selection_loss\n";

   NeuralNetwork nn;
   DataSet ds;
   SumSquaredError sse(&nn, &ds);

   double selection_objective;

   // Test

   nn.set();

   nn.construct_multilayer_perceptron();

   ds.set();

   selection_objective = sse.calculate_selection_error();
   
   assert_true(selection_objective == 0.0, LOG);
}


void SumSquaredErrorTest::test_calculate_squared_errors()
{
   message += "test_calculate_squared_errors\n";

   NeuralNetwork nn;

   DataSet ds;

   SumSquaredError sse(&nn, &ds);

   Vector<double> squared_errors;

   double objective;

   // Test 

   nn.set(1,1,1);

   nn.initialize_parameters(0.0);

   ds.set(1,1,1);

   ds.initialize_data(0.0);

   squared_errors = sse.calculate_squared_errors();

   assert_true(squared_errors.size() == 1, LOG);
   assert_true(squared_errors == 0.0, LOG);   

   // Test

   nn.set(2,2,2);

   nn.randomize_parameters_normal();

   ds.set(2,2,2);

   ds.randomize_data_normal();

   squared_errors = sse.calculate_squared_errors();

   objective = sse.calculate_error();

   assert_true(fabs(squared_errors.calculate_sum() - objective) < 1.0e-12, LOG);

}


void SumSquaredErrorTest::test_to_XML()   
{
    message += "test_to_XML\n";

    SumSquaredError sse;

    tinyxml2::XMLDocument* document;

    // Test

    document = sse.to_XML();

    assert_true(document != NULL, LOG);

    delete document;
}


void SumSquaredErrorTest::test_from_XML()
{
    message += "test_from_XML\n";

    SumSquaredError sse1;
    SumSquaredError sse2;

   tinyxml2::XMLDocument* document;

   // Test

   sse1.set_display(false);

   document = sse1.to_XML();

   sse2.from_XML(*document);

   delete document;

   assert_true(sse2.get_display() == false, LOG);
}


void SumSquaredErrorTest::run_test_case()
{
   message += "Running sum squared error test case...\n";

   // Constructor and destructor methods

//   test_constructor();
//   test_destructor();

   // Get methods

   // Set methods

   // Objective methods

   test_calculate_loss();
   test_calculate_selection_loss();

   test_calculate_gradient();

//   test_calculate_Hessian();

   // Objective terms methods

   test_calculate_terms();

//   test_calculate_terms_Jacobian();

   // Serialization methods

//   test_to_XML();
//   test_from_XML();

   message += "End of sum squared error test case.\n";
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
