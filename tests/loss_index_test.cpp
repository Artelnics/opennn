/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   L O S S   I N D E X   T E S T   C L A S S                                                                  */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "loss_index_test.h"

using namespace OpenNN;


// GENERAL CONSTRUCTOR

LossIndexTest::LossIndexTest() : UnitTesting() 
{
}


// DESTRUCTOR

LossIndexTest::~LossIndexTest()
{
}


// METHODS

void LossIndexTest::test_constructor()
{
   message += "test_constructor\n";
/*
   LossIndex pf1;

   assert_true(pf1.has_neural_network() == false, LOG);
   assert_true(pf1.has_data_set() == false, LOG);
*/
}


void LossIndexTest::test_destructor()
{
   message += "test_destructor\n";
}


void LossIndexTest::test_get_neural_network_pointer()
{
   message += "test_get_neural_network_pointer\n";

   SumSquaredError sse;
   NeuralNetwork nn;

   // Test

   sse.set_neural_network_pointer(&nn);
   assert_true(sse.get_neural_network_pointer() != nullptr,	LOG);
}


void LossIndexTest::test_get_mathematical_model_pointer()
{
   message += "test_get_mathematical_model_pointer\n";
}


void LossIndexTest::test_get_data_set_pointer()
{
   message += "test_get_data_set_pointer\n";
}


void LossIndexTest::test_get_user_pointer()
{
   message += "test_get_user_pointer\n";
}


void LossIndexTest::test_get_user_regularization_pointer()
{
   message += "test_get_user_regularization_pointer\n";
}


void LossIndexTest::test_get_user_constraints_pointer()
{
   message += "test_get_user_constraints_pointer\n";
}


void LossIndexTest::test_get_numerical_differentiation_pointer()
{
   message += "test_get_numerical_differentiation_pointer\n";
}


void LossIndexTest::test_get_display()
{
   message += "test_get_display\n";

   SumSquaredError sse;

   // Test

   sse.set_display(true);
   assert_true(sse.get_display() == true, LOG);

   sse.set_display(false);
   assert_true(sse.get_display() == false, LOG);
}


void LossIndexTest::test_set_neural_network_pointer()
{
   message += "test_set_neural_network_pointer\n";

   SumSquaredError sse;
   NeuralNetwork nn;

   // Test

   sse.set_neural_network_pointer(&nn);
   assert_true(sse.get_neural_network_pointer() != nullptr, LOG);
}


void LossIndexTest::test_set_numerical_differentiation()
{
   message += "test_set_numerical_differentiation\n";
}


void LossIndexTest::test_set_default()
{
   message += "test_set_default\n";

   SumSquaredError sse;

   // Test

   sse.set_default();
}


void LossIndexTest::test_set_display()
{
   message += "test_set_display\n";
}


void LossIndexTest::test_calculate_loss()
{
   message += "test_calculate_loss\n";
/*
   DataSet ds;

   Vector<size_t> instances_indices;

   NeuralNetwork nn;

   Vector<double> parameters;

   SumSquaredError sse(&nn);

   Vector<double> direction;
   double rate;

   // Test

   sse.set_regularization_method(LossIndex::L2);

   double neural_parameters_norm_weight = sse.get_regularization_weight();

   nn.set(1, 1);

   nn.initialize_parameters(1.0);

   parameters = nn.get_parameters();

   assert_true(fabs(sse.calculate_loss() - neural_parameters_norm_weight*sqrt(2.0)) < 1.0e-3, LOG);

   assert_true(fabs(sse.calculate_loss() - sse.calculate_loss(parameters)) < 1.0e-3, LOG);

   // Test

   parameters = nn.get_parameters();

   assert_true(fabs(sse.calculate_loss() - sse.calculate_loss(parameters*2.0)) < numeric_limits<double>::min(), LOG);

   // Test

   direction.set(2, -0.5);
   rate = 2.0;

//   assert_true(sse.calculate_loss(instances_indices, direction, rate) == 0.0, LOG);

   // Test

   parameters = nn.get_parameters();

   direction.set(2, -1.5);
   rate = 2.3;

//   assert_true(fabs(sse.calculate_loss(instances_indices, direction, rate) - sse.calculate_loss(instances_indices, parameters + direction*rate)) < numeric_limits<double>::min(), LOG);

   // Test

   ds.set(1, 1, 1);
   ds.randomize_data_normal();

   sse.set_data_set_pointer(&ds);

   nn.set(1, 1);

   nn.initialize_parameters(1.0);

   parameters = nn.get_parameters();

   assert_true(fabs(sse.calculate_loss() - sse.calculate_loss(parameters)) < 1.0e-3, LOG);

   // Test

   parameters = nn.get_parameters();

   assert_true(sse.calculate_loss() != sse.calculate_loss(parameters*2.0), LOG);

   // Test

   parameters = nn.get_parameters();

   direction.set(2, -1.5);
   rate = 2.3;

//   assert_true(sse.calculate_loss(instances_indices, direction, rate) == sse.calculate_loss(instances_indices, parameters + direction*rate), LOG);
*/
}


void LossIndexTest::test_calculate_gradient()
{
   message += "test_calculate_gradient\n";
/*
   DataSet ds;
   NeuralNetwork nn;

   size_t parameters_number;
   Vector<double> parameters;

   SumSquaredError sse(&nn, &ds);

   sse.set_regularization_method(LossIndex::L2);

   Vector<double> gradient;

   // Test

   nn.set(1, 1, 1);

   nn.initialize_parameters(0.0);

   parameters = nn.get_parameters();

   gradient = sse.calculate_gradient(parameters);

   assert_true(gradient == 0.0, LOG);

   // Test

   parameters_number = nn.get_parameters_number();
   nn.initialize_parameters(0.0);

   MockErrorTerm* mptp = new MockErrorTerm(&nn);

   sse.set_user_error_pointer(mptp);

   gradient = sse.calculate_gradient();

   assert_true(gradient.size() == parameters_number, LOG);
   assert_true(gradient == 0.0, LOG);
*/
}


void LossIndexTest::test_calculate_layers_delta()
{
   message += "test_calculate_layers_delta\n";
/*
   DataSet ds;
   NeuralNetwork nn;
   NumericalDifferentiation nd;
//   LossIndex li(&nn, &ds);

   size_t parameters_number;
   Vector<double> parameters;

   SumSquaredError sse(&nn, &ds);

   sse.set_regularization_method(LossIndex::L2);

   Vector<double> gradient;

   // Test

   nn.set(1,1,1);
   nn.initialize_parameters(1.0);

   ds.set(1,1,1);

   ds.initialize_data(1.0);

   const Matrix<double> inputs = ds.get_inputs();
   const Matrix<double> targets = ds.get_targets();

   const Vector<size_t> indices(1,0);

   const Vector<Matrix<double>> layers_combination = nn.get_multilayer_perceptron_pointer()->calculate_layers_combinations(inputs);
   const Matrix<double> outputs = nn.get_multilayer_perceptron_pointer()->calculate_outputs(inputs);

   cout << "layers_combination: " << layers_combination[0] << endl;

   const Matrix<double> output_gradient = (outputs - targets)*2.0;

   const Vector<Matrix<double>> layers_delta =
           sse.calculate_layers_delta(layers_combination, output_gradient);

//   nd.calculate_gradient(sse, &SumSquaredError::calculate_points_errors_layer_combinations, 0, layers_combination[0]);

   // Test

//   nn.set(1, 1, 1);

//   nn.initialize_parameters(0.0);

//   parameters = nn.get_parameters();

//   gradient = sse.calculate_gradient(parameters);

//   assert_true(gradient == 0.0, LOG);

//   // Test
//   parameters_number = nn.get_parameters_number();
//   nn.initialize_parameters(0.0);

//   MockErrorTerm* mptp = new MockErrorTerm(&nn);

//   sse.set_user_error_pointer(mptp);

//   gradient = sse.calculate_gradient();

//   assert_true(gradient.size() == parameters_number, LOG);
//   assert_true(gradient == 0.0, LOG);
*/
}


void LossIndexTest::test_calculate_gradient_norm()
{
   message += "test_calculate_gradient_norm\n";
}

// @todo

void LossIndexTest::test_calculate_Hessian()
{
   message += "test_calculate_Hessian\n";
/*
   DataSet ds;
   NeuralNetwork nn;
   size_t parameters_number;
   Vector<double> parameters;
   
   SumSquaredError sse(&nn, &ds);

   sse.set_regularization_method(LossIndex::L2);

   Matrix<double> Hessian;

   nn.set(1, 1, 1);

   nn.initialize_parameters(0.0);

   parameters_number = nn.get_parameters_number();
   parameters = nn.get_parameters();

   Hessian = sse.calculate_Hessian(parameters);

   assert_true(Hessian.get_rows_number() == parameters_number, LOG);
   assert_true(Hessian.get_columns_number() == parameters_number, LOG);

   nn.set();

   nn.initialize_parameters(0.0);

   parameters_number = nn.get_parameters_number();
   parameters = nn.get_parameters();

   Hessian = sse.calculate_Hessian(parameters);

   assert_true(Hessian.get_rows_number() == parameters_number, LOG);
   assert_true(Hessian.get_columns_number() == parameters_number, LOG);

   nn.set(1, 1);

   nn.initialize_parameters(0.0);

   parameters_number = nn.get_parameters_number();
   parameters = nn.get_parameters();

   Hessian = sse.calculate_Hessian(parameters);

   assert_true(Hessian.get_rows_number() == parameters_number, LOG);
   assert_true(Hessian.get_columns_number() == parameters_number, LOG);

   // Test

   parameters_number = nn.get_parameters_number();
   nn.initialize_parameters(0.0);
/*
   MockErrorTerm* mptp = new MockErrorTerm(&nn);

   sse.set_user_error_pointer(mptp);

   Hessian = sse.calculate_Hessian();

   assert_true(Hessian.get_rows_number() == parameters_number, LOG);
   assert_true(Hessian.get_columns_number() == parameters_number, LOG);
*/
}

// @todo

void LossIndexTest::test_calculate_inverse_Hessian()
{
   message += "test_calculate_inverse_Hessian\n";

//   NeuralNetwork nn(1, 1);

//   SumSquaredError sse(&nn);

//   Matrix<double> Hessian = sse.calculate_Hessian();

//   assert_true(sse.calculate_inverse_Hessian() == Hessian.calculate_inverse(), LOG);

}


// @todo

void LossIndexTest::test_calculate_vector_dot_Hessian()
{
   message += "test_calculate_vector_dot_Hessian\n";

//   NeuralNetwork nn(1, 1);

//   size_t parameters_number = nn.get_parameters_number();

//   SumSquaredError sse(&nn);

//   Vector<double> vector(0.0, 1.0, parameters_number-1.0);

//   Matrix<double> Hessian = sse.calculate_Hessian();

//   assert_true(sse.calculate_vector_dot_Hessian(vector) == vector.dot(Hessian), LOG);
}


void LossIndexTest::test_calculate_error_terms()
{
   message += "test_calculate_error_terms\n";
/*
   DataSet ds;
   NeuralNetwork nn;
   SumSquaredError sse(&nn, &ds);

   sse.set_loss_method(LossIndex::SUM_SQUARED_ERROR);

   Vector<double> terms;

    // Test

   ds.set(2,1,1);
   ds.initialize_data(0.0);

   nn.set(1,1);
   nn.initialize_parameters(0.0);

   terms = sse.calculate_error_terms();

   assert_true(terms.size() == 2, LOG);
   assert_true(terms == 0.0, LOG);
*/
}


void LossIndexTest::test_calculate_error_terms_Jacobian()
{
   message += "test_calculate_error_terms_Jacobian\n";
/*
   DataSet ds;
   NeuralNetwork nn;
   SumSquaredError sse(&nn, &ds);

   sse.set_loss_method(LossIndex::SUM_SQUARED_ERROR);

   Matrix<double> terms_Jacobian;

    // Test

   ds.set(3,1,1);
   ds.initialize_data(0.0);

   nn.set(1,1);
   nn.initialize_parameters(0.0);

   //@bug
   terms_Jacobian = sse.calculate_error_terms_Jacobian();

   assert_true(terms_Jacobian.get_rows_number() == 3, LOG);
   assert_true(terms_Jacobian.get_columns_number() == 2, LOG);
   assert_true(terms_Jacobian == 0.0, LOG);
*/
}


void LossIndexTest::test_calculate_zero_order_Taylor_approximation()
{
   message += "test_calculate_zero_order_Taylor_approximation\n";
}


void LossIndexTest::test_calculate_first_order_Taylor_approximation()
{
   message += "test_calculate_first_order_Taylor_approximation\n";
}


void LossIndexTest::test_calculate_second_order_Taylor_approximation()
{
   message += "test_calculate_second_order_Taylor_approximation\n";
}


void LossIndexTest::test_calculate_directional_loss()
{
   message += "test_calculate_directional_loss\n";

   DataSet ds;
   NeuralNetwork nn;

   Vector<double> direction;
   double rate;

   SumSquaredError sse(&nn, &ds);

   // Test

   nn.set(1, 1);

   sse.set_regularization_method(LossIndex::L2);

   direction.set(2, 1.0e3);

   rate = 1.0e3;

//   assert_true(sse.calculate_loss(Vector<size_t>(), direction, rate) != sse.calculate_loss(), LOG);
}


void LossIndexTest::test_calculate_directional_loss_derivatives()
{
   message += "test_calculate_directional_loss_derivative\n";

   DataSet ds;

   Vector<size_t> instances_indices;

   NeuralNetwork nn;

   Vector<double> direction;
   double rate;

   SumSquaredError sse(&nn, &ds);

   // Test

   nn.set(1, 1);
   nn.initialize_parameters(0.0);

   sse.set_regularization_method(LossIndex::L2);

   direction.set(2, 0.0);

   rate = 0.0;

//   assert_true(sse.calculate_directional_loss_derivatives(instances_indices, direction, rate) == 0.0, LOG);
}


void LossIndexTest::test_calculate_directional_loss_second_derivatives()
{
   message += "test_calculate_directional_loss_second_derivative\n";

   Vector<size_t> instances_indices;

   NeuralNetwork nn;

   Vector<double> direction;
   double rate;

   SumSquaredError sse(&nn);

   // Test

   nn.set(1, 1);
   nn.initialize_parameters(0.0);

   sse.set_regularization_method(LossIndex::L2);

   direction.set(2, 0.0);

   rate = 0.0;

//   assert_true(sse.calculate_loss_second_derivatives(direction, rate) == 0.0, LOG);

}


void LossIndexTest::test_to_XML()
{
   message += "test_to_XML\n";

   SumSquaredError sse;

   sse.set_regularization_method(LossIndex::L2);

   tinyxml2::XMLDocument* document = sse.to_XML();

   assert_true(document != nullptr, LOG);

   delete document;

}


void LossIndexTest::test_from_XML()
{
   message += "test_from_XML\n";

//   LossIndex li1;
//   LossIndex li2;

//   pf1.set_regularization_method(LossIndex::L2);

//   tinyxml2::XMLDocument* document = pf1.to_XML();

//    pf2.from_XML(*document);

//   delete document;

//    assert_true(pf2.get_error_type() == LossIndex::MINKOWSKI_ERROR, LOG);
//    assert_true(pf2.get_regularization_type() == LossIndex::L2, LOG);
}


void LossIndexTest::test_print()
{
   message += "test_print\n";

   SumSquaredError sse;

//   sse.print();
}


void LossIndexTest::test_save()
{
   message += "test_save\n";
/*
   string file_name = "../data/loss_index.xml";

   SumSquaredError sse;

   sse.set_loss_method(LossIndex::MINKOWSKI_ERROR);
   sse.set_regularization_method(LossIndex::L2);

   sse.save(file_name);
*/
}


void LossIndexTest::test_load()
{
   message += "test_load\n";

   string file_name = "../data/loss_index.xml";

//   LossIndex pf1;
//   LossIndex pf2;

   // Test
/*
   pf1.set_loss_method(LossIndex::MINKOWSKI_ERROR);
   pf1.set_regularization_method(LossIndex::L2);

   pf1.save(file_name);

   pf2.load(file_name);

   assert_true(pf2.get_error_type() == LossIndex::MINKOWSKI_ERROR, LOG);
   assert_true(pf2.get_regularization_type() == LossIndex::L2, LOG);
*/
}


void LossIndexTest::test_write_information()
{
   message += "test_write_information\n";

   DataSet ds;
   NeuralNetwork nn;

   SumSquaredError sse(&nn, &ds);

   string information;

   // Test

   ds.set(2, 1, 1);
   ds.randomize_data_normal();
   nn.set(1, 1);

   information = sse.write_information();

   assert_true(information.empty(), LOG);
}


void LossIndexTest::run_test_case()
{
   message += "Running loss index test case...\n";

   // Constructor and destructor methods
/*
   test_constructor();
   test_destructor();

   // Get methods

   test_get_neural_network_pointer();
   test_get_mathematical_model_pointer();
   test_get_data_set_pointer();

   test_get_user_pointer();
   test_get_user_regularization_pointer();
   test_get_user_constraints_pointer();

   test_get_numerical_differentiation_pointer();

   test_get_display();

//   // Set methods

   test_set_neural_network_pointer();
   test_set_numerical_differentiation();

   test_set_default();

   test_set_display();

//   // Performance methods

   test_calculate_loss();

   test_calculate_gradient();
*/
   test_calculate_layers_delta();
/*
   test_calculate_gradient_norm();

   test_calculate_Hessian();

   test_calculate_directional_loss();
   test_calculate_directional_loss_derivatives();
   test_calculate_directional_loss_second_derivatives();

   test_calculate_inverse_Hessian();

   test_calculate_vector_dot_Hessian();

   test_calculate_error_terms();
   test_calculate_error_terms_Jacobian();

//   // Taylor approximation methods

   test_calculate_zero_order_Taylor_approximation();
   test_calculate_first_order_Taylor_approximation();
   test_calculate_second_order_Taylor_approximation();

//   // Serialization methods

   test_to_XML();
   test_from_XML();

   test_print();
   test_save();
   test_load();

   test_write_information();
*/
   message += "End of loss index test case.\n";
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
