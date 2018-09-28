/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P E R F O R M A N C E   F U N C T I O N A L   T E S T   C L A S S                                          */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
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

   LossIndex pf1;

   assert_true(pf1.has_neural_network() == false, LOG);
   assert_true(pf1.has_data_set() == false, LOG);
   assert_true(pf1.has_mathematical_model() == false, LOG);
}


void LossIndexTest::test_destructor()
{
   message += "test_destructor\n";
}


void LossIndexTest::test_get_neural_network_pointer()
{
   message += "test_get_neural_network_pointer\n";

   LossIndex pf;
   NeuralNetwork nn;

   // Test

   pf.set_neural_network_pointer(&nn);
   assert_true(pf.get_neural_network_pointer() != NULL,	LOG);
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

   LossIndex pf;

   // Test

   pf.set_display(true);
   assert_true(pf.get_display() == true, LOG);

   pf.set_display(false);
   assert_true(pf.get_display() == false, LOG);
}


void LossIndexTest::test_set_neural_network_pointer()
{
   message += "test_set_neural_network_pointer\n";

   LossIndex pf;
   NeuralNetwork nn;

   // Test

   pf.set_neural_network_pointer(&nn);
   assert_true(pf.get_neural_network_pointer() != NULL, LOG);
}


void LossIndexTest::test_set_numerical_differentiation()
{
   message += "test_set_numerical_differentiation\n";
}


void LossIndexTest::test_set_default()
{
   message += "test_set_default\n";

   LossIndex pf;

   // Test

   pf.set_default();
}


void LossIndexTest::test_set_display()
{
   message += "test_set_display\n";
}


void LossIndexTest::test_calculate_loss()
{
   message += "test_calculate_loss\n";

   DataSet ds;

   NeuralNetwork nn;

   Vector<double> parameters;

   LossIndex pf(&nn);

   double loss;

   Vector<double> direction;
   double rate;

   // Test

   pf.destruct_all_terms();
   pf.set_regularization_type(LossIndex::NEURAL_PARAMETERS_NORM);

   NeuralParametersNorm* neural_parameters_norm = pf.get_neural_parameters_norm_pointer();

   double neural_parameters_norm_weight = neural_parameters_norm->get_neural_parameters_norm_weight();

   nn.set(1, 1);

   nn.initialize_parameters(1.0);

   parameters = nn.arrange_parameters();

   assert_true(fabs(pf.calculate_loss() - neural_parameters_norm_weight*sqrt(2.0)) < 1.0e-3, LOG);

   assert_true(fabs(pf.calculate_loss() - pf.calculate_loss(parameters)) < 1.0e-3, LOG);

   // Test

   parameters = nn.arrange_parameters();

   assert_true(pf.calculate_loss() != pf.calculate_loss(parameters*2.0), LOG);

   // Test

   direction.set(2, -0.5);
   rate = 2.0;

   assert_true(pf.calculate_loss(direction, rate) == 0.0, LOG);

   // Test

   parameters = nn.arrange_parameters();

   direction.set(2, -1.5);
   rate = 2.3;

   assert_true(pf.calculate_loss(direction, rate) == pf.calculate_loss(parameters + direction*rate), LOG);

   // Test

   ds.set(1, 1, 1);
   ds.randomize_data_normal();

   pf.set_data_set_pointer(&ds);

   pf.destruct_all_terms();
   pf.set_error_type(LossIndex::SUM_SQUARED_ERROR);

   nn.set(1, 1);

   nn.initialize_parameters(1.0);

   parameters = nn.arrange_parameters();

   assert_true(fabs(pf.calculate_loss() - pf.calculate_loss(parameters)) < 1.0e-3, LOG);

   // Test

   parameters = nn.arrange_parameters();

   assert_true(pf.calculate_loss() != pf.calculate_loss(parameters*2.0), LOG);

   // Test

   parameters = nn.arrange_parameters();

   direction.set(2, -1.5);
   rate = 2.3;

   assert_true(pf.calculate_loss(direction, rate) == pf.calculate_loss(parameters + direction*rate), LOG);

   // Test

   nn.initialize_parameters(0.0);

   MockErrorTerm* mptp = new MockErrorTerm(&nn);

   pf.set_user_error_pointer(mptp);

   loss = pf.calculate_loss();

   assert_true(loss == 0.0, LOG);
}


void LossIndexTest::test_calculate_gradient()
{
   message += "test_calculate_gradient\n";

   DataSet ds;
   NeuralNetwork nn;

   size_t parameters_number;
   Vector<double> parameters;

   LossIndex pf(&nn, &ds);

   pf.destruct_all_terms();
   pf.set_regularization_type(LossIndex::NEURAL_PARAMETERS_NORM);

   Vector<double> gradient;

   // Test

   nn.set(1, 1, 1);

   nn.initialize_parameters(0.0);

   parameters = nn.arrange_parameters();

   gradient = pf.calculate_gradient(parameters);

   assert_true(gradient == 0.0, LOG);

   // Test

   parameters_number = nn.count_parameters_number();
   nn.initialize_parameters(0.0);

   MockErrorTerm* mptp = new MockErrorTerm(&nn);

   pf.set_user_error_pointer(mptp);

   gradient = pf.calculate_gradient();

   assert_true(gradient.size() == parameters_number, LOG);
   assert_true(gradient == 0.0, LOG);
}


void LossIndexTest::test_calculate_gradient_norm()
{
   message += "test_calculate_gradient_norm\n";
}

// @todo

void LossIndexTest::test_calculate_Hessian()
{
   message += "test_calculate_Hessian\n";

   DataSet ds;
   NeuralNetwork nn;
   size_t parameters_number;
   Vector<double> parameters;
   
   LossIndex pf(&nn, &ds);

   pf.destruct_all_terms();
   pf.set_regularization_type(LossIndex::NEURAL_PARAMETERS_NORM);

   Matrix<double> Hessian;

   nn.set(1, 1, 1);

   nn.initialize_parameters(0.0);

   parameters_number = nn.count_parameters_number();
   parameters = nn.arrange_parameters();

   Hessian = pf.calculate_Hessian(parameters);

   assert_true(Hessian.get_rows_number() == parameters_number, LOG);
   assert_true(Hessian.get_columns_number() == parameters_number, LOG);

   nn.set();

   nn.initialize_parameters(0.0);

   parameters_number = nn.count_parameters_number();
   parameters = nn.arrange_parameters();

   Hessian = pf.calculate_Hessian(parameters);

   assert_true(Hessian.get_rows_number() == parameters_number, LOG);
   assert_true(Hessian.get_columns_number() == parameters_number, LOG);

   nn.set(1, 1);

   nn.initialize_parameters(0.0);

   parameters_number = nn.count_parameters_number();
   parameters = nn.arrange_parameters();

   Hessian = pf.calculate_Hessian(parameters);

   assert_true(Hessian.get_rows_number() == parameters_number, LOG);
   assert_true(Hessian.get_columns_number() == parameters_number, LOG);

   // Test

   parameters_number = nn.count_parameters_number();
   nn.initialize_parameters(0.0);

   MockErrorTerm* mptp = new MockErrorTerm(&nn);

   pf.set_user_error_pointer(mptp);

   Hessian = pf.calculate_Hessian();

   assert_true(Hessian.get_rows_number() == parameters_number, LOG);
   assert_true(Hessian.get_columns_number() == parameters_number, LOG);

}

// @todo

void LossIndexTest::test_calculate_inverse_Hessian()
{
   message += "test_calculate_inverse_Hessian\n";

//   NeuralNetwork nn(1, 1);

//   LossIndex pf(&nn);

//   Matrix<double> Hessian = pf.calculate_Hessian();

//   assert_true(pf.calculate_inverse_Hessian() == Hessian.calculate_inverse(), LOG);

}


// @todo

void LossIndexTest::test_calculate_vector_dot_Hessian()
{
   message += "test_calculate_vector_dot_Hessian\n";

//   NeuralNetwork nn(1, 1);

//   size_t parameters_number = nn.count_parameters_number();

//   LossIndex pf(&nn);

//   Vector<double> vector(0.0, 1.0, parameters_number-1.0);

//   Matrix<double> Hessian = pf.calculate_Hessian();

//   assert_true(pf.calculate_vector_dot_Hessian(vector) == vector.dot(Hessian), LOG);
}


void LossIndexTest::test_calculate_terms()
{
   message += "test_calculate_terms\n";

   DataSet ds;
   NeuralNetwork nn;
   LossIndex pf(&nn, &ds);

   pf.set_error_type(LossIndex::SUM_SQUARED_ERROR);

   Vector<double> terms;

    // Test

   ds.set(2,1,1);
   ds.initialize_data(0.0);

   nn.set(1,1);
   nn.initialize_parameters(0.0);

   terms = pf.calculate_terms();

   assert_true(terms.size() == 2, LOG);
   assert_true(terms == 0.0, LOG);

}


void LossIndexTest::test_calculate_terms_Jacobian()
{
   message += "test_calculate_terms_Jacobian\n";

   DataSet ds;
   NeuralNetwork nn;
   LossIndex pf(&nn, &ds);

   pf.set_error_type(LossIndex::SUM_SQUARED_ERROR);

   Matrix<double> terms_Jacobian;

    // Test

   ds.set(3,1,1);
   ds.initialize_data(0.0);

   nn.set(1,1);
   nn.initialize_parameters(0.0);

   //@bug
   terms_Jacobian = pf.calculate_terms_Jacobian();

   assert_true(terms_Jacobian.get_rows_number() == 3, LOG);
   assert_true(terms_Jacobian.get_columns_number() == 2, LOG);
   assert_true(terms_Jacobian == 0.0, LOG);

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

   LossIndex pf(&nn, &ds);

   // Test

   nn.set(1, 1);

   pf.destruct_all_terms();
   pf.set_regularization_type(LossIndex::NEURAL_PARAMETERS_NORM);

   direction.set(2, 1.0e3);

   rate = 1.0e3;

   assert_true(pf.calculate_loss(direction, rate) != pf.calculate_loss(), LOG);
}


void LossIndexTest::test_calculate_directional_loss_derivative()
{
   message += "test_calculate_directional_loss_derivative\n";

   DataSet ds;
   NeuralNetwork nn;

   Vector<double> direction;
   double rate;

   LossIndex pf(&nn, &ds);

   // Test

   nn.set(1, 1);
   nn.initialize_parameters(0.0);

   pf.destruct_all_terms();
   pf.set_regularization_type(LossIndex::NEURAL_PARAMETERS_NORM);

   direction.set(2, 0.0);

   rate = 0.0;

   assert_true(pf.calculate_loss_derivative(direction, rate) == 0.0, LOG);
}


void LossIndexTest::test_calculate_directional_loss_second_derivative()
{
   message += "test_calculate_directional_loss_second_derivative\n";

   NeuralNetwork nn;

   Vector<double> direction;
   double rate;

   LossIndex pf(&nn);

   // Test

   nn.set(1, 1);
   nn.initialize_parameters(0.0);

   pf.destruct_all_terms();
   pf.set_regularization_type(LossIndex::NEURAL_PARAMETERS_NORM);

   direction.set(2, 0.0);

   rate = 0.0;

   assert_true(pf.calculate_loss_second_derivative(direction, rate) == 0.0, LOG);
}


void LossIndexTest::test_to_XML()
{
   message += "test_to_XML\n";

   LossIndex pf;

   pf.set_error_type(LossIndex::MINKOWSKI_ERROR);
   pf.set_regularization_type(LossIndex::NEURAL_PARAMETERS_NORM);

   tinyxml2::XMLDocument* document = pf.to_XML();

   assert_true(document != NULL, LOG);

   delete document;
}


void LossIndexTest::test_from_XML()
{
   message += "test_from_XML\n";

   LossIndex pf1;
   LossIndex pf2;

   pf1.set_error_type(LossIndex::MINKOWSKI_ERROR);
   pf1.set_regularization_type(LossIndex::NEURAL_PARAMETERS_NORM);

   tinyxml2::XMLDocument* document = pf1.to_XML();

    pf2.from_XML(*document);

   delete document;

    assert_true(pf2.get_error_type() == LossIndex::MINKOWSKI_ERROR, LOG);
    assert_true(pf2.get_regularization_type() == LossIndex::NEURAL_PARAMETERS_NORM, LOG);

}


void LossIndexTest::test_print()
{
   message += "test_print\n";

   LossIndex pf;

//   pf.print();
}


void LossIndexTest::test_save()
{
   message += "test_save\n";

   string file_name = "../data/loss_index.xml";

   LossIndex pf;

   pf.set_error_type(LossIndex::MINKOWSKI_ERROR);
   pf.set_regularization_type(LossIndex::NEURAL_PARAMETERS_NORM);

   pf.save(file_name);
}


void LossIndexTest::test_load()
{
   message += "test_load\n";

   string file_name = "../data/loss_index.xml";

   LossIndex pf1;
   LossIndex pf2;

   // Test

   pf1.set_error_type(LossIndex::MINKOWSKI_ERROR);
   pf1.set_regularization_type(LossIndex::NEURAL_PARAMETERS_NORM);

   pf1.save(file_name);

   pf2.load(file_name);

   assert_true(pf2.get_error_type() == LossIndex::MINKOWSKI_ERROR, LOG);
   assert_true(pf2.get_regularization_type() == LossIndex::NEURAL_PARAMETERS_NORM, LOG);
}


void LossIndexTest::test_write_information()
{
   message += "test_write_information\n";

   DataSet ds;
   NeuralNetwork nn;

   LossIndex pf(&nn, &ds);

   string information;

   // Test

   ds.set(2, 1, 1);
   ds.randomize_data_normal();
   nn.set(1, 1);

   information = pf.write_information();

   assert_true(information.empty(), LOG);
}


void LossIndexTest::run_test_case()
{
   message += "Running loss functional test case...\n";

   // Constructor and destructor methods

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

   test_calculate_gradient_norm();

   test_calculate_Hessian();

   test_calculate_directional_loss();
   test_calculate_directional_loss_derivative();
   test_calculate_directional_loss_second_derivative();

   test_calculate_inverse_Hessian();

   test_calculate_vector_dot_Hessian();

   test_calculate_terms();
   test_calculate_terms_Jacobian();

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

   message += "End of loss functional test case.\n";
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
