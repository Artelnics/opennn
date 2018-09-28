/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   C O N D I T I O N S   L A Y E R   T E S T   C L A S S                                                      */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "conditions_layer_test.h"

using namespace OpenNN;

// GENERAL CONSTRUCTOR

ConditionsLayerTest::ConditionsLayerTest() : UnitTesting()
{
}


// DESTRUCTOR

ConditionsLayerTest::~ConditionsLayerTest()
{
}


// METHODS

void ConditionsLayerTest::test_constructor()
{
   message += "test_constructor\n";

   // Default constructor

   ConditionsLayer cl1;

   assert_true(cl1.get_external_inputs_number() == 0, LOG);
   assert_true(cl1.get_conditions_neurons_number() == 0, LOG);

   // Copy constructor

}


void ConditionsLayerTest::test_destructor()
{
   message += "test_destructor\n";
}


void ConditionsLayerTest::test_assignment_operator()
{
   message += "test_assignment_operator\n";

//   ConditionsLayer cl_1;
//   ConditionsLayer cl_2 = cl_1;
}


void ConditionsLayerTest::test_count_inputs_number()
{
   message += "test_count_inputs_number\n";

   ConditionsLayer cl;

   // Test

   cl.set();
   assert_true(cl.get_external_inputs_number() == 0, LOG);
}


void ConditionsLayerTest::test_get_display()
{
   message += "test_get_display\n";
}


// @todo

void ConditionsLayerTest::test_set()
{
   message += "test_set\n";

   ConditionsLayer cl;

   size_t inputs_number;
   size_t conditions_neurons_number;

   // Test

   cl.set(1, 1);

   inputs_number = cl.get_external_inputs_number();
   conditions_neurons_number = cl.get_conditions_neurons_number();

   assert_true(inputs_number == 1, LOG);
   assert_true(conditions_neurons_number == 1, LOG);
}


void ConditionsLayerTest::test_set_default()
{
   message += "test_set_default\n";
}


void ConditionsLayerTest::test_set_display()
{
   message += "test_set_display\n";
}


void ConditionsLayerTest::test_initialize_random()
{
   message += "test_initialize_random\n";

   ConditionsLayer cl;

   // Test

   cl.initialize_random();
}


// @todo

void ConditionsLayerTest::test_calculate_particular_solution()
{
   message += "test_calculate_particular_solution\n";

   ConditionsLayer cl;

   Vector<double> inputs;
   Vector<double> particular_solution;

   // Test 

   cl.set(1, 1);

   cl.set_external_input_value(0, 0.0);
   cl.set_external_input_value(1, 1.0);

   cl.set_output_value(0, 0, 0.0);
   cl.set_output_value(0, 1, 1.0);

   inputs.set(1, 0.0);

   particular_solution = cl.calculate_particular_solution(inputs);

   assert_true(particular_solution.size() == 1, LOG);
   assert_true(particular_solution == 0.0, LOG);

   inputs.set(1, 1.0);

   particular_solution = cl.calculate_particular_solution(inputs);

   assert_true(particular_solution.size() == 1, LOG);
   assert_true(particular_solution == 1.0, LOG);

   // Test 

   cl.set(1, 2);

   cl.set_external_input_value(0, 0.0);
   cl.set_external_input_value(1, 1.0);

   cl.set_output_value(0, 0, 0.0);
   cl.set_output_value(0, 1, 1.0);
   cl.set_output_value(1, 0, 0.0);
   cl.set_output_value(1, 1, 1.0);

   inputs.set(1, 0.0);

   particular_solution = cl.calculate_particular_solution(inputs);

   assert_true(particular_solution.size() == 2, LOG);
   assert_true(particular_solution == 0.0, LOG);
}


// @todo

void ConditionsLayerTest::test_calculate_particular_solution_Jacobian()
{
   message += "test_calculate_particular_solution_Jacobian\n";

   ConditionsLayer cl;

   Vector<double> inputs;
   Matrix<double> particular_solution_Jacobian;

   // Test 

   cl.set(1, 1);

   cl.set_external_input_value(0, 0.0);
   cl.set_external_input_value(1, 1.0);

   cl.set_output_value(0, 0, 0.0);
   cl.set_output_value(0, 1, 1.0);

   inputs.set(1, 0.0);

   particular_solution_Jacobian = cl.calculate_particular_solution_Jacobian(inputs);

   assert_true(particular_solution_Jacobian.get_rows_number() == 1, LOG);
   assert_true(particular_solution_Jacobian.get_columns_number() == 1, LOG);
   assert_true(particular_solution_Jacobian == 1.0, LOG);

   // Test 

   cl.set(1, 3);

   cl.set_external_input_value(0, 0.0);
   cl.set_external_input_value(1, 1.0);

   cl.set_output_value(0, 0, 0.0);
   cl.set_output_value(0, 1, 1.0);

   inputs.set(1, 1.0);

   particular_solution_Jacobian = cl.calculate_particular_solution_Jacobian(inputs);

//   assert_true(particular_solution_Jacobian.get_rows_number() == 3, LOG);
//   assert_true(particular_solution_Jacobian.get_columns_number() == 2, LOG);
   assert_true(particular_solution_Jacobian == 1.0, LOG);
}


// @todo

void ConditionsLayerTest::test_calculate_homogeneous_solution()
{
   message += "test_calculate_homogeneous_solution\n";

   ConditionsLayer cl;

   Vector<double> inputs;
   Vector<double> homogeneous_solution;

   // Test 

   cl.set(1, 1);

   cl.set_external_input_value(0, 0.0);
   cl.set_external_input_value(1, 1.0);

   cl.set_output_value(0, 0, 0.0);
   cl.set_output_value(0, 1, 1.0);

   inputs.set(1, 0.0);

   homogeneous_solution = cl.calculate_homogeneous_solution(inputs);

   assert_true(homogeneous_solution.size() == 1, LOG);
   assert_true(homogeneous_solution == 0.0, LOG);

   inputs.set(1, 0.5);

   homogeneous_solution = cl.calculate_homogeneous_solution(inputs);

   assert_true(homogeneous_solution.size() == 1, LOG);
   assert_true(homogeneous_solution != 0.0, LOG);

   inputs.set(1, 1.0);

   homogeneous_solution = cl.calculate_homogeneous_solution(inputs);

   assert_true(homogeneous_solution.size() == 1, LOG);
   assert_true(homogeneous_solution == 0.0, LOG);

   // Test 

   cl.set(1, 2);

   cl.set_external_input_value(0, 0.0);
   cl.set_external_input_value(1, 1.0);

   cl.set_output_value(0, 0, 0.0);
   cl.set_output_value(0, 1, 1.0);
   cl.set_output_value(1, 0, 0.0);
   cl.set_output_value(1, 1, 1.0);

   inputs.set(1, 0.0);

   homogeneous_solution = cl.calculate_homogeneous_solution(inputs);

   assert_true(homogeneous_solution.size() == 2, LOG);
   assert_true(homogeneous_solution == 0.0, LOG); 
}


// @todo

void ConditionsLayerTest::test_calculate_homogeneous_solution_Jacobian()
{
   message += "test_calculate_homogeneous_solution_Jacobian\n";

   ConditionsLayer cl;

   Vector<double> inputs;
   Matrix<double> homogeneous_solution_Jacobian;

   // Test 

   cl.set(1, 1);

   cl.set_external_input_value(0, 0.0);
   cl.set_external_input_value(1, 1.0);

   cl.set_output_value(0, 0, 0.0);
   cl.set_output_value(0, 1, 1.0);

   inputs.set(1, 1.0);

//   homogeneous_solution_Jacobian = cl.calculate_homogeneous_solution_Jacobian(inputs);

//   assert_true(homogeneous_solution_Jacobian.get_rows_number() == 1, LOG);
//   assert_true(homogeneous_solution_Jacobian.get_columns_number() == 1, LOG);
//   assert_true(homogeneous_solution_Jacobian == 0.0, LOG);

   // Test 

   cl.set(1,3);
   inputs.set(1, 0.0);

//   homogeneous_solution_Jacobian = mbcl.calculate_homogeneous_solution_Jacobian(inputs);

//   assert_true(homogeneous_solution_Jacobian.get_rows_number() == 3, LOG);
//   assert_true(homogeneous_solution_Jacobian.get_columns_number() == 2, LOG);
//   assert_true(homogeneous_solution_Jacobian == 0.0, LOG);
}


void ConditionsLayerTest::test_calculate_outputs()
{
   message += "test_calculate_outputs\n";

   ConditionsLayer cl;

   Vector<double> inputs1;
   Vector<double> inputs2;
   Vector<double> outputs;

   // Test 

   cl.set(1, 1);

   cl.set_external_input_value(0, 0.0);
   cl.set_external_input_value(1, 1.0);

   cl.set_output_value(0, 0, 0.0);
   cl.set_output_value(0, 1, 1.0);

   inputs1.set(1, 1.0);
   inputs2.set(1, 1.0);

   outputs = cl.calculate_outputs(inputs1, inputs2);

   assert_true(outputs.size() == 1, LOG);
}


void ConditionsLayerTest::test_calculate_Jacobian()
{
   message += "test_calculate_Jacobian\n";
}


void ConditionsLayerTest::test_calculate_Hessian_form()
{
   message += "test_calculate_Hessian_form\n";
}


// @todo

void ConditionsLayerTest::test_to_XML()
{
   message += "test_to_XML\n";

   ConditionsLayer  cl;

   tinyxml2::XMLDocument* document = cl.to_XML();

   assert_true(document != NULL, LOG);

   delete document;
}


void ConditionsLayerTest::test_from_XML()
{
   message += "test_to_XML\n";

   ConditionsLayer  cl;
}


void ConditionsLayerTest::test_write_particular_solution_expression()
{
   message += "test_write_particular_solution_expression\n";
}


void ConditionsLayerTest::test_write_homogeneous_solution_expression()
{
   message += "test_write_homogeneous_solution_expression\n";
}


// @todo

void ConditionsLayerTest::test_write_expression()
{
   message += "test_write_expression\n";

   ConditionsLayer cl(1,1);

   Vector<string> x1(1, "x1");
   Vector<string> x2(1, "x2");
   Vector<string> y(1, "y");

   string expression = cl.write_expression(x1, x2, y);
}


void ConditionsLayerTest::run_test_case()
{
   message += "Running conditions layer test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Get methods

   // Display warnings

   test_get_display();

   // Set methods

   test_set();
   test_set_default();

   // Display messages

   test_set_display();

   // PerceptronLayer initialization methods

   test_initialize_random();

   // Conditions 

   test_calculate_particular_solution();
   test_calculate_particular_solution_Jacobian();

   test_calculate_homogeneous_solution();
   test_calculate_homogeneous_solution_Jacobian();

   test_calculate_outputs();
   test_calculate_Jacobian();
   test_calculate_Hessian_form();

   // Expression methods
   
   test_write_particular_solution_expression();
   test_write_homogeneous_solution_expression();

   test_write_expression();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   message += "End of conditions layer test case.\n";
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
