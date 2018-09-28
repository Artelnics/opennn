/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   C O N D I T I O N S   L A Y E R   T E S T   C L A S S   H E A D E R                                        */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __CONDITIONSLAYERTEST_H__
#define __CONDITIONSLAYERTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class ConditionsLayerTest : public UnitTesting
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // GENERAL CONSTRUCTOR

   explicit ConditionsLayerTest();


   // DESTRUCTOR

   virtual ~ConditionsLayerTest();

   // METHODS

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Assignment operators methods

   void test_assignment_operator();

   // Get methods

   // PerceptronLayer arrangement 

   void test_count_inputs_number();
   void test_count_outputs_number();
   
   // Display warnings 

   void test_get_display();

   // SET METHODS

   void test_set();
   void test_set_default();

   // Multilayer perceptron architecture

   void test_set_size();

   // Display messages

   void test_set_display();

   // Neural network initialization methods

   void test_initialize_random();

   // Conditions 

   void test_calculate_particular_solution();
   void test_calculate_particular_solution_Jacobian();
   void test_calculate_particular_solution_Hessian_form();

   void test_calculate_homogeneous_solution();
   void test_calculate_homogeneous_solution_Jacobian();
   void test_calculate_homogeneous_solution_Hessian_form();

   void test_calculate_outputs();
   void test_calculate_Jacobian();
   void test_calculate_Hessian_form();

   // Expression methods

   void test_write_particular_solution_expression();
   void test_write_homogeneous_solution_expression();

   void test_write_expression();

   // Serialization methods

   void test_to_XML();
   void test_from_XML();

   // Unit testing methods

   void run_test_case();
};


#endif



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
