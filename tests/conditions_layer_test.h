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

   explicit ConditionsLayerTest(void);


   // DESTRUCTOR

   virtual ~ConditionsLayerTest(void);

   // METHODS

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void);

   // Assignment operators methods

   void test_assignment_operator(void);

   // Get methods

   // PerceptronLayer arrangement 

   void test_count_inputs_number(void);
   void test_count_outputs_number(void);
   
   // Display warnings 

   void test_get_display(void);

   // SET METHODS

   void test_set(void);
   void test_set_default(void);

   // Multilayer perceptron architecture

   void test_set_size(void);

   // Display messages

   void test_set_display(void);

   // Neural network initialization methods

   void test_initialize_random(void);

   // Conditions 

   void test_calculate_particular_solution(void);
   void test_calculate_particular_solution_Jacobian(void);
   void test_calculate_particular_solution_Hessian_form(void);

   void test_calculate_homogeneous_solution(void);
   void test_calculate_homogeneous_solution_Jacobian(void);
   void test_calculate_homogeneous_solution_Hessian_form(void);

   void test_calculate_outputs(void);
   void test_calculate_Jacobian(void);
   void test_calculate_Hessian_form(void);

   // Expression methods

   void test_write_particular_solution_expression(void);
   void test_write_homogeneous_solution_expression(void);

   void test_write_expression(void);

   // Serialization methods

   void test_to_XML(void);
   void test_from_XML(void);

   // Unit testing methods

   void run_test_case(void);
};


#endif



// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2016 Roberto Lopez.
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
