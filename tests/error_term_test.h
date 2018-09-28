/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   E R R O R   T E R M   T E S T   C L A S S   H E A D E R                                                    */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __ERRORTERMTEST_H__
#define __ERRORTERMTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class ErrorTermTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // GENERAL CONSTRUCTOR

   explicit ErrorTermTest();


   // DESTRUCTOR

   virtual ~ErrorTermTest();


   // METHODS

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Operators

   void test_assingment_operator();
   void test_equal_to_operator();

   // Get methods

   void test_get_neural_network_pointer();

   void test_get_data_set_pointer();

   void test_get_numerical_differentiation_pointer();

   // Serialization methods

   void test_get_display();

   // Set methods

   void test_set_neural_network_pointer();

   void test_set_data_set_pointer();

   void test_set_numerical_differentiation_pointer();
   
   void test_set_default();

   // Serialization methods

   void test_set_display();

   // delta methods

   void test_calculate_layers_delta();
   void test_calculate_interlayers_Delta();

   // Point objective function methods

   void test_calculate_point_objective();
   void test_calculate_point_gradient();
   void test_calculate_point_Hessian();

   // Objective methods

   void test_calculate_error();

   void test_calculate_selection_error();

   void test_calculate_gradient(); 
   void test_calculate_Hessian(); 

   void test_calculate_terms();
   void test_calculate_terms_Jacobian();


   // Serialization methods

   void test_to_XML();   
   void test_from_XML();   

   void test_write_information();

   // Unit testing methods

   void run_test_case();

};


#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Roberto Lopez
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
