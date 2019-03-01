/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P E R F O R M A N C E   F U N C T I O N A L   T E S T   C L A S S   H E A D E R                            */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __PERFORMANCEFUNCTIONALTEST_H__
#define __PERFORMANCEFUNCTIONALTEST_H__

// Unit testing includes

#include "unit_testing.h"
//#include "mock_error_term.h"

using namespace OpenNN;


class LossIndexTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // GENERAL CONSTRUCTOR

   explicit LossIndexTest();


   // DESTRUCTOR

   virtual ~LossIndexTest();


   // METHODS

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Get methods

   void test_get_neural_network_pointer();

   void test_get_mathematical_model_pointer();
   void test_get_data_set_pointer();

   void test_get_user_pointer();
   void test_get_user_regularization_pointer();
   void test_get_user_constraints_pointer();

   void test_get_numerical_differentiation_pointer();

   // Serialization methods

   void test_get_display();

   // Set methods

   void test_set_neural_network_pointer();
   void test_set_numerical_differentiation();
   
   void test_set_default();

   // Serialization methods

   void test_set_display();

   // Loss index methods

   void test_calculate_loss();

   void test_calculate_gradient();

   void test_calculate_layers_delta();

   void test_calculate_gradient_norm();

   void test_calculate_Hessian();

   void test_calculate_inverse_Hessian();

   void test_calculate_vector_dot_Hessian();

   void test_calculate_error_terms();
   void test_calculate_error_terms_Jacobian();

   // Taylor approximation methods

   void test_calculate_zero_order_Taylor_approximation();
   void test_calculate_first_order_Taylor_approximation();
   void test_calculate_second_order_Taylor_approximation();

   // Directional loss methods
   
   void test_calculate_directional_loss();
   void test_calculate_directional_loss_derivatives();
   void test_calculate_directional_loss_second_derivatives();

   // Serialization methods

   void test_to_XML();   
   void test_from_XML();   

   void test_print();   
   void test_save();   
   void test_load();   

   void test_write_information();

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
