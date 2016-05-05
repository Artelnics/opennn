/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P E R C E P T R O N   T E S T   C L A S S   H E A D E R                                                    */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __PERCEPTRONTEST_H__
#define __PERCEPTRONTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class PerceptronTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // DEFAULT CONSTRUCTOR

   PerceptronTest(void);


   // DESTRUCTOR

   virtual ~PerceptronTest(void);

   // METHODS

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void); 

   // Assignment operator 

   void test_assignment_operator(void);

   // Get methods

   void test_count_inputs_number(void);

   void test_get_activation_function(void);
   void test_get_bias(void);   
   void test_arrange_synaptic_weights(void);
   void test_get_synaptic_weight(void);

   void test_count_parameters_number(void);
   void test_arrange_parameters(void);

   void test_get_display(void);

   // Set methods

   void test_set(void);

   void test_set_activation_function(void);

   void test_set_inputs_number(void);

   void test_set_bias(void);
   void test_set_synaptic_weights(void);
   void test_set_synaptic_weight(void);

   void test_set_parameters_number(void);
   void test_set_parameters(void);

   void test_set_display(void);

   // Growing and pruning

   void test_grow_input(void);

   void test_prune_input(void);

   // Initialization methods

   void test_initialize_bias_uniform(void);
   void test_initialize_bias_normal(void);

   void test_initialize_synaptic_weights_uniform(void);
   void test_initialize_synaptic_weights_normal(void);

   void test_initialize_parameters(void);

   // Combination methods

   void test_calculate_combination(void);

   void test_calculate_combination_gradient(void);
   void test_calculate_combination_Hessian(void);

   void test_calculate_combination_parameters_gradient(void);
   void test_calculate_combination_parameters_Hessian(void);

   // Activation methods

   void test_calculate_activation(void);
   void test_calculate_activation_derivative(void);
   void test_calculate_activation_second_derivative(void);

   // Output methods

   void test_calculate_output(void);
   void test_calculate_gradient(void);
   void test_calculate_Hessian(void);

   void test_calculate_parameters_outputs(void);
   void test_calculate_parameters_gradient(void);
   void test_calculate_parameters_Hessian(void);

   // Serialization methods

   void test_print(void);
   void test_save(void);
   void test_load(void);

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

