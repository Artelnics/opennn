/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P E R C E P T R O N   L A Y E R   T E S T   C L A S S   H E A D E R                                        */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __PERCEPTRONLAYERTEST_H__
#define __PERCEPTRONLAYERTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class PerceptronLayerTest : public UnitTesting
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // GENERAL CONSTRUCTOR

   explicit PerceptronLayerTest(void);


   // DESTRUCTOR

   virtual ~PerceptronLayerTest(void);

   // METHODS

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void);

   // Assignment operators methods

   void test_assignment_operator(void);

   // Get methods

   // PerceptronLayer arrangement

   void test_is_empty(void);

   void test_count_inputs_number(void);
   void test_get_perceptrons_number(void);

   void test_get_perceptrons(void);
   void test_get_perceptron(void);

   // Parameters

   void test_arrange_biases(void);
   void test_arrange_synaptic_weights(void);

   void test_count_parameters_number(void);
   void test_arrange_parameters(void);
   void test_calculate_parameters_norm(void);

   void test_count_cumulative_parameters_number(void);

   // Activation functions

   void test_get_activation_function(void);
   void test_get_activation_function_name(void);
   
   // Display messages

   void test_get_display(void);

   // SET METHODS

   void test_set(void);
   void test_set_default(void);

   // Architecture

   void test_set_size(void);

   // Parameters

   void test_set_biases(void);
   void test_set_synaptic_weights(void);
   void test_set_parameters(void);

   // Activation functions

   void test_set_activation_function(void);

   // Display messages

   void test_set_display(void);

   // Growing and pruning

   void test_grow_inputs(void);
   void test_grow_perceptrons(void);

   void test_prune_input(void);
   void test_prune_perceptron(void);

   // Initialization methods

   void test_initialize_random(void);

   // Parameters initialization methods

   void test_initialize_parameters(void);

   void test_initialize_biases(void);    
   void test_initialize_synaptic_weights(void);
   void test_randomize_parameters_uniform(void);
   void test_randomize_parameters_normal(void);

   // PerceptronLayer combination

   void test_calculate_combination(void);

   void test_calculate_combination_Jacobian(void);
   void test_calculate_combination_Hessian_form(void);

   void test_calculate_combination_parameters_Jacobian(void);
   void test_calculate_combination_parameters_Hessian_form(void);

   // PerceptronLayer activation 

   void test_calculate_activation(void);
   void test_calculate_activation_derivative(void);
   void test_calculate_activation_second_derivative(void);

   // PerceptronLayer outputs 

   void test_calculate_outputs(void);

   void test_calculate_Jacobian(void);   
   void test_calculate_Hessian_form(void);

   void test_calculate_parameters_Jacobian(void);
   void test_calculate_parameters_Hessian_form(void);

   // Expression methods

   void test_get_activation_function_expression(void);

   void test_write_expression(void);

   void test_get_network_architecture_expression(void);

   void test_get_inputs_scaling_expression(void);
   void test_get_outputs_unscaling_expression(void);

   void test_get_boundary_conditions_expression(void);

   void test_get_bounded_output_expression(void);

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
