/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M U L T I L A Y E R   P E R C E P T R O N   T E S T   C L A S S   H E A D E R                              */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __MULTILAYERPERCEPTRONTEST_H__
#define __MULTILAYERPERCEPTRONTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class MultilayerPerceptronTest : public UnitTesting
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // GENERAL CONSTRUCTOR

   explicit MultilayerPerceptronTest(void);


   // DESTRUCTOR

   virtual ~MultilayerPerceptronTest(void);

   // METHODS

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void);

   // Assignment operators methods

   void test_assignment_operator(void);

   // Get methods

   // Multilayer perceptron architecture

   void test_count_inputs_number(void);

   void test_get_layers_number(void);
   void test_count_layers_perceptrons_number(void);

   void test_count_outputs_number(void);

   void test_count_perceptrons_number(void);
   void test_count_cumulative_perceptrons_number(void);

   void test_get_layers(void);
   void test_get_layer(void);

   // Multilayer Perceptron parameters

   void test_arrange_layers_parameters_number(void);

   void test_count_parameters_number(void);
   void test_get_cumulative_parameters_number(void);

   void test_arrange_parameters(void);   

   void test_arrange_layers_biases(void);

   void test_arrange_layers_synaptic_weights(void);

   void test_get_layers_parameters(void);

   void test_get_parameter_indices(void);
   void test_arrange_parameters_indices(void);

   void test_get_layers_activation_function(void);
   void test_get_layers_activation_function_name(void);

   // Display messages

   void test_get_display(void);

   // SET METHODS

   void test_set(void);
   void test_set_default(void);

   // Multilayer perceptron architecture

   void test_set_layers_perceptrons_number(void);

   // Multilayer perceptron parameters

   void test_set_parameters(void);

   void test_set_layers_biases(void);

   void test_set_layers_synaptic_weights(void);

   void test_set_layers_parameters(void);

   // Activation functions

   void test_set_layers_activation_function(void);

   // Display messages

   void test_set_display(void);

   // Check methods

   void test_is_empty(void);

   // Growing and pruning

   void test_grow_input(void);
   void test_grow_layer(void);

   void test_prune_input(void);
   void test_prune_output(void);

   void test_prune_layer(void);

   // Initialization methods

   void test_initialize_random(void);

   // Parameters initialization methods

   void test_initialize_parameters(void);

   void test_initialize_biases(void);    
   void test_initialize_synaptic_weights(void);
   void test_randomize_parameters_uniform(void);
   void test_randomize_parameters_normal(void);

   // Parameters norm 

   void test_calculate_parameters_norm(void);   

   // Multilayer perceptron architecture outputs

   void test_calculate_outputs(void);

   void test_calculate_Jacobian(void);
   void test_calculate_Hessian_form(void);

   void test_calculate_parameters_Jacobian(void);
   void test_calculate_parameters_Hessian_form(void);

   // PerceptronLayer combination combination

   void test_calculate_layer_combination_combination(void);
   void test_calculate_layer_combination_combination_Jacobian(void);

   // Interlayer combination combination

   void test_calculate_interlayer_combination_combination(void);
   void test_calculate_interlayer_combination_combination_Jacobian(void);

   // Forward propagation

   void test_calculate_layers_combination(void);

   void test_calculate_layers_combination_Jacobian(void);
   void test_calculate_layers_combination_parameters_Jacobian(void);
   void test_calculate_perceptrons_combination_parameters_gradient(void);

   void test_calculate_layers_activation(void);
   void test_calculate_layers_activation_derivative(void);
   void test_calculate_layers_activation_second_derivative(void);

   void test_calculate_first_order_forward_propagation(void);
   void test_calculate_second_order_forward_propagation(void);
 
   void test_calculate_layers_Jacobian(void);
   void test_calculate_layers_Hessian_form(void);

   void test_calculate_output_layers_delta(void);
   void test_calculate_output_interlayers_Delta(void);

   void test_calculate_interlayers_combination_combination_Jacobian(void);

   // Expression methods

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
