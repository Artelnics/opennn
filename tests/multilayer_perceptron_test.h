/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M U L T I L A Y E R   P E R C E P T R O N   T E S T   C L A S S   H E A D E R                              */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
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

   explicit MultilayerPerceptronTest();


   // DESTRUCTOR

   virtual ~MultilayerPerceptronTest();

   // METHODS

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Assignment operators methods

   void test_assignment_operator();

   // Get methods

   // Multilayer perceptron architecture

   void test_get_inputs_number();

   void test_get_layers_number();
   void test_count_layers_perceptrons_number();

   void test_get_outputs_number();

   void test_get_perceptrons_number();
   void test_count_cumulative_perceptrons_number();

   void test_get_layers();
   void test_get_layer();

   // Multilayer Perceptron parameters

   void test_get_layers_parameters_number();

   void test_get_parameters_number();
   void test_get_cumulative_parameters_number();

   void test_get_parameters();   

   void test_get_layers_biases();

   void test_get_layers_synaptic_weights();

   void test_get_layers_parameters();

   void test_get_parameter_indices();
   void test_get_parameters_indices();

   void test_get_layers_activation_function();
   void test_get_layers_activation_function_name();

   // Display messages

   void test_get_display();

   // SET METHODS

   void test_set();
   void test_set_default();

   // Multilayer perceptron architecture

   void test_set_layers_perceptrons_number();

   // Multilayer perceptron parameters

   void test_set_parameters();

   void test_set_layers_biases();

   void test_set_layers_synaptic_weights();

   void test_set_layers_parameters();

   // Activation functions

   void test_set_layers_activation_function();

   // Display messages

   void test_set_display();

   // Check methods

   void test_is_empty();

   // Growing and pruning

   void test_grow_input();
   void test_grow_layer();

   void test_prune_input();
   void test_prune_output();

   void test_prune_layer();

   // Initialization methods

   void test_initialize_random();

   // Parameters initialization methods

   void test_initialize_parameters();

   void test_initialize_biases();    
   void test_initialize_synaptic_weights();
   void test_randomize_parameters_uniform();
   void test_randomize_parameters_normal();

   // Parameters norm 

   void test_calculate_parameters_norm();   

   // Multilayer perceptron architecture outputs

   void test_calculate_outputs();

   void test_calculate_Jacobian();
   void test_calculate_Hessian();

   void test_calculate_parameters_Jacobian();
   void test_calculate_parameters_Hessian();

   // Combination-combination

   void test_calculate_layer_combination_combination();
   void test_calculate_layer_combination_combination_Jacobian();

   // Interlayer combination combination

   void test_calculate_interlayer_combination_combination();
   void test_calculate_interlayer_combination_combination_Jacobian();

   // Forward propagation

   void test_calculate_layers_combination();

   void test_calculate_layers_combination_Jacobian();
   void test_calculate_layers_combination_parameters_Jacobian();
   void test_calculate_perceptrons_combination_parameters_gradient();

   void test_calculate_layers_activation();
   void test_calculate_layers_activation_derivatives();
   void test_calculate_layers_activation_second_derivatives();

   void test_calculate_first_order_forward_propagation();
   void test_calculate_second_order_forward_propagation();
 
   void test_calculate_layers_Jacobian();
   void test_calculate_layers_Hessian();

   void test_calculate_output_layers_delta();
   void test_calculate_output_interlayers_Delta();

   void test_calculate_interlayers_combination_combination_Jacobian();

   // Expression methods

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
