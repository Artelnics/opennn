/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   N E U R A L   N E T W O R K   T E S T   C L A S S   H E A D E R                                            */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __NEURALNETWORKTEST_H__
#define __NEURALNETWORKTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class NeuralNetworkTest : public UnitTesting
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // GENERAL CONSTRUCTOR

   explicit NeuralNetworkTest();


   // DESTRUCTOR

   virtual ~NeuralNetworkTest();

   // METHODS

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Assignment operators methods

   void test_assignment_operator();

   // Get methods

   void test_get_inputs_pointer();
   void test_get_outputs_pointer();

   void test_get_multilayer_perceptron_pointer();
   void test_get_scaling_layer_pointer();
   void test_get_unscaling_layer_pointer();   
   void test_get_bounding_layer_pointer();
   void test_get_probabilistic_layer_pointer();

   // Display warning 

  // Parameters methods

   void test_get_parameters_number();
   void test_get_parameters();   

   // Display messages

   void test_get_display();

   // SET METHODS

   void test_set();
   void test_set_default();

   // Multilayer perceptron architecture

   void test_set_network();

   // Variables

   void test_set_variables();

   // Variables statistics

   void test_set_variables_statistics();

   // Parameters

   void test_set_parameters();

   // Display messages

   void test_set_display_inputs_warning();
   void test_set_display();

   // Growing and pruning

   void test_prune_input();
   void test_prune_output();

   // Neural network initialization methods

   void test_initialize_random();

   // Parameters initialization methods

   void test_initialize_parameters();
   void test_randomize_parameters_uniform();
   void test_randomize_parameters_normal();

   // Parameters norm 

   void test_calculate_parameters_norm();

   // Output 

   void test_calculate_outputs();

   void test_calculate_Jacobian();
   void test_calculate_Jacobian_data();

   void test_calculate_parameters_Jacobian();
   void test_calculate_parameters_Jacobian_data();

   // Expression methods

   // XML expression methods

   void test_write_expression();
   void test_save_expression();

   // Hinton diagram methods

   void test_get_Hinton_diagram_XML();   
   void test_save_Hinton_diagram();

   // Serialization methods

   void test_to_XML();
   void test_from_XML();

   void test_print();

   void test_save();
   void test_load();

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
