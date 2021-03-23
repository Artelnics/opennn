//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   T E S T   C L A S S   H E A D E R       
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NEURALNETWORKTEST_H
#define NEURALNETWORKTEST_H

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class NeuralNetworkTest : public UnitTesting
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   explicit NeuralNetworkTest();   

   virtual ~NeuralNetworkTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Display messages

   void test_get_display();

   // Appending layers

   void test_add_layer();
   void check_layer_type();

   // Get methods

   void test_has_methods();

   void test_get_inputs();
   void test_get_outputs();

   void test_get_trainable_layers();
   void test_get_layers_type_pointers();
   void test_get_layer_pointer();

   // Set methods

   void test_set();

   void test_set_names();
   void test_set_inputs_number();

   void test_set_pointers();

   void test_set_default();

   // Layers

   void test_get_layers_number();

   // Architecture

   void test_inputs_outputs_number();
   void test_get_architecture();

   // Architecture

   void test_set_network();

   // Parameters

   // Parameters methods

   void test_get_parameters();
   void test_get_trainable_layers_parameters();

   void test_set_parameters();

   // Parameters initialization methods

   void test_set_parameters_constant();
   void test_set_parameters_random();

   // Display messages

   void test_set_display_inputs_warning();
   void test_set_display();

   // Parameters norm / descriptives / histogram

   void test_calculate_parameters_norm();
   void test_calculate_parameters_descriptives();
   void test_calculate_parameters_histogram();

   void test_perturbate_parameters();

   // Output 

   void test_calculate_trainable_outputs();
   void test_calculate_outputs();

   void test_calculate_directional_inputs();
   void test_calculate_outputs_histograms();

   // Expression methods

   // XML expression methods

   void test_write_expression();
   void test_save_expression();

   // Serialization methods

   void test_to_XML();
   void test_from_XML();

   void test_print();

   void test_save();
   void test_load();

   // Forward propagation

   void test_forward_propagate();

   // Unit testing methods

   void run_test_case();
};

#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
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
