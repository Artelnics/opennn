//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PROBABILISTICLAYERTEST_H
#define PROBABILISTICLAYERTEST_H

// Unit testing includes

#include "unit_testing.h"

class ProbabilisticLayerTest : public UnitTesting
{

public:

   explicit ProbabilisticLayerTest();

   virtual ~ProbabilisticLayerTest();

   // Constructor and destructor methods

   void test_constructor();

   // Get methods

   void test_get_biases();
   void test_get_synaptic_weights();
   void test_get_parameters();
   void test_get_decision_threshold();

   // Probabilistic layer

   void test_get_inputs_number();
   void test_get_neurons_number();

   // Output variables probabilistic postprocessing


   // Display messages

   

   // Set methods

   void test_set();
   void test_set_default();
   void test_set_biases();
   void test_set_synaptic_weights();
   void test_set_parameters();
   void test_set_decision_threshold();

   // Activation function


   void test_write_activation_function();
   void test_write_activation_function_text();
   void test_set_activation_function();


   // Display messages

   

  // Probabilistic post-processing

   void test_calculate_combinations();
   void test_calculate_activations();
   void test_calculate_activations_derivatives();
   void test_calculate_outputs();

   // Expression methods

   void test_get_layer_expression();

   // Serialization methods

   void test_to_XML();
   void test_from_XML();

   // Forward propagate

   void test_forward_propagate();

   // Write expression

   void test_write_expression();

   // Unit testing methods

   void run_test_case();

private:

   Index inputs_number;
   Index neurons_number;
   Index samples_number;

   ProbabilisticLayer probabilistic_layer;

   NumericalDifferentiation numerical_differentiation;
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
