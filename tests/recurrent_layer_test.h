//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef RECURRENTLAYERTEST_H
#define RECURRENTLAYERTEST_H

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class RecurrentLayerTest : public UnitTesting
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   explicit RecurrentLayerTest();

   virtual ~RecurrentLayerTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Assignment operators methods

   void  test_assignment_operator();

   // Inputs and perceptrons

   void test_is_empty();

   void test_get_inputs_number();
   void test_get_neurons_number();

   void test_get_perceptrons();
   void test_get_perceptron();

   // Parameters

   void test_get_biases();
   void test_get_weights();
   void test_get_recurrent_initializer();

   void test_get_parameters_number();

   void test_get_parameters();

   //void test_calculate_parameters_norm();

   //void test_get_perceptrons_parameters();


   void test_calculate_activations_derivatives();

   void test_calculate_combinations();

   void test_calculate_outputs();

   // Unit testing methods

   void run_test_case();
};


#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
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
