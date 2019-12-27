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

using namespace OpenNN;


class ProbabilisticLayerTest : public UnitTesting
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   explicit ProbabilisticLayerTest();

   virtual ~ProbabilisticLayerTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Assignment operators methods

   void test_assignment_operator();

   // Get methods

   // Probabilistic layer

   void test_get_neurons_number();

   // Output variables probabilistic postprocessing


   // Display messages

   void test_get_display();

   // Set methods

   void test_set();

   void test_set_default();

   // Display messages

   void test_set_display();

  // Probabilistic post-processing

   void test_calculate_outputs();

   // Expression methods

   void test_get_layer_expression();

   // Serialization methods

   void test_to_XML();

   void test_from_XML();

   // Activations

   void test_calculate_activation_derivatives();


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
