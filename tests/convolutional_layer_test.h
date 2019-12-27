//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CONVOLUTIONALLAYERTEST_H
#define CONVOLUTIONALLAYERTEST_H

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class ConvolutionalLayerTest : public UnitTesting
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   explicit ConvolutionalLayerTest();

   virtual ~ConvolutionalLayerTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Get methods

   void test_get_parameters();

   void test_get_outputs_dimensions();

   void test_get_parameters_number();

   // Set methods

   void test_set();

   void test_set_parameters();

   // Combinations

   void test_calculate_image_convolution();

   void test_calculate_convolutions();

   // Activation

   void test_calculate_activations();

   void test_calculate_activations_derivatives();

  // Outputs

  void test_calculate_outputs();

  void test_insert_padding();

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
