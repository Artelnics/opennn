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

#include "../opennn/unit_testing.h"
#include "../opennn/convolutional_layer.h"
#include "../opennn/numerical_differentiation.h"

namespace opennn
{

class ConvolutionalLayerTest : public UnitTesting
{

public:

   explicit ConvolutionalLayerTest();

   virtual ~ConvolutionalLayerTest();

   void test_eigen_convolution();

   // Constructor and destructor

   void test_constructor();
   void test_destructor();


   // Combinations

   void test_calculate_convolutions();

   // Activation

   void test_calculate_activations();

   void test_calculate_activations_derivatives();

  // Outputs

  void test_insert_padding();

  // Forward propagate

  void test_forward_propagate();

  //Back propagate

  void test_back_propagate();

  // Utils

  void test_memcpy_approach();

  // Unit testing

  void run_test_case();

private:

    ConvolutionalLayer convolutional_layer;

    ConvolutionalLayerForwardPropagation forward_propagation;
    ConvolutionalLayerBackPropagation back_propagation;

    NumericalDifferentiation numerical_differentiation;
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
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
