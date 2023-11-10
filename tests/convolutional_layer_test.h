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

class ConvolutionalLayerTest : public UnitTesting
{

public:

   explicit ConvolutionalLayerTest();

   virtual ~ConvolutionalLayerTest();

   void test_eigen_convolution();
   void test_eigen_convolution_3d();
   void test_read_bmp();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Set methods

   void test_set();

   void test_set_parameters();

   // Combinations

   void test_calculate_combinations();

   ///@move to polling
   void test_calculate_average_pooling_outputs();
   void test_calculate_max_pooling_outputs();

   // Activation

   void test_calculate_activations();

   void test_calculate_activations_derivatives();

   void test_forward_propagation();

  // Outputs

  void test_insert_padding();

  // Forward propagate

  void test_forward_propagate();

  //Back propagate

  void test_calculate_hidden_delta_perceptron_test();

  // Utils

  void test_memcpy_approach();

  // Unit testing methods

  void run_test_case();

private:

    ConvolutionalLayer convolutional_layer;

    ConvolutionalLayerForwardPropagation forward_propagation;

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
