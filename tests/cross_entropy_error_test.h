//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   T E S T   C L A S S   H E A D E R      
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CROSSENTROPYERRORTEST_H
#define CROSSENTROPYERRORTEST_H

// Unit testing includes

#include "unit_testing.h"

class CrossEntropyErrorTest : public UnitTesting
{

public:

   explicit CrossEntropyErrorTest();   

   virtual ~CrossEntropyErrorTest();

   // Error methods

    void test_calculate_error_binary_classification();
    void test_calculate_error_multiple_classification();

   void test_calculate_error();

   void test_calculate_error_gradient_binary_classification();
   void test_calculate_error_gradient_multiple_classification();

   void test_calculate_error_gradient_recurrent();
   void test_calculate_error_gradient_long_short_term_memory();
   void test_calculate_error_gradient_convolutional();

   void test_calculate_error_gradient();

   // Unit testing methods

   void run_test_case();

private:

   DataSet data_set;

   NeuralNetwork neural_network;

   CrossEntropyError cross_entropy_error;

   DataSetBatch batch;

   NeuralNetworkForwardPropagation forward_propagation;

   LossIndexBackPropagation back_propagation;

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
