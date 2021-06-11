//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z E D   S Q U A R E D   E R R O R   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL (Artelnics)                     
//   artelnics@artelnics.com

#ifndef NORMALIZEDSQUAREDERRORTEST_H
#define NORMALIZEDSQUAREDERRORTEST_H

// Unit testing includes

#include "unit_testing.h"

class NormalizedSquaredErrorTest : public UnitTesting 
{

public:

   explicit NormalizedSquaredErrorTest(); 

   virtual ~NormalizedSquaredErrorTest();

   // Constructor and destructor methods

   void test_constructor();

   // Normalization coefficient

   void test_calculate_normalization_coefficient();

   void test_calculate_training_normalization_coefficient();   
   void test_calculate_selection_normalization_coefficient();   

   // Error methods

   void test_calculate_error();

   void test_calculate_error_gradient();









   // Unit testing methods

   void run_test_case();

private:

   DataSet data_set;

   NeuralNetwork neural_network;

   NormalizedSquaredError normalized_squared_error;

   DataSetBatch batch;

   NeuralNetworkForwardPropagation forward_propagation;

   LossIndexBackPropagation back_propagation;

   LossIndexBackPropagationLM back_propagation_lm;
};


#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques SL.
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
