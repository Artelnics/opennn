//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S U M   S Q U A R E D   E R R O R   T E S T   C L A S S   H E A D E R 
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef SUMSQUAREDERRORTEST_H
#define SUMSQUAREDERRORTEST_H

// Unit testing includes

#include "unit_testing.h"

class SumSquaredErrorTest : public UnitTesting 
{

public: 

   explicit SumSquaredErrorTest(); 

   virtual ~SumSquaredErrorTest();

   // Constructor and destructor methods

   void test_constructor();

   // Gradient methods

   void test_back_propagate_approximation_zero();
   void test_back_propagate_approximation_random();

   void test_back_propagate_binary_classification_zero();
   void test_back_propagate_binary_classification_random();

   void test_back_propagate_forecasting_zero();
   void test_back_propagate_forecasting_random();

   void test_calculate_error_gradient_lm();

   void test_back_propagate_lm_approximation_random();

   // Unit testing methods

   void run_test_case();

private:

   Index samples_number;
   Index inputs_number;
   Index neurons_number;
   Index outputs_number;

   Tensor<Index, 1> samples_indices;
   Tensor<Index, 1> input_variables_indices;
   Tensor<Index, 1> target_variables_indices;

   DataSet data_set;

   NeuralNetwork neural_network;

   SumSquaredError sum_squared_error;

   DataSetBatch batch;  

   NeuralNetworkForwardPropagation forward_propagation;

   LossIndexBackPropagation back_propagation;

   LossIndexBackPropagationLM back_propagation_lm;

   Tensor<type, 1> gradient_numerical_differentiation;
   Tensor<type, 2> jacobian_numerical_differentiation;

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
