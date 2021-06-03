//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E A N   S Q U A R E D   E R R O R    T E S T   C L A S S   H E A D E R       
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef MEANSQUAREDERRORTEST_H
#define MEANSQUAREDERRORTEST_H

// Unit testing includes

#include "unit_testing.h"

class MeanSquaredErrorTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   explicit MeanSquaredErrorTest();

   virtual ~MeanSquaredErrorTest();

   // Constructor and destructor methods

   void test_constructor();

   // Error methods

   void test_calculate_error();

   void test_calculate_error_gradient();

   // Squared errors methods 

   void test_calculate_squared_errors();

   void test_calculate_squared_errors_jacobian();

   // Unit testing methods

   void run_test_case();

private:

   DataSet data_set;

   NeuralNetwork neural_network;

   MeanSquaredError mean_squared_error;

   DataSetBatch batch;

   NeuralNetworkForwardPropagation forward_propagation;

   LossIndexBackPropagation back_propagation;

   LossIndexBackPropagationLM back_propagation_lm;
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
