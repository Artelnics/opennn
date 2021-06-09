//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N J U G A T E   G R A D I E N T   T E S T   C L A S S   H E A D E R        
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CONJUGATEGRADIENTTEST_H
#define CONJUGATEGRADIENTTEST_H

// Unit testing includes

#include "unit_testing.h"

class ConjugateGradientTest : public UnitTesting 
{

public:

   explicit ConjugateGradientTest();    

   virtual ~ConjugateGradientTest();

   // Constructor and destructor methods

   void test_constructor();

   // Get methods

   void test_get_training_direction_method();

   // Set methods

   void test_set_training_direction_method();

   // Training methods

   void test_calculate_PR_parameter();
   void test_calculate_FR_parameter();

   void test_calculate_FR_training_direction();
   void test_calculate_PR_training_direction();

   void test_perform_training();

   // Unit testing methods

   void run_test_case();

private:

   DataSet data_set;

   NeuralNetwork neural_network;

   SumSquaredError sum_squared_error;

   ConjugateGradient conjugate_gradient;

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

