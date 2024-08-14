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

#include "../opennn/unit_testing.h"
#include "../opennn/data_set.h"
#include "../opennn/neural_network.h"
#include "../opennn/sum_squared_error.h"
#include "../opennn/conjugate_gradient.h"

namespace opennn
{

class ConjugateGradientTest : public UnitTesting 
{

public:

   explicit ConjugateGradientTest();    

   virtual ~ConjugateGradientTest();

   // Constructor and destructor methods

   void test_constructor();

   void test_destructor();

   // Training methods

   void test_calculate_PR_parameter();
   void test_calculate_FR_parameter();

   void test_calculate_FR_training_direction();
   void test_calculate_PR_training_direction();

   void test_perform_training();

   // Unit testing methods

   void run_test_case();

private:

   Index samples_number;
   Index inputs_number;
   Index outputs_number;

   DataSet data_set;

   NeuralNetwork neural_network;

   SumSquaredError sum_squared_error;

   ConjugateGradient conjugate_gradient;

   TrainingResults training_results;
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

