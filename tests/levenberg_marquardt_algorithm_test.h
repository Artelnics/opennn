//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E V E N B E R G   M A R Q U A R D T   A L G O R I T H M   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LEVENBERGMARQUARDTALGORITHMTEST_H
#define LEVENBERGMARQUARDTALGORITHMTEST_H

// Unit testing includes

#include "unit_testing.h"

class LevenbergMarquardtAlgorithmTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   explicit LevenbergMarquardtAlgorithmTest(); 

   virtual ~LevenbergMarquardtAlgorithmTest();   

   // Constructor and destructor methods

   void test_constructor();

   // Training methods

   void test_perform_training();

   // Training history methods

   void test_resize_training_error_history();

   // Linear algebraic equations methods

   void test_perform_Householder_QR_decomposition();

   // Unit testing methods

   void run_test_case();

private:

   DataSet data_set;

   NeuralNetwork neural_network;

   SumSquaredError sum_squared_error;

   LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm;
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

