//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E S T I N G   A N A L Y S I S   T E S T   C L A S S   H E A D E R   
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TENSORUTILITIESTEST_H
#define TENSORUTILITIESTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"

class TensorUtilitiesTest : public UnitTesting
{

public:  

   explicit TensorUtilitiesTest();

   virtual ~TensorUtilitiesTest();

    void test_fill_submatrix();

    void test_calculate_rank();

   // Unit testing methods

   void run_test_case();

private:

   Tensor<type, 0> scalar;
   Tensor<type, 1> vector;
   Tensor<type, 2> matrix;


};


#endif

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the s of the GNU Lesser General Public
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
