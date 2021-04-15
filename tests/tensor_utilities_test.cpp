//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E S T I N G   A N A L Y S I S   T E S T   C L A S S                 
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensor_utilities_test.h"

TensorUtilitiesTest::TensorUtilitiesTest() : UnitTesting()
{
}


TensorUtilitiesTest::~TensorUtilitiesTest()
{
}


void TensorUtilitiesTest::test_calculate_rank()
{
    cout << "test_calculate_rank\n";

    Tensor<type, 1> vector;
    Tensor<Index, 1> rank;

    // Test

    vector.resize(3);
    vector.setValues({4,2,3});

    rank = calculate_rank_greater(vector);

    cout << vector << endl << endl;
    cout << rank << endl;
}


void TensorUtilitiesTest::run_test_case()
{
   cout << "Running tensor utilities test case...\n";

   test_calculate_rank();

   cout << "End of tensor utilities test case.\n\n";
}


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
