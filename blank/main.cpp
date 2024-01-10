//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <chrono>

// OpenNN includes

#include "../opennn/opennn.h"
#include <iostream>

using namespace std;
using namespace OpenNN;
using namespace Eigen;

int main()
{
   try
   {
        cout << "Blank\n";

        srand(static_cast<unsigned>(time(nullptr)));

        // Create a 2D tensor (matrix)
        Eigen::Tensor<double, 2> matrix(3, 3); // Replace the dimensions with your actual matrix size

        // Create a 1D tensor (vector)
        Eigen::Tensor<double, 1> vector(3); // Replace the size with your actual vector size

        // Fill matrix and vector with some values
        matrix.setValues({{1, 2, 3},
                          {4, 5, 6},
                          {7, 8, 9}});

        vector.setValues({2, 3, 4});

        // Divide columns of the matrix by corresponding elements of the vector
        Eigen::array<int, 2> dimensions = {0, 1}; // Along columns
/*
        auto result = matrix / vector.reshape(dimensions);
/*
        // Display the result
        std::cout << "Original Matrix:\n" << matrix << "\n\n";
        std::cout << "Vector:\n" << vector << "\n\n";
        std::cout << "Result Matrix:\n" << result << "\n";
*/
        cout << "Bye!" << endl;

        return 0;
   }
   catch (const exception& e)
   {
       cerr << e.what() << endl;

       return 1;
   }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
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
