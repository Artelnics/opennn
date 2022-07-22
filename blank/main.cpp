//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

// OpenNN includes

#include "../opennn/opennn.h"
#include "../opennn/tensor_utilities.h"

using namespace opennn;
using namespace std;
using namespace Eigen;


int main()
{
    try
    {
        cout << "Hello OpenNN!" << endl;

        const int rows = 2;
        const int cols = 4;
        const int channels = 3;
        const int batch = 2;

        const int dims[] = {rows, cols, channels, batch};
        Tensor<float,4> test(rows,cols,channels,batch);

        test.setConstant(0.);

        test.chip(0,3).chip(0,2).setConstant(1.);
        test.chip(0,3).chip(1,2).setConstant(2.);

        test.chip(1,3).chip(0,2).setConstant(3.);
        test.chip(1,3).chip(1,2).setConstant(4.);

        print_tensor(test.data(), dims);

        cout << "Goodbye!" << endl;


        return 0;
    }
    catch(const exception& e)
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
