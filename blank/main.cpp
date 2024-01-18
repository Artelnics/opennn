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
/*
        Tensor<type, 2> x(2, 2);
        Tensor<type, 2> y(2, 2);

        Tensor<type, 2> dy_dx(2, 2);

        x.setValues({{2, -3},{4, 2}});
        y.setValues({{-1, 3},{-5, 1}});

        // cout << x.cwiseMax(type(2)) << endl;
        // cout << x.cwiseMin(type(2)) << endl;

        cout << y.cwiseMax(y > type(0)) << endl;

        dy_dx.setConstant(type(0.2));
        dy_dx = dy_dx.cwiseMax(x > type(-2.5)).cwiseMax(x < type(2.5));

        cout << dy_dx << endl;
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
