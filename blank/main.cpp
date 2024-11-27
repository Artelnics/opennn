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
using namespace opennn;


int main()
{
   try
   {
        cout << "Blank\n";

        srand(static_cast<unsigned>(time(nullptr)));
        
        Eigen::Tensor<int, 1> tensor1(5);
        Eigen::Tensor<int, 1> tensor2(5);

        // Inicializando los tensores con algunos valores
        tensor1.setValues({1, 2, 3, 4, 5});
        tensor2.setValues({1, 2, 3, 4, 5});

        Tensor<bool, 0> is_equal_tensor = (tensor1 == tensor2).all();

        if(is_equal_tensor(0))
        {
            cout << "Tensors are equal" << endl;
        }
        else
        {
            cout << "Tensors are different" << endl;
        }
        
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
