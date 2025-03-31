//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>

#include "../opennn/opennn.h"

using namespace std;
using namespace opennn;
using namespace std::chrono;
using namespace Eigen;

int main()
{
    try
    {
        cout << "OpenNN. Blank." << endl;

        Index batch_size = 1;
        Index seq_len = 3;
        Index embed_dim = 4;
        Index num_heads = 2;

        Tensor<type, 3> x(1,3,4);
        x(0,0,0) = 1;
        x(0,0,1) = 2;
        x(0,0,2) = 3;
        x(0,0,3) = 4;
        x(0,1,0) = 5;
        x(0,1,1) = 6;
        x(0,1,2) = 7;
        x(0,1,3) = 8;
        x(0,2,0) = 9;
        x(0,2,1) = 10;
        x(0,2,2) = 11;
        x(0,2,3) = 12;

        NeuralNetwork nn;

        nn.add_layer(make_unique<MultiHeadAttention>(seq_len,seq_len,embed_dim, num_heads, false));

        nn.set_parameters_constant(0.1);



        Tensor<type,3 > outputs = nn.calculate_outputs(x);

        cout << "Outputs:\n" << outputs << endl;


        cout << "Bye!" << endl;

        cout << nn.get_parameters()<<endl;

        return 0;
    }
    catch(const exception& e)
    {
        cout << e.what() << endl;

        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2025 Artificial Intelligence Techniques SL
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
