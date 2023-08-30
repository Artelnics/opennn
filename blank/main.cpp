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

using namespace opennn;
using namespace std;

int main(int argc, char *argv[])
{
   try
   {
        cout << "Blank\n";

        VGG16 vgg16;

        const Index batch_samples_number = 1;

        const Index inputs_rows_number = 224;
        const Index inputs_columns_number = 224;
        const Index inputs_channels_number = 3;

        Tensor<type, 4> inputs(batch_samples_number,
                               inputs_rows_number,
                               inputs_columns_number,
                               inputs_channels_number);
//        inputs.setConstant(1.f);
        inputs.setConstant(type(5));

        vgg16.set_parameters_constant(0.01);

        const Tensor<type, 2> outputs = vgg16.calculate_outputs(inputs);

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
