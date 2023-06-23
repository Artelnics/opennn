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
        cout << "OpenNN. Conv2D Example." << endl;
/*
        const Index batch_samples_number = 1;

        const Index inputs_channels_number = 3;
        const Index inputs_rows_number = 8;
        const Index inputs_columns_number = 8;

        const Index kernels_number = 4;
        const Index kernels_channels_number = inputs_channels_number;
        const Index kernels_rows_number = 2;
        const Index kernels_columns_number = 2;

        Tensor<Index, 1> input_variables_dimensions(3);
        input_variables_dimensions.setValues({inputs_rows_number, inputs_columns_number, inputs_channels_number});

        Tensor<Index, 1> kernels_dimensions(4);
        kernels_dimensions.setValues({kernels_number, kernels_rows_number, kernels_columns_number, kernels_channels_number});

        Tensor<type, 4> inputs(batch_samples_number, inputs_rows_number, inputs_columns_number, inputs_channels_number);
        Tensor<type, 4> kernels(kernels_number, kernels_rows_number, kernels_columns_number, kernels_channels_number);

        cout << "inputs dimensions: " << inputs.dimensions() << endl;

        cout << "kernels dimensions: " << kernels.dimensions() << endl;

        const Eigen::array<ptrdiff_t, 2> convolution_dimensions = {2, 3};

        Tensor<type, 4> convolved_image = inputs.convolve(kernels, convolution_dimensions);

        cout << "convolved_image: " << convolved_image.dimensions() << endl;
*/

        Tensor<float, 4> input(1, 3, 3, 3);
        Tensor<float, 4> kernel(3, 2, 2, 1);

        input.setConstant(1.0f);
        kernel.setConstant(2.0f);

        Eigen::array<ptrdiff_t, 4> dims({3,1,2,0});  // Specify second and third dimension for convolution.
        Tensor<float, 4> output = input.convolve(kernel, dims);

        cout << "output_dims: " << output.dimensions() << endl;

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
