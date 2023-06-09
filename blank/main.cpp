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
        srand(static_cast<unsigned>(time(nullptr)));

        const Index batch_samples_number = 3;

        const Index inputs_channels_number = 3;
        const Index inputs_rows_number = 28;
        const Index inputs_columns_number = 12;

        const Index kernels_number = 27;
        const Index kernels_channels_number = inputs_channels_number;
        const Index kernels_rows_number = 3;
        const Index kernels_columns_number = 3;

        Tensor<Index, 1> input_variables_dimensions(3);
        input_variables_dimensions(0) = inputs_channels_number;
        input_variables_dimensions(1) = inputs_rows_number;
        input_variables_dimensions(2) = inputs_columns_number;

        Tensor<Index, 1> kernels_dimensions(4);
        kernels_dimensions(0) = kernels_number;
        kernels_dimensions(1) = kernels_channels_number;
        kernels_dimensions(2) = kernels_rows_number;
        kernels_dimensions(3) = kernels_columns_number;

        ConvolutionalLayer convolutional_layer(input_variables_dimensions, kernels_dimensions);
        convolutional_layer.set_activation_function(ConvolutionalLayer::ActivationFunction::Linear);
        convolutional_layer.set_biases_constant(1.0);
        convolutional_layer.set_synaptic_weights_constant(1.0);

        ConvolutionalLayerForwardPropagation convolutional_layer_forward_propagation(batch_samples_number,
                                                                                     &convolutional_layer);

        Tensor<type, 4> x(batch_samples_number,
                          inputs_channels_number,
                          inputs_rows_number,
                          inputs_columns_number);

        x.setConstant(type(1));

        bool switch_train = false;

        convolutional_layer.forward_propagate(x.data(),
                                              input_variables_dimensions,
                                              &convolutional_layer_forward_propagation,
                                              switch_train);


    convolutional_layer_forward_propagation.print();
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
