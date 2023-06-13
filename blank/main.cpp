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

        const Index batch_samples_number = 1;

        const Index inputs_channels_number = 3;
        const Index inputs_rows_number = 8;
        const Index inputs_columns_number = 8;

        const Index targets_number = 1;

        const Index kernels_number = 3;
        const Index kernels_channels_number = inputs_channels_number;
        const Index kernels_rows_number = 3;
        const Index kernels_columns_number = 3;

        Tensor<type, 4> x(batch_samples_number,
                          inputs_channels_number,
                          inputs_rows_number,
                          inputs_columns_number);
        x.setConstant(1);

        DataSet data_set(batch_samples_number,
                         inputs_channels_number,
                         inputs_rows_number,
                         inputs_columns_number,
                         targets_number);

        data_set.set_training();

        Tensor<Index, 1> input_variables_dimensions(3);
        input_variables_dimensions.setValues({inputs_channels_number,
                                              inputs_rows_number,
                                              inputs_columns_number});

        Tensor<Index, 1> kernels_dimensions(4);
        kernels_dimensions.setValues({kernels_number,
                                      kernels_channels_number,
                                      kernels_rows_number,
                                      kernels_columns_number});

        NeuralNetwork neural_network;

        ConvolutionalLayer convolutional_layer(input_variables_dimensions, kernels_dimensions);
        neural_network.add_layer(&convolutional_layer);
        convolutional_layer.set_activation_function(ConvolutionalLayer::ActivationFunction::Linear);

        Tensor<Index, 1> convolutional_layer_outputs_dimensions = convolutional_layer.get_outputs_dimensions();

        FlattenLayer flatten_layer(convolutional_layer_outputs_dimensions);
        neural_network.add_layer(&flatten_layer);

        Tensor<Index, 1> flatten_layer_outputs_dimensions = flatten_layer.get_outputs_dimensions();

        PerceptronLayer perceptron_layer(flatten_layer_outputs_dimensions(0), 1);
        perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Logistic);
        neural_network.add_layer(&perceptron_layer);

        neural_network.set_parameters_constant(1.0);


        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.perform_training();

//        Tensor<type, 2> inputs = data_set.get_input_data();
//        Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

//        Tensor<type, 2> outputs(neural_network.get_outputs_number(), inputs_dimensions[0]);

//        outputs = neural_network.calculate_outputs(inputs.data(), inputs_dimensions);

//        cout << "outputs: " << outputs << endl;

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
