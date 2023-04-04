//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M N I S T    A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com


#ifndef _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#endif


// System includes

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <ctime>
#include <exception>
#include <random>
#include <regex>
#include <map>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <list>
#include <vector>

// OpenNN includes

#include "../../opennn/opennn.h"
#include "../../opennn/opennn_strings.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. National Institute of Standards and Techonology (MNIST) Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set;

        data_set.set_data_file_name("../data/images/");

        data_set.read_bmp();

        data_set.scale_input_variables();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        const Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
        const Index samples_number = samples_indices.size();

        const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
        const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

        const Tensor<Index, 1> input_variables_dimensions = data_set.get_input_variables_dimensions();
        const Index inputs_channels_number = input_variables_dimensions[0];
        const Index inputs_rows_number = input_variables_dimensions[1];
        const Index inputs_columns_number = input_variables_dimensions[2];

        Tensor<Index, 1> convolutional_layer_inputs_dimensions(4);
        convolutional_layer_inputs_dimensions[0] = inputs_rows_number;
        convolutional_layer_inputs_dimensions[1] = inputs_columns_number;
        convolutional_layer_inputs_dimensions[2] = inputs_channels_number;
        convolutional_layer_inputs_dimensions[3] = samples_number;

        const Index kernels_rows_number = 1;
        const Index kernels_columns_number = 1;
        const Index kernels_number = 1;
        const Index kernels_channels_number = inputs_channels_number;

        Tensor<Index, 1> convolutional_layer_kernels_dimensions(4);
        convolutional_layer_kernels_dimensions(0) = kernels_rows_number;
        convolutional_layer_kernels_dimensions(1) = kernels_columns_number;
        convolutional_layer_kernels_dimensions(2) = kernels_number;
        convolutional_layer_kernels_dimensions(3) = kernels_channels_number;

        Tensor<Index, 1> flatten_layer_inputs_dimensions(4);
        flatten_layer_inputs_dimensions(0) = inputs_rows_number-kernels_rows_number+1;
        flatten_layer_inputs_dimensions(1) = inputs_columns_number-kernels_columns_number+1;
        flatten_layer_inputs_dimensions(2) = kernels_number;
        flatten_layer_inputs_dimensions(3) = samples_number;

        // Neural network

        NeuralNetwork neural_network;

        ScalingLayer scaling_layer(input_variables_dimensions);
        neural_network.add_layer(&scaling_layer);

        ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(convolutional_layer_inputs_dimensions, convolutional_layer_kernels_dimensions);
        neural_network.add_layer(convolutional_layer);

        FlattenLayer flatten_layer(flatten_layer_inputs_dimensions);
        neural_network.add_layer(&flatten_layer);

        PerceptronLayer perceptron_layer(input_variables_number, target_variables_number);
        neural_network.add_layer(&perceptron_layer);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        training_strategy.get_adaptive_moment_estimation_pointer()->set_batch_samples_number(1000);
        training_strategy.get_adaptive_moment_estimation_pointer()->set_maximum_epochs_number(10000);

        cout << "Before training " << endl;
        training_strategy.perform_training();

        // Testing analysis
/*
        Tensor<type, 4> inputs_4d;

        const TestingAnalysis testing_analysis(&neural_network, &data_set);

        Tensor<unsigned char,1> zero = data_set.read_bmp_image("../data/images/zero/0_1.bmp");
        Tensor<unsigned char,1> one = data_set.read_bmp_image("../data/images/one/1_1.bmp");

        vector<type> zero_int(zero.size()); ;
        vector<type> one_int(one.size());

        for(Index i = 0 ; i < zero.size() ; i++ )
        {
            zero_int[i]=(type)zero[i];
            one_int[i]=(type)one[i];
        }

        Tensor<type, 2> inputs(2, zero.size());
        Tensor<type, 2> outputs(2, neural_network.get_outputs_number());

        Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);
        Tensor<Index, 1> outputs_dimensions = get_dimensions(outputs);

        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

        outputs = neural_network.calculate_outputs(inputs.data(), inputs_dimensions);

        cout << "\nInputs:\n" << inputs << endl;

        cout << "\nOutputs:\n" << outputs << endl;

        cout << "\nConfusion matrix:\n" << confusion << endl;

*/
        cout << "Bye!" << endl;

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques SL
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
