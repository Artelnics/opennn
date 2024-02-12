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

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. National Institute of Standards and Techonology (MNIST) Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        ImageDataSet image_data_set;

        image_data_set.set_data_source_path("../data/images/");

        image_data_set.scale_input_variables();

//        augmentation = false;
//        random_reflection_axis_x = false;
//        random_reflection_axis_y = false;
//        random_rotation_minimum = 0;
//        random_rotation_maximum = 0;
//        random_rescaling_minimum = 1;
//        random_rescaling_maximum = 1;
//        random_horizontal_translation = 0;
//        random_vertical_translation = 0;

        const Index target_variables_number = image_data_set.get_target_variables_number();

        const Tensor<Index, 1> samples_indices = image_data_set.get_training_samples_indices();
        const Index samples_number = samples_indices.size();

        const Tensor<Index, 1> input_variables_indices = image_data_set.get_input_variables_indices();
        const Tensor<Index, 1> target_variables_indices = image_data_set.get_target_variables_indices();

        const Tensor<Index, 1> input_variables_dimensions = image_data_set.get_input_variables_dimensions();
        const Index inputs_channels_number = input_variables_dimensions[0];
        const Index inputs_rows_number = input_variables_dimensions[1];
        const Index inputs_raw_variables_number = input_variables_dimensions[2];

        Tensor<Index, 1> convolutional_layer_inputs_dimensions(4);
        convolutional_layer_inputs_dimensions[0] = inputs_rows_number;
        convolutional_layer_inputs_dimensions[1] = inputs_raw_variables_number;
        convolutional_layer_inputs_dimensions[2] = inputs_channels_number;
        convolutional_layer_inputs_dimensions[3] = samples_number;

        const Index kernels_rows_number = 2;
        const Index kernels_raw_variables_number = 2;
        const Index kernels_number = 1;
        const Index kernels_channels_number = inputs_channels_number;

        Tensor<Index, 1> convolutional_layer_kernels_dimensions(4);
        convolutional_layer_kernels_dimensions(0) = kernels_rows_number;
        convolutional_layer_kernels_dimensions(1) = kernels_raw_variables_number;
        convolutional_layer_kernels_dimensions(2) = kernels_number;
        convolutional_layer_kernels_dimensions(3) = kernels_channels_number;

        Tensor<Index, 1> flatten_layer_inputs_dimensions(4);
        flatten_layer_inputs_dimensions(0) = inputs_rows_number-kernels_rows_number+1;
        flatten_layer_inputs_dimensions(1) = inputs_raw_variables_number-kernels_raw_variables_number+1;
        flatten_layer_inputs_dimensions(2) = kernels_number;
        flatten_layer_inputs_dimensions(3) = samples_number;

        // Neural network

        NeuralNetwork neural_network;

        ScalingLayer4D scaling_layer(input_variables_dimensions);
        neural_network.add_layer(&scaling_layer);

        ConvolutionalLayer* convolutional_layer
            = new ConvolutionalLayer(convolutional_layer_inputs_dimensions, convolutional_layer_kernels_dimensions);
        neural_network.add_layer(convolutional_layer);

        FlattenLayer flatten_layer(flatten_layer_inputs_dimensions);
        neural_network.add_layer(&flatten_layer);

        PerceptronLayer perceptron_layer(flatten_layer.get_outputs_dimensions()[0], target_variables_number);
        neural_network.add_layer(&perceptron_layer);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &image_data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::L2);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(1000);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(10000);

        training_strategy.perform_training();

        // Testing analysis

        Tensor<type, 4> inputs_4d;

        const TestingAnalysis testing_analysis(&neural_network, &image_data_set);

        Tensor<unsigned char,1> zero = image_data_set.read_bmp_image("../data/images/zero/0_1.bmp");
        Tensor<unsigned char,1> one = image_data_set.read_bmp_image("../data/images/one/1_1.bmp");

        vector<type> zero_int(zero.size()); ;
        vector<type> one_int(one.size());

        for(Index i = 0 ; i < zero.size() ; i++ )
        {
            zero_int[i]=(type)zero[i];
            one_int[i]=(type)one[i];
        }

        Tensor<type, 2> inputs(2, zero.size());
        Tensor<type, 2> outputs(2, neural_network.get_outputs_number());

        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

        outputs = neural_network.calculate_outputs(inputs);

        cout << "\nInputs:\n" << inputs << endl;

        cout << "\nOutputs:\n" << outputs << endl;

        cout << "\nConfusion matrix:\n" << confusion << endl;

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
// Copyright (C) 2005-2024 Artificial Intelligence Techniques SL
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
