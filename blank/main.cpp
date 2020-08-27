//   OpenNN: Open Neural Networks Library
//   www.opennn.org
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <stdint.h>
#include <limits.h>
#include <statistics.h>
#include <regex>

// Systems Complementaries

#include <cmath>
#include <cstdlib>
#include <ostream>


// OpenNN includes

#include "../opennn/opennn.h"

using namespace OpenNN;
using namespace std;
using namespace chrono;


int main(void)
{
    try
    {
        cout << "OpenNN. Convolutional Example." << endl;


        //////////////////////////////////////////////////////////////////////////////////////
        /// MNIST EXAMPLE
        //////////////////////////////////////////////////////////////////////////////////////


        const string mnist_data_file = "D:\\opennn\\examples\\mnist\\data\\mnist_train_2.csv";

        const Index image_rows_number = 28;
        const Index image_columns_number = 28;
        const Index image_channels_number = 1;
        const Index images_number = 5145;

        const Index kernel_rows_number = 5;
        const Index kernel_columns_number = 5;
        const Index kernel_channels_number = image_channels_number;
        const Index kernels_number = 5;

        const Eigen::array<Eigen::Index, 4> four_dims = {image_rows_number,
                                                         image_columns_number,
                                                         image_channels_number,
                                                         images_number};

        const Index max_epochs = 5;

        DataSet data_set(mnist_data_file, ',', false);

        data_set.set_column_use(0, DataSet::Target);
        data_set.get_columns()(0).set_type("Categorical");

        data_set.print_summary();

        Tensor<Index, 1> inputs_dimension(3);
        inputs_dimension(0) = image_rows_number;
        inputs_dimension(1) = image_columns_number;
        inputs_dimension(2) = image_channels_number;
        data_set.set_input_variables_dimensions(inputs_dimension);

        Tensor<type, 4> kernels(kernel_rows_number,
                                kernel_columns_number,
                                kernel_channels_number,
                                kernels_number);
        kernels.setRandom();

        Tensor<type, 1> biases(kernels_number);
        biases.setRandom();

        ConvolutionalLayer convolutional_layer;
        convolutional_layer.set(data_set.get_input_data().reshape(four_dims),
                                kernels,
                                biases);

        ProbabilisticLayer probabilistic_layer;
        probabilistic_layer.set(300, 10);

        Tensor<Layer*, 1> layers(2);
        layers(0) = &convolutional_layer;
        layers(1) = &probabilistic_layer;

        NeuralNetwork neural_network(layers);

        TrainingStrategy training_strategy(&neural_network, &data_set);
        training_strategy.set_optimization_method(TrainingStrategy::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.set_loss_method(TrainingStrategy::CROSS_ENTROPY_ERROR);

        training_strategy.get_adaptive_moment_estimation_pointer()->set_maximum_epochs_number(max_epochs);

        training_strategy.perform_training();

/*
*/


/*
        //////////////////////////////////////////////////////////////////////////////////////
        /// RANDOM DATA EXAMPLE
        //////////////////////////////////////////////////////////////////////////////////////

        const Index image_rows_number = 10;
        const Index image_columns_number = 10;
        const Index image_channels_number = 3;
        const Index images_number = 100;

        const Index kernel_rows_number = 3;
        const Index kernel_columns_number = 3;
        const Index kernel_channels_number = 3;
        const Index kernels_number = 2;

        const Index max_epochs = 100;

        Tensor<type, 4> image_data(image_rows_number,
                                   image_columns_number,
                                   image_channels_number,
                                   images_number);
        image_data.setRandom();

        Tensor<type, 2> image_data_2D;
        Eigen::array<Eigen::Index, 2> two_dims = {image_rows_number * image_columns_number * image_channels_number, images_number};
        Eigen::array<Eigen::Index, 2> shuffle_dims = {1, 0};
        image_data_2D = image_data.reshape(two_dims);
        image_data_2D = image_data_2D.shuffle(shuffle_dims);

        Tensor<type, 2> image_classes(images_number, 1);
        image_classes.setConstant(1);

        DataSet data_set(image_data_2D.concatenate(image_classes, 1));
        data_set.get_columns()(image_data_2D.dimension(1)).set_type("Categorical");

        Tensor<Index, 1> inputs_dimension(3);
        inputs_dimension(0) = image_rows_number;
        inputs_dimension(1) = image_columns_number;
        inputs_dimension(2) = image_channels_number;
        data_set.set_input_variables_dimensions(inputs_dimension);
        data_set.set_column_use(image_data_2D.dimension(1), DataSet::Target);


        Tensor<type, 4> kernels(kernel_rows_number,
                                kernel_columns_number,
                                kernel_channels_number,
                                kernels_number);
        kernels.setRandom();

        Tensor<type, 1> biases(kernels_number);
        biases.setRandom();

        ConvolutionalLayer convolutional_layer;
        convolutional_layer.set(image_data, kernels, biases);

        ProbabilisticLayer probabilistic_layer;
        probabilistic_layer.set(300, 1);

        Tensor<Layer*, 1> layers(2);
        layers(0) = &convolutional_layer;
        layers(1) = &probabilistic_layer;

        NeuralNetwork neural_network(layers);


        TrainingStrategy training_strategy(&neural_network, &data_set);
        training_strategy.set_optimization_method(TrainingStrategy::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.set_loss_method(TrainingStrategy::CROSS_ENTROPY_ERROR);

        training_strategy.get_adaptive_moment_estimation_pointer()->set_maximum_epochs_number(max_epochs);

        training_strategy.perform_training();
/*
 */
        cout << "End" << endl;
        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
