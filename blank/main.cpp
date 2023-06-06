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
       cout << "OpenNN. Custom Convolutional Neural Network Example." << endl;

       srand(static_cast<unsigned>(time(nullptr)));

       // Data set

       Index images_number = 1;
       Index channels_number = 1;
       Index height = 1;
       Index width = 1;
       Index outputs_number = 1;

       DataSet data_set(images_number,
                        channels_number,
                        height,
                        width,
                        outputs_number);

       data_set.set_data_constant(1);

       data_set.set_training();

       Tensor<Index, 1> input_variables_dimensions = data_set.get_input_variables_dimensions();

       // Neural network

       NeuralNetwork neural_network;

//        ScalingLayer scaling_layer(input_variables_dimensions);
//        neural_network.add_layer(&scaling_layer);

       Tensor<Index, 1> kernels_dimensions(4); // rows, cols, channels, number => // number, channels, rows, cols,
       kernels_dimensions.setConstant(1);

       ConvolutionalLayer convolutional_layer(input_variables_dimensions, kernels_dimensions);

       neural_network.add_layer(&convolutional_layer);

       Tensor<Index, 1> outputs_dimensions_convolutional_layer = convolutional_layer.get_outputs_dimensions();

//       const Index flattened_dimensions_to_perceptron = outputs_dimensions_convolutional_layer(0) *
//               outputs_dimensions_convolutional_layer(1) *
//               outputs_dimensions_convolutional_layer(2);

       FlattenLayer flatten_layer(outputs_dimensions_convolutional_layer);
       neural_network.add_layer(&flatten_layer);

       cout << "outputs_dimensions_convolutional_layer: " << outputs_dimensions_convolutional_layer << endl;
       Tensor<Index, 1> flatten_layer_outputs_dimensions = flatten_layer.get_outputs_dimensions();

       cout << "flatten_layer_outputs_dimensions: " << flatten_layer_outputs_dimensions << endl;

       PerceptronLayer perceptron_layer(flatten_layer_outputs_dimensions(0), 1);
       perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Logistic);

       neural_network.add_layer(&perceptron_layer);

       neural_network.set_parameters_constant(1.0);

//       // Training strategy

       TrainingStrategy training_strategy(&neural_network, &data_set);

       training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
       training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
       training_strategy.set_maximum_epochs_number(5);
       training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
       training_strategy.get_adaptive_moment_estimation_pointer()->set_batch_samples_number(1);

       training_strategy.perform_training();

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
