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
#include <string>
#include <exception>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. National Institute of Standards and Techonology (MNIST) Example." << endl;
    
        const Index kernel_height = 1;
        const Index kernel_width = 1;
        const Index kernel_channels = 1;
        const Index kernels_number = 1;

        const Index pool_height = 1;
        const Index pool_width = 1;

        // Data set

        ImageDataSet image_data_set(3,3,3,1,3);

        image_data_set.set_image_data_random();

        //image_data_set.set_data_source_path("../data");
        //image_data_set.set_data_source_path("C:/mnist/binary");
        //image_data_set.set_data_source_path("C:/cifar10");

        //image_data_set.read_bmp();


        image_data_set.set_training();

        image_data_set.print();

        //image_data_set.print_data();

        // Neural network

        NeuralNetwork neural_network;

        ScalingLayer4D* scaling_layer = new ScalingLayer4D(image_data_set.get_input_dimensions());
        neural_network.add_layer(scaling_layer);

        //ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(image_data_set.get_input_dimensions(),
        //                                                                 { kernel_height, kernel_width, kernel_channels, kernels_number });
        //neural_network.add_layer(convolutional_layer);

        //ConvolutionalLayer* convolutional_layer_2 = new ConvolutionalLayer(convolutional_layer->get_output_dimensions(),
        //                                                                   { 1,1,kernels_number,kernels_number } );
        //neural_network.add_layer(convolutional_layer_2);

        //PoolingLayer* pooling_layer = new PoolingLayer(convolutional_layer_2->get_output_dimensions(),
        //                                                 {pool_height , pool_width} );
        //neural_network.add_layer(pooling_layer);

        FlattenLayer* flatten_layer = new FlattenLayer(image_data_set.get_input_dimensions());
        neural_network.add_layer(flatten_layer);

        ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(flatten_layer->get_output_dimensions(),
                                                                         image_data_set.get_target_dimensions());
        neural_network.add_layer(probabilistic_layer);

        //cout << neural_network.get_parameters_number() << endl;
        //neural_network.print();

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &image_data_set);
        
        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(100000);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(1);
        training_strategy.get_adaptive_moment_estimation()->set_learning_rate(type(0.02));
        training_strategy.set_display_period(1);
        
        training_strategy.perform_training();

        // Testing analysis
/*
        const TestingAnalysis testing_analysis(&neural_network, &image_data_set);
        
        cout << "Calculating confusion...." << endl;
        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
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
