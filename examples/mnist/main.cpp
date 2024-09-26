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

        const Index kernel_height = 5;
        const Index kernel_width = 5;
        const Index channels = 3;
        const Index kernels_number = 3;

        const Index pool_height = 2;
        const Index pool_width = 2;

        // Data set
        
        /* //Random image data set 
        const Index samples_number = 2;
        const Index image_height = 4;
        const Index image_width = 4;
        ImageDataSet image_data_set(samples_number, image_height, image_width, channels, 2);
        image_data_set.set_image_data_random();
        */

        ImageDataSet image_data_set;
        //image_data_set.set_data_source_path("data");
        //image_data_set.set_data_source_path("C:/mnist/train");
        //image_data_set.set_data_source_path("C:/binary_mnist");
        //image_data_set.set_data_source_path("C:/melanoma_dataset_bmp");
        image_data_set.set_data_source_path("C:/melanoma_dataset_bmp_small");

        image_data_set.read_bmp();

        //image_data_set.set_training();

        image_data_set.print();

        // Neural network

        NeuralNetwork neural_network;

        ScalingLayer4D* scaling_layer = new ScalingLayer4D(image_data_set.get_input_dimensions());
        neural_network.add_layer(scaling_layer);

        //ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(image_data_set.get_input_dimensions(),
        //                                                                { kernel_height, kernel_width, channels, kernels_number });
        //neural_network.add_layer(convolutional_layer);

        //ConvolutionalLayer* convolutional_layer_2 = new ConvolutionalLayer(convolutional_layer->get_output_dimensions(),
        //                                                                  { kernel_height, kernel_width, channels, kernels_number } );
        //neural_network.add_layer(convolutional_layer_2);

        //PoolingLayer* pooling_layer = new PoolingLayer(convolutional_layer->get_output_dimensions(),
        //                                              { pool_height , pool_width } );
        //neural_network.add_layer(pooling_layer);

        //PoolingLayer* pooling_layer_2 = new PoolingLayer(pooling_layer->get_output_dimensions(),
        //                                                { pool_height , pool_width });
        //pooling_layer_2->set_pooling_method("MaxPooling");
        //neural_network.add_layer(pooling_layer_2);

        FlattenLayer* flatten_layer = new FlattenLayer(image_data_set.get_input_dimensions());
        neural_network.add_layer(flatten_layer);

        ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(flatten_layer->get_output_dimensions(),
                                                                         image_data_set.get_target_dimensions());
        neural_network.add_layer(probabilistic_layer);

        //neural_network.print();

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &image_data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(1000);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(20);
        //training_strategy.get_adaptive_moment_estimation()->set_learning_rate(type(0.02));
        training_strategy.set_display_period(1);

        training_strategy.perform_training();

        // Testing analysis
        
        const TestingAnalysis testing_analysis(&neural_network, &image_data_set);
        
        cout << "Calculating confusion...." << endl;
        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
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
