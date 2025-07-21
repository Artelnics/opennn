//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V I T   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <fstream>
#include <sstream>
#include <cstring>
#include <time.h>

#include "tensors.h"
#include "../../opennn/opennn.h"

using namespace std;
using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. National Institute of Standards and Techonology (ViT) Example." << endl;

        // Data set

        
        const Index samples_number = 15000;

        const Index image_height = 32;
        const Index image_width = 32;
        const Index channel_number = 3;
        const Index targets = 3;
        
        ImageDataSet image_data_set(samples_number, { image_height, image_width, channel_number }, { targets });
        
        image_data_set.set_data_path("data_prueba");

        image_data_set.read_bmp();
        
        
        // Neural network

        const Index patch_size = 4;
        const Index embedding_dimension = 128;
        const Index hidden_dimension = 512;
        const Index heads_number = 4;
        const Index number_of_layers = 4;
        
        // Neural network
        
        VisionTransformer visiontransformer(image_height, image_width, channel_number, patch_size, targets, embedding_dimension, hidden_dimension, heads_number, number_of_layers);

        visiontransformer.set_dropout_rate(0);
        
        cout << "Total number of layers: " << visiontransformer.get_layers_number() << endl;
        cout << "Total number of parameters: " << visiontransformer.get_parameters_number() << endl;
        
        // Training strategy
        
        TrainingStrategy training_strategy(&visiontransformer, &image_data_set);
        
        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);

        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        //training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::L2);

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        
        training_strategy.get_adaptive_moment_estimation()->set_custom_learning_rate(embedding_dimension);

        training_strategy.get_adaptive_moment_estimation()->set_loss_goal(0.6);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(1000);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_time(36000);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(128);

        training_strategy.get_adaptive_moment_estimation()->set_display(true);
        training_strategy.get_adaptive_moment_estimation()->set_display_period(1);

        TrainingResults training_results = training_strategy.perform_training();
        

        
        cout << "Saving model..." << endl;
        visiontransformer.save("weights_data.xml");
        
        

        /*
        VisionTransformer visiontransformer;
        
        visiontransformer.load("weights_data.xml");

        Index prediction = visiontransformer.calculate_image_output("frog.bmp");
        cout << "Prediction : " << prediction << endl;     
        */
        

        // Testing analysis
        
        const TestingAnalysis testing_analysis(&visiontransformer, &image_data_set);
        
        cout << "Calculating confusion...." << endl;
        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
        cout << "\nConfusion matrix:\n" << confusion << endl;
        
        cout << "Bye!" << endl;

        return 0;
    }
    catch (exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2025 Artificial Intelligence Techniques SL
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