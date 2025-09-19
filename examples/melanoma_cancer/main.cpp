//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E L A N O M A   C A N C E R   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <iostream>
#include <string>
#include <vector>
#include <exception>

#include "../../opennn/image_dataset.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/optimization_algorithm.h"
#include "../../opennn/adaptive_moment_estimation.h"

using namespace opennn;

int main()
{
    try
    {   
        cout << "OpenNN. Melanoma Cancer CUDA Example." << endl;

        #ifndef OPENNN_CUDA
            throw runtime_error("CUDA is required for this example. Enable in pch.h");
        #endif      

        // Data set

        ImageDataset image_dataset("../data/melanoma_cancer_data");

        image_dataset.split_samples_random(0.8, 0.0, 0.2);

        // Neural network

        ImageClassificationNetwork image_classification_network(
            image_dataset.get_dimensions("Input"),
            {64,128,32},
            image_dataset.get_dimensions("Target"));

        // Training strategy

        TrainingStrategy training_strategy(&image_classification_network, &image_dataset);

        training_strategy.set_loss_index("CrossEntropyError2d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        training_strategy.get_loss_index()->set_regularization_method("None");

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_display_period(1);
        adam->set_batch_size(16);
        adam->set_maximum_epochs_number(5);

        //training_strategy.train();
#ifdef OPENNN_CUDA
        training_strategy.train_cuda();
#endif

        // Testing analysis

        const TestingAnalysis testing_analysis(&image_classification_network, &image_dataset);
        testing_analysis.print_binary_classification_tests();

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
