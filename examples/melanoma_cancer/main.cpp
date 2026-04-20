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
#include "../../opennn/optimizer.h"
#include "../../opennn/adaptive_moment_estimation.h"

using namespace opennn;

int main()
{
    try
    {   
        cout << "OpenNN. Melanoma Cancer CUDA Example." << endl;

        set_seed(42);

        #ifdef OPENNN_WITH_CUDA

        // Data set

        ImageDataset image_dataset("../data/melanoma_cancer_data");

        image_dataset.split_samples_random(0.8, 0.0, 0.2);

        // Neural network

        ImageClassificationNetwork image_classification_network(
            image_dataset.get_shape("Input"),
            { 32, 64, 16 },
            image_dataset.get_shape("Target"));

        // Training strategy

        TrainingStrategy training_strategy(&image_classification_network, &image_dataset);

        training_strategy.set_loss("CrossEntropy");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        training_strategy.get_loss()->set_regularization("None");

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_display_period(10);
        adam->set_batch_size(16);
        adam->set_maximum_epochs(50);

        Device::instance().set(DeviceType::Gpu);

        training_strategy.train();


        // Testing analysis

        cout << "Calculating Binary classification tests..." << endl;
        TestingAnalysis testing_analysis(&image_classification_network, &image_dataset);
        testing_analysis.set_batch_size(16);
        cout << testing_analysis.calculate_confusion() << endl;
        testing_analysis.print_binary_classification_tests();

        #endif

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
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
