//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M N I S T    A P P L I C A T I O N
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
#include "../../opennn/random_utilities.h"
#include "../../opennn/dense_layer.h"

using namespace opennn;

int main()
{
    try
    {

        cout << "OpenNN. National Institute of Standards and Techonology (MNIST) Example." << endl;

        // Dataset

        ImageDataset image_dataset("../data/mnist_data");

        // Neural network

        ImageClassificationNetwork image_classification_network(image_dataset.get_shape("Input"),
            {4},
            image_dataset.get_shape("Target"));

        opennn::Dense<2>* hidden_dense = dynamic_cast<opennn::Dense<2>*>(image_classification_network.get_first("Dense2d"));
        if (hidden_dense) hidden_dense->set_dropout_rate(type(0.2));

        // Training strategy

        TrainingStrategy training_strategy(&image_classification_network, &image_dataset);

        training_strategy.set_loss("CrossEntropy");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        training_strategy.get_loss()->set_regularization("None");

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_maximum_epochs(20);
        adam->set_display_period(5);

        Configuration::instance().set(DeviceType::CPU, TrainingPrecision::Float32, InferencePrecision::Float32);

        training_strategy.train();

        // Testing analysis

        const TestingAnalysis testing_analysis(&image_classification_network, &image_dataset);

        cout << "Calculating confusion..." << endl;
        cout << "\nConfusion matrix:\n" << testing_analysis.calculate_confusion() << endl;

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
