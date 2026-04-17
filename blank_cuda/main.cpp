//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   C U D A
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <chrono>

#include "../opennn/image_dataset.h"
#include "../opennn/neural_network.h"
#include "../opennn/standard_networks.h"
#include "../opennn/training_strategy.h"
#include "../opennn/testing_analysis.h"
#include "../opennn/optimizer.h"
#include "../opennn/adaptive_moment_estimation.h"
#include "../opennn/random_utilities.h"

using namespace opennn;
using namespace std::chrono;

int main()
{
    try
    {
        cout << "OpenNN. Melanoma Cancer benchmark (refactor)." << endl;

        set_seed(42);

#ifdef OPENNN_WITH_CUDA

        // Data set

        ImageDataset image_dataset("/home/artelnics/Documents/melanoma_dataset_bmp");

        cout << "[PARITY] samples=" << image_dataset.get_samples_number()
             << " input_shape=" << image_dataset.get_shape("Input")
             << " target_shape=" << image_dataset.get_shape("Target") << endl;

        image_dataset.split_samples_random(0.8, 0.0, 0.2);

        cout << "[PARITY] train_samples=" << image_dataset.get_samples_number("Training")
             << " test_samples=" << image_dataset.get_samples_number("Testing") << endl;

        // Neural network

        ImageClassificationNetwork image_classification_network(
            image_dataset.get_shape("Input"),
            {32, 64, 16},
            image_dataset.get_shape("Target"));

        cout << "[PARITY] layers=" << image_classification_network.get_layers_number()
             << " params_size=" << image_classification_network.get_parameters_size()
             << " params_sum=" << image_classification_network.get_parameters().sum() << endl;

        // Training strategy

        TrainingStrategy training_strategy(&image_classification_network, &image_dataset);

        training_strategy.set_loss("CrossEntropy");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        training_strategy.get_loss()->set_regularization("None");

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_batch_size(16);
        adam->set_maximum_epochs(5);
        adam->set_display_period(1);

        Device::instance().set(DeviceType::Gpu);

        const auto t0 = steady_clock::now();
        training_strategy.train();
        const auto t1 = steady_clock::now();

        const double training_seconds = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;

        // Testing analysis

        TestingAnalysis testing_analysis(&image_classification_network, &image_dataset);
        testing_analysis.set_batch_size(16);

        cout << "\nConfusion matrix (CPU test):\n" << testing_analysis.calculate_confusion() << endl;
        testing_analysis.print_binary_classification_tests();

        cout << "\nTotal training time (refactor): " << training_seconds << " s" << endl;

#endif

        cout << "Bye!" << endl;

        return 0;
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
