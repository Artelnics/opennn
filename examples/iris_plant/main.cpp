//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I R I S   P L A N T   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <iostream>
#include <string>

#include "../../opennn/image_dataset.h"
#include "../../opennn/adaptive_moment_estimation.h"

#include "../../opennn/dataset.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/optimization_algorithm.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Iris Plant Example." << endl;

        // Data set

        // Dataset dataset("../data/iris_plant_original.csv", ";", true, false);

        // const Index inputs_number = dataset.get_variables_number("Input");
        // const Index targets_number = dataset.get_variables_number("Target");

        // // Neural network

        // const Index neurons_number = 6;

        // ClassificationNetwork classification_network({inputs_number}, {neurons_number}, {targets_number});

        // TrainingStrategy training_strategy(&classification_network, &dataset);

        // training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        // training_strategy.perform_training();

        // const TestingAnalysis testing_analysis(&classification_network, &dataset);

        // cout << "Confusion matrix:\n"
        //      << testing_analysis.calculate_confusion() << endl;

        ImageDataset dataset;

        dataset.set_data_path("../../mnist/data_bin");
        //dataset.set_data_path("../examples/mnist/data");

        dataset.read_bmp();

        dataset.split_samples_random(0.8, 0.0, 0.2);

        const dimensions input_dimensions  = dataset.get_dimensions("Input");
        const dimensions output_dimensions = dataset.get_dimensions("Target");

        // Neural network

        ImageClassificationNetwork neural_network(
            input_dimensions,
            { 16 },
            output_dimensions);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &dataset);
        training_strategy.set_loss_index("CrossEntropyError2d");
        // training_strategy.get_loss_index()->set_regularization_method("NoRegularization");
        training_strategy.get_loss_index()->set_regularization_method("L2");
        training_strategy.get_optimization_algorithm()->set_display_period(1);
        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_batch_size(2048);
        adam->set_maximum_epochs_number(5);

        training_strategy.perform_training();

        TestingAnalysis testing_analysis(&neural_network, &dataset);

        cout << "Calculating confusion...." << endl;
        Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
        cout << "\nConfusion matrix:\n" << confusion << endl;

        cout << "Good bye!" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cout << e.what() << endl;

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
