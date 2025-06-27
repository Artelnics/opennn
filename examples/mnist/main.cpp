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

using namespace opennn;

int main()
{
    try
    {   
        cout << "OpenNN. National Institute of Standards and Techonology (MNIST) Example." << endl;

        // Data set

        ImageDataset image_dataset("../data");

        // Neural network

        ImageClassificationNetwork image_classification_network(image_dataset.get_dimensions(Dataset::VariableUse::Input),
            {8, 4},
            image_dataset.get_dimensions(Dataset::VariableUse::Target));

        // Training strategy

        TrainingStrategy training_strategy(&image_classification_network, &image_dataset);

        training_strategy.set_loss_index("CrossEntropyError2d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        //training_strategy.get_adaptive_moment_estimation()->set_batch_size(512);
        //training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(10);
        //training_strategy.set_display_period(1);

        training_strategy.perform_training();

        // Testing analysis

        const TestingAnalysis testing_analysis(&image_classification_network, &image_dataset);
        
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
