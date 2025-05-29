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

#include "../../opennn/image_data_set.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"

using namespace opennn;

int main()
{
    try
    {   
        cout << "OpenNN. National Institute of Standards and Techonology (MNIST) Example." << endl;

        // Data set

        ImageDataSet image_data_set;

        image_data_set.set_data_path("data");

        image_data_set.read_bmp();

        // Neural network

        NeuralNetwork neural_network(NeuralNetwork::ModelType::ImageClassification,
            image_data_set.get_dimensions(DataSet::VariableUse::Input),
            { 8,4 },
            image_data_set.get_dimensions(DataSet::VariableUse::Target));

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &image_data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(512);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(10);
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
