//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>

#include "../opennn/opennn.h"

using namespace std;
using namespace opennn;
using namespace std::chrono;
using namespace Eigen;

int main()
{
    try
    {
        cout << "OpenNN. Blank." << endl;

        DataSet data_set("C:/Users/davidgonzalez/Documents/iris_plant_original.csv", ";", true, true);

        data_set.print();

        data_set.save("data_set_test.xml");

        DataSet test;

        test.load("data_set_test.xml");

        test.print();

        /*
        ImageDataSet image_data_set(0,{0,0,0},{0});

        image_data_set.set_data_path("/Users/artelnics/Desktop/Datasets/melanoma_dataset_bmp_testing");
        // image_data_set.set_data_path("/Users/artelnics/Desktop/Datasets/pepsico_resized_testing");

        image_data_set.read_bmp();

        // image_data_set.set(DataSet::SampleUse::Training);


        // image_data_set.print();

        // Neural network

        NeuralNetwork neural_network(NeuralNetwork::ModelType::ImageClassification,
                                     image_data_set.get_dimensions(DataSet::VariableUse::Input),
                                     { 3,9,9,9 },
                                     image_data_set.get_dimensions(DataSet::VariableUse::Target));


        // neural_network.print();
        // throw runtime_error("Checking the parameters of the network");

        // Training strategy
        /*
        TrainingStrategy training_strategy(&neural_network, &image_data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(1000);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(400);
        training_strategy.set_maximum_time(3600000);
        training_strategy.set_display_period(1);

        training_strategy.perform_training();

        neural_network.save("/Users/artelnics/Desktop/neural_network_save.xml");
        neural_network.load("/Users/artelnics/Desktop/neural_network_save.xml");

        // neural_network.print();
        // image_data_set.set(DataSet::SampleUse::Testing);

        // Testing analysis
        /*
        const TestingAnalysis testing_analysis(&neural_network, &image_data_set);

        testing_analysis.print_binary_classification_tests();
        // TestingAnalysis::RocAnalysis roc_analysis = testing_analysis.perform_roc_analysis();

        // cout << "Area under the curve: " << roc_analysis.area_under_curve << endl << "Roc curve:\n" << roc_analysis.roc_curve << endl;

        // cout << "Confidence limit: " << roc_analysis.confidence_limit << endl << "Optimal threshold: " << roc_analysis.optimal_threshold << endl;

        // cout << "Calculating confusion...." << endl;
        // const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
        // cout << "\nConfusion matrix:\n" << confusion << endl;
        */
        cout << "Bye!" << endl;

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
