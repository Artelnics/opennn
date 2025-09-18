//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A M A Z O N   R E V I E W S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

#include "../../opennn/language_dataset.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/adaptive_moment_estimation.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Amazon reviews example." << endl;

        // Data Set

        LanguageDataset language_dataset("../data/amazon_cells_labelled.txt");

        // Neural Network

        const Index embedding_dimension = 64;
        const Index neurons_number = 8;

        const Index input_vocabulary_size = language_dataset.get_input_vocabulary_size();
        const Index target_vocabulary_size = language_dataset.get_target_vocabulary_size();

        const Index input_sequence_length = language_dataset.get_input_sequence_length();
        const Index targets_number = language_dataset.get_target_sequence_length();
        const Index reserved_tokens = language_dataset.reserved_tokens.size();

        dimensions input_dimensions = {input_vocabulary_size, input_sequence_length, embedding_dimension};
        dimensions complexity_dimensions = {neurons_number};
        dimensions output_dimensions = {target_vocabulary_size - reserved_tokens};

        TextClassificationNetwork text_classification_network(
            input_dimensions,
            complexity_dimensions,
            output_dimensions
            );

        // Training Strategy

        TrainingStrategy training_strategy(&text_classification_network, &language_dataset);

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_display_period(2);
        adam->set_maximum_epochs_number(20);
        adam->set_batch_size(100);

        training_strategy.train();

        // Testing Analysis

        TestingAnalysis testing_analysis(&text_classification_network, &language_dataset);

        cout << testing_analysis.calculate_confusion() << endl;

        TestingAnalysis::RocAnalysis roc_analysis = testing_analysis.perform_roc_analysis();

        cout << "perform_roc_analysis:\n"
            << "  AUC: " << roc_analysis.area_under_curve << "\n"
            << "  Confidence Limit: " << roc_analysis.confidence_limit << "\n"
            << "  Optimal Threshold: " << roc_analysis.optimal_threshold << "\n";

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
// Copyright (C) Artificial Intelligence Techniques SL.
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
// License along with this library; if not, write to the Free Software Foundation.
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA), language_dataset.get_target_dimensions(), Dense2d::Activation::Logistic));
