//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A M A Z O N   R E V I E W S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cstring>
#include <iostream>
//#include <fstream>
//#include <sstream>
//#include <string>
//#include <time.h>

#include "../../opennn/language_dataset.h"
#include "../../opennn/standard_networks.h"
//#include "../../opennn/neural_network.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/adaptive_moment_estimation.h"
#include "../../opennn/weighted_squared_error.h"
//#include "../../opennn/mean_squared_error.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Amazon reviews example." << endl;

        // Settings

        const Index embedding_dimension = 64;
        const Index heads_number = 8;

        // Data Set

        LanguageDataset language_dataset("../data/amazon_cells_labelled.txt");

        const Index input_vocabulary_size = language_dataset.get_input_vocabulary_size();
        const Index input_sequence_length = language_dataset.get_maximum_input_sequence_length();
        const Index targets_number = language_dataset.get_maximum_target_sequence_length();

        // Neural Network
/*
        TextClassificationNetwork text_classification_network({input_vocabulary_size, input_sequence_length, embedding_dimension},
                                                              {heads_number},
                                                              {targets_number});


        text_classification_network.print();

        // Training Strategy

        TrainingStrategy training_strategy(&text_classification_network, &language_dataset);

        //training_strategy.train();

        // Testing Analysis

        TestingAnalysis testing_analysis(&text_classification_network, &language_dataset);

        cout << testing_analysis.calculate_confusion() << endl;

        TestingAnalysis::RocAnalysis roc_analysis = testing_analysis.perform_roc_analysis();

        roc_analysis.print();
*/

        // Deployment

        string document = "This is great!";


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
