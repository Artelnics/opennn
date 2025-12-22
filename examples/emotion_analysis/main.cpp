//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M O T I O M   A N A L Y S I S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cstring>
#include <iostream>

#include "../../opennn/language_dataset.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/adaptive_moment_estimation.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Emotion analysis example." << endl;

        // Settings

        const Index embedding_dimension = 64;
        const Index heads_number = 8;

        // Data Set

        LanguageDataset language_dataset("../data/emotion_analysis_tiny.txt");

        language_dataset.print();

//        print_vector(language_dataset.get_input_vocabulary());

        const Index input_vocabulary_size = language_dataset.get_input_vocabulary_size();
        const Index target_vocabulary_size = language_dataset.get_target_vocabulary_size();

        const Index maximum_input_sequence_length = language_dataset.get_maximum_input_sequence_length();

        const Index targets_number = language_dataset.get_maximum_target_sequence_length();

        // Neural Network
/*
        TextClassificationNetwork text_classification_network({input_vocabulary_size, maximum_input_sequence_length, embedding_dimension},
                                                              {heads_number},
                                                              {targets_number});

        text_classification_network.print();

        //MeanSquaredError mean_squared_error(&text_classification_network, &language_dataset);

        //const Tensor<type, 1> analytical_gradient = mean_squared_error.calculate_gradient();
        //const Tensor<type, 1> numerical_gradient = mean_squared_error.calculate_numerical_gradient();
        //const bool gradients_are_equal = are_equal(analytical_gradient, numerical_gradient, type(1.0e-3));

        //if (gradients_are_equal)
        //{
        //    cout << "SUCCESS: The analytical and numerical gradients are equal.\n";
        //}
        //else
        //{
        //    cout << "FAILURE: The analytical and numerical gradients DO NOT match.\n";
        //}

        // Training Strategy

        TrainingStrategy training_strategy(&text_classification_network, &language_dataset);

        AdaptiveMomentEstimation* adam = static_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_display_period(10);
        adam->set_maximum_epochs_number(100);
        //adam->set_batch_size(100);

        training_strategy.train();

        // Testing Analysis

        const TestingAnalysis testing_analysis(&text_classification_network, &language_dataset);

        cout << "Confusion matrix:\n"
             << testing_analysis.calculate_confusion() << endl;
*/
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
