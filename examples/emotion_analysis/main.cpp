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
        const Index heads_number = 4;

        // Data Set

        LanguageDataset language_dataset("../data/emotion_analysis.txt");

        const Index input_vocabulary_size = language_dataset.get_input_vocabulary_size();
        const Index target_vocabulary_size = language_dataset.get_target_vocabulary_size();

        const Index maximum_input_sequence_length = language_dataset.get_maximum_input_sequence_length();

        const Index targets_number = language_dataset.get_maximum_target_sequence_length();

        // Neural Network

        TextClassificationNetwork text_classification_network({input_vocabulary_size, maximum_input_sequence_length, embedding_dimension},
                                                              {heads_number},
                                                              {targets_number});

        // Training Strategy

        TrainingStrategy training_strategy(&text_classification_network, &language_dataset);
        training_strategy.get_loss()->set_regularization_method("L2");
        training_strategy.get_loss()->set_regularization_weight(type(0.001));

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_maximum_epochs(200);
        adam->set_learning_rate(type(0.0001));
        adam->set_display_period(20);

    #ifdef OPENNN_CUDA
        training_strategy.train_cuda();
    #else
        cout << "Training with CPU, it might take some time: "<< endl;
        training_strategy.train();
    #endif

        // Testing Analysis

        const TestingAnalysis testing_analysis(&text_classification_network, &language_dataset);

        cout << "Confusion matrix:\n"
             << testing_analysis.calculate_confusion() << endl;

        // Prediction

        const vector<string>& emotions = language_dataset.get_target_vocabulary();

        Tensor<string, 1> documents(1);
        documents[0] = "I feel so sad and lonely today";

        MatrixR outputs = text_classification_network.calculate_text_outputs(documents);

        Index predicted_class = 0;
        type max_value = outputs(0, 0);
        for(Index i = 1; i < outputs.cols(); i++)
        {
            if(outputs(0, i) > max_value)
            {
                max_value = outputs(0, i);
                predicted_class = i;
            }
        }

        const Index reserved_offset = 4; // [PAD], [UNK], [START], [END]

        cout << "Prediction for '" << documents[0] << "': "
             << emotions[predicted_class + reserved_offset] << " (" << max_value << ")" << endl;

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
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software Foundation.
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
