//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M O T I O N   A N A L Y S I S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>

#include "opennn/dataset/language_dataset.h"
#include "opennn/models/models.h"
#include "opennn/training_strategy/training_strategy.h"
#include "opennn/testing_analysis/testing_analysis.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Emotion analysis example." << endl;

        Configuration::instance().set(Device::Auto, Type::FP32);

        const Index embedding_dimension = 64;
        const Index heads_number = 4;

        LanguageDataset language_dataset("../data/emotion_analysis/emotion_analysis.txt");

        const Index input_vocabulary_size = language_dataset.get_input_vocabulary_size();
        const Index maximum_input_sequence_length = language_dataset.get_maximum_input_sequence_length();
        const Index targets_number = language_dataset.get_features_number("Target");

        TextClassificationNetwork text_classification_network(
            {input_vocabulary_size, maximum_input_sequence_length, embedding_dimension},
            {heads_number},
            {targets_number});

        TrainingStrategy training_strategy(&text_classification_network, &language_dataset);

        training_strategy.train();

        const TestingAnalysis testing_analysis(&text_classification_network, &language_dataset);

        testing_analysis.print_multiple_classification_tests();

        const string document = "I feel so sad and lonely today";
        const auto prediction = text_classification_network.classify(document);

        cout << "Prediction for '" << document << "': "
             << prediction.category << " (" << prediction.confidence << ')' << endl;

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
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
