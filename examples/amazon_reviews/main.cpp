//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A M A Z O N   R E V I E W S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cstring>
#include <iostream>

#include "opennn/dataset/language_dataset.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/training_strategy/training_strategy.h"
#include "opennn/testing_analysis/testing_analysis.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/core/random_utilities.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Amazon reviews example." << endl;

        Configuration::instance().set(Device::Auto, Type::Auto);

        const Index embedding_dimension = 64;
        const Index heads_number = 4;

        LanguageDataset language_dataset("../data/amazon_reviews/amazon_cells_labelled.txt");
        const Index input_vocabulary_size = language_dataset.get_input_vocabulary_size();
        const Index input_sequence_length = language_dataset.get_maximum_input_sequence_length();
        const Index targets_number = language_dataset.get_features_number("Target");

        TextClassificationNetwork text_classification_network(
            {input_vocabulary_size, input_sequence_length, embedding_dimension},
            {heads_number},
            {targets_number});

        text_classification_network.set_tokenizer(language_dataset.get_input_tokenizer().clone());

        TrainingStrategy training_strategy(&text_classification_network, &language_dataset);

        training_strategy.set_loss("CrossEntropy");
        training_strategy.get_loss()->set_regularization("L2");

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_maximum_epochs(50);
        adam->set_display_period(10);

        cout << "Training network..." << endl;
        training_strategy.train();

        TestingAnalysis testing_analysis(&text_classification_network, &language_dataset);
        cout << "Confusion Matrix:" << endl;
        cout << testing_analysis.calculate_confusion() << endl;

        Tensor<string, 1> documents(1);
        documents[0] = "This product is amazing and I love it!";
        MatrixR outputs = text_classification_network.calculate_text_outputs(documents);

        cout << "Prediction for '" << documents[0] << "': " << outputs(0,0) << endl;

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
