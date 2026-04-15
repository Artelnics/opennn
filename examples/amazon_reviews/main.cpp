//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A M A Z O N   R E V I E W S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cstring>
#include <iostream>

#include "../../opennn/language_dataset.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/loss.h"
#include "../../opennn/adaptive_moment_estimation.h"
#include "../../opennn/random_utilities.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Amazon reviews example." << endl;

        set_seed(42);

        // Settings

        const Index embedding_dimension = 64;
        const Index heads_number = 4;

        // Dataset

        LanguageDataset language_dataset("../data/amazon_cells_labelled.txt");
        const Index input_vocabulary_size = language_dataset.get_input_vocabulary_size();
        const Index input_sequence_length = language_dataset.get_maximum_input_sequence_length();
        const Index targets_number = language_dataset.get_maximum_target_sequence_length();

        // Neural Network

        TextClassificationNetwork text_classification_network(
            {input_vocabulary_size, input_sequence_length, embedding_dimension},
            {heads_number},
            {targets_number});

        // Training Strategy

        TrainingStrategy training_strategy(&text_classification_network, &language_dataset);

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_maximum_epochs(50);
        adam->set_display_period(10);

        training_strategy.set_loss("CrossEntropy");

        Device::instance().set(DeviceType::Gpu);

        cout << "Training network..." << endl;
        training_strategy.train();
/*
        // Testing Analysis

        TestingAnalysis testing_analysis(&text_classification_network, &language_dataset);
        cout << "Confusion Matrix:" << endl;
        cout << testing_analysis.calculate_confusion() << endl;

        Tensor<string, 1> documents(1);
        documents[0] = "This product is amazing and I love it!";
        MatrixR outputs = text_classification_network.calculate_text_outputs(documents);

        cout << "Prediction for '" << documents[0] << "': " << outputs(0,0) << endl;
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
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
