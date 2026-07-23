//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M O T I O N   A N A L Y S I S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>

#include "opennn/text_dataset.h"
#include "opennn/standard_networks.h"
#include "opennn/training_strategy.h"
#include "opennn/testing_analysis.h"
#include "opennn/loss.h"
#include "opennn/adaptive_moment_estimation.h"

using namespace opennn;

int main(int argc, char* argv[])
{
    try
    {
        cout << "OpenNN. Emotion analysis example." << endl;

        Configuration::instance().set(Device::Auto, Type::FP32);

        // Settings

        const Index embedding_dimension = 64;
        const Index heads_number = 4;

        // Data Set

        unique_ptr<TextDataset> language_dataset =
            TextDataset::from_classification("../data/emotion_analysis/emotion_analysis.txt");

        const Index input_vocabulary_size = language_dataset->get_vocabulary_size();
        const Index maximum_input_sequence_length = language_dataset->get_sequence_length();
        const Index targets_number = language_dataset->get_features_number("Target");

        // Neural Network

        TextClassificationNetwork text_classification_network(
            {input_vocabulary_size, maximum_input_sequence_length, embedding_dimension},
            {heads_number},
            {targets_number});

        text_classification_network.set_tokenizer(language_dataset->clone_input_tokenizer());

        // Training Strategy

        TrainingStrategy training_strategy(&text_classification_network, language_dataset.get());
        training_strategy.set_loss("CrossEntropy");
        training_strategy.get_loss()->set_regularization("L2");

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        const Index maximum_epochs = argc > 1 ? Index(stol(argv[1])) : 200;
        adam->set_maximum_epochs(maximum_epochs);
        adam->set_batch_size(1000);
        adam->set_learning_rate(float(0.0001));
        adam->set_display_period(20);

        cout << "Training with "
             << (Configuration::instance().is_gpu() ? "GPU" : "CPU")
             << ", it might take some time: " << endl;
        training_strategy.train();

        // Testing Analysis

        const TestingAnalysis testing_analysis(&text_classification_network, language_dataset.get());

        cout << "Confusion matrix:\n"
             << testing_analysis.calculate_confusion() << endl;

        // Prediction

        const vector<string>& emotions = text_classification_network.get_output_variables()[0].categories;

        Tensor<string, 1> documents(1);
        documents[0] = "I feel so sad and lonely today";

        MatrixR outputs = text_classification_network.calculate_text_outputs(documents);

        Index predicted_class = 0;
        const float max_value = outputs.row(0).maxCoeff(&predicted_class);

        cout << "Prediction for '" << documents[0] << "': "
             << emotions[predicted_class] << " (" << max_value << ")" << endl;

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
