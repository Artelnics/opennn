//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B E R T   S S T - 2   E X A M P L E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

//   Fine-tunes a pretrained BERT (bert-base-uncased) for binary sentiment on SST-2.
//   The pretrained factory downloads and loads the matching architecture, weights,
//   and WordPiece vocabulary.
//
//   usage: bert [sst2.txt] [model_dir]
//     sst2.txt    text<TAB>label file (default: bundled ../data/bert/sst2.txt)
//     model_dir   pretrained model cache (default: ../data/bert)

#include <iostream>
#include <string>

#include "opennn/dataset/bert_dataset.h"
#include "opennn/models/models.h"
#include "opennn/training_strategy/training_strategy.h"
#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/testing_analysis/testing_analysis.h"
#include "opennn/core/configuration.h"

using namespace opennn;

int main(int argc, char* argv[])
{
    try
    {
        cout << "OpenNN. BERT SST-2 example." << endl;

        const string text_path = argc > 1 ? argv[1] : "../data/bert/sst2.txt";
        const string model_directory = argc > 2 ? argv[2] : "../data/bert";

        Configuration::instance().set(Device::Auto, Type::FP32);

        cout << "Loading pretrained model..." << endl;
        auto pretrained =
            BertForSequenceClassification::from_pretrained(model_directory);
        BertForSequenceClassification& model = *pretrained.model;

        BertDataset dataset(text_path, pretrained.vocabulary_path,
                            pretrained.sequence_length);

        model.set_dropout_rate(0.1f);

        TrainingStrategy training_strategy(&model, &dataset);

        auto& optimizer = dynamic_cast<AdaptiveMomentEstimation&>(
            *training_strategy.get_optimization_algorithm());
        optimizer.set_maximum_epochs(3);
        optimizer.set_batch_size(32);
        optimizer.set_learning_rate(2.0e-5f);

        training_strategy.train();

        TestingAnalysis testing_analysis(&model, &dataset);
        testing_analysis.set_batch_size(256);
        testing_analysis.print_binary_classification_tests();

        cout << "Good bye!" << endl;
        return 0;
    }
    catch (const exception& e)
    {
        cout << e.what() << endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
