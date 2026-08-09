//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B E R T   S S T - 2   E X A M P L E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

//   Fine-tunes a pretrained BERT (bert-base-uncased) for binary sentiment on SST-2.
//   The architecture is a standard OpenNN network; only the weights are downloaded
//   (a .bin GitHub release asset) and loaded with load_parameters_binary().
//
//   usage: bert [sst2.txt] [vocab.txt] [weights.bin] [seq]
//     sst2.txt    text<TAB>label file (default: bundled ../data/bert/sst2.txt)
//     vocab.txt   WordPiece vocabulary; downloaded from the GitHub release if missing
//     weights.bin pretrained weights; downloaded from the GitHub release if missing
//     seq         sequence length; must match the weights (default 64)

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "opennn/dataset/bert_dataset.h"
#include "opennn/core/io_utilities.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/training_strategy/training_strategy.h"
#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/testing_analysis/testing_analysis.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/core/configuration.h"

using namespace opennn;

int main(int argc, char* argv[])
{
    try
    {
        const string weights_url =
            "https://github.com/Artelnics/opennn/releases/download/bert-weights-v1/bert-base-uncased-seq64.bin";
        const string vocabulary_url =
            "https://github.com/Artelnics/opennn/releases/download/bert-weights-v1/bert-base-uncased-vocab.txt";

        constexpr Index vocabulary_size = 30522;
        constexpr Index hidden_size = 768;
        constexpr Index heads_number = 12;
        constexpr Index intermediate = 3072;
        constexpr Index layers_number = 12;

        cout << "OpenNN. BERT SST-2 example." << endl;

        const string text_path       = argc > 1 ? argv[1] : "../data/bert/sst2.txt";
        const string vocab_path      = argc > 2 ? argv[2] : "../data/bert/bert-base-uncased-vocab.txt";
        const string weights_path    = argc > 3 ? argv[3] : "../data/bert/bert-base-uncased-seq64.bin";
        const Index  sequence_length = argc > 4 ? Index(stol(argv[4])) : 64;

        Configuration::instance().set(Device::Auto, Type::FP32);

        download_if_missing(vocab_path, vocabulary_url);
        download_if_missing(weights_path, weights_url);

        BertDataset dataset(text_path, vocab_path, sequence_length);
        const Index labels = dataset.get_features_number("Target");
        cout << "Samples: " << dataset.get_samples_number()
             << "  seq: " << sequence_length << "  labels: " << labels << endl;

        BertForSequenceClassification model(sequence_length, vocabulary_size, hidden_size,
                                            heads_number, intermediate, layers_number, labels);

        model.set_dropout_rate(0.1f);

        cout << "Loading pretrained weights..." << endl;
        model.load_parameters_binary(weights_path);

        TrainingStrategy training_strategy(&model, &dataset);
        training_strategy.set_loss("CrossEntropy");

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_maximum_epochs(3);
        adam->set_batch_size(32);
        adam->set_learning_rate(2.0e-5f);
        adam->set_display_period(1);

        cout << "Fine-tuning (Adam, lr=2e-5, batch=32, 3 epochs)..." << endl;
        training_strategy.train();

        TestingAnalysis testing_analysis(&model, &dataset);
        testing_analysis.set_batch_size(256);
        const MatrixI confusion = testing_analysis.calculate_confusion();

        Index correct = 0;
        for (Index i = 0; i < confusion.rows() - 1; ++i) correct += confusion(i, i);

        cout << "Confusion matrix (rows = target, cols = predicted, last row/col = totals):\n"
             << confusion << endl;
        cout << "Test accuracy: "
             << 100.0 * double(correct) / double(dataset.get_sample_indices("Testing").size())
             << " %" << endl;

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
