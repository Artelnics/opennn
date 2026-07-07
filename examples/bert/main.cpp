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
//     sst2.txt    text<TAB>label file (default: bundled ../data/sst2.txt)
//     vocab.txt   WordPiece vocabulary; downloaded from the GitHub release if missing
//     weights.bin pretrained weights; downloaded from the GitHub release if missing
//     seq         sequence length; must match the weights (default 64)

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "../../opennn/bert_dataset.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/adaptive_moment_estimation.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/configuration.h"

using namespace opennn;

namespace
{
const string WEIGHTS_URL =
    "https://github.com/Artelnics/opennn/releases/download/bert-weights-v1/bert-base-uncased-seq64.bin";
const string VOCABULARY_URL =
    "https://github.com/Artelnics/opennn/releases/download/bert-weights-v1/bert-base-uncased-vocab.txt";

constexpr Index VOCABULARY_SIZE = 30522;
constexpr Index HIDDEN_SIZE     = 768;
constexpr Index HEADS_NUMBER    = 12;
constexpr Index INTERMEDIATE    = 3072;
constexpr Index LAYERS_NUMBER   = 12;

void download_if_missing(const string& path, const string& url)
{
    if (filesystem::exists(path)) return;

    cout << "Downloading " << url << " -> " << path << " ..." << endl;
    const string command = "curl -L --fail -o \"" + path + "\" " + url;
    if (system(command.c_str()) != 0 || !filesystem::exists(path))
        throw runtime_error("Download failed. Get it manually from:\n  " + url);
}

void evaluate(NeuralNetwork& model, Dataset& dataset, Index sequence_length)
{
    const vector<Index> samples = dataset.get_sample_indices("Testing");
    const vector<Index> target_columns = dataset.get_variable_indices("Target");
    const MatrixR& data = dataset.get_data();

    const Index samples_number = ssize(samples);
    if (samples_number == 0) { cout << "No testing samples." << endl; return; }

    const Index labels = ssize(target_columns);

    MatrixR outputs(samples_number, labels);
    MatrixR targets(samples_number, labels);

    const Index chunk = 256;

    for (Index start = 0; start < samples_number; start += chunk)
    {
        const Index batch = min(chunk, samples_number - start);

        vector<float> input_ids(size_t(batch * sequence_length));
        vector<float> token_type(size_t(batch * sequence_length));

        for (Index r = 0; r < batch; ++r)
        {
            const Index sample = samples[size_t(start + r)];
            for (Index c = 0; c < sequence_length; ++c)
            {
                input_ids[size_t(r * sequence_length + c)]  = data(sample, c);
                token_type[size_t(r * sequence_length + c)] = data(sample, sequence_length + c);
            }
        }

        vector<TensorView> inputs = {
            TensorView(input_ids.data(),  {batch, sequence_length}),
            TensorView(token_type.data(), {batch, sequence_length})
        };
        const MatrixR batch_outputs = model.calculate_outputs(inputs);

        for (Index r = 0; r < batch; ++r)
        {
            const Index sample = samples[size_t(start + r)];
            for (Index c = 0; c < labels; ++c)
            {
                outputs(start + r, c) = batch_outputs(r, c);
                targets(start + r, c) = data(sample, target_columns[size_t(c)]);
            }
        }
    }

    TestingAnalysis testing_analysis(&model, &dataset);
    const MatrixI confusion = testing_analysis.calculate_confusion(targets, outputs);

    const Index classes = confusion.rows() - 1;
    Index correct = 0;
    for (Index c = 0; c < classes; ++c) correct += confusion(c, c);

    cout << "Confusion matrix (rows = target, cols = predicted, last row/col = totals):\n"
         << confusion << endl;
    cout << "Test accuracy: " << 100.0 * double(correct) / double(samples_number) << " %" << endl;
}
}


int main(int argc, char* argv[])
{
    try
    {
        cout << "OpenNN. BERT SST-2 example." << endl;

        const string text_path       = argc > 1 ? argv[1] : "../data/sst2.txt";
        const string vocab_path      = argc > 2 ? argv[2] : "../data/bert-base-uncased-vocab.txt";
        const string weights_path    = argc > 3 ? argv[3] : "../data/bert-base-uncased-seq64.bin";
        const Index  sequence_length = argc > 4 ? Index(stol(argv[4])) : 64;

        Configuration::instance().set(Device::Auto, Type::FP32);   // weights .bin is FP32

        download_if_missing(vocab_path, VOCABULARY_URL);
        download_if_missing(weights_path, WEIGHTS_URL);

        // Dataset: WordPiece-tokenizes text<TAB>label into BERT input views (cached CSV).

        BertDataset dataset(text_path, vocab_path, sequence_length);
        const Index labels = dataset.get_features_number("Target");
        cout << "Samples: " << dataset.get_samples_number()
             << "  seq: " << sequence_length << "  labels: " << labels << endl;

        // Neural network: the bert-base-uncased architecture, weights from the .bin.

        BertForSequenceClassification model(sequence_length, VOCABULARY_SIZE, HIDDEN_SIZE,
                                            HEADS_NUMBER, INTERMEDIATE, LAYERS_NUMBER, labels);

        if (model.get_parameters_size() != Index(filesystem::file_size(weights_path) / sizeof(float)))
            throw runtime_error("Weights size mismatch: the .bin was exported for a different seq/labels. "
                                "Use seq=64 and a binary-label dataset, or re-export the weights.");

        cout << "Loading pretrained weights..." << endl;
        model.load_parameters_binary(weights_path);

        // Fine-tuning with Adam.

        TrainingStrategy training_strategy(&model, &dataset);
        training_strategy.set_loss("CrossEntropy");

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_maximum_epochs(3);
        adam->set_batch_size(32);
        adam->set_learning_rate(2.0e-5f);
        adam->set_display_period(1);

        cout << "Fine-tuning (Adam, lr=2e-5, batch=32, 3 epochs)..." << endl;
        training_strategy.train();

        // Testing Analysis

        evaluate(model, dataset, sequence_length);

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
