//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M O T I O N   A N A L Y S I S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// Six-class emotion classification of short English messages with a small
// transformer encoder: token and learned positional embeddings, two post-norm
// encoder blocks, masked mean pooling and a softmax output.
//
// Usage: emotion_analysis [--device auto|cpu|cuda] [--seed N] [--epochs E] [--data FILE]

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>

#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/text_dataset.h"
#include "opennn/evaluation/evaluation.h"
#include "opennn/network/network.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/network/layers/embedding_layer.h"
#include "opennn/network/layers/multihead_attention_layer.h"
#include "opennn/network/layers/normalization_layer_3d.h"
#include "opennn/network/layers/pooling_layer_3d.h"
#include "opennn/network/layers/tokenizer_layer.h"
#include "opennn/training/adam.h"
#include "opennn/training/training.h"

using namespace opennn;

namespace
{

// Stratified 80/10/10 split, which the dataset does not provide. Each class gives
// the same largest-remainder quota to testing and to validation, taken after a
// per-class Fisher-Yates shuffle with explicit modulo draws, so the split is the
// same with every standard library.

vector<SampleRole> stratified_split(const vector<Index>& classes, Index classes_number, unsigned seed)
{
    const Index samples_number = ssize(classes);

    vector<vector<Index>> members(static_cast<size_t>(classes_number));

    for (Index i = 0; i < samples_number; ++i)
        members[size_t(classes[size_t(i)])].push_back(i);

    vector<Index> quota(static_cast<size_t>(classes_number));
    vector<pair<double, Index>> remainders;

    for (Index c = 0; c < classes_number; ++c)
    {
        const double exact = 0.1 * double(members[size_t(c)].size());
        quota[size_t(c)] = Index(floor(exact));
        remainders.push_back({exact - floor(exact), c});
    }

    stable_sort(remainders.begin(), remainders.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

    const Index holdout_samples = Index(llround(0.1 * double(samples_number)));
    throw_if(holdout_samples == 0, "Emotion analysis: not enough samples for validation and testing.");
    const Index missing = holdout_samples - accumulate(quota.begin(), quota.end(), Index(0));

    for (Index k = 0; k < missing; ++k)
        ++quota[size_t(remainders[size_t(k)].second)];

    mt19937 generator(seed);
    vector<SampleRole> roles(size_t(samples_number), SampleRole::Training);

    for (Index c = 0; c < classes_number; ++c)
    {
        vector<Index>& indices = members[size_t(c)];
        const Index q = quota[size_t(c)];
        throw_if(2 * q >= ssize(indices),
                 "Emotion analysis: not enough samples in class {} for the stratified split.", c);

        for (Index i = ssize(indices) - 1; i > 0; --i)
            swap(indices[size_t(i)], indices[size_t(generator() % uint32_t(i + 1))]);

        for (Index k = 0; k < 2 * q; ++k)
            roles[size_t(indices[size_t(k)])] = k < q ? SampleRole::Testing : SampleRole::Validation;
    }

    return roles;
}

}

int main(int argc, char* argv[])
{
    try
    {
        cout << "OpenNN. Emotion analysis example." << endl;

        unsigned seed = 1;
        string device = "auto";
        string data_path = "../data/emotion_analysis/emotion_analysis.txt";
        Index maximum_epochs = 30;

        for (int i = 1; i < argc; i += 2)
        {
            const string flag = argv[i];
            const string value = i + 1 < argc ? argv[i + 1] : "";

            if (flag == "--seed" && !value.empty()) seed = unsigned(stoul(value));
            else if (flag == "--device" && !value.empty()) device = value;
            else if (flag == "--epochs" && !value.empty()) maximum_epochs = stoll(value);
            else if (flag == "--data" && !value.empty()) data_path = value;
            else throw runtime_error("Usage: emotion_analysis [--device auto|cpu|cuda] [--seed N] [--epochs E] [--data FILE]");
        }

        Configuration::instance().set(device == "cpu" ? Device::CPU : device == "cuda" ? Device::CUDA : Device::Auto,
                                      Type::FP32);

        // One "text<TAB>label" line per message. Word-level tokens, a vocabulary of at
        // most 10,000 entries including the reserved tokens, sequences of at most 64.

        TextDataset dataset({.sequence_length = 64, .maximum_vocabulary_size = 10000});
        dataset.set_storage_mode(Dataset::StorageMode::Matrix);
        dataset.set_separator(Dataset::Separator::Tab);
        dataset.set_has_header(false);
        dataset.read_txt(data_path);

        const Index samples_number = dataset.get_samples_number();
        const Index vocabulary_size = dataset.get_vocabulary_size();
        const Index sequence_length = dataset.get_sequence_length();
        const Index classes_number = dataset.get_features_number(VariableRole::Target);
        const vector<string> classes = dataset.get_variables(VariableRole::Target)[0].categories;
        throw_if(classes_number < 3 || classes_number != ssize(classes),
                 "Emotion analysis requires at least three distinct emotion labels.");

        // Class of every row from its one-hot target columns, then the stratified split
        // (with a fixed seed, so the test set is the same in every run).

        const MatrixR& data = dataset.get_data();
        vector<Index> sample_classes(static_cast<size_t>(samples_number));

        for (Index i = 0; i < samples_number; ++i)
            data.row(i).segment(sequence_length, classes_number).maxCoeff(&sample_classes[size_t(i)]);

        const vector<SampleRole> roles = stratified_split(sample_classes, classes_number, 20000);

        for (Index i = 0; i < samples_number; ++i)
            dataset.set_sample_role(i, roles[size_t(i)]);

        cout << "Training, validation, testing samples: " << dataset.get_samples_number(SampleRole::Training) << ", "
             << dataset.get_samples_number(SampleRole::Validation) << ", "
             << dataset.get_samples_number(SampleRole::Testing) << endl;

        for (Index c = 0; c < classes_number; ++c)
        {
            Index counts[3] = {0, 0, 0};

            for (Index i = 0; i < samples_number; ++i)
                if (sample_classes[size_t(i)] == c) ++counts[size_t(roles[size_t(i)])];

            cout << "  " << classes[size_t(c)] << ": " << counts[0] << ", " << counts[1] << ", " << counts[2] << endl;
        }

        // Encoder 2:4:128 with a feed-forward width of 256, assembled layer by layer.

        set_seed(seed);

        const Index blocks = 2;
        const Index heads = 4;
        const Index dimension = 128;
        const Index feed_forward = 256;
        const Shape sequence_shape{sequence_length, dimension};

        Network network;
        network.set_task(NetworkTask::TextClassification);

        Index current = network.add_layer(make_unique<Tokenizer>(Shape{sequence_length}, "tokenizer"), {-1});

        auto embedding = make_unique<Embedding>(Shape{vocabulary_size, sequence_length}, dimension, "embedding");
        embedding->set_learned_positional(true);
        embedding->set_export_valid_lengths(true);   // non-padding lengths for attention and pooling
        current = network.add_layer(std::move(embedding), {current});

        for (Index block = 1; block <= blocks; ++block)
        {
            const Index attention = network.add_layer(
                make_unique<MultiHeadAttention>(sequence_shape, heads, format("self_attention_{}", block)), {current});

            auto attention_normalization = make_unique<Normalization3d>(sequence_shape, format("attention_normalization_{}", block));
            attention_normalization->set_fuse_add(true);   // LayerNorm(x + Attention(x))
            const Index attended = network.add_layer(std::move(attention_normalization), {current, attention});

            const Index hidden = network.add_layer(
                make_unique<opennn::Dense>(sequence_shape, Shape{feed_forward}, "ReLU", BatchNormalization::No,
                                           format("feed_forward_hidden_{}", block)), {attended});

            const Index projected = network.add_layer(
                make_unique<opennn::Dense>(Shape{sequence_length, feed_forward}, Shape{dimension}, "Identity",
                                           BatchNormalization::No, format("feed_forward_output_{}", block)), {hidden});

            auto feed_forward_normalization = make_unique<Normalization3d>(sequence_shape, format("feed_forward_normalization_{}", block));
            feed_forward_normalization->set_fuse_add(true);   // LayerNorm(x + FeedForward(x))
            current = network.add_layer(std::move(feed_forward_normalization), {attended, projected});
        }

        current = network.add_layer(make_unique<Pooling3d>(sequence_shape, PoolingMethod::AveragePooling, "masked_mean_pooling"), {current});

        network.add_layer(make_unique<opennn::Dense>(Shape{dimension}, Shape{classes_number}, "Softmax", BatchNormalization::No,
                                                     "classification_layer"), {current});

        network.compile();
        network.set_parameters_glorot();

        cout << "Parameters: " << network.get_parameters_number() << endl;

        // Cross-entropy and Adam; stop after 3 epochs without validation improvement
        // and restore the parameters of the best epoch.

        Training training(&network, &dataset);
        training.set_loss("CrossEntropy");
        training.set_optimization_algorithm("Adam");

        Adam* adam = dynamic_cast<Adam*>(training.get_optimization_algorithm());
        adam->set_learning_rate(5.0e-4f);
        adam->set_batch_size(64);
        adam->set_maximum_epochs(maximum_epochs);
        adam->set_maximum_validation_failures(3);
        adam->set_restore_best(true);
        adam->set_shuffle(true);
        adam->set_bf16_first_moment(false);

        const TrainingResult training_result = training.train();

        cout << "Epochs: " << training_result.get_epochs_number() << ", restored epoch: "
             << (training_result.restored_epoch ? *training_result.restored_epoch + 1 : 0) << endl;

        // Testing: accuracy, macro F1 and the confusion matrix (rows true, columns predicted).

        const Evaluation evaluation(&network, &dataset);

        const auto [targets, outputs] = evaluation.get_targets_and_outputs("Testing");
        const MatrixI confusion = evaluation.calculate_confusion(targets, outputs);

        const double accuracy = double(confusion.topLeftCorner(classes_number, classes_number).trace()) / double(targets.rows());
        double macro_f1 = 0.0;

        for (Index c = 0; c < classes_number; ++c)
        {
            const double precision = confusion(classes_number, c) > 0 ? double(confusion(c, c)) / double(confusion(classes_number, c)) : 0.0;
            const double recall = confusion(c, classes_number) > 0 ? double(confusion(c, c)) / double(confusion(c, classes_number)) : 0.0;

            macro_f1 += (precision + recall > 0.0 ? 2.0 * precision * recall / (precision + recall) : 0.0) / double(classes_number);
        }

        cout << "Test accuracy: " << accuracy << "  macro F1: " << macro_f1 << endl
             << "Confusion matrix:" << endl
             << confusion.topLeftCorner(classes_number, classes_number) << endl;

        // Deployment: tokenize a raw message with the network's tokenizer and take the most probable class.

        const auto* tokenizer = dynamic_cast<const Tokenizer*>(network.get_layer("tokenizer").get())->get_tokenizer();

        const string message = "I feel so sad and lonely today";
        const vector<Index> ids = tokenizer->encode_sequence(string_view(message), sequence_length);

        MatrixR inputs = MatrixR::Zero(1, sequence_length);

        for (Index j = 0; j < min(ssize(ids), sequence_length); ++j)
            inputs(0, j) = float(ids[size_t(j)]);

        const MatrixR probabilities = network.calculate_outputs(inputs);

        Index predicted_class = 0;
        probabilities.row(0).maxCoeff(&predicted_class);

        cout << "Prediction for '" << message << "': " << classes[size_t(predicted_class)]
             << " (" << probabilities(0, predicted_class) << ')' << endl;

        cout << "Good bye!" << endl;

        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
