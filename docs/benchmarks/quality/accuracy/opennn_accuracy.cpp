// OpenNN accuracy-parity benchmark on the HIGGS classification task.
//
// Trains the canonical HIGGS dense classifier (28 -> 1024 -> 1024 -> 1, ReLU
// hidden, sigmoid output, binary cross entropy, Adam, fixed epochs) on the
// shared prepared split and prints the test-set quality so the parity between
// OpenNN, PyTorch, and TensorFlow can be checked at a fixed training budget.
//
// Usage:
//   opennn_accuracy <train_csv> <test_csv> [epochs] [batch] [hidden] [hidden_layers]
//
// Reads $OPENNN_BENCH_DATA/higgs/{higgs_train.csv,higgs_test.csv} by default.
// Prints (one key=value per line):
//   test_accuracy, test_log_loss, test_roc_auc, RESULT=OK

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "opennn/adaptive_moment_estimation.h"
#include "opennn/configuration.h"
#include "opennn/dense_layer.h"
#include "opennn/forward_propagation.h"
#include "opennn/neural_network.h"
#include "opennn/random_utilities.h"
#include "opennn/tabular_dataset.h"
#include "opennn/training_strategy.h"

using namespace opennn;

namespace
{

float clamp_probability(float value)
{
    if (value < 1.0e-7f) return 1.0e-7f;
    if (value > 1.0f - 1.0e-7f) return 1.0f - 1.0e-7f;
    return value;
}

string bench_data_path(const string& leaf)
{
    const char* root = getenv("OPENNN_BENCH_DATA");
    const string base = root && *root
        ? string(root)
        : string(getenv("HOME") ? getenv("HOME") : ".") + "/opennn-benchmark-data";
    return base + "/higgs/" + leaf;
}

unique_ptr<NeuralNetwork> make_network(const Shape& input_shape,
                                            const Shape& target_shape,
                                            Index hidden,
                                            Index hidden_layers)
{
    auto network = make_unique<NeuralNetwork>();
    Shape current = input_shape;

    for (Index i = 0; i < hidden_layers; ++i)
    {
        network->add_layer(make_unique<opennn::Dense>(
            current,
            Shape{hidden},
            "ReLU",
            false,
            "higgs_dense_" + to_string(i + 1)));
        current = network->get_output_shape();
    }

    network->add_layer(make_unique<opennn::Dense>(
        current,
        target_shape,
        "Sigmoid",
        false,
        "higgs_output"));

    network->compile();
    network->set_parameters_glorot();
    return network;
}

double calculate_auc(const vector<pair<float, int>>& scored)
{
    const Index n = Index(scored.size());
    if (n == 0) return 0.0;

    Index positives = 0;
    for (const auto& item : scored)
        positives += item.second ? 1 : 0;
    const Index negatives = n - positives;
    if (positives == 0 || negatives == 0) return 0.0;

    vector<pair<float, int>> sorted = scored;
    sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    double positive_rank_sum = 0.0;
    Index i = 0;
    while (i < n)
    {
        Index j = i + 1;
        while (j < n && sorted[j].first == sorted[i].first) ++j;
        const double average_rank = (double(i + 1) + double(j)) * 0.5;
        for (Index k = i; k < j; ++k)
            if (sorted[k].second) positive_rank_sum += average_rank;
        i = j;
    }

    return (positive_rank_sum - double(positives) * double(positives + 1) * 0.5)
         / (double(positives) * double(negatives));
}

struct BinaryMetrics
{
    double accuracy = 0.0;
    double log_loss = 0.0;
    double auc = 0.0;
    Index samples = 0;
};

BinaryMetrics evaluate(NeuralNetwork& network,
                       const string& test_path,
                       Index batch)
{
    TabularDataset test_dataset(test_path, ",", false, false);
    test_dataset.set_sample_roles("Testing");
    const MatrixR& all = test_dataset.get_data();
    const Index samples = all.rows();
    const Index inputs_number = test_dataset.get_input_shape()[0];
    const Index processed = (samples / batch) * batch;
    const MatrixR inputs = all.leftCols(inputs_number);

    ForwardPropagation forward_propagation(batch, &network);
    vector<pair<float, int>> scored;
    scored.reserve(size_t(processed));

    double log_loss = 0.0;
    Index correct = 0;
    for (Index i = 0; i + batch <= samples; i += batch)
    {
        float* batch_data = const_cast<float*>(inputs.data()) + i * inputs_number;
        TensorView view(batch_data, Shape{batch, inputs_number}, Type::FP32);
        network.forward_propagate({view}, forward_propagation, false);
        const MatrixMap outputs = forward_propagation.get_outputs().as_matrix();

        for (Index r = 0; r < batch; ++r)
        {
            const float probability = clamp_probability(outputs(r, 0));
            const int label = all(i + r, inputs_number) >= 0.5f ? 1 : 0;
            const int predicted = probability >= 0.5f ? 1 : 0;
            correct += predicted == label ? 1 : 0;
            log_loss += label
                ? -log(double(probability))
                : -log(double(1.0f - probability));
            scored.emplace_back(probability, label);
        }
    }

    BinaryMetrics metrics;
    metrics.samples = processed;
    if (processed > 0)
    {
        metrics.accuracy = double(correct) / double(processed);
        metrics.log_loss = log_loss / double(processed);
        metrics.auc = calculate_auc(scored);
    }
    return metrics;
}

}

int main(int argc, char* argv[])
{
    try
    {
        const string train_path = argc > 1 ? argv[1] : bench_data_path("higgs_train.csv");
        const string test_path = argc > 2 ? argv[2] : bench_data_path("higgs_test.csv");
        const Index epochs = argc > 3 ? Index(stoll(argv[3])) : 5;
        const Index batch = argc > 4 ? Index(stoll(argv[4])) : 1024;
        const Index hidden = argc > 5 ? Index(stoll(argv[5])) : 1024;
        const Index hidden_layers = argc > 6 ? Index(stoll(argv[6])) : 2;

        set_seed(42);
        Configuration::instance().set(Device::CPU, Type::FP32);

        TabularDataset dataset(train_path, ",", false, false);
        dataset.set_sample_roles("Training");
        const Index samples = dataset.get_samples_number();

        auto network = make_network(dataset.get_input_shape(),
                                    dataset.get_target_shape(),
                                    hidden,
                                    hidden_layers);

        TrainingStrategy training_strategy(network.get(), &dataset);
        training_strategy.set_loss("CrossEntropy");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        adam->set_batch_size(batch);
        adam->set_display_period(1000000);
        adam->set_gradient_clip_norm(0.0f);
        adam->set_maximum_epochs(epochs);

        training_strategy.train();

        BinaryMetrics metrics = evaluate(*network, test_path, batch);

        cout << "engine=opennn\n";
        cout << "device=cpu\n";
        cout << "samples=" << samples << "\n";
        cout << "batch=" << batch << "\n";
        cout << "epochs=" << epochs << "\n";
        cout << "hidden=" << hidden << "\n";
        cout << "hidden_layers=" << hidden_layers << "\n";
        cout << "activation=relu\n";
        cout << "test_samples=" << metrics.samples << "\n";
        cout << "test_accuracy=" << metrics.accuracy << "\n";
        cout << "test_log_loss=" << metrics.log_loss << "\n";
        cout << "test_roc_auc=" << metrics.auc << "\n";
        cout << "RESULT=OK\n";
        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << "\n";
        cout << "RESULT=ERROR\n";
        return 1;
    }
}
