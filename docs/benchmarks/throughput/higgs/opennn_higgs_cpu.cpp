// OpenNN CPU HIGGS dense benchmark.
//
// Modes:
//   opennn_higgs_cpu train <train_csv> <test_csv> [epochs] [batch] [hidden] [hidden_layers] [activation] [warmup_epochs]
//   opennn_higgs_cpu infer <test_csv> [reps] [batch] [hidden] [hidden_layers] [activation]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
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
using clock_type = chrono::steady_clock;

namespace
{

float clamp_probability(float value)
{
    if (value < 1.0e-7f) return 1.0e-7f;
    if (value > 1.0f - 1.0e-7f) return 1.0f - 1.0e-7f;
    return value;
}

unique_ptr<NeuralNetwork> make_network(const Shape& input_shape,
                                            const Shape& target_shape,
                                            Index hidden,
                                            Index hidden_layers,
                                            const string& activation)
{
    auto network = make_unique<NeuralNetwork>();
    Shape current = input_shape;
    const string hidden_activation = (activation == "relu" || activation == "ReLU")
        ? "ReLU"
        : "Tanh";

    for (Index i = 0; i < hidden_layers; ++i)
    {
        network->add_layer(make_unique<opennn::Dense>(
            current,
            Shape{hidden},
            hidden_activation,
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

struct BinaryMetrics
{
    double accuracy = 0.0;
    double log_loss = 0.0;
    double auc = 0.0;
    Index samples = 0;
};

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

int train_mode(int argc, char* argv[])
{
    if (argc < 4)
    {
        cerr << "usage: opennn_higgs_cpu train <train_csv> <test_csv> [epochs] [batch] [hidden] [hidden_layers] [activation] [warmup_epochs]\n";
        return 2;
    }

    const string train_path = argv[2];
    const string test_path = argv[3];
    const Index epochs = argc > 4 ? Index(stoll(argv[4])) : 1;
    const Index batch = argc > 5 ? Index(stoll(argv[5])) : 1024;
    const Index hidden = argc > 6 ? Index(stoll(argv[6])) : 1024;
    const Index hidden_layers = argc > 7 ? Index(stoll(argv[7])) : 2;
    const string activation = argc > 8 ? argv[8] : "relu";
    const Index warmup_epochs = argc > 9 ? Index(stoll(argv[9])) : 0;

    set_seed(42);
    Configuration::instance().set(Device::CPU, Type::FP32);

    TabularDataset dataset(train_path, ",", false, false);
    dataset.set_sample_roles("Training");
    const Index samples = dataset.get_samples_number();

    cout << "engine=opennn\n";
    cout << "mode=train\n";
    cout << "device=cpu\n";
    cout << "samples=" << samples << "\n";
    cout << "batch=" << batch << "\n";
    cout << "epochs=" << epochs << "\n";
    cout << "hidden=" << hidden << "\n";
    cout << "hidden_layers=" << hidden_layers << "\n";
    cout << "activation=" << activation << "\n";

    auto network = make_network(dataset.get_input_shape(),
                                dataset.get_target_shape(),
                                hidden,
                                hidden_layers,
                                activation);

    TrainingStrategy training_strategy(network.get(), &dataset);
    training_strategy.set_loss("CrossEntropy");
    training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

    auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
        training_strategy.get_optimization_algorithm());
    adam->set_batch_size(batch);
    adam->set_display_period(1000000);
    adam->set_gradient_clip_norm(0.0f);

    if (warmup_epochs > 0)
    {
        adam->set_maximum_epochs(warmup_epochs);
        training_strategy.train();
    }

    adam->set_maximum_epochs(epochs);
    const auto t0 = clock_type::now();
    training_strategy.train();
    const auto t1 = clock_type::now();

    const double total_s = chrono::duration<double>(t1 - t0).count();
    const double median_epoch_s = total_s / double(epochs);
    const double samples_per_sec = double(samples) / median_epoch_s;

    BinaryMetrics metrics = evaluate(*network, test_path, batch);

    cout << "median_epoch_s=" << median_epoch_s << "\n";
    cout << "samples_per_sec=" << long(samples_per_sec) << "\n";
    cout << "test_samples=" << metrics.samples << "\n";
    cout << "test_accuracy=" << metrics.accuracy << "\n";
    cout << "test_log_loss=" << metrics.log_loss << "\n";
    cout << "test_roc_auc=" << metrics.auc << "\n";
    cout << "RESULT=OK\n";
    return 0;
}

int infer_mode(int argc, char* argv[])
{
    if (argc < 3)
    {
        cerr << "usage: opennn_higgs_cpu infer <test_csv> [reps] [batch] [hidden] [hidden_layers] [activation]\n";
        return 2;
    }

    const string test_path = argv[2];
    const Index reps = argc > 3 ? Index(stoll(argv[3])) : 10;
    const Index batch = argc > 4 ? Index(stoll(argv[4])) : 1024;
    const Index hidden = argc > 5 ? Index(stoll(argv[5])) : 1024;
    const Index hidden_layers = argc > 6 ? Index(stoll(argv[6])) : 2;
    const string activation = argc > 7 ? argv[7] : "relu";

    set_seed(42);
    Configuration::instance().set(Device::CPU, Type::FP32);

    TabularDataset dataset(test_path, ",", false, false);
    dataset.set_sample_roles("Testing");
    const MatrixR& all = dataset.get_data();
    const Index samples = dataset.get_samples_number();
    const Index inputs_number = dataset.get_input_shape()[0];
    const Index processed = (samples / batch) * batch;
    const MatrixR inputs = all.leftCols(inputs_number);

    auto network = make_network(dataset.get_input_shape(),
                                dataset.get_target_shape(),
                                hidden,
                                hidden_layers,
                                activation);
    ForwardPropagation forward_propagation(batch, network.get());

    auto run_pass = [&]()
    {
        double sink = 0.0;
        for (Index i = 0; i + batch <= samples; i += batch)
        {
            float* batch_data = const_cast<float*>(inputs.data()) + i * inputs_number;
            TensorView view(batch_data, Shape{batch, inputs_number}, Type::FP32);
            network->forward_propagate({view}, forward_propagation, false);
            const MatrixMap outputs = forward_propagation.get_outputs().as_matrix();
            sink += outputs(0, 0);
        }
        return sink;
    };

    volatile double sink = run_pass();
    sink += run_pass();

    vector<double> times;
    times.reserve(size_t(reps));
    for (Index r = 0; r < reps; ++r)
    {
        const auto t0 = clock_type::now();
        sink += run_pass();
        const auto t1 = clock_type::now();
        times.push_back(chrono::duration<double>(t1 - t0).count());
    }
    (void)sink;

    sort(times.begin(), times.end());
    const double median_pass_s = times[times.size() / 2];
    const double samples_per_sec = double(processed) / median_pass_s;

    cout << "engine=opennn\n";
    cout << "mode=infer\n";
    cout << "device=cpu\n";
    cout << "samples=" << processed << "\n";
    cout << "batch=" << batch << "\n";
    cout << "reps=" << reps << "\n";
    cout << "hidden=" << hidden << "\n";
    cout << "hidden_layers=" << hidden_layers << "\n";
    cout << "activation=" << activation << "\n";
    cout << "median_pass_s=" << median_pass_s << "\n";
    cout << "samples_per_sec=" << long(samples_per_sec) << "\n";
    cout << "RESULT=OK\n";
    return 0;
}

}

int main(int argc, char* argv[])
{
    try
    {
        if (argc < 2)
        {
            cerr << "usage: opennn_higgs_cpu <train|infer> ...\n";
            return 2;
        }

        const string mode = argv[1];
        if (mode == "train") return train_mode(argc, argv);
        if (mode == "infer") return infer_mode(argc, argv);

        cerr << "unknown mode: " << mode << "\n";
        return 2;
    }
    catch (const exception& e)
    {
        cerr << e.what() << "\n";
        cout << "RESULT=ERROR\n";
        return 1;
    }
}
