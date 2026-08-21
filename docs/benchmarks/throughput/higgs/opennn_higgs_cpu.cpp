// OpenNN CPU HIGGS dense benchmark.
//
// Modes:
//   opennn_higgs_cpu train <train_csv> <test_csv> [epochs] [batch] [hidden] [hidden_layers] [activation] [warmup_epochs]
//   opennn_higgs_cpu infer <test_csv> [reps] [batch[,batch...]] [hidden] [hidden_layers] [activation]
//
// A comma-separated batch list is measured in one process. That matters on a
// laptop that drifts ten per cent over a sweep: the batch sizes have to share
// one load and one thermal window if they are to be comparable with each other.

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/core/configuration.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/training_strategy/training_strategy.h"

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

// Parsing the 500k-row split costs about 1.4 s of a 9 s run - measured, 8.891
// against 7.503 with the cache warm. Small, but it is 1.4 s of full-machine
// work immediately before a measurement, so it is worth not doing twice. The
// parsed floats are cached beside the CSV, the way the Python drivers already
// cache theirs as .npy, and re-read whenever the cache is newer than the CSV.
// Nothing timed changes: the cache holds exactly what the parser produced.
struct Table
{
    Index rows = 0;
    Index columns = 0;
    vector<float> values;
};

const char table_magic[8] = {'O', 'N', 'N', 'T', 'B', 'L', '0', '1'};

bool read_table_cache(const filesystem::path& cache, Table& table)
{
    ifstream file(cache, ios::binary);
    if (!file) return false;

    char magic[sizeof(table_magic)] = {};
    int64_t rows = 0;
    int64_t columns = 0;

    file.read(magic, sizeof(magic));
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&columns), sizeof(columns));

    if (!file || memcmp(magic, table_magic, sizeof(magic)) != 0 || rows <= 0 || columns <= 0)
        return false;

    table.rows = Index(rows);
    table.columns = Index(columns);
    table.values.resize(size_t(rows) * size_t(columns));
    file.read(reinterpret_cast<char*>(table.values.data()),
              streamsize(table.values.size() * sizeof(float)));

    return bool(file);
}

void write_table_cache(const filesystem::path& cache, const Table& table)
{
    ofstream file(cache, ios::binary);
    if (!file) return;                          // read-only data directory: parse next time

    const int64_t rows = int64_t(table.rows);
    const int64_t columns = int64_t(table.columns);

    file.write(table_magic, sizeof(table_magic));
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
    file.write(reinterpret_cast<const char*>(table.values.data()),
               streamsize(table.values.size() * sizeof(float)));
}

Table load_table(const string& csv_path)
{
    const filesystem::path csv(csv_path);
    const filesystem::path cache = filesystem::path(csv_path + ".bin");

    Table table;

    error_code error;
    const auto cache_time = filesystem::last_write_time(cache, error);

    if (!error && cache_time >= filesystem::last_write_time(csv)
        && read_table_cache(cache, table))
        return table;

    TabularDataset dataset(csv_path, ",", false, false);
    const MatrixR& data = dataset.get_data();

    table.rows = data.rows();
    table.columns = data.cols();
    table.values.assign(data.data(), data.data() + data.size());

    write_table_cache(cache, table);

    return table;
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
    const Table table = load_table(test_path);
    const Eigen::Map<const MatrixR> all(table.values.data(), table.rows, table.columns);
    const Index samples = table.rows;
    const Index inputs_number = table.columns - 1;
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

vector<Index> parse_batches(const string& text)
{
    vector<Index> batches;
    size_t start = 0;

    while (start <= text.size())
    {
        const size_t comma = text.find(',', start);
        const string item = text.substr(start, comma == string::npos ? string::npos : comma - start);

        if (!item.empty()) batches.push_back(Index(stoll(item)));
        if (comma == string::npos) break;
        start = comma + 1;
    }

    if (batches.empty()) batches.push_back(1024);

    return batches;
}

int train_mode(int argc, char* argv[])
{
    if (argc < 4)
    {
        cerr << "usage: opennn_higgs_cpu train <train_csv> <test_csv> [epochs] [batch[,batch...]] [hidden] [hidden_layers] [activation] [warmup_epochs]\n";
        return 2;
    }

    const string train_path = argv[2];
    const string test_path = argv[3];
    const Index epochs = argc > 4 ? Index(stoll(argv[4])) : 1;
    const vector<Index> batches = parse_batches(argc > 5 ? argv[5] : "1024");
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
    cout << "epochs=" << epochs << "\n";
    cout << "warmup_epochs=" << warmup_epochs << "\n";
    cout << "hidden=" << hidden << "\n";
    cout << "hidden_layers=" << hidden_layers << "\n";
    cout << "activation=" << activation << "\n";

    for (const Index batch : batches)
    {
        // A fresh network per rung. A batch size that inherited the previous
        // rung's weights would be training a different problem, and its
        // held-out metrics would not be comparable with the other engines',
        // which build one model per rung as well.
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

        // The warmup epochs run inside the SAME train() call as the timed ones
        // and are discarded by the callback, rather than in a train() of their
        // own. Two calls would leave the second one's setup - allocating the
        // batch arena, building the forward contexts - inside the timed window,
        // which is exactly what the PyTorch and TensorFlow drivers keep out of
        // theirs. It used to be timed as one wall-clock span over a whole
        // train() divided by the epoch count, which is a mean with the setup
        // folded in, not a median of epochs.
        vector<double> epoch_seconds;
        auto previous_mark = clock_type::now();

        adam->set_maximum_epochs(warmup_epochs + epochs);
        adam->post_epoch_callback = [&](Index epoch, float, float, NeuralNetwork*)
        {
            const auto now = clock_type::now();
            const double elapsed = chrono::duration<double>(now - previous_mark).count();
            previous_mark = now;

            if (epoch >= warmup_epochs) epoch_seconds.push_back(elapsed);
        };

        training_strategy.train();

        if (Index(epoch_seconds.size()) != epochs)
        {
            cerr << "epoch timing marks missing: " << epoch_seconds.size()
                 << " of " << epochs << "\n";
            cout << "RESULT=ERROR\n";
            return 1;
        }

        // In temporal order, before the sort: a median hides a drifting machine
        // entirely. If these fall monotonically, the run is measuring the clock.
        cout << "batch_" << batch << "_epoch_times=";
        for (size_t i = 0; i < epoch_seconds.size(); ++i)
            cout << (i ? "," : "") << epoch_seconds[i];
        cout << "\n";

        vector<double> sorted_seconds = epoch_seconds;
        sort(sorted_seconds.begin(), sorted_seconds.end());
        const double median_epoch_s = sorted_seconds[sorted_seconds.size() / 2];

        // An epoch runs whole batches only and drops the remainder. Dividing the
        // whole split by the epoch time would overstate throughput by up to one
        // batch: nothing at 1,024 rows, 1.6% at 16,384.
        const Index samples_per_epoch = (samples / batch) * batch;
        const double samples_per_sec = double(samples_per_epoch) / median_epoch_s;

        const BinaryMetrics metrics = evaluate(*network, test_path, batch);

        if (batches.size() == 1)
        {
            cout << "batch=" << batch << "\n";
            cout << "samples_per_epoch=" << samples_per_epoch << "\n";
            cout << "median_epoch_s=" << median_epoch_s << "\n";
            cout << "samples_per_sec=" << long(samples_per_sec) << "\n";
            cout << "test_samples=" << metrics.samples << "\n";
            cout << "test_accuracy=" << metrics.accuracy << "\n";
            cout << "test_log_loss=" << metrics.log_loss << "\n";
            cout << "test_roc_auc=" << metrics.auc << "\n";
        }
        else
        {
            cout << "batch_" << batch << "_samples_per_sec=" << long(samples_per_sec)
                 << " median_epoch_s=" << median_epoch_s
                 << " samples_per_epoch=" << samples_per_epoch << "\n";
            cout << "batch_" << batch << "_test_accuracy=" << metrics.accuracy
                 << " test_log_loss=" << metrics.log_loss
                 << " test_roc_auc=" << metrics.auc << "\n";
        }

        cout.flush();
    }

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
    const vector<Index> batches = parse_batches(argc > 4 ? argv[4] : "1024");
    const Index hidden = argc > 5 ? Index(stoll(argv[5])) : 1024;
    const Index hidden_layers = argc > 6 ? Index(stoll(argv[6])) : 2;
    const string activation = argc > 7 ? argv[7] : "relu";

    set_seed(42);
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Table table = load_table(test_path);
    const Index samples = table.rows;
    const Index inputs_number = table.columns - 1;
    const MatrixR inputs = Eigen::Map<const MatrixR>(table.values.data(),
                                                    table.rows,
                                                    table.columns).leftCols(inputs_number);

    auto network = make_network(Shape{inputs_number},
                                Shape{table.columns - inputs_number},
                                hidden,
                                hidden_layers,
                                activation);

    cout << "engine=opennn\n";
    cout << "mode=infer\n";
    cout << "device=cpu\n";
    cout << "reps=" << reps << "\n";
    cout << "hidden=" << hidden << "\n";
    cout << "hidden_layers=" << hidden_layers << "\n";
    cout << "activation=" << activation << "\n";

    for (const Index batch : batches)
    {
        const Index processed = (samples / batch) * batch;
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

        // In temporal order, before the sort: a median hides a drifting machine
        // entirely, and this one drifts - the first thing measured after an idle
        // gap runs in the processor's boost window and everything after it does
        // not. If these fall monotonically, the run is measuring the clock.
        cout << "batch_" << batch << "_pass_times=";
        for (size_t i = 0; i < times.size(); ++i)
            cout << (i ? "," : "") << times[i];
        cout << "\n";

        sort(times.begin(), times.end());
        const double median_pass_s = times[times.size() / 2];
        const double samples_per_sec = double(processed) / median_pass_s;

        if (batches.size() == 1)
        {
            cout << "samples=" << processed << "\n";
            cout << "batch=" << batch << "\n";
            cout << "median_pass_s=" << median_pass_s << "\n";
            cout << "samples_per_sec=" << long(samples_per_sec) << "\n";
        }
        else
        {
            cout << "batch_" << batch << "_samples_per_sec=" << long(samples_per_sec)
                 << " median_pass_s=" << median_pass_s << "\n";
        }

        cout.flush();
    }

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
