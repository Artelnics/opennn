// OpenNN CPU HIGGS dense benchmark - the application code a user writes, plus
// a timer. The measurement protocol (rotation, soaking, medians over rounds)
// lives in run_higgs_cpu_sweep.py.
//
//   opennn_higgs_cpu train <train_csv> <test_csv> [epochs] [batch,...] [hidden] [layers] [activation] [warmup]
//   opennn_higgs_cpu infer <test_csv> [reps] [batch,...] [hidden] [layers] [activation]
//
// The batch sizes run inside one process so they share one data load and one
// thermal window; each prints its own batch_<B>_... lines for the runner.

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/layers/scaling_layer.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/testing_analysis/testing_analysis.h"
#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/training_strategy/training_strategy.h"

using namespace opennn;
using clock_type = chrono::steady_clock;

namespace
{

vector<Index> parse_batches(const string& text)
{
    vector<Index> batches;

    for (size_t start = 0; start < text.size();)
    {
        const size_t end = min(text.find(',', start), text.size());
        if (end > start) batches.push_back(Index(stoll(text.substr(start, end - start))));
        start = end + 1;
    }

    return batches.empty() ? vector<Index>{1024} : batches;
}

int train_mode(int argc, char* argv[])
{
    const string train_path = argv[2];
    const string test_path = argv[3];
    const Index epochs = argc > 4 ? stoll(argv[4]) : 1;
    const vector<Index> batches = parse_batches(argc > 5 ? argv[5] : "1024");
    const Index hidden = argc > 6 ? stoll(argv[6]) : 1024;
    const Index hidden_layers = argc > 7 ? stoll(argv[7]) : 2;
    const string activation = argc > 8 && string(argv[8]) != "relu" ? "Tanh" : "ReLU";
    const Index warmup_epochs = argc > 9 ? stoll(argv[9]) : 0;

    Configuration::instance().set(Device::CPU, Type::FP32);

    TabularDataset dataset(train_path, ",", false, false);
    dataset.set_sample_roles("Training");
    dataset.set_variable_scalers("None");   // prepare_higgs.py already normalized

    TabularDataset test_dataset(test_path, ",", false, false);
    test_dataset.set_sample_roles("Testing");

    const Index samples = dataset.get_samples_number();

    cout << "engine=opennn\nmode=train\ndevice=cpu\n"
         << "samples=" << samples << "\nepochs=" << epochs << "\n";

    for (const Index batch : batches)
    {
        set_seed(42);

        ClassificationNetwork network(dataset.get_input_shape(),
                                      Shape(size_t(hidden_layers), hidden),
                                      Shape{1},
                                      activation);

        TrainingStrategy training_strategy(&network, &dataset);
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

        // The strategy owns the epoch loop, so only the epochs' mean time is
        // observable from user code.
        const double epoch_s = chrono::duration<double>(t1 - t0).count() / double(epochs);

        TestingAnalysis analysis(&network, &test_dataset);
        const float accuracy = analysis.calculate_binary_classification_tests()[0];
        const float auc = analysis.perform_roc_analysis().area_under_curve;

        cout << "batch_" << batch << "_samples_per_sec=" << long(double(samples) / epoch_s)
             << " median_epoch_s=" << epoch_s << "\n";
        cout << "batch_" << batch << "_epoch_times=" << epoch_s << "\n";
        cout << "batch_" << batch << "_test_accuracy=" << accuracy
             << " test_roc_auc=" << auc << "\n";
        cout.flush();
    }

    cout << "RESULT=OK\n";
    return 0;
}

int infer_mode(int argc, char* argv[])
{
    const string test_path = argv[2];
    const Index reps = argc > 3 ? stoll(argv[3]) : 10;
    const vector<Index> batches = parse_batches(argc > 4 ? argv[4] : "1024");
    const Index hidden = argc > 5 ? stoll(argv[5]) : 1024;
    const Index hidden_layers = argc > 6 ? stoll(argv[6]) : 2;
    const string activation = argc > 7 && string(argv[7]) != "relu" ? "Tanh" : "ReLU";

    Configuration::instance().set(Device::CPU, Type::FP32);
    set_seed(42);

    TabularDataset dataset(test_path, ",", false, false);
    dataset.set_sample_roles("Testing");
    const MatrixR& data = dataset.get_data();
    const Index samples = dataset.get_samples_number();
    const Index inputs_number = dataset.get_input_shape()[0];
    const MatrixR inputs = data.leftCols(inputs_number);

    ClassificationNetwork network(Shape{inputs_number},
                                  Shape(size_t(hidden_layers), hidden),
                                  Shape{1},
                                  activation);

    // Nothing trains here, so nothing computes scaling statistics; the CSV is
    // already normalized, and None makes the scaling layer a passthrough.
    dynamic_cast<Scaling&>(*network.get_layer(0)).set_scalers("None");

    cout << "engine=opennn\nmode=infer\ndevice=cpu\nreps=" << reps << "\n";

    for (const Index batch : batches)
    {
        const Index processed = (samples / batch) * batch;
        ForwardPropagation forward_propagation(batch, &network);

        double sink = 0.0;              // reading one output keeps LTO honest

        auto run_pass = [&]()
        {
            for (Index i = 0; i + batch <= samples; i += batch)
            {
                TensorView view(const_cast<float*>(inputs.data()) + i * inputs_number,
                                Shape{batch, inputs_number}, Type::FP32);
                network.forward_propagate({view}, forward_propagation, false);
                sink += forward_propagation.get_outputs().as_matrix()(0, 0);
            }
        };

        run_pass();
        run_pass();

        vector<double> times;

        for (Index r = 0; r < reps; ++r)
        {
            const auto t0 = clock_type::now();
            run_pass();
            times.push_back(chrono::duration<double>(clock_type::now() - t0).count());
        }

        (void)sink;
        cout << "batch_" << batch << "_pass_times=";
        for (size_t i = 0; i < times.size(); ++i)
            cout << (i ? "," : "") << times[i];
        cout << "\n";

        sort(times.begin(), times.end());
        const double median_pass_s = times[times.size() / 2];

        cout << "batch_" << batch << "_samples_per_sec=" << long(double(processed) / median_pass_s)
             << " median_pass_s=" << median_pass_s << "\n";
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
        const string mode = argc > 1 ? argv[1] : "";

        if (mode == "train" && argc > 3) return train_mode(argc, argv);
        if (mode == "infer" && argc > 2) return infer_mode(argc, argv);

        cerr << "usage: opennn_higgs_cpu <train|infer> ...\n";
        return 2;
    }
    catch (const exception& e)
    {
        cerr << e.what() << "\n";
        cout << "RESULT=ERROR\n";
        return 1;
    }
}
