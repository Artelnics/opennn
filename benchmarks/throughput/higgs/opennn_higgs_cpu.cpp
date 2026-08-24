// OpenNN CPU HIGGS dense benchmark: one main function, written the way a user
// writes an OpenNN application, plus a timer. The measurement protocol lives
// in run_higgs_cpu_sweep.py.
//
//   opennn_higgs_cpu train <train_csv> <test_csv> [epochs] [batch,...] [hidden] [layers] [activation] [warmup]
//   opennn_higgs_cpu infer <test_csv> [reps] [batch,...] [hidden] [layers] [activation]
//
// The batch sizes run inside one process so they share one data load and one
// thermal window; each prints its own batch_<B>_... lines for the runner.

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
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

int main(int argc, char* argv[])
{
    const string mode = argc > 1 ? argv[1] : "";

    if ((mode != "train" || argc < 4) && (mode != "infer" || argc < 3))
    {
        cerr << "usage: opennn_higgs_cpu train <train_csv> <test_csv> [epochs] [batch,...] [hidden] [layers] [activation] [warmup]\n"
                "       opennn_higgs_cpu infer <test_csv> [reps] [batch,...] [hidden] [layers] [activation]\n";
        return 2;
    }

    Configuration::instance().set(Device::CPU, Type::FP32);

    const int shift = mode == "train" ? 1 : 0;        // train takes two paths
    const Index count = argc > 3 + shift ? stoll(argv[3 + shift]) : 1;   // epochs or reps
    const Index hidden = argc > 5 + shift ? stoll(argv[5 + shift]) : 1024;
    const Index layers = argc > 6 + shift ? stoll(argv[6 + shift]) : 2;
    const string activation =
        argc > 7 + shift && string(argv[7 + shift]) != "relu" ? "Tanh" : "ReLU";

    vector<Index> batches;
    stringstream batch_text(argc > 4 + shift ? argv[4 + shift] : "1024");
    for (string item; getline(batch_text, item, ',');)
        batches.push_back(stoll(item));

    cout << "engine=opennn\nmode=" << mode << "\ndevice=cpu\n";

    if (mode == "train")
    {
        TabularDataset dataset(argv[2], ",", false, false);
        dataset.set_sample_roles("Training");
        dataset.set_variable_scalers("None");   // prepare_higgs.py already normalized

        TabularDataset test_dataset(argv[3], ",", false, false);
        test_dataset.set_sample_roles("Testing");

        const Index samples = dataset.get_samples_number();
        const Index warmup_epochs = argc > 9 ? stoll(argv[9]) : 0;

        for (const Index batch : batches)
        {
            set_seed(42);

            ClassificationNetwork network(dataset.get_input_shape(),
                                          Shape::filled(size_t(layers), hidden),
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

            adam->set_maximum_epochs(count);
            const auto t0 = clock_type::now();
            training_strategy.train();
            const auto t1 = clock_type::now();

            // The strategy owns the epoch loop, so only the epochs' mean time
            // is observable from here.
            const double epoch_s = chrono::duration<double>(t1 - t0).count() / double(count);

            TestingAnalysis analysis(&network, &test_dataset);

            cout << "batch_" << batch << "_samples_per_sec=" << long(double(samples) / epoch_s)
                 << " median_epoch_s=" << epoch_s << "\n"
                 << "batch_" << batch << "_epoch_times=" << epoch_s << "\n"
                 << "batch_" << batch
                 << "_test_accuracy=" << analysis.calculate_binary_classification_tests()[0]
                 << " test_roc_auc=" << analysis.perform_roc_analysis().area_under_curve
                 << "\n" << flush;
        }
    }
    else
    {
        TabularDataset dataset(argv[2], ",", false, false);
        dataset.set_sample_roles("Testing");
        const MatrixR& data = dataset.get_data();
        const Index samples = dataset.get_samples_number();
        const Index inputs_number = dataset.get_input_shape()[0];
        const MatrixR inputs = data.leftCols(inputs_number);

        set_seed(42);

        ClassificationNetwork network(Shape{inputs_number},
                                      Shape::filled(size_t(layers), hidden),
                                      Shape{1},
                                      activation);

        // Nothing trains here, so nothing computes scaling statistics; the CSV
        // is already normalized, and None makes the layer a passthrough.
        dynamic_cast<Scaling&>(*network.get_layer(0)).set_scalers("None");

        for (const Index batch : batches)
        {
            const Index processed = (samples / batch) * batch;
            ForwardPropagation forward_propagation(batch, &network);

            double sink = 0.0;          // reading one output keeps LTO honest

            const auto run_pass = [&]
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

            for (Index r = 0; r < count; ++r)
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
                 << " median_pass_s=" << median_pass_s << "\n" << flush;
        }
    }

    cout << "RESULT=OK\n";
    return 0;
}
