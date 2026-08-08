// OpenNN convergence-gate benchmark on the HIGGS classification dataset.
//
// MLPerf-style metric: WALL-CLOCK TIME TO REACH A FIXED QUALITY TARGET, not
// throughput at a fixed epoch count. Trains the canonical HIGGS dense classifier
// (28 -> 1024 -> 1024 -> 1, ReLU, sigmoid, BCE, Adam) and, after each short
// training chunk, evaluates the HELD-OUT (test) log-loss. When the held-out
// log-loss reaches the target, the clock stops and we report the wall-clock
// time, the epochs taken, and the final held-out metric.
//
// This answers the reviewer question "are you fast because you do not actually
// learn?": every engine must reach the same held-out quality, and we time how
// long that takes. Same data, arch, optimizer, and target as the PyTorch/TF
// drivers. Per-chunk evaluation is excluded from the clock.
//
//   usage: opennn_convergence <train_csv> <test_csv> [target_log_loss]
//                             [max_epochs] [batch] [hidden] [hidden_layers]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
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

double evaluate_log_loss(NeuralNetwork& network,
                         const MatrixR& all,
                         Index inputs_number,
                         Index batch)
{
    const Index samples = all.rows();
    const MatrixR inputs = all.leftCols(inputs_number);

    ForwardPropagation forward_propagation(batch, &network);

    double log_loss = 0.0;
    Index processed = 0;
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
            log_loss += label
                ? -log(double(probability))
                : -log(double(1.0f - probability));
        }
        processed += batch;
    }

    return processed > 0 ? log_loss / double(processed) : NAN;
}

}

int main(int argc, char* argv[])
{
    try
    {
        if (argc < 3)
        {
            cerr << "usage: opennn_convergence <train_csv> <test_csv> "
                         "[target_log_loss] [max_epochs] [batch] [hidden] [hidden_layers]\n";
            return 2;
        }

        const string train_path    = argv[1];
        const string test_path     = argv[2];
        const float target_log_loss     = argc > 3 ? stof(argv[3])          : 0.60f;
        const Index max_epochs          = argc > 4 ? Index(stoll(argv[4]))  : 50;
        const Index batch               = argc > 5 ? Index(stoll(argv[5]))  : 1024;
        const Index hidden              = argc > 6 ? Index(stoll(argv[6]))  : 1024;
        const Index hidden_layers       = argc > 7 ? Index(stoll(argv[7]))  : 2;

        set_seed(42);
        Configuration::instance().set(Device::CPU, Type::FP32);

        TabularDataset dataset(train_path, ",", false, false);
        dataset.set_sample_roles("Training");
        const Index samples = dataset.get_samples_number();
        const Index inputs_number = dataset.get_input_shape()[0];

        TabularDataset test_dataset(test_path, ",", false, false);
        test_dataset.set_sample_roles("Testing");
        const MatrixR& test_all = test_dataset.get_data();

        auto network = make_network(dataset.get_input_shape(),
                                    dataset.get_target_shape(),
                                    hidden,
                                    hidden_layers);

        TrainingStrategy training_strategy(network.get(), &dataset);
        training_strategy.set_loss("CrossEntropy");
        training_strategy.get_loss()->set_regularization("NoRegularization");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        adam->set_batch_size(batch);
        adam->set_display(false);
        adam->set_display_period(1000000);
        adam->set_gradient_clip_norm(0.0f);
        adam->set_loss_goal(0.0f);

        const Index chunk = 1;
        adam->set_maximum_epochs(chunk);

        bool reached = false;
        Index epochs = 0;
        double test_log_loss = NAN;
        double train_s = 0.0;

        while (epochs < max_epochs)
        {
            const auto t0 = clock_type::now();
            training_strategy.train();
            train_s += chrono::duration<double>(clock_type::now() - t0).count();
            epochs += chunk;

            test_log_loss = evaluate_log_loss(*network, test_all, inputs_number, batch);
            if (test_log_loss <= target_log_loss) { reached = true; break; }
        }

        cout.precision(10);
        cout << "engine=opennn\n";
        cout << "device=cpu\n";
        cout << "dataset=HIGGS\n";
        cout << "train_samples=" << samples << "\n";
        cout << "batch=" << batch << "\n";
        cout << "hidden=" << hidden << "\n";
        cout << "hidden_layers=" << hidden_layers << "\n";
        cout << "target_log_loss=" << target_log_loss << "\n";
        cout << "reached_goal=" << (reached ? 1 : 0) << "\n";
        cout << "epochs_to_target=" << epochs << "\n";
        cout << "test_log_loss=" << test_log_loss << "\n";
        cout << "time_to_target_s=" << train_s << "\n";
        cout << "RESULT=" << (reached ? "OK" : "DID_NOT_CONVERGE") << "\n";

        return reached ? 0 : 1;
    }
    catch (const exception& e)
    {
        cerr << e.what() << "\n";
        cout << "RESULT=ERROR\n";
        return 1;
    }
}
