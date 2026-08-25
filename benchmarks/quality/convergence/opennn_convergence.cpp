// OpenNN convergence-gate benchmark on the HIGGS classification dataset: one
// main function, written the way a user writes an OpenNN application, plus a
// timer. The protocol lives in run_convergence.py.
//
// MLPerf-style metric: WALL-CLOCK TIME TO REACH A FIXED QUALITY TARGET, not
// throughput at a fixed epoch count. Trains the canonical HIGGS dense
// classifier (28 -> 1024 -> 1024 -> 1, ReLU, sigmoid, BCE, Adam) in a single
// train() call and, after each epoch (the optimizer's post-epoch callback, so
// Adam state persists across epochs exactly as in the PyTorch/TF drivers),
// evaluates the HELD-OUT (test) log-loss. When it reaches the target, training
// stops and the wall-clock time, the epochs taken, and the final held-out
// metric are reported. Per-epoch evaluation is excluded from the clock.
//
//   usage: opennn_convergence <train_csv> <test_csv> [target_log_loss]
//                             [max_epochs] [batch] [hidden] [hidden_layers]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/training_strategy/training_strategy.h"

using namespace opennn;
using clock_type = chrono::steady_clock;

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

        const string train_path = argv[1];
        const string test_path = argv[2];
        const float target_log_loss = argc > 3 ? stof(argv[3]) : 0.60f;
        const Index max_epochs = argc > 4 ? Index(stoll(argv[4])) : 50;
        const Index batch = argc > 5 ? Index(stoll(argv[5])) : 1024;
        const Index hidden = argc > 6 ? Index(stoll(argv[6])) : 1024;
        const Index hidden_layers = argc > 7 ? Index(stoll(argv[7])) : 2;

        set_seed(42);
        Configuration::instance().set(Device::CPU, Type::FP32);

        TabularDataset dataset(train_path, ",", false, false);
        dataset.set_sample_roles("Training");
        const Index samples = dataset.get_samples_number();
        const Index inputs_number = dataset.get_input_shape()[0];

        TabularDataset test_dataset(test_path, ",", false, false);
        test_dataset.set_sample_roles("Testing");
        const MatrixR& test_data = test_dataset.get_data();
        const Index test_samples = test_data.rows();
        const MatrixR test_inputs = test_data.leftCols(inputs_number);

        NeuralNetwork network;
        Shape current = dataset.get_input_shape();

        for (Index i = 0; i < hidden_layers; ++i)
        {
            network.add_layer(make_unique<opennn::Dense>(current,
                                                         Shape{hidden},
                                                         "ReLU",
                                                         false,
                                                         "higgs_dense_" + to_string(i + 1)));
            current = network.get_output_shape();
        }

        network.add_layer(make_unique<opennn::Dense>(current,
                                                     dataset.get_target_shape(),
                                                     "Sigmoid",
                                                     false,
                                                     "higgs_output"));
        network.compile();
        network.set_parameters_glorot();

        TrainingStrategy training_strategy(&network, &dataset);
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
        adam->set_maximum_epochs(max_epochs);

        bool reached = false;
        Index epochs = 0;
        double test_log_loss = NAN;
        double evaluation_s = 0.0;

        // The gate: after each epoch, the mean clamped binary cross entropy over
        // the held-out split (whole batches only), with its wall time kept out
        // of the clock. Reaching the target zeroes the remaining epoch budget,
        // which ends the single train() call.
        adam->post_epoch_callback = [&](Index epoch, float, float, NeuralNetwork* trained_network)
        {
            const auto evaluation_start = clock_type::now();

            ForwardPropagation forward_propagation(batch, trained_network);
            double log_loss = 0.0;
            Index processed = 0;

            for (Index i = 0; i + batch <= test_samples; i += batch)
            {
                TensorView view(const_cast<float*>(test_inputs.data()) + i * inputs_number,
                                Shape{batch, inputs_number}, Type::FP32);
                trained_network->forward_propagate({view}, forward_propagation, ForwardPropagationMode::Inference);
                const MatrixMap outputs = forward_propagation.get_outputs().as_matrix();

                for (Index r = 0; r < batch; ++r)
                {
                    const float probability = clamp(outputs(r, 0), 1.0e-7f, 1.0f - 1.0e-7f);
                    const int label = test_data(i + r, inputs_number) >= 0.5f ? 1 : 0;
                    log_loss += label ? -log(double(probability)) : -log(double(1.0f - probability));
                }
                processed += batch;
            }

            test_log_loss = processed > 0 ? log_loss / double(processed) : NAN;
            epochs = epoch + 1;

            if (test_log_loss <= target_log_loss)
            {
                reached = true;
                adam->set_maximum_epochs(0);
            }

            evaluation_s += chrono::duration<double>(clock_type::now() - evaluation_start).count();
        };

        const auto t0 = clock_type::now();
        training_strategy.train();
        const double train_s = chrono::duration<double>(clock_type::now() - t0).count() - evaluation_s;

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
