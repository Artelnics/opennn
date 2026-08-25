// OpenNN accuracy-parity benchmark on the HIGGS classification task: one main
// function, written the way a user writes an OpenNN application. Trains the
// canonical HIGGS dense classifier (28 -> 1024 -> 1024 -> 1, ReLU hidden,
// sigmoid output, binary cross entropy, Adam, fixed epochs) on the shared
// prepared split and prints the test-set quality, so the parity between
// OpenNN, PyTorch, and TensorFlow can be checked at a fixed training budget.
// The protocol lives in run_accuracy.py.
//
//   opennn_accuracy [train_csv] [test_csv] [epochs] [batch] [hidden] [hidden_layers]
//
// Reads $OPENNN_BENCH_DATA/higgs/{higgs_train.csv,higgs_test.csv} by default.
// Prints (one key=value per line):
//   test_accuracy, test_log_loss, test_roc_auc, RESULT=OK
//
// The three metrics are computed here exactly as in metrics.py (probabilities
// clamped to [1e-7, 1 - 1e-7], rank-based ROC AUC with tied scores sharing
// their average rank, whole batches only), so every engine is scored the same
// way.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/training_strategy/training_strategy.h"

using namespace opennn;

int main(int argc, char* argv[])
{
    try
    {
        const char* bench_data = getenv("OPENNN_BENCH_DATA");
        const string higgs_dir = (bench_data && *bench_data
            ? string(bench_data)
            : string(getenv("HOME") ? getenv("HOME") : ".") + "/opennn-benchmark-data") + "/higgs/";

        const string train_path = argc > 1 ? argv[1] : higgs_dir + "higgs_train.csv";
        const string test_path = argc > 2 ? argv[2] : higgs_dir + "higgs_test.csv";
        const Index epochs = argc > 3 ? Index(stoll(argv[3])) : 5;
        const Index batch = argc > 4 ? Index(stoll(argv[4])) : 1024;
        const Index hidden = argc > 5 ? Index(stoll(argv[5])) : 1024;
        const Index hidden_layers = argc > 6 ? Index(stoll(argv[6])) : 2;

        set_seed(42);
        Configuration::instance().set(Device::CPU, Type::FP32);

        TabularDataset dataset(train_path, ",", false, false);
        dataset.set_sample_roles("Training");
        const Index samples = dataset.get_samples_number();

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
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        adam->set_batch_size(batch);
        adam->set_display_period(1000000);
        adam->set_gradient_clip_norm(0.0f);
        adam->set_maximum_epochs(epochs);

        training_strategy.train();

        // The test CSV is loaded after training on purpose: TabularDataset's
        // constructor draws a random sample split, so loading it earlier would
        // shift the random stream behind the Glorot initialization and the
        // batch order, and with it the published numbers.
        TabularDataset test_dataset(test_path, ",", false, false);
        test_dataset.set_sample_roles("Testing");
        const MatrixR& test_data = test_dataset.get_data();
        const Index test_samples = test_data.rows();
        const Index inputs_number = test_dataset.get_input_shape()[0];
        const Index processed = (test_samples / batch) * batch;
        const MatrixR inputs = test_data.leftCols(inputs_number);

        ForwardPropagation forward_propagation(batch, &network);
        vector<pair<float, int>> scored;      // (clamped probability, label)
        scored.reserve(size_t(processed));

        double log_loss = 0.0;
        Index correct = 0;

        for (Index i = 0; i + batch <= test_samples; i += batch)
        {
            TensorView view(const_cast<float*>(inputs.data()) + i * inputs_number,
                            Shape{batch, inputs_number}, Type::FP32);
            network.forward_propagate({view}, forward_propagation, ForwardPropagationMode::Inference);
            const MatrixMap outputs = forward_propagation.get_outputs().as_matrix();

            for (Index r = 0; r < batch; ++r)
            {
                const float probability = clamp(outputs(r, 0), 1.0e-7f, 1.0f - 1.0e-7f);
                const int label = test_data(i + r, inputs_number) >= 0.5f ? 1 : 0;
                const int predicted = probability >= 0.5f ? 1 : 0;
                correct += predicted == label;
                log_loss += label ? -log(double(probability)) : -log(double(1.0f - probability));
                scored.emplace_back(probability, label);
            }
        }

        Index positives = 0;
        for (const auto& item : scored)
            positives += item.second;
        const Index negatives = processed - positives;

        sort(scored.begin(), scored.end(),
             [](const auto& a, const auto& b) { return a.first < b.first; });

        double positive_rank_sum = 0.0;
        for (Index i = 0; i < processed;)
        {
            Index j = i + 1;
            while (j < processed && scored[j].first == scored[i].first) ++j;
            const double average_rank = (double(i + 1) + double(j)) * 0.5;
            for (Index k = i; k < j; ++k)
                if (scored[k].second) positive_rank_sum += average_rank;
            i = j;
        }

        const double accuracy = processed > 0 ? double(correct) / double(processed) : 0.0;
        const double mean_log_loss = processed > 0 ? log_loss / double(processed) : 0.0;
        const double auc = positives == 0 || negatives == 0 ? 0.0
            : (positive_rank_sum - double(positives) * double(positives + 1) * 0.5)
              / (double(positives) * double(negatives));

        cout << "engine=opennn\n";
        cout << "device=cpu\n";
        cout << "samples=" << samples << "\n";
        cout << "batch=" << batch << "\n";
        cout << "epochs=" << epochs << "\n";
        cout << "hidden=" << hidden << "\n";
        cout << "hidden_layers=" << hidden_layers << "\n";
        cout << "activation=relu\n";
        cout << "test_samples=" << processed << "\n";
        cout << "test_accuracy=" << accuracy << "\n";
        cout << "test_log_loss=" << mean_log_loss << "\n";
        cout << "test_roc_auc=" << auc << "\n";
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
