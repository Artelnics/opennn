//   OpenNN GPU HIGGS dense training-speed benchmark.
//
//   Trains the canonical HIGGS dense classifier
//   (28 -> hidden -> hidden -> 1, ReLU hidden, sigmoid output, binary cross
//   entropy -- see docs/benchmarks/throughput/higgs/README.md) on the GPU with
//   Adam, then reports training throughput and a test-set quality gate.
//
//   The training split is loaded once and made device-resident
//   (StorageMode::GPUPersistantData); Adam runs for N epochs at the given batch
//   with the CUDA-graph training step. After training the test CSV is scored on
//   the same network and accuracy / log-loss / ROC-AUC are computed exactly like
//   the CPU reference (../higgs/opennn_higgs_cpu.cpp).
//
//   Precision is selectable fp32 or bf16 (bf16 is OpenNN's mixed-precision
//   training path), matching the autocast / mixed_bfloat16 used on the PyTorch
//   and TensorFlow sides. It is selected exactly like opennn_higgs_infer.cpp:
//   Configuration::instance().set(Device::CUDA, type).
//
//   The CLI arg order matches run_higgs_dense.py's opennn command exactly:
//
//   usage:  opennn_speed <train_csv> <epochs> <batch> <fp32|bf16>
//                        <hidden> <activation> <hidden_layers> <test_csv>
//                        <min_accuracy> <max_log_loss> <min_auc>
//
//   The three threshold args are "none" when unset; a threshold is enforced only
//   when a finite number is given.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../../opennn/adaptive_moment_estimation.h"
#include "../../../opennn/configuration.h"
#include "../../../opennn/dense_layer.h"
#include "../../../opennn/forward_propagation.h"
#include "../../../opennn/neural_network.h"
#include "../../../opennn/random_utilities.h"
#include "../../../opennn/tabular_dataset.h"
#include "../../../opennn/tensor_types.h"
#include "../../../opennn/training_strategy.h"

using namespace opennn;
using clock_type = std::chrono::steady_clock;

namespace
{

float clamp_probability(float value)
{
    if (value < 1.0e-7f) return 1.0e-7f;
    if (value > 1.0f - 1.0e-7f) return 1.0f - 1.0e-7f;
    return value;
}

std::unique_ptr<NeuralNetwork> make_network(const Shape& input_shape,
                                            const Shape& target_shape,
                                            Index hidden,
                                            Index hidden_layers,
                                            const std::string& activation)
{
    auto network = std::make_unique<NeuralNetwork>();
    Shape current = input_shape;
    const std::string hidden_activation = (activation == "relu" || activation == "ReLU")
        ? "ReLU"
        : "Tanh";

    for (Index i = 0; i < hidden_layers; ++i)
    {
        network->add_layer(std::make_unique<opennn::Dense>(
            current,
            Shape{hidden},
            hidden_activation,
            false,
            "higgs_dense_" + std::to_string(i + 1)));
        current = network->get_output_shape();
    }

    network->add_layer(std::make_unique<opennn::Dense>(
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

double calculate_auc(const std::vector<std::pair<float, int>>& scored)
{
    const Index n = Index(scored.size());
    if (n == 0) return 0.0;

    Index positives = 0;
    for (const auto& item : scored)
        positives += item.second ? 1 : 0;
    const Index negatives = n - positives;
    if (positives == 0 || negatives == 0) return 0.0;

    std::vector<std::pair<float, int>> sorted = scored;
    std::sort(sorted.begin(), sorted.end(),
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

// Score the test CSV on the trained network. calculate_outputs dispatches to
// the device forward automatically when Configuration is CUDA and copies the
// results back to a host matrix, so the same host-side metric computation as the
// CPU reference works unchanged.
BinaryMetrics evaluate(NeuralNetwork& network,
                       const std::string& test_path,
                       Index batch)
{
    TabularDataset test_dataset(test_path, ",", false, false);
    test_dataset.set_sample_roles("Testing");
    const MatrixR& all = test_dataset.get_data();
    const Index samples = all.rows();
    const Index inputs_number = test_dataset.get_input_shape()[0];
    const Index processed = (samples / batch) * batch;
    const MatrixR inputs = all.leftCols(inputs_number);

    std::vector<std::pair<float, int>> scored;
    scored.reserve(size_t(processed));

    double log_loss = 0.0;
    Index correct = 0;
    for (Index i = 0; i + batch <= samples; i += batch)
    {
        float* batch_data = const_cast<float*>(inputs.data()) + i * inputs_number;
        const TensorView view(batch_data, Shape{batch, inputs_number}, Type::FP32);
        const MatrixR outputs = network.calculate_outputs({view});

        for (Index r = 0; r < batch; ++r)
        {
            const float probability = clamp_probability(float(outputs(r, 0)));
            const int label = all(i + r, inputs_number) >= 0.5f ? 1 : 0;
            const int predicted = probability >= 0.5f ? 1 : 0;
            correct += predicted == label ? 1 : 0;
            log_loss += label
                ? -std::log(double(probability))
                : -std::log(double(1.0f - probability));
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

bool has_threshold(const std::string& value)
{
    if (value.empty() || value == "none" || value == "None" || value == "nan")
        return false;
    return true;
}

}

int main(int argc, char* argv[])
{
    std::cout << std::unitbuf;

    try
    {
        if (argc < 5)
        {
            std::cerr << "usage: opennn_speed <train_csv> <epochs> <batch> <fp32|bf16>"
                         " <hidden> <activation> <hidden_layers> <test_csv>"
                         " <min_accuracy> <max_log_loss> <min_auc>\n";
            return 2;
        }

        const std::string train_path = argv[1];
        const Index epochs = Index(std::stoll(argv[2]));
        const Index batch = Index(std::stoll(argv[3]));
        const std::string precision = argv[4];
        const Index hidden = argc > 5 ? Index(std::stoll(argv[5])) : 1024;
        const std::string activation = argc > 6 ? argv[6] : "relu";
        const Index hidden_layers = argc > 7 ? Index(std::stoll(argv[7])) : 2;
        const std::string test_path = argc > 8 ? argv[8] : "";

        const std::string min_accuracy_arg = argc > 9 ? argv[9] : "none";
        const std::string max_log_loss_arg = argc > 10 ? argv[10] : "none";
        const std::string min_auc_arg = argc > 11 ? argv[11] : "none";

        if (test_path.empty())
            throw std::runtime_error("test CSV path is required for the quality gate");

        set_seed(42);
        const Type training_type = (precision == "bf16") ? Type::BF16 : Type::FP32;
        Configuration::instance().set(Device::CUDA, training_type);

        // Training split, device-resident: no per-step H2D copies, matching the
        // PyTorch / TensorFlow protocols where the whole tensor lives on the GPU.
        TabularDataset dataset(train_path, ",", false, false);
        dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);
        dataset.set_sample_roles("Training");
        if (!std::getenv("OPENNN_BENCH_SCALERS"))
            dataset.set_variable_scalers("None");  // PyTorch/TF train on the prepared file as-is
        const Index samples = dataset.get_samples_number();

        std::cout << "engine=opennn\n";
        std::cout << "mode=train\n";
        std::cout << "device=cuda\n";
        std::cout << "samples=" << samples << "\n";
        std::cout << "batch=" << batch << "\n";
        std::cout << "epochs=" << epochs << "\n";
        std::cout << "hidden=" << hidden << "\n";
        std::cout << "hidden_layers=" << hidden_layers << "\n";
        std::cout << "activation=" << activation << "\n";
        std::cout << "precision=" << precision << "\n";

        auto network = make_network(dataset.get_input_shape(),
                                    dataset.get_target_shape(),
                                    hidden,
                                    hidden_layers,
                                    activation);
        std::cout << "parameters=" << network->get_parameters_number() << "\n";

        TrainingStrategy training_strategy(network.get(), &dataset);
        training_strategy.set_loss("CrossEntropy");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        adam->set_batch_size(batch);
        adam->set_cuda_graph(true);          // capture/replay the training step
        adam->set_display_period(1000000);   // silence per-epoch printing
        adam->set_gradient_clip_norm(0.0f);  // match PyTorch/TF default (no clipping)

        // Warmup: a couple of epochs to absorb kernel/cuDNN autotuning and the
        // CUDA-graph capture. Excluded from timing.
        adam->set_maximum_epochs(2);
        training_strategy.train();

        // Timed run.
        adam->set_maximum_epochs(epochs);
        const auto t0 = clock_type::now();
        training_strategy.train();
        const auto t1 = clock_type::now();

        const double total_s = std::chrono::duration<double>(t1 - t0).count();
        const double median_epoch_s = total_s / double(epochs);
        const double samples_per_sec = double(samples) / median_epoch_s;

        const BinaryMetrics metrics = evaluate(*network, test_path, batch);

        std::cout << "median_epoch_s=" << median_epoch_s << "\n";
        std::cout << "samples_per_sec=" << long(samples_per_sec) << "\n";
        std::cout << "test_samples=" << metrics.samples << "\n";
        std::cout << "test_accuracy=" << metrics.accuracy << "\n";
        std::cout << "test_log_loss=" << metrics.log_loss << "\n";
        std::cout << "test_roc_auc=" << metrics.auc << "\n";

        bool gate_pass = true;
        if (has_threshold(min_accuracy_arg)
            && metrics.accuracy < std::stod(min_accuracy_arg))
            gate_pass = false;
        if (has_threshold(max_log_loss_arg)
            && metrics.log_loss > std::stod(max_log_loss_arg))
            gate_pass = false;
        if (has_threshold(min_auc_arg)
            && (!std::isfinite(metrics.auc) || metrics.auc < std::stod(min_auc_arg)))
            gate_pass = false;

        if (has_threshold(min_accuracy_arg)
            || has_threshold(max_log_loss_arg)
            || has_threshold(min_auc_arg))
            std::cout << "quality_gate=" << (gate_pass ? "PASS" : "FAIL") << "\n";

        std::cout << "RESULT=OK\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        std::cout << "RESULT=ERROR\n";
        return 1;
    }
}
