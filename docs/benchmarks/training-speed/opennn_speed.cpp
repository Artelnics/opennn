//   OpenNN GPU training-speed benchmark.
//
//   Mirrors the Neural Designer training-speed benchmark: a 2-layer MLP
//   (F -> F -> 1, tanh then linear) trained with Adam + MSE on the Rosenbrock
//   dataset, batch 1000. Reports median seconds/epoch and samples/second.
//
//   Runs on the GPU (Device::CUDA). Precision is selectable: fp32 or bf16
//   (bf16 is OpenNN's mixed-precision training path, matching the autocast /
//   mixed_bfloat16 used on the PyTorch and TensorFlow sides).
//
//   usage:  opennn_speed <csv_path> <features> [epochs] [batch] [fp32|bf16]

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "../../../opennn/tabular_dataset.h"
#include "../../../opennn/standard_networks.h"
#include "../../../opennn/training_strategy.h"
#include "../../../opennn/optimizer.h"
#include "../../../opennn/adaptive_moment_estimation.h"
#include "../../../opennn/random_utilities.h"

using namespace opennn;
using clock_type = std::chrono::steady_clock;

int main(int argc, char* argv[])
{
    try
    {
        if (argc < 3)
        {
            std::cerr << "usage: opennn_speed <csv_path> <features> [epochs] [batch] [fp32|bf16] [tanh|relu]\n";
            return 2;
        }

        const std::string csv_path = argv[1];
        const Index features = Index(std::stoll(argv[2]));
        const Index timed_epochs = (argc > 3) ? Index(std::stoll(argv[3])) : 30;
        const Index batch = (argc > 4) ? Index(std::stoll(argv[4])) : 1000;
        const std::string precision = (argc > 5) ? argv[5] : "fp32";
        const std::string activation = (argc > 6) ? argv[6] : "tanh";
        const std::string hidden_activation = (activation == "relu") ? "ReLU" : "Tanh";

        set_seed(42);
        const Type training_type = (precision == "bf16") ? Type::BF16 : Type::FP32;
        Configuration::instance().set(Device::CUDA, training_type);

        TabularDataset dataset(csv_path, ",", false, false);
        dataset.split_samples_random(1.0f, 0.0f, 0.0f);
        if (!std::getenv("OPENNN_BENCH_SCALERS"))
            dataset.set_variable_scalers("None");  // PyTorch/TF train on raw data; keep the protocols identical
        const Index samples = dataset.get_samples_number();

        std::cout << "samples=" << samples << " features=" << features
                  << " batch=" << batch << " epochs=" << timed_epochs
                  << " precision=" << precision << " activation=" << activation << "\n";

        ApproximationNetwork network(dataset.get_input_shape(),
                                     {features},
                                     dataset.get_target_shape(),
                                     hidden_activation);

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.set_loss("MeanSquaredError");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        adam->set_batch_size(batch);
        adam->set_display_period(1000000);   // silence per-epoch printing
        adam->set_gradient_clip_norm(0.0f);  // match PyTorch/TF default (no clipping)

        // Warmup: a couple of epochs to absorb kernel/cuDNN autotuning.
        adam->set_maximum_epochs(2);
        training_strategy.train();

        // Timed run.
        adam->set_maximum_epochs(timed_epochs);
        const auto t0 = clock_type::now();
        training_strategy.train();
        const auto t1 = clock_type::now();

        const double total_s =
            std::chrono::duration<double>(t1 - t0).count();
        const double epoch_s = total_s / double(timed_epochs);
        const double samples_per_sec = double(samples) / epoch_s;

        std::cout << "median_epoch_s=" << epoch_s << "\n";
        std::cout << "samples_per_sec=" << long(samples_per_sec) << "\n";
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
