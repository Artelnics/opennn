//   OpenNN GPU CNN training-speed benchmark on MNIST.
//
//   One convolutional and one pooling layer, as simple as a CNN gets:
//   28x28x1 -> Conv 16@3x3 (Same, ReLU) -> MaxPool 2x2 -> Flatten ->
//   Dense 10 (Softmax), cross-entropy, Adam, batch 128.
//   Reads the BMP class folders (examples/mnist/data), trains on the GPU,
//   and reports seconds/epoch and samples/second after a warmup.
//
//   usage:  opennn_cnn_speed <data_path> [epochs] [batch] [fp32|bf16]

#include <chrono>
#include <iostream>
#include <string>

#include "opennn/image_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/scaling_layer.h"
#include "opennn/convolutional_layer.h"
#include "opennn/pooling_layer.h"
#include "opennn/flatten_layer.h"
#include "opennn/dense_layer.h"
#include "opennn/training_strategy.h"
#include "opennn/adaptive_moment_estimation.h"
#include "opennn/random_utilities.h"
#include "opennn/configuration.h"

using namespace opennn;
using clock_type = std::chrono::steady_clock;

int main(int argc, char* argv[])
{
    try
    {
        const std::string data_path = argc > 1 ? argv[1] : "../../../../examples/mnist/data";
        const Index timed_epochs = argc > 2 ? Index(std::stoll(argv[2])) : 10;
        const Index batch = argc > 3 ? Index(std::stoll(argv[3])) : 128;
        const std::string precision = argc > 4 ? argv[4] : "fp32";

        set_seed(42);
        const Type training_type = (precision == "bf16") ? Type::BF16 : Type::FP32;
        Configuration::instance().set(Device::CUDA, training_type);

        ImageDataset dataset(data_path);
        dataset.set_sample_roles("Training");
        // Keep the whole dataset mirrored on the GPU and gather batches device-side
        // -- the analogue of the GPU-resident tensors in the PyTorch/TF scripts.
        // Enabled in code; there is no environment switch.
        dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);
        const Index samples = dataset.get_samples_number();

        std::cout << "samples=" << samples << " batch=" << batch
                  << " epochs=" << timed_epochs << " precision=" << precision << "\n";

        NeuralNetwork network;

        auto scaling = make_unique<Scaling>(dataset.get_shape("Input"));
        scaling->set_scalers("ImageMinMax");
        network.add_layer(move(scaling));

        network.add_layer(make_unique<Convolutional>(network.get_output_shape(),
                                                     Shape{3, 3, 1, 16},
                                                     "ReLU",
                                                     Shape{1, 1},
                                                     "Same"));

        network.add_layer(make_unique<Pooling>(network.get_output_shape(),
                                               Shape{2, 2},
                                               Shape{2, 2},
                                               Shape{0, 0},
                                               "MaxPooling"));

        network.add_layer(make_unique<Flatten>(network.get_output_shape()));

        network.add_layer(make_unique<opennn::Dense>(network.get_output_shape(),
                                                     dataset.get_shape("Target"),
                                                     "Softmax"));

        network.compile();
        network.set_parameters_random();

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.set_loss("CrossEntropy");
        training_strategy.get_loss()->set_regularization("NoRegularization");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        adam->set_batch_size(batch);
        adam->set_display_period(1000000);
        adam->set_gradient_clip_norm(0.0f);

        // Warmup: absorb cuDNN autotuning and first-touch allocations.
        adam->set_maximum_epochs(2);
        training_strategy.train();

        adam->set_maximum_epochs(timed_epochs);
        const auto t0 = clock_type::now();
        const TrainingResult results = training_strategy.train();
        const auto t1 = clock_type::now();

        const double total_s = std::chrono::duration<double>(t1 - t0).count();
        const double epoch_s = total_s / double(timed_epochs);

        std::cerr << "final_training_error " << results.get_training_error() << "\n";
        std::cout << "epoch_s=" << epoch_s << "\n";
        std::cout << "samples_per_sec=" << long(double(samples) / epoch_s) << "\n";
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
