//   OpenNN GPU ResNet-50 training-speed benchmark.
//
//   Standard ResNet-50 v1.5 (opennn::ResNet, bottleneck blocks [3,4,6,3],
//   stride on the 3x3 convolution as torchvision builds it). Cross-entropy,
//   Adam, fp32/bf16. By default the image shape comes from the first file
//   (CIFAR path); pass image_size=224 for the full ImageNet benchmark.
//
//   usage:  opennn_resnet50_speed <data_path> [epochs] [batch] [fp32|bf16] [image_size]

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#include "../../../opennn/image_dataset.h"
#include "../../../opennn/standard_networks.h"
#include "../../../opennn/training_strategy.h"
#include "../../../opennn/adaptive_moment_estimation.h"
#include "../../../opennn/random_utilities.h"
#include "../../../opennn/configuration.h"

using namespace opennn;
using clock_type = std::chrono::steady_clock;

int main(int argc, char* argv[])
{
    try
    {
        const std::string data_path = argc > 1 ? argv[1] : "cifar10/train";
        const Index timed_epochs = argc > 2 ? Index(std::stoll(argv[2])) : 5;
        const Index batch = argc > 3 ? Index(std::stoll(argv[3])) : 128;
        const std::string precision = argc > 4 ? argv[4] : "fp32";
        const Index image_size = argc > 5 ? Index(std::stoll(argv[5])) : 0;

        set_seed(42);
        const Type training_type = (precision == "bf16") ? Type::BF16 : Type::FP32;
        Configuration::instance().set(Device::CUDA, training_type);

        std::unique_ptr<ImageDataset> dataset_ptr =
            image_size > 0
                ? std::make_unique<ImageDataset>(data_path, Shape{image_size, image_size, 3})
                : std::make_unique<ImageDataset>(data_path);
        ImageDataset& dataset = *dataset_ptr;
        dataset.set_sample_roles("Training");
        const Index samples = dataset.get_samples_number();

        std::cout << "samples=" << samples << " batch=" << batch
                  << " epochs=" << timed_epochs << " precision=" << precision;
        if (image_size > 0) std::cout << " image_size=" << image_size;
        std::cout << "\n";

        ResNet network(dataset.get_shape("Input"),
                       {3, 4, 6, 3},
                       Shape{64, 128, 256, 512},
                       dataset.get_shape("Target"),
                       /*use_bottleneck=*/true);

        std::cout << "layers=" << network.get_layers_number()
                  << " parameters=" << network.get_parameters_size() << "\n";

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.set_loss("CrossEntropy");
        training_strategy.get_loss()->set_regularization("NoRegularization");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        adam->set_batch_size(batch);
        adam->set_display_period(1000000);
        adam->set_gradient_clip_norm(0.0f);

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
