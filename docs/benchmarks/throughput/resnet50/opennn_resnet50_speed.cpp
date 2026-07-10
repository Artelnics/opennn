//   OpenNN GPU ResNet-50 training-speed benchmark.
//
//   Standard ResNet-50 v1.5 (opennn::ResNet, bottleneck blocks [3,4,6,3],
//   stride on the 3x3 convolution as torchvision builds it). Cross-entropy,
//   Adam, fp32/bf16. By default the image shape comes from the first file
//   (CIFAR path); pass image_size=224 for the full ImageNet benchmark.
//
//   CUDA graph capture and GPU-resident data are enabled from this code, not
//   from environment variables. The graph is on by default and can be turned off
//   with the optional [cuda_graph 0|1] argument (used by the ImageNet runner's
//   --no-cuda-graph). Data is kept GPU-resident automatically for the small CIFAR
//   path (image_size==0); the 224px ImageNet path is too large and stays host-staged.
//
//   usage:  opennn_resnet50_speed <data_path> [epochs] [batch] [fp32|bf16] [image_size] [cuda_graph 0|1] [cache_dir]

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#include "opennn/image_dataset.h"
#include "opennn/standard_networks.h"
#include "opennn/training_strategy.h"
#include "opennn/adaptive_moment_estimation.h"
#include "opennn/random_utilities.h"
#include "opennn/configuration.h"
#include "opennn/device_backend.h"
#include "opennn/memory_debug.h"

using namespace opennn;
using clock_type = std::chrono::steady_clock;

int main(int argc, char* argv[])
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    try
    {
        const std::string data_path = argc > 1 ? argv[1] : "cifar10/train";
        const Index timed_epochs = argc > 2 ? Index(std::stoll(argv[2])) : 5;
        const Index batch = argc > 3 ? Index(std::stoll(argv[3])) : 128;
        const std::string precision = argc > 4 ? argv[4] : "fp32";
        // A negative image_size resizes to |image_size| AND keeps the set device
        // resident (fits smaller crops in VRAM so BF16/FP32 both take the gather
        // mega-graph). Positive stays host-staged (large-image ImageNet path).
        const Index image_size_arg = argc > 5 ? Index(std::stoll(argv[5])) : 0;
        const Index image_size = image_size_arg < 0 ? -image_size_arg : image_size_arg;
        const bool force_resident = image_size_arg < 0;
        const bool cuda_graph = argc > 6 ? (std::stoi(argv[6]) != 0) : true;
        const std::string cache_dir = argc > 7 ? argv[7] : "";
        // Conv workspace cap A/B: "off"/"0" = autotune, "auto" = AUTO cap, N = MiB cap.
        const std::string workspace_arg = argc > 8 ? argv[8] : "off";

        memory_debug::reset();

        set_seed(42);
        const Type training_type = (precision == "bf16") ? Type::BF16 : Type::FP32;
        Configuration::instance().set(Device::CUDA, training_type);

        // off = autotune (cap off); heur = plain heuristic (autotune off, cap off);
        // auto = AUTO cap (heuristic); N = explicit MiB cap (heuristic).
        if (workspace_arg == "off" || workspace_arg == "0")
            { device::set_conv_autotune(true);  device::set_conv_workspace_cap(0); }
        else if (workspace_arg == "heur")
            { device::set_conv_autotune(false); device::set_conv_workspace_cap(0); }
        else if (workspace_arg == "auto")
            device::set_conv_workspace_cap(-1);
        else
            device::set_conv_workspace_cap(std::stoll(workspace_arg) * 1024 * 1024);
        std::cout << "workspace_mode=" << workspace_arg << "\n";

        // A custom image-cache directory is no longer configurable from OpenNN
        // (the setter was removed when backend toggles became code-only). The
        // cache always lands in <data_path>/.cache, which stays outside the repo
        // because datasets live under $OPENNN_BENCH_DATA.
        if (!cache_dir.empty())
            std::cerr << "note: custom cache dir ignored (OpenNN caches in "
                         "<data_path>/.cache): " << cache_dir << "\n";

        std::unique_ptr<ImageDataset> dataset_ptr =
            image_size > 0
                ? std::make_unique<ImageDataset>(data_path, Shape{image_size, image_size, 3})
                : std::make_unique<ImageDataset>(data_path);
        ImageDataset& dataset = *dataset_ptr;
        dataset.set_sample_roles("Training");

        // CIFAR (image_size==0) fits in VRAM: keep it GPU-resident so each batch is
        // a device-side gather. The 224px ImageNet path is too large, so it stays
        // host-staged. Enabled in code; there is no environment switch.
        const bool gpu_resident = (image_size == 0) || force_resident;
        if (gpu_resident)
            dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);
        else
            dataset.set_storage_mode(Dataset::StorageMode::BinaryFile);

        const Index samples = dataset.get_samples_number();

        std::cout << "samples=" << samples << " batch=" << batch
                  << " epochs=" << timed_epochs << " precision=" << precision
                  << " cuda_graph=" << cuda_graph << " gpu_resident=" << gpu_resident;
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
        adam->set_cuda_graph(cuda_graph);

        // Probe mode (epochs<=0): only the CUDA warmup runs (peak allocation for
        // this batch), then report memory + fit. maximum_epochs<0 makes the epoch
        // loop run zero times, so no full epoch is trained -- fast OOM/fit check.
        if (timed_epochs <= 0)
        {
            adam->set_display(false);
            adam->set_maximum_epochs(0);
            training_strategy.train();
            memory_debug::print(std::cout);
            std::cout << "RESULT=OK\n";
            return 0;
        }

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
        memory_debug::print(std::cout);
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
