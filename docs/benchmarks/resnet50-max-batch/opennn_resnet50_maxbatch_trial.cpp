//   OpenNN GPU ResNet-50 max-batch trial.
//
//   One invocation = one batch attempt in its own process. The Python driver
//   grows and binary-searches the batch size around this program so CUDA OOMs
//   cannot poison later trials.
//
//   CUDA graph, sample shuffle and cuDNN conv autotune are all turned off in
//   code (no environment variables); the prefetch-pool depth is set with the
//   optional [batch_pool] argument (the pool1 engine passes 1).
//
//   usage: opennn_resnet50_maxbatch_trial <cifar10_dir> <batch> [fp32] [batch_pool] [workspace_mib]
//          workspace_mib: 0 (default) = AUTO library policy; >0 = explicit conv workspace cap

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include "../../../opennn/adaptive_moment_estimation.h"
#include "../../../opennn/configuration.h"
#include "../../../opennn/device_backend.h"
#include "../../../opennn/image_dataset.h"
#include "../../../opennn/memory_debug.h"
#include "../../../opennn/random_utilities.h"
#include "../../../opennn/standard_networks.h"
#include "../../../opennn/training_strategy.h"

using namespace opennn;

namespace
{

constexpr Index kClasses = 10;

struct TempImageTree
{
    std::filesystem::path root;

    ~TempImageTree()
    {
        if (!root.empty())
        {
            std::error_code ec;
            std::filesystem::remove_all(root, ec);
        }
    }
};

std::vector<std::pair<std::filesystem::path, std::string>>
collect_cifar_images(const std::filesystem::path& train_dir)
{
    namespace fs = std::filesystem;

    throw_if(!fs::is_directory(train_dir), "Missing CIFAR-10 train directory: " + train_dir.string());

    std::vector<fs::path> class_dirs;
    for (const fs::directory_entry& entry : fs::directory_iterator(train_dir))
        if (entry.is_directory() && !entry.path().filename().string().starts_with('.'))
            class_dirs.push_back(entry.path());
    std::ranges::sort(class_dirs);

    throw_if(ssize(class_dirs) != kClasses,
             "Expected 10 CIFAR-10 class folders under: " + train_dir.string());

    std::vector<std::pair<fs::path, std::string>> samples;
    for (const fs::path& class_dir : class_dirs)
    {
        std::vector<fs::path> files;
        for (const fs::directory_entry& entry : fs::directory_iterator(class_dir))
            if (entry.is_regular_file() || entry.is_symlink())
                files.push_back(entry.path());
        std::ranges::sort(files);

        const std::string class_name = class_dir.filename().string();
        for (const fs::path& file : files)
            samples.emplace_back(file, class_name);
    }

    throw_if(samples.empty(), "No CIFAR-10 images found under: " + train_dir.string());
    return samples;
}

std::filesystem::path make_repeated_image_tree(const std::string& data_dir,
                                               Index batch,
                                               TempImageTree& temp)
{
    namespace fs = std::filesystem;

    const fs::path train_dir = fs::path(data_dir) / "train";
    const auto samples = collect_cifar_images(train_dir);

    temp.root = fs::temp_directory_path()
              / ("opennn_resnet50_maxbatch_"
                 + std::to_string(static_cast<long long>(getpid()))
                 + "_" + std::to_string(static_cast<long long>(batch)));

    fs::create_directories(temp.root);

    for (const auto& sample : samples)
        fs::create_directories(temp.root / sample.second);

    for (Index i = 0; i < batch; ++i)
    {
        const auto& [source, class_name] = samples[size_t(i % ssize(samples))];
        const fs::path link = temp.root / class_name
            / ("sample_" + std::to_string(static_cast<long long>(i)) + source.extension().string());

        std::error_code ec;
        fs::create_symlink(fs::absolute(source), link, ec);
        if (ec)
            fs::copy_file(source, link, fs::copy_options::overwrite_existing);
    }

    return temp.root;
}

} // namespace

int main(int argc, char* argv[])
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    const std::string data_dir = argc > 1 ? argv[1] : "../resnet50-training-speed/cifar10";
    const Index batch = argc > 2 ? Index(std::stoll(argv[2])) : 128;
    const std::string precision = argc > 3 ? argv[3] : "fp32";
    const int batch_pool = argc > 4 ? std::stoi(argv[4]) : 0;   // 0 = library default
    // Conv workspace cap A/B: "off"/"0" = autotune (uncapped, colleague default),
    // "auto" = AUTO cap (largest layer activation), or a positive integer = MiB cap.
    const std::string workspace_arg = argc > 5 ? argv[5] : "off";

    try
    {
        memory_debug::reset();

        throw_if(batch <= 0, "Batch size must be positive.");
        throw_if(precision != "fp32", "Only fp32 is supported by this benchmark trial.");

        set_seed(42);
        Configuration::instance().set(Device::CUDA, Type::FP32);

        if (workspace_arg == "off" || workspace_arg == "0")
            device::set_conv_workspace_cap(0);
        else if (workspace_arg == "auto")
            device::set_conv_workspace_cap(-1);
        else
            device::set_conv_workspace_cap(std::stoll(workspace_arg) * 1024 * 1024);
        std::cout << "workspace_mode=" << workspace_arg << "\n";

        TempImageTree temp_images;
        const std::filesystem::path trial_data_path =
            make_repeated_image_tree(data_dir, batch, temp_images);

        ImageDataset dataset(trial_data_path);
        dataset.set_sample_roles("Training");
        dataset.set_display(false);

        ResNet network(dataset.get_shape("Input"),
                       {3, 4, 6, 3},
                       Shape{64, 128, 256, 512},
                       dataset.get_shape("Target"),
                       true);
        memory_debug::record("model", "NeuralNetwork::parameters",
                             network.get_parameters_size() * Index(sizeof(float)),
                             "planned");
        memory_debug::record("model", "NeuralNetwork::states",
                             network.get_states_buffer_size() * Index(sizeof(float)),
                             "planned");

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.set_loss("CrossEntropy");
        training_strategy.get_loss()->set_regularization("NoRegularization");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        throw_if(!adam, "AdaptiveMomentEstimation optimizer was not created.");

        adam->set_batch_size(batch);
        adam->set_maximum_epochs(0);
        adam->set_display(false);
        adam->set_display_period(1000000);
        adam->set_gradient_clip_norm(0.0f);

        // Max-batch probe: one batch == the whole set, so there is no step-to-step
        // overlap for a CUDA graph to amortise and nothing to shuffle. Both are
        // therefore left off in code (no environment switch).
        adam->set_cuda_graph(false);
        adam->set_shuffle(false);
        // Prefetch-pool depth (0 = library default); the pool1 engine passes 1 to
        // hold the fewest device batch copies and reach the largest batch.
        adam->set_batch_pool_size(batch_pool);

        const TrainingResult result = training_strategy.train();
        const float training_error = result.get_training_error();
        throw_if(!std::isfinite(training_error), "Training error is not finite.");

        std::cout << "engine=opennn\n";
        std::cout << "model=ResNet-50-v1.5-CIFAR\n";
        std::cout << "samples=" << batch << " batch=" << batch
                  << " precision=" << precision << "\n";
        std::cout << "storage=ImageDataset BinaryFile cache\n";
        std::cout << "gpu_resident_data=0\n";
        std::cout << "parameters=" << network.get_parameters_size() << "\n";
        std::cout << "training_error=" << training_error << "\n";
        memory_debug::print(std::cout);
        std::cout << "RESULT=OK\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "FAIL batch=" << batch << " : " << e.what() << "\n";
        memory_debug::print(std::cout);
        std::cout << "RESULT=ERROR\n";
        return 1;
    }
}
