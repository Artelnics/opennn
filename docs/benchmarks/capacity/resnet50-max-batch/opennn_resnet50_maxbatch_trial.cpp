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
//   usage: opennn_resnet50_maxbatch_trial <cifar10_dir> <batch> [fp32|bf16] [batch_pool] [workspace_mib]
//          workspace_mib: 0 (default) = AUTO library policy; >0 = explicit conv workspace cap

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include "opennn/adaptive_moment_estimation.h"
#include "opennn/configuration.h"
#include "opennn/device_backend.h"
#include "opennn/image_dataset.h"
#include "opennn/memory_debug.h"
#include "opennn/random_utilities.h"
#include "opennn/standard_networks.h"
#include "opennn/training_strategy.h"

using namespace opennn;

namespace
{

constexpr Index kClasses = 10;

struct TempImageTree
{
    filesystem::path root;

    ~TempImageTree()
    {
        if (!root.empty())
        {
            error_code ec;
            filesystem::remove_all(root, ec);
        }
    }
};

vector<pair<filesystem::path, string>>
collect_cifar_images(const filesystem::path& train_dir)
{
    namespace fs = filesystem;

    throw_if(!fs::is_directory(train_dir), "Missing CIFAR-10 train directory: " + train_dir.string());

    vector<fs::path> class_dirs;
    for (const fs::directory_entry& entry : fs::directory_iterator(train_dir))
        if (entry.is_directory() && !entry.path().filename().string().starts_with('.'))
            class_dirs.push_back(entry.path());
    ranges::sort(class_dirs);

    throw_if(ssize(class_dirs) != kClasses,
             "Expected 10 CIFAR-10 class folders under: " + train_dir.string());

    vector<pair<fs::path, string>> samples;
    for (const fs::path& class_dir : class_dirs)
    {
        vector<fs::path> files;
        for (const fs::directory_entry& entry : fs::directory_iterator(class_dir))
            if (entry.is_regular_file() || entry.is_symlink())
                files.push_back(entry.path());
        ranges::sort(files);

        const string class_name = class_dir.filename().string();
        for (const fs::path& file : files)
            samples.emplace_back(file, class_name);
    }

    throw_if(samples.empty(), "No CIFAR-10 images found under: " + train_dir.string());
    return samples;
}

filesystem::path make_repeated_image_tree(const string& data_dir,
                                               Index batch,
                                               TempImageTree& temp)
{
    namespace fs = filesystem;

    const fs::path train_dir = fs::path(data_dir) / "train";
    const auto samples = collect_cifar_images(train_dir);

    temp.root = fs::temp_directory_path()
              / ("opennn_resnet50_maxbatch_"
                 + to_string(static_cast<long long>(getpid()))
                 + "_" + to_string(static_cast<long long>(batch)));

    fs::create_directories(temp.root);

    for (const auto& sample : samples)
        fs::create_directories(temp.root / sample.second);

    for (Index i = 0; i < batch; ++i)
    {
        const auto& [source, class_name] = samples[size_t(i % ssize(samples))];
        const fs::path link = temp.root / class_name
            / ("sample_" + to_string(static_cast<long long>(i)) + source.extension().string());

        error_code ec;
        fs::create_symlink(fs::absolute(source), link, ec);
        if (ec)
            fs::copy_file(source, link, fs::copy_options::overwrite_existing);
    }

    return temp.root;
}

}

int main(int argc, char* argv[])
{
    cout << unitbuf;
    cerr << unitbuf;

    const char* bench_data_env = getenv("OPENNN_BENCH_DATA");
    const char* home_env = getenv("HOME");
    const string default_data_dir =
        (bench_data_env && *bench_data_env
             ? string(bench_data_env)
             : string(home_env ? home_env : ".") + "/opennn-benchmark-data")
        + "/cifar10";
    const string data_dir = argc > 1 ? argv[1] : default_data_dir;
    const Index batch = argc > 2 ? Index(stoll(argv[2])) : 128;
    const string precision = argc > 3 ? argv[3] : "fp32";
    const int batch_pool = argc > 4 ? stoi(argv[4]) : 0;
    const string workspace_arg = argc > 5 ? argv[5] : "off";

    try
    {
        memory_debug::reset();

        throw_if(batch <= 0, "Batch size must be positive.");
        throw_if(precision != "fp32" && precision != "bf16",
                 "Precision must be fp32 or bf16.");

        set_seed(42);
        const Type training_type = (precision == "bf16") ? Type::BF16 : Type::FP32;
        Configuration::instance().set(Device::CUDA, training_type);

        if (workspace_arg == "off" || workspace_arg == "0")
            { device::set_conv_autotune(true);  device::set_conv_workspace_cap(0); }
        else if (workspace_arg == "heur")
            { device::set_conv_autotune(false); device::set_conv_workspace_cap(0); }
        else if (workspace_arg == "auto")
            device::set_conv_workspace_cap(-1);
        else
            device::set_conv_workspace_cap(stoll(workspace_arg) * 1024 * 1024);
        cout << "workspace_mode=" << workspace_arg << "\n";

        TempImageTree temp_images;
        const filesystem::path trial_data_path =
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

        adam->set_cuda_graph(false);
        adam->set_shuffle(false);
        adam->set_batch_pool_size(batch_pool);

        const TrainingResult result = training_strategy.train();
        const float training_error = result.get_training_error();
        throw_if(!isfinite(training_error), "Training error is not finite.");

        cout << "engine=opennn\n";
        cout << "model=ResNet-50-v1.5-CIFAR\n";
        cout << "samples=" << batch << " batch=" << batch
                  << " precision=" << precision << "\n";
        cout << "storage=ImageDataset BinaryFile cache\n";
        cout << "gpu_resident_data=0\n";
        cout << "parameters=" << network.get_parameters_size() << "\n";
        cout << "training_error=" << training_error << "\n";
        memory_debug::print(cout);
        cout << "RESULT=OK\n";
        return 0;
    }
    catch (const exception& e)
    {
        cerr << "FAIL batch=" << batch << " : " << e.what() << "\n";
        memory_debug::print(cout);
        cout << "RESULT=ERROR\n";
        return 1;
    }
}
