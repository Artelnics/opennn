//   OpenNN GPU ResNet-50 max-batch trial: one main function, written the way a
//   user writes an OpenNN application, plus a timer.
//
//   One invocation = one batch attempt in its own process. The Python driver
//   (run_resnet50_maxbatch.py) grows and binary-searches the batch size around
//   this program so CUDA OOMs cannot poison later trials.
//
//   CUDA graph and sample shuffle are turned off in code. The prefetch-pool
//   depth and convolution workspace policy are explicit trial arguments so a
//   capacity run cannot accidentally inherit the throughput configuration.
//
//   usage: opennn_resnet50_maxbatch_trial <cifar10_dir> <batch> [fp32|bf16] [batch_pool] [workspace_mib] [recompute]
//          workspace_mib: positive integer (default 16) = explicit cap
//                         auto = library auto cap, heur = uncapped heuristic,
//                         off = uncapped autotune (throughput/debug only)
//          recompute:     1 (default) = training activation recomputation,
//                         0 = the non-recomputed control
//   env:   OPENNN_BENCH_DATA   -> default <cifar10_dir> is $OPENNN_BENCH_DATA/cifar10
//                                (else ~/opennn-benchmark-data/cifar10)
//          OPENNN_BENCH_SEED=N -> seed (default 42)
//          OPENNN_TARGET_LOSS=x
//              stop at the first step at or below x (OPENNN_MAX_STEPS is the
//              ceiling, default 1) and print the TRAIN_*_UNIX markers and
//              history a time/energy-to-target harness reads

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/memory_debug.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/image_dataset.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/training_strategy/training_strategy.h"

using namespace opennn;

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
    const string workspace_arg = argc > 5 ? argv[5] : "16";
    const bool recompute_activations = argc <= 6 || stoi(argv[6]) != 0;

    // The trial's own image tree (built below), removed on every exit path.
    filesystem::path trial_data_path;
    int exit_code = 0;

    try
    {
        memory_debug::reset();

        throw_if(batch <= 0, "Batch size must be positive.");
        throw_if(precision != "fp32" && precision != "bf16",
                 "Precision must be fp32 or bf16.");

        const char* seed_env = getenv("OPENNN_BENCH_SEED");
        set_seed(seed_env && *seed_env ? stoi(seed_env) : 42);
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
        cout << "workspace_cap_mib="
             << device::conv_workspace_limit_bytes() / (1024 * 1024) << "\n";
        cout << "conv_autotune=" << (device::conv_autotune_enabled() ? 1 : 0) << "\n";

        // ImageDataset reads a class-folder tree, so the trial builds one of
        // exactly `batch` samples from the CIFAR-10 training images: cycling
        // through the ten classes and, within a class, through its images in
        // index order, repeating modulo when the batch exceeds the source --
        // the same convention as the PyTorch and TensorFlow trials. Symlinks
        // where the filesystem allows them, copies otherwise.
        namespace fs = filesystem;
        constexpr Index classes_number = 10;

        const fs::path train_dir = fs::path(data_dir) / "train";
        throw_if(!fs::is_directory(train_dir), "Missing CIFAR-10 train directory: " + train_dir.string());

        vector<fs::path> class_dirs;
        for (const fs::directory_entry& entry : fs::directory_iterator(train_dir))
            if (entry.is_directory() && !entry.path().filename().string().starts_with('.'))
                class_dirs.push_back(entry.path());
        ranges::sort(class_dirs);

        throw_if(ssize(class_dirs) != classes_number,
                 "Expected 10 CIFAR-10 class folders under: " + train_dir.string());

        vector<vector<fs::path>> images_by_class(class_dirs.size());
        Index images_number = 0;
        for (size_t c = 0; c < class_dirs.size(); ++c)
        {
            vector<fs::path>& files = images_by_class[c];
            for (const fs::directory_entry& entry : fs::directory_iterator(class_dirs[c]))
                if (entry.is_regular_file() || entry.is_symlink())
                    files.push_back(entry.path());
            ranges::sort(files, [](const fs::path& left, const fs::path& right)
            {
                const auto sample_index = [](const fs::path& path)
                {
                    const string stem = path.stem().string();
                    const size_t separator = stem.rfind('_');
                    return separator == string::npos
                        ? Index(0) : Index(stoll(stem.substr(separator + 1)));
                };
                return sample_index(left) < sample_index(right);
            });
            images_number += ssize(files);
        }

        throw_if(images_number == 0, "No CIFAR-10 images found under: " + train_dir.string());

#ifdef _WIN32
        const int pid = _getpid();
#else
        const int pid = getpid();
#endif
        trial_data_path = fs::temp_directory_path()
            / ("opennn_resnet50_maxbatch_" + to_string(static_cast<long long>(pid))
               + "_" + to_string(static_cast<long long>(batch)));

        fs::create_directories(trial_data_path);

        for (size_t c = 0; c < class_dirs.size(); ++c)
            if (!images_by_class[c].empty())
                fs::create_directories(trial_data_path / class_dirs[c].filename());

        for (Index i = 0; i < batch; ++i)
        {
            const size_t class_index = size_t(i % classes_number);
            const vector<fs::path>& sources = images_by_class[class_index];
            const fs::path& source = sources[size_t(i / classes_number) % sources.size()];
            const fs::path link = trial_data_path / class_dirs[class_index].filename()
                / ("sample_" + to_string(static_cast<long long>(i)) + source.extension().string());

            error_code ec;
            fs::create_symlink(fs::absolute(source), link, ec);
            if (ec)
                fs::copy_file(source, link, fs::copy_options::overwrite_existing);
        }

        ImageDataset dataset(trial_data_path);
        dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);
        dataset.set_sample_roles("Training");
        dataset.set_display(false);

        ResNet network(dataset.get_shape("Input"),
                       {3, 4, 6, 3},
                       Shape{64, 128, 256, 512},
                       dataset.get_shape("Target"),
                       true);
        network.set_training_activation_recomputation(recompute_activations);
        memory_debug::record("model", "NeuralNetwork::parameters",
                             network.get_parameters_buffer_size() * Index(sizeof(float)),
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

        // OPENNN_TARGET_LOSS turns the capacity probe into a time-to-target
        // run: up to OPENNN_MAX_STEPS epochs, stopping at the first at or
        // below the target, with the markers and history a harness reads.
        const char* target_env = getenv("OPENNN_TARGET_LOSS");
        const bool target_mode = target_env && *target_env;
        const float target = target_mode ? stof(target_env) : 0.0f;
        const char* steps_env = getenv("OPENNN_MAX_STEPS");
        const Index max_steps = steps_env && *steps_env
            ? max<Index>(Index(1), Index(stoll(steps_env)))
            : Index(1);
        if (target_mode)
        {
            adam->set_maximum_epochs(max_steps);
            adam->set_loss_goal(target);
        }

        if (target_mode)
            cout << "TRAIN_START_UNIX=" << fixed << setprecision(3)
                 << chrono::duration<double>(chrono::system_clock::now().time_since_epoch()).count()
                 << "\n" << defaultfloat;
        const auto train_start = chrono::steady_clock::now();
        const TrainingResult result = training_strategy.train();
        const auto train_end = chrono::steady_clock::now();
        if (target_mode)
            cout << "TRAIN_END_UNIX=" << fixed << setprecision(3)
                 << chrono::duration<double>(chrono::system_clock::now().time_since_epoch()).count()
                 << "\n" << defaultfloat;
        const float training_error = result.get_training_error();
        throw_if(!isfinite(training_error), "Training error is not finite.");

        cout << "engine=opennn\n";
        cout << "model=ResNet-50-v1.5-CIFAR\n";
        cout << "samples=" << batch << " batch=" << batch
             << " precision=" << precision << "\n";
        cout << "storage=ImageDataset GPU-persistent cache\n";
        cout << "gpu_resident_data=1\n";
        cout << "training_activation_recomputation="
             << (recompute_activations ? 1 : 0) << "\n";
        cout << "parameters=" << network.get_parameters_buffer_size() << "\n";
        cout << "training_error=" << training_error << "\n";
        if (target_mode)
        {
            const Index steps_run = result.get_epochs_number();
            const double wall_s =
                chrono::duration<double>(train_end - train_start).count();
            cout << "target=" << target << "\n";
            cout << "steps_run=" << steps_run << "\n";
            cout << "epochs_run=" << steps_run << "\n";
            cout << "final_error=" << training_error << "\n";
            cout << "reached_goal=" << (training_error <= target ? 1 : 0) << "\n";
            cout << "loss_history=";
            for (Index step = 0; step < result.training_error_history.size(); ++step)
                cout << (step ? "," : "") << result.training_error_history(step);
            cout << "\n";
            cout << "wall_s=" << wall_s << "\n";
            cout << "samples_per_sec="
                 << double(batch) * double(steps_run) / wall_s << "\n";
        }
        memory_debug::print(cout);
        cout << "RESULT=OK\n";
    }
    catch (const exception& e)
    {
        cerr << "FAIL batch=" << batch << " : " << e.what() << "\n";
        memory_debug::print(cout);
        cout << "RESULT=ERROR\n";
        exit_code = 1;
    }

    if (!trial_data_path.empty())
    {
        error_code ec;
        filesystem::remove_all(trial_data_path, ec);
    }

    return exit_code;
}
