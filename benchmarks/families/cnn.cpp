// The CNN family, defined once, driven four ways.
//
// PLAN.md. ResNet-50 v1.5: bottleneck blocks [3,4,6,3], widths
// [64,128,256,512], on the pinned ImageNet subset -- 1000 classes at 50
// images each, 224x224. All 1000 classes are kept so the head is the real
// 2048x1000; a ten-class subset would be a different network.
//
//   cnn train    <train_dir> <test_dir> [epochs] [batch,...] [size] [dev] [prec]
//   cnn infer    <test_dir>             [reps]   [batch,...] [size] [dev] [prec]
//   cnn capacity <train_dir>            [batch]              [size] [dev] [prec]
//   cnn quality  <train_dir> <test_dir> [epochs] [batch]      [size] [dev] [prec]
//
// Images are lazy-loaded per batch from class folders, which both engines do,
// so this measures convolution throughput *plus* input-pipeline efficiency.
// That is deliberate: at 50,000 x 224x224x3 the split cannot be resident, so
// pretending otherwise would measure a workload nobody runs.

#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#endif

#include "opennn/core/configuration.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/tensor_types.h"
#include "opennn/dataset/image_dataset.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/models/models.h"
#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/training_strategy/training_strategy.h"

using namespace opennn;
using clock_type = chrono::steady_clock;

namespace
{

double resident_mb()
{
#ifdef __linux__
    // statm reports pages; the second field is resident.
    ifstream statm("/proc/self/statm");
    long total = 0, resident = 0;
    if (statm >> total >> resident)
        return double(resident) * double(sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0);
#endif
    return 0.0;
}

constexpr Index SEED = 42;

struct Options
{
    Index image_size = 224;
    Device device = Device::CUDA;
    Type precision = Type::FP32;
};

unique_ptr<ResNet> build(ImageDataset& dataset)
{
    set_seed(SEED);

    return make_unique<ResNet>(dataset.get_shape("Input"),
                               vector<Index>{3, 4, 6, 3},
                               Shape{64, 128, 256, 512},
                               dataset.get_shape("Target"),
                               true);
}

unique_ptr<ImageDataset> open_dataset(const string& path, const Options& options)
{
    return options.image_size > 0
        ? make_unique<ImageDataset>(path, Shape{options.image_size, options.image_size, 3})
        : make_unique<ImageDataset>(path);
}

vector<Index> parse_batches(const string& text)
{
    vector<Index> batches;
    stringstream stream(text);

    for (string item; getline(stream, item, ',');)
        if (!item.empty())
            batches.push_back(stoll(item));

    return batches;
}

Options parse_options(int argc, char* argv[], int first)
{
    Options options;

    if (argc > first)     options.image_size = stoll(argv[first]);
    if (argc > first + 1) options.device = string(argv[first + 1]) == "cpu" ? Device::CPU : Device::CUDA;
    if (argc > first + 2) options.precision = string(argv[first + 2]) == "bf16" ? Type::BF16 : Type::FP32;

    return options;
}

AdaptiveMomentEstimation* configure(TrainingStrategy& strategy, Index batch)
{
    strategy.set_loss("CrossEntropy");
    strategy.get_loss()->set_regularization("NoRegularization");
    strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

    auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(strategy.get_optimization_algorithm());
    adam->set_batch_size(batch);
    adam->set_display_period(1000000);
    adam->set_gradient_clip_norm(0.0f);

    return adam;
}

int usage()
{
    cerr << "usage: cnn train    <train_dir> <test_dir> [epochs] [batch,...] [size] [dev] [prec]\n"
            "       cnn infer    <test_dir>             [reps]   [batch,...] [size] [dev] [prec]\n"
            "       cnn capacity <train_dir>            [batch]              [size] [dev] [prec]\n";
    return 2;
}

}   // namespace

int main(int argc, char* argv[])
{
    // Each engine at its best, as PROTOCOL.md requires. The library defaults to
    // Eigen so a plain build behaves like a plain build; a build that has the
    // MKL kernels is told to use them here rather than inheriting them.
    Configuration::instance().set_blas(Blas::Mkl);
    cout << "blas=" << (blas_mkl_available() ? "mkl" : "eigen") << "\n";

    const string mode = argc > 1 ? argv[1] : "";

    if (mode == "train" || mode == "quality")
    {
        if (argc < 4) return usage();

        const Index epochs = argc > 4 ? stoll(argv[4]) : 1;
        const vector<Index> batches = parse_batches(argc > 5 ? argv[5] : "128");
        const Options options = parse_options(argc, argv, 6);

        Configuration::instance().set(options.device, options.precision);
        cout << "baseline_rss_mib=" << resident_mb() << "\n";
        cout << "engine=opennn\nmode=" << mode
             << "\ndevice=" << (options.device == Device::CPU ? "cpu" : "cuda") << "\n";

        auto dataset = open_dataset(argv[2], options);
        dataset->set_sample_roles("Training");

        const Index all_samples = dataset->get_samples_number();
        const bool timing = mode == "train";
        const Index warmup = timing ? 2 : 0;

        cout << "samples=" << all_samples << "\n";

        for (const Index batch : batches)
        {
            // Whole batches only, matching the PyTorch driver's
            // range(0, n - batch + 1, batch). A tail batch is real work the
            // throughput figure does not count, and the library keeps a second
            // set of activation contexts for it.
            dataset->set_sample_roles("Training");
            for (Index sample = (all_samples / batch) * batch; sample < all_samples; ++sample)
                dataset->set_sample_role(sample, SampleRole::None);

            auto network = build(*dataset);
            if (batch == batches.front())
                cout << "parameters=" << network->get_parameters_number() << "\n" << flush;

            const bool graph = options.device == Device::CUDA
                               && getenv("OPENNN_NO_CUDA_GRAPH") == nullptr;

            TrainingStrategy strategy(network.get(), dataset.get());
            auto* adam = configure(strategy, batch);
            adam->set_cuda_graph(graph);
            adam->set_maximum_epochs(warmup + epochs);

            const auto unix_now = []
            {
                return chrono::duration<double>(
                    chrono::system_clock::now().time_since_epoch()).count();
            };

            vector<double> epoch_seconds;
            auto previous_mark = clock_type::now();

            adam->post_epoch_callback = [&](Index epoch, float, float, NeuralNetwork*)
            {
                const auto now = clock_type::now();
                const double elapsed = chrono::duration<double>(now - previous_mark).count();
                previous_mark = now;

                if (epoch == warmup - 1)
                    cout << "TIMED_START_UNIX=" << fixed << setprecision(3)
                         << unix_now() << "\n" << defaultfloat;
                else if (epoch >= warmup)
                    epoch_seconds.push_back(elapsed);

                if (epoch == warmup + epochs - 1)
                    cout << "TIMED_END_UNIX=" << fixed << setprecision(3)
                         << unix_now() << "\n" << defaultfloat;
            };

            strategy.train();

            if (Index(epoch_seconds.size()) != epochs)
            {
                cerr << "epoch timing marks missing\n";
                return 1;
            }

            sort(epoch_seconds.begin(), epoch_seconds.end());
            const double median_epoch_s = epoch_seconds[epoch_seconds.size() / 2];
            const Index samples_per_epoch = (all_samples / batch) * batch;

            cout << "batch_" << batch << "_samples_per_sec="
                 << long(double(samples_per_epoch) / median_epoch_s)
                 << " median_epoch_s=" << median_epoch_s << "\n"
                 << "batch_" << batch << "_epoch_times=";
            for (size_t i = 0; i < epoch_seconds.size(); ++i)
                cout << (i ? "," : "") << epoch_seconds[i];
            cout << "\n"
                 << "batch_" << batch << "_cuda_graph="
                 << (!graph ? "off"
                     : adam->get_cuda_graph_capture_failed() ? "failed" : "captured")
                 << "\n" << flush;
        }
    }
    else if (mode == "infer")
    {
        if (argc < 3) return usage();

        const Index reps = argc > 3 ? stoll(argv[3]) : 1;
        const vector<Index> batches = parse_batches(argc > 4 ? argv[4] : "128");
        const Options options = parse_options(argc, argv, 5);

        Configuration::instance().set(options.device, options.precision);
        cout << "baseline_rss_mib=" << resident_mb() << "\n";
        cout << "engine=opennn\nmode=infer\ndevice="
             << (options.device == Device::CPU ? "cpu" : "cuda") << "\n";

        auto dataset = open_dataset(argv[2], options);
        dataset->set_sample_roles("Testing");

        const Index samples = dataset->get_samples_number();
        auto network = build(*dataset);

        cout << "samples=" << samples
             << " parameters=" << network->get_parameters_number() << "\n" << flush;

        for (const Index batch : batches)
        {
            const Index processed = (samples / batch) * batch;
            ForwardPropagation forward_propagation(batch, network.get(),
                                                  ForwardPropagationMode::Inference);
            forward_propagation.set_cuda_graph(options.device == Device::CUDA);

            // One batch is filled once and replayed: this measures the
            // resident forward pass, the same thing the PyTorch driver times
            // after its own warmup, rather than the image decode that
            // training already accounts for.
            vector<Index> indices(size_t(batch), Index(0));
            iota(indices.begin(), indices.end(), Index(0));

            Batch data(batch, dataset.get(), network->get_config());
            data.fill(indices, dataset->get_feature_selection(), FillMode::Inference);

            const vector<TensorView>& inputs = data.get_inputs();

            const auto run_pass = [&]
            {
                for (Index i = 0; i + batch <= samples; i += batch)
                    network->calculate_outputs_resident(inputs, forward_propagation, false);

                // The clock stops when the work is done, not when it is
                // queued, matching torch.cuda.synchronize() in the PyTorch
                // driver. Without it this cell reported passes of 0.0019 s,
                // 9.69 s and 25.3 s for identical work: the first pass timed
                // the launches and the ones after it paid for the backlog.
                if (options.device == Device::CUDA)
                    device::synchronize(device::get_compute_stream());
            };

            network->calculate_outputs_resident(inputs, forward_propagation, true);
            run_pass();

            const auto unix_now = []
            {
                return chrono::duration<double>(
                    chrono::system_clock::now().time_since_epoch()).count();
            };

            cout << "TIMED_START_UNIX=" << fixed << setprecision(3)
                 << unix_now() << "\n" << defaultfloat << flush;

            vector<double> times;
            for (Index r = 0; r < reps; ++r)
            {
                const auto t0 = clock_type::now();
                run_pass();
                times.push_back(chrono::duration<double>(clock_type::now() - t0).count());
            }

            cout << "TIMED_END_UNIX=" << fixed << setprecision(3)
                 << unix_now() << "\n" << defaultfloat << flush;

            cout << "batch_" << batch << "_pass_times=";
            for (size_t i = 0; i < times.size(); ++i) cout << (i ? "," : "") << times[i];
            cout << "\n";

            sort(times.begin(), times.end());
            const double median_pass_s = times[times.size() / 2];

            cout << "batch_" << batch << "_samples_per_sec="
                 << long(double(processed) / median_pass_s)
                 << " median_pass_s=" << median_pass_s << "\n" << flush;
        }
    }
    else if (mode == "capacity")
    {
        if (argc < 3) return usage();

        const Index batch = argc > 3 ? stoll(argv[3]) : 128;
        const Options options = parse_options(argc, argv, 4);

        Configuration::instance().set(options.device, options.precision);
        cout << "baseline_rss_mib=" << resident_mb() << "\n";
        cout << "engine=opennn\nmode=capacity\ndevice="
             << (options.device == Device::CPU ? "cpu" : "cuda")
             << "\nbatch=" << batch << "\n";

        // One attempt, then exit: an out-of-memory fault leaves the CUDA
        // context unusable, so a second attempt here would measure the wreck
        // of the first.
        try
        {
            auto dataset = open_dataset(argv[2], options);
            dataset->set_sample_roles("Training");

            auto network = build(*dataset);
            cout << "parameters=" << network->get_parameters_number() << "\n" << flush;

            TrainingStrategy strategy(network.get(), dataset.get());
            configure(strategy, batch)->set_maximum_epochs(1);
            strategy.train();
        }
        catch (const exception& error)
        {
            cout << "fits=0\nreason=" << error.what() << "\nRESULT=OOM\n" << flush;
            return 1;
        }

        cout << "fits=1\nRESULT=OK\n" << flush;
        return 0;
    }
    else
    {
        return usage();
    }

    cout << "RESULT=OK\n";

    return 0;
}
