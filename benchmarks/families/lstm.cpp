// The LSTM family, defined once, driven four ways.
//
// LSTM forecasting on UCI Beijing PM2.5, hourly, predicting the next
// reading from a window of past ones.
//
//   lstm train    <csv> <csv> [epochs] [batch,...] [hidden] [past] [dev] [prec]
//   lstm infer    <csv>       [reps]   [batch,...] [hidden] [past] [dev] [prec]
//   lstm capacity <csv>       [batch]              [hidden] [past] [dev] [prec]
//
// LSTM rather than the plain recurrent layer, for two reasons. It is the
// architecture sequence-forecasting results are reported on, and both engines
// route it to the *same* NVIDIA kernel -- OpenNN through
// `cudnn_rnn_forward_`, PyTorch through cuDNN behind `nn.LSTM`. That makes
// this the cleanest cell in the matrix: with the arithmetic identical, what is
// left to measure is the surrounding machinery -- data movement, launch
// overhead, the optimiser -- rather than two teams' hand-written kernels.
//
// `set_parameters_pytorch()` initialises to PyTorch's convention, so the two
// engines start from the same distribution rather than merely the same seed.

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifdef __linux__
#include <unistd.h>
#endif

#include "opennn/core/configuration.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/tensor_types.h"
#include "opennn/dataset/time_series_dataset.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/layers/long_short_term_memory_layer.h"
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

bool enable_flush_denormals()
{
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    // MXCSR bit 15 is flush-to-zero and bit 6 is denormals-are-zero. They are
    // thread-local, so set them before any OpenMP team is created; workers then
    // inherit the same policy. PyTorch's driver calls set_flush_denormal(true).
    constexpr unsigned int ftz_and_daz = (1U << 15U) | (1U << 6U);
    _mm_setcsr(_mm_getcsr() | ftz_and_daz);
    return (_mm_getcsr() & ftz_and_daz) == ftz_and_daz;
#else
    return false;
#endif
}

struct Options
{
    Index hidden = 128;
    Index past = 24;            // one day of hourly readings
    Device device = Device::CUDA;
    Type precision = Type::FP32;
};

unique_ptr<ForecastingLstmNetwork> build(TimeSeriesDataset& dataset, const Options& options)
{
    set_seed(SEED);

    auto network = make_unique<ForecastingLstmNetwork>(dataset.get_shape("Input"),
                                                       Shape{options.hidden},
                                                       dataset.get_shape("Target"));
    network->set_parameters_pytorch();

    return network;
}

unique_ptr<NeuralNetwork> build_inference(const TimeSeriesDataset& dataset,
                                          const Options& options)
{
    set_seed(SEED);

    auto network = make_unique<NeuralNetwork>();
    network->set_task(NetworkTask::Forecasting);

    auto recurrent = make_unique<LongShortTermMemory>(dataset.get_shape("Input"),
                                                       Shape{options.hidden},
                                                       "Tanh", "Sigmoid",
                                                       "long_short_term_memory_layer");
    recurrent->set_return_sequences(false);
    network->add_layer(std::move(recurrent));

    network->add_layer(make_unique<opennn::Dense>(network->get_output_shape(),
                                                  dataset.get_shape("Target"),
                                                  "Identity",
                                                  BatchNormalization::No,
                                                  "forecasting_layer"));

    network->compile();
    network->set_parameters_pytorch();

    return network;
}

unique_ptr<TimeSeriesDataset> open_dataset(const string& path, const Options& options)
{
    cout << "dataset_opened=" << filesystem::absolute(path).string() << "\n" << flush;
    auto dataset = make_unique<TimeSeriesDataset>(path, ",", true, false);
    dataset->set_past_time_steps(options.past);
    dataset->set_future_time_steps(1);

    if (options.device == Device::CUDA)
        dataset->set_storage_mode(Dataset::StorageMode::GPUPersistantData);

    return dataset;
}

Index use_all_valid_windows(TimeSeriesDataset& dataset, SampleRole role,
                            const Options& options)
{
    const Index windows = max(Index(0), dataset.get_samples_number()
                                      - options.past
                                      - dataset.get_future_time_steps() + 1);
    vector<Index> valid(static_cast<size_t>(windows));
    iota(valid.begin(), valid.end(), Index(0));

    // TimeSeriesDataset normally installs a chronological 60/20/20 split.
    // This benchmark, like the PyTorch driver, uses the complete CSV. Keep
    // only starts whose full input window and target are in range; marking
    // every raw row used to append `past` zero-padded pseudo-windows.
    dataset.set_sample_roles(SampleRole::None);
    dataset.set_sample_roles(valid, role);
    return windows;
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

    if (argc > first)     options.hidden = stoll(argv[first]);
    if (argc > first + 1) options.past = stoll(argv[first + 1]);
    if (argc > first + 2) options.device = string(argv[first + 2]) == "cpu" ? Device::CPU : Device::CUDA;
    if (argc > first + 3) options.precision = string(argv[first + 3]) == "bf16" ? Type::BF16 : Type::FP32;

    return options;
}

AdaptiveMomentEstimation* configure(TrainingStrategy& strategy, Index batch)
{
    strategy.set_loss("MeanSquaredError");
    strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

    auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(strategy.get_optimization_algorithm());
    adam->set_batch_size(batch);
    adam->set_display(false);
    adam->set_display_period(1000000);
    adam->set_gradient_clip_norm(0.0f);

    return adam;
}

void describe(const TimeSeriesDataset& dataset, const NeuralNetwork& network, const Options& options)
{
    cout << "samples=" << dataset.get_used_samples_number()
         << " inputs=" << dataset.get_shape("Input").back()
         << " past=" << options.past
         << " hidden=" << options.hidden
         << " parameters=" << network.get_parameters_number() << "\n" << flush;
}

int usage()
{
    cerr << "usage: lstm train    <csv> <csv> [epochs] [batch,...] [hidden] [past] [dev] [prec]\n"
            "       lstm infer    <csv>       [reps]   [batch,...] [hidden] [past] [dev] [prec]\n"
            "       lstm capacity <csv>       [batch]              [hidden] [past] [dev] [prec]\n";
    return 2;
}

}   // namespace

int main(int argc, char* argv[])
{
    cout << "flush_denormals="
         << (enable_flush_denormals() ? "on" : "unsupported") << "\n";

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
        const vector<Index> batches = parse_batches(argc > 5 ? argv[5] : "256");
        const Options options = parse_options(argc, argv, 6);

        Configuration::instance().set(options.device, options.precision);
        cout << "baseline_rss_mib=" << resident_mb() << "\n";
        cout << "engine=opennn\nmode=" << mode
             << "\ndevice=" << (options.device == Device::CPU ? "cpu" : "cuda") << "\n";

        auto dataset = open_dataset(argv[2], options);
        const Index samples = use_all_valid_windows(*dataset, SampleRole::Training, options);

        const bool timing = mode == "train";
        const Index warmup = timing ? 2 : 0;

        for (const Index batch : batches)
        {
            auto network = build(*dataset, options);
            if (batch == batches.front()) describe(*dataset, *network, options);

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

            cout << "batch_" << batch << "_samples_per_sec="
                 << long(double((samples / batch) * batch) / median_epoch_s)
                 << " median_epoch_s=" << median_epoch_s << "\n"
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
        const vector<Index> batches = parse_batches(argc > 4 ? argv[4] : "256");
        const Options options = parse_options(argc, argv, 5);

        Configuration::instance().set(options.device, options.precision);
        cout << "baseline_rss_mib=" << resident_mb() << "\n";
        cout << "engine=opennn\nmode=infer\ndevice="
             << (options.device == Device::CPU ? "cpu" : "cuda") << "\n";

        auto dataset = open_dataset(argv[2], options);
        const Index samples = use_all_valid_windows(*dataset, SampleRole::Testing, options);

        for (const Index batch : batches)
        {
            // Match PyTorch's timed nn.LSTM + nn.Linear exactly. The full
            // ForecastingLstmNetwork remains the train/quality/capacity model;
            // its Scaling, Unscaling, and Clamping endpoints are deliberately
            // absent from the inference throughput graph.
            auto network = build_inference(*dataset, options);
            if (batch == batches.front()) describe(*dataset, *network, options);

            const Index processed = (samples / batch) * batch;

            ForwardPropagation forward_propagation(batch, network.get(),
                                                   ForwardPropagationMode::Inference);
            const bool graph = options.device == Device::CUDA
                               && getenv("OPENNN_NO_CUDA_GRAPH") == nullptr;
            forward_propagation.set_cuda_graph(graph);

            vector<Index> indices(size_t(batch), Index(0));
            for (Index k = 0; k < batch; ++k) indices[size_t(k)] = k;

            Batch data(batch, dataset.get(), network->get_config());
            data.fill(indices, dataset->get_feature_selection(), FillMode::Inference);

            // fill() stages a CUDA batch on the host; the transfer is a
            // separate stream operation, issued here once, before the clock,
            // so the replayed pass runs on the windows it was filled with.
            if (options.device == Device::CUDA)
            {
                data.upload_to_device_batch_async(data, device::get_transfer_stream());
                data.wait_h2d_on_compute_stream();
            }

            const vector<TensorView>& inputs = data.get_inputs();

            const auto run_once = [&](bool upload_parameters)
            {
                if (options.device == Device::CUDA)
                    network->calculate_outputs_resident(inputs, forward_propagation,
                                                        upload_parameters);
                else
                    network->forward_propagate(inputs, forward_propagation,
                                               ForwardPropagationMode::Inference);
            };

            const auto run_pass = [&]
            {
                for (Index i = 0; i + batch <= samples; i += batch)
                    run_once(false);

                // CUDA calls are asynchronous. End the pass at completion so
                // this measures kernel execution, as the PyTorch benchmark's
                // torch.cuda.synchronize() does, rather than launch latency.
                if (options.device == Device::CUDA)
                    device::synchronize(device::get_compute_stream());
            };

            // Uploading parameters is a device operation -- a CPU-compiled
            // network has no device copy and `copy_parameters_device` throws,
            // which aborted every CPU inference cell in this family. The
            // warm-up pass still has to happen, so it is the upload that is
            // conditional, not the pass.
            run_once(options.device == Device::CUDA);
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

            sort(times.begin(), times.end());
            const double median_pass_s = times[times.size() / 2];

            cout << "batch_" << batch << "_samples_per_sec="
                 << long(double(processed) / median_pass_s)
                 << " median_pass_s=" << median_pass_s << "\n"
                 << "batch_" << batch << "_cuda_graph="
                 << (!graph ? "off"
                     : forward_propagation.cuda_graph_failed ? "failed"
                     : forward_propagation.inference_graph_exec ? "captured" : "warming")
                 << "\n" << flush;
        }
    }
    else if (mode == "capacity")
    {
        if (argc < 3) return usage();

        const Index batch = argc > 3 ? stoll(argv[3]) : 256;
        const Options options = parse_options(argc, argv, 4);

        Configuration::instance().set(options.device, options.precision);
        cout << "baseline_rss_mib=" << resident_mb() << "\n";
        cout << "engine=opennn\nmode=capacity\ndevice="
             << (options.device == Device::CPU ? "cpu" : "cuda")
             << "\nbatch=" << batch << "\n";

        try
        {
            auto dataset = open_dataset(argv[2], options);
            use_all_valid_windows(*dataset, SampleRole::Training, options);

            auto network = build(*dataset, options);
            describe(*dataset, *network, options);

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
