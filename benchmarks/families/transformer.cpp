// The transformer family, defined once, driven four ways.
//
// PLAN.md. The "Attention Is All You Need" base model -- d_model 512, 8
// heads, feed-forward 2048, 6 layers -- on WMT14 English-German, which is the
// corpus that paper trained and reported on and therefore the citable one.
// Heads and feed-forward width follow d_model by the paper's own ratios
// (d_model/64 heads, 4*d_model feed-forward) rather than being separate
// knobs, so a change to d_model cannot silently produce a model the paper
// would not recognise.
//
//   transformer train    <corpus> <corpus> [epochs] [batch,...] [d_model] [layers] [dev] [prec]
//   transformer infer    <corpus>          [reps]   [batch,...] [d_model] [layers] [dev] [prec]
//   transformer capacity <corpus>          [batch]              [d_model] [layers] [dev] [prec]
//
// Throughput is reported both per sequence and per token. Per token is what
// the literature quotes; per sequence is what the runner compares, and the
// two differ by the padded sequence length, which the corpus fixes.

#include <algorithm>
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
#include "opennn/core/random_utilities.h"
#include "opennn/core/tensor_types.h"
#include "opennn/dataset/language_dataset.h"
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
    Index d_model = 512;
    Index layers = 6;
    Device device = Device::CUDA;
    Type precision = Type::FP32;

    Index heads() const { return max(Index(1), d_model / 64); }
    Index feed_forward() const { return 4 * d_model; }
};

unique_ptr<Transformer> build(LanguageDataset& dataset, const Options& options)
{
    set_seed(SEED);

    return make_unique<Transformer>(dataset.get_shape("Input")[0],
                                    dataset.get_shape("Decoder")[0],
                                    dataset.get_input_vocabulary_size(),
                                    dataset.get_target_vocabulary_size(),
                                    options.d_model,
                                    options.heads(),
                                    options.feed_forward(),
                                    options.layers);
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

    if (argc > first)     options.d_model = stoll(argv[first]);
    if (argc > first + 1) options.layers = stoll(argv[first + 1]);
    if (argc > first + 2) options.device = string(argv[first + 2]) == "cpu" ? Device::CPU : Device::CUDA;
    if (argc > first + 3) options.precision = string(argv[first + 3]) == "bf16" ? Type::BF16 : Type::FP32;

    return options;
}

AdaptiveMomentEstimation* configure(TrainingStrategy& strategy, Index batch)
{
    strategy.set_loss("CrossEntropyError3d");
    strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

    auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(strategy.get_optimization_algorithm());
    adam->set_batch_size(batch);
    adam->set_display(false);
    adam->set_display_period(1000000);
    adam->set_learning_rate(0.0001f);

    return adam;
}

int usage()
{
    cerr << "usage: transformer train    <corpus> <corpus> [epochs] [batch,...] [d_model] [layers] [dev] [prec]\n"
            "       transformer infer    <corpus>          [reps]   [batch,...] [d_model] [layers] [dev] [prec]\n"
            "       transformer capacity <corpus>          [batch]              [d_model] [layers] [dev] [prec]\n";
    return 2;
}

void report(Index batch, const vector<double>& seconds, Index sequences, Index tokens_per_sequence)
{
    vector<double> sorted = seconds;
    sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size() / 2];

    cout << "batch_" << batch << "_samples_per_sec=" << long(double(sequences) / median)
         << " median_epoch_s=" << median << "\n"
         << "batch_" << batch << "_tokens_per_sec="
         << long(double(sequences) * double(tokens_per_sequence) / median) << "\n"
         << "batch_" << batch << "_epoch_times=";
    for (size_t i = 0; i < seconds.size(); ++i) cout << (i ? "," : "") << seconds[i];
    cout << "\n" << flush;
}

}   // namespace

int main(int argc, char* argv[])
{
    const string mode = argc > 1 ? argv[1] : "";

    if (mode == "train" || mode == "quality")
    {
        if (argc < 4) return usage();

        const Index epochs = argc > 4 ? stoll(argv[4]) : 1;
        const vector<Index> batches = parse_batches(argc > 5 ? argv[5] : "32");
        const Options options = parse_options(argc, argv, 6);

        Configuration::instance().set(options.device, options.precision);
        cout << "baseline_rss_mib=" << resident_mb() << "\n";
        cout << "engine=opennn\nmode=" << mode
             << "\ndevice=" << (options.device == Device::CPU ? "cpu" : "cuda") << "\n";

        LanguageDataset dataset(argv[2]);
        dataset.set_sample_roles("Training");

        const Index samples = dataset.get_samples_number();
        const Index sequence = dataset.get_shape("Input")[0];

        cout << "samples=" << samples << " sequence=" << sequence
             << " input_vocab=" << dataset.get_input_vocabulary_size()
             << " target_vocab=" << dataset.get_target_vocabulary_size()
             << " d_model=" << options.d_model << " heads=" << options.heads()
             << " ff=" << options.feed_forward() << " layers=" << options.layers << "\n";

        const bool timing = mode == "train";
        const Index warmup = timing ? 1 : 0;

        for (const Index batch : batches)
        {
            auto network = build(dataset, options);
            if (batch == batches.front())
                cout << "parameters=" << network->get_parameters_number() << "\n" << flush;

            const bool graph = options.device == Device::CUDA
                               && getenv("OPENNN_NO_CUDA_GRAPH") == nullptr;

            TrainingStrategy strategy(network.get(), &dataset);
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

            cout << "batch_" << batch << "_cuda_graph="
                 << (!graph ? "off"
                     : adam->get_cuda_graph_capture_failed() ? "failed" : "captured") << "\n";

            report(batch, epoch_seconds, (samples / batch) * batch, sequence);
        }
    }
    else if (mode == "infer")
    {
        if (argc < 3) return usage();

        const Index reps = argc > 3 ? stoll(argv[3]) : 1;
        const vector<Index> batches = parse_batches(argc > 4 ? argv[4] : "32");
        const Options options = parse_options(argc, argv, 5);

        Configuration::instance().set(options.device, options.precision);
        cout << "baseline_rss_mib=" << resident_mb() << "\n";
        cout << "engine=opennn\nmode=infer\ndevice="
             << (options.device == Device::CPU ? "cpu" : "cuda") << "\n";

        LanguageDataset dataset(argv[2]);
        dataset.set_sample_roles("Testing");

        const Index samples = dataset.get_samples_number();
        const Index sequence = dataset.get_shape("Input")[0];

        cout << "samples=" << samples << " sequence=" << sequence
             << " input_vocab=" << dataset.get_input_vocabulary_size()
             << " target_vocab=" << dataset.get_target_vocabulary_size() << "\n";

        for (const Index batch : batches)
        {
            auto network = build(dataset, options);
            if (batch == batches.front())
                cout << "parameters=" << network->get_parameters_number() << "\n" << flush;
            const Index processed = (samples / batch) * batch;

            ForwardPropagation forward_propagation(batch, network.get(),
                                                   ForwardPropagationMode::Inference);

            vector<Index> indices(size_t(batch), Index(0));
            for (Index k = 0; k < batch; ++k) indices[size_t(k)] = k;

            Batch data(batch, &dataset, network->get_config());
            data.fill(indices, dataset.get_feature_selection(), FillMode::Inference);

            const vector<TensorView>& inputs = data.get_inputs();

            const auto run_pass = [&]
            {
                for (Index i = 0; i + batch <= samples; i += batch)
                    network->calculate_outputs_resident(inputs, forward_propagation, false);
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

            report(batch, times, processed, sequence);
        }
    }
    else if (mode == "capacity")
    {
        if (argc < 3) return usage();

        const Index batch = argc > 3 ? stoll(argv[3]) : 32;
        const Options options = parse_options(argc, argv, 4);

        Configuration::instance().set(options.device, options.precision);
        cout << "baseline_rss_mib=" << resident_mb() << "\n";
        cout << "engine=opennn\nmode=capacity\ndevice="
             << (options.device == Device::CPU ? "cpu" : "cuda")
             << "\nbatch=" << batch << "\n";

        try
        {
            LanguageDataset dataset(argv[2]);
            dataset.set_sample_roles("Training");

            auto network = build(dataset, options);
            cout << "parameters=" << network->get_parameters_number() << "\n" << flush;

            TrainingStrategy strategy(network.get(), &dataset);
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
