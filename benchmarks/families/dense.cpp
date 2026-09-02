// The dense family, defined once, driven four ways.
//
// REORGANIZATION_PLAN.md sections 4 and 8. This replaces the model
// construction that stood in six programs -- opennn_speed, opennn_higgs_cpu,
// opennn_higgs_infer, opennn_higgs_maxbatch_trial, opennn_accuracy and
// opennn_convergence, 1,717 lines between them. Only ~100 of those lines were
// the model; the rest is driver logic, which becomes the modes below rather
// than disappearing.
//
// The point of one definition is that the definitions had already drifted.
// DUPLICATION_LEDGER.md records the capacity site seeding with 0 while the
// other five seeded with 42, so the capacity benchmark had never measured the
// same initialised network as the speed and quality ones. Here `build` is the
// only way to make the network, so that cannot recur.
//
//   model_opennn train    <train_csv> <test_csv> [epochs] [batch,...] [opts]
//   model_opennn infer    <test_csv>             [reps]   [batch,...] [opts]
//   model_opennn capacity <train_csv>            [batch]              [opts]
//   model_opennn quality  <train_csv> <test_csv> [epochs] [batch]     [opts]
//
//   opts: [hidden] [layers] [relu|tanh] [cpu|cuda] [fp32|bf16]
//
// train and infer take a comma-separated batch list and run it inside one
// process, so every batch shares one data load and one thermal window --
// section 6 only trusts comparisons taken back to back.
//
// capacity takes exactly one batch and exits, because a CUDA out-of-memory
// fault leaves the context unusable: the next attempt in the same process
// would measure the wreckage of the last one. The runner re-launches per
// attempt and reads the exit code -- 0 fits, 1 does not.

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

#ifdef __linux__
#include <unistd.h>
#endif

#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/core/memory_debug.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/tensor_types.h"
#include "opennn/dataset/batch.h"
#include "opennn/dataset/kernel_gather.cuh"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/testing_analysis/testing_analysis.h"
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

constexpr Index SEED = 42;        // the value five of six sites already used

struct Options
{
    Index hidden = 1024;
    Index layers = 2;
    string activation = "ReLU";
    Device device = Device::CUDA;
    Type precision = Type::FP32;
};

// The dense family. Every mode below goes through here, and nothing else
// constructs the network.
//
// Layers only, no Scaling: the six sites disagreed on this too. Four reached
// for `ClassificationNetwork`, which prepends a Scaling layer, while
// opennn_speed built the layers directly. PyTorch's definition is
// `Linear -> activation -> ... -> Linear(1)` with no scaling stage, so the
// bare stack is the like-for-like one and the wrapper was quietly measuring a
// layer the other engine did not have. prepare_higgs.py normalises the CSV
// beforehand, so the layer was a passthrough that still cost per batch.
//
// Glorot initialisation is set explicitly rather than left to whatever the
// wrapper defaulted to, for the same reason: it is a property of the
// comparison, so it belongs where the comparison can see it.
unique_ptr<NeuralNetwork> build(const Shape& inputs, const Shape& targets, const Options& options)
{
    set_seed(SEED);

    auto network = make_unique<NeuralNetwork>();
    Shape current = inputs;

    for (Index i = 0; i < options.layers; ++i)
    {
        network->add_layer(make_unique<opennn::Dense>(current, Shape{options.hidden},
                                                     options.activation, BatchNormalization::No,
                                                     "dense_" + to_string(i + 1)));
        current = network->get_output_shape();
    }

    network->add_layer(make_unique<opennn::Dense>(current, targets, "Sigmoid", BatchNormalization::No, "output"));
    network->compile();
    network->set_parameters_glorot();

    return network;
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

// Trailing options are positional and shared by every mode, so each mode only
// says where its own arguments stop.
Options parse_options(int argc, char* argv[], int first)
{
    Options options;

    if (argc > first)     options.hidden = stoll(argv[first]);
    if (argc > first + 1) options.layers = stoll(argv[first + 1]);
    if (argc > first + 2) options.activation = string(argv[first + 2]) == "tanh" ? "Tanh" : "ReLU";
    if (argc > first + 3) options.device = string(argv[first + 3]) == "cpu" ? Device::CPU : Device::CUDA;
    if (argc > first + 4) options.precision = string(argv[first + 4]) == "bf16" ? Type::BF16 : Type::FP32;

    return options;
}

AdaptiveMomentEstimation* configure(TrainingStrategy& strategy, Index batch)
{
    strategy.set_loss("CrossEntropy");
    strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

    auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(strategy.get_optimization_algorithm());
    adam->set_batch_size(batch);
    adam->set_display_period(1000000);
    adam->set_gradient_clip_norm(0.0f);

    return adam;
}

int usage()
{
    cerr << "usage: model_opennn train    <train_csv> <test_csv> [epochs] [batch,...] [opts]\n"
            "       model_opennn infer    <test_csv>             [reps]   [batch,...] [opts]\n"
            "       model_opennn capacity <train_csv>            [batch]              [opts]\n"
            "       model_opennn quality  <train_csv> <test_csv> [epochs] [batch]     [opts]\n"
            "       opts: [hidden] [layers] [relu|tanh] [cpu|cuda] [fp32|bf16]\n";
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
        const vector<Index> batches = parse_batches(argc > 5 ? argv[5] : "1024");
        const Options options = parse_options(argc, argv, 6);

        Configuration::instance().set(options.device, options.precision);
        cout << "baseline_rss_mib=" << resident_mb() << "\n";
        cout << "engine=opennn\nmode=" << mode
             << "\ndevice=" << (options.device == Device::CPU ? "cpu" : "cuda") << "\n";

        cout << "dataset_train=" << filesystem::absolute(argv[2]).string() << "\n" << flush;
        TabularDataset dataset(argv[2], ",", false, false);

        // Contract item 3 again: the training split lives on the device, so an
        // epoch is not measuring the host-to-device copy. Without this the
        // same model measured 9.41M samples/s against 11.32M.
        if (options.device == Device::CUDA)
            dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);

        dataset.set_sample_roles("Training");
        dataset.set_variable_scalers("None");     // prepare_higgs.py already normalised

        cout << "dataset_test=" << filesystem::absolute(argv[3]).string() << "\n" << flush;
        TabularDataset test_dataset(argv[3], ",", false, false);
        test_dataset.set_sample_roles("Testing");

        const Index samples = dataset.get_samples_number();

        // Whole batches only, matching the PyTorch driver's
        // range(0, n - batch + 1, batch): the remainder is left out of the
        // epoch rather than trained as a short tail batch. Otherwise OpenNN
        // does work the throughput figure does not count, and keeps a second
        // set of activation contexts for the tail -- at large batches that is
        // the difference between fitting and paging.
        const auto drop_tail = [&](Index batch)
        {
            dataset.set_sample_roles("Training");

            for (Index sample = (samples / batch) * batch; sample < samples; ++sample)
                dataset.set_sample_role(sample, SampleRole::None);
        };

        // Contract item 3: OpenNN is timed at its best, which means the
        // captured CUDA graph and warmup epochs excluded from the window.
        // Dropping either understates it badly -- the first epoch carries
        // allocation and graph capture, and without this the same model
        // measured 1.51M samples/s against 11.3M.
        //
        // quality takes neither: warmup epochs would train a different network
        // than the one whose accuracy is being reported.
        const bool timing = mode == "train";
        const Index warmup = timing ? 2 : 0;

        for (const Index batch : batches)
        {
            drop_tail(batch);

            auto network = build(dataset.get_input_shape(), dataset.get_target_shape(), options);
            if (batch == batches.front())
                cout << "parameters=" << network->get_parameters_number() << "\n" << flush;

            TrainingStrategy strategy(network.get(), &dataset);
            const bool graph = options.device == Device::CUDA
                               && getenv("OPENNN_NO_CUDA_GRAPH") == nullptr;

            auto* adam = configure(strategy, batch);
            adam->set_cuda_graph(graph);
            adam->set_maximum_epochs(warmup + epochs);

            // The strategy owns the epoch loop, so epochs are timed from its
            // post-epoch callback. The wall clock at the edges of the timed
            // window is printed for the energy protocol, which integrates
            // board power between the two marks.
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

            // An epoch runs whole batches only and drops the remainder, so
            // dividing the full split by the epoch time overstates throughput
            // by up to one batch -- 6.5% at batch 896,000.
            const Index samples_per_epoch = (samples / batch) * batch;

            TestingAnalysis analysis(network.get(), &test_dataset);

            // Evaluate at the training batch, which is what the PyTorch driver
            // does. Left alone, TestingAnalysis defaults to the whole split in
            // one batch on CPU, so a 500,000-row test set built a 1,024 MiB
            // activation arena against PyTorch's 64 MiB -- a 16x difference in
            // the memory column that had nothing to do with training.
            analysis.set_batch_size(batch);

            cout << "batch_" << batch << "_samples_per_sec="
                 << long(double(samples_per_epoch) / median_epoch_s)
                 << " median_epoch_s=" << median_epoch_s << "\n"
                 << "batch_" << batch << "_epoch_times=";
            for (size_t i = 0; i < epoch_seconds.size(); ++i)
                cout << (i ? "," : "") << epoch_seconds[i];
            cout << "\n"
                 << "batch_" << batch << "_cuda_graph="
                 << (!graph ? "off"
                     : adam->get_cuda_graph_capture_failed() ? "failed" : "captured") << "\n"
                 << "batch_" << batch
                 << "_test_accuracy=" << analysis.calculate_binary_classification_tests()[0]
                 << " test_roc_auc=" << analysis.perform_roc_analysis().area_under_curve
                 << "\n" << flush;
        }
    }
    else if (mode == "infer")
    {
        if (argc < 3) return usage();

        const Index reps = argc > 3 ? stoll(argv[3]) : 1;
        const vector<Index> batches = parse_batches(argc > 4 ? argv[4] : "1024");
        const Options options = parse_options(argc, argv, 5);

        Configuration::instance().set(options.device, options.precision);
        cout << "baseline_rss_mib=" << resident_mb() << "\n";
        cout << "engine=opennn\nmode=infer\ndevice="
             << (options.device == Device::CPU ? "cpu" : "cuda") << "\n";

        cout << "dataset_test=" << filesystem::absolute(argv[2]).string() << "\n" << flush;
        TabularDataset dataset(argv[2], ",", false, false);

        // The PyTorch driver uploads the whole test split once, before the
        // timed window, and slices it on the device; streaming it per batch
        // instead times PCIe rather than the network. Profiled on this cell
        // the per-batch copy and the synchronisation behind it left the GPU
        // idle for 39% of the pass -- 12.9 ms of kernels inside a 21.3 ms
        // pass. The training path above already keeps its split resident for
        // exactly this reason, and so does the recurrent family.
        //
        // set_storage_mode only requests residency; the optimizer is what
        // enables it for training, and drops it when it returns. Inference
        // has no optimizer, so it asks here, after the roles are set.
        if (options.device == Device::CUDA)
            dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);

        dataset.set_sample_roles("Testing");

        if (options.device == Device::CUDA)
            dataset.enable_device_residency();

        const MatrixR& data = dataset.get_data();
        const Index samples = dataset.get_samples_number();
        const Index inputs_number = dataset.get_input_shape()[0];
        const MatrixR inputs = data.leftCols(inputs_number);

        auto network = build(Shape{inputs_number}, Shape{1}, options);
        cout << "parameters=" << network->get_parameters_number() << "\n" << flush;

        for (const Index batch : batches)
        {
            const Index processed = (samples / batch) * batch;
            // Inference mode, not the default. A training arena keeps every
            // layer's activations alive for the backward pass that never
            // comes; inference can reuse buffers between layers. Measured on
            // this cell the difference is 32 MiB against 16.
            ForwardPropagation forward_propagation(batch, network.get(),
                                                   ForwardPropagationMode::Inference);

            // Touching the output keeps LTO from deleting the forward pass.
            // Only CPU can read a scalar out of it: `as_matrix` requires CPU
            // FP32 storage, and on CUDA the outputs are still on the device,
            // where dereferencing one would add a synchronising copy per batch
            // to the very thing being timed.
            const bool on_cpu = options.device == Device::CPU;
            double sink = 0.0;

            // Device-resident batches on CUDA, host views on CPU: the same
            // slices either way, differing only in which side of the bus they
            // are already on when the clock starts.
            //
            // The CUDA split is resident in fp32 with the target beside the
            // inputs (set_storage_mode above). The network wants the inputs
            // alone, in its own precision, so they are gathered once here,
            // outside the clock, into one buffer that every batch is a view
            // into -- what the PyTorch driver does with x.to(bfloat16) and
            // x[start:start + batch]. The alternative, kept behind
            // OPENNN_DENSE_INFER_GATHER=1, is the training-style path: fill()
            // stages a batch's row indices, upload_to_device_batch_async()
            // copies them and gathers the rows into the batch's fixed buffer,
            // and a CUDA graph replays the forward pass on that buffer. Fixed
            // addresses are what a graph needs and the only thing this pass
            // gets from it: an index copy and a gather kernel per batch, for
            // three launches that a 0.2 ms GEMM hides anyway.
            const bool per_batch_gather = getenv("OPENNN_DENSE_INFER_GATHER") != nullptr;
            forward_propagation.set_cuda_graph(!on_cpu && per_batch_gather);

            Batch device_batch(batch, &dataset, network->get_config());
            vector<Index> indices(size_t(batch), Index(0));

            const Type input_type = on_cpu ? Type::FP32
                                           : device_batch.get_inputs().front().get_type();
            const Index row_bytes = inputs_number * type_bytes(input_type);
            Buffer resident(Device::CUDA);

            if (!on_cpu)
            {
                iota(indices.begin(), indices.end(), Index(0));
                device_batch.fill(indices, dataset.get_feature_selection(),
                                  FillMode::Inference);
                device_batch.upload_to_device_batch_async(device_batch,
                                                          device::get_compute_stream());
                network->calculate_outputs_resident(device_batch.get_inputs(),
                                                    forward_propagation, true);

                const FeatureSelection features = dataset.get_feature_selection();
                throw_if(!is_contiguous(features.inputs),
                         "dense infer: the resident gather needs contiguous input columns.");

                vector<int> rows(static_cast<size_t>(samples), 0);
                iota(rows.begin(), rows.end(), 0);
                Buffer rows_device(Device::CUDA);
                rows_device.resize_bytes(samples * Index(sizeof(int)), Device::CUDA);
                resident.resize_bytes(samples * row_bytes, Device::CUDA);

                const cudaStream_t stream = device::get_compute_stream();
                device::copy_async(rows_device.data(), rows.data(), samples * Index(sizeof(int)),
                                   device::CopyKind::HostToDevice, stream);
                gather_rows_cuda(dataset.get_device_data(), rows_device.as<int>(), resident.data(),
                                 input_type == Type::BF16, samples, inputs_number,
                                 dataset.get_device_data_columns(), features.inputs.front(),
                                 stream);
                device::synchronize(stream);
            }

            const auto run_pass = [&]
            {
                for (Index i = 0; i + batch <= samples; i += batch)
                {
                    if (on_cpu)
                    {
                        TensorView view(const_cast<float*>(inputs.data()) + i * inputs_number,
                                        Shape{batch, inputs_number}, Type::FP32);
                        network->forward_propagate({view}, forward_propagation,
                                                   ForwardPropagationMode::Inference);
                        sink += forward_propagation.get_outputs().as_matrix()(0, 0);
                        continue;
                    }

                    if (per_batch_gather)
                    {
                        for (Index k = 0; k < batch; ++k) indices[size_t(k)] = i + k;
                        device_batch.fill(indices, dataset.get_feature_selection(),
                                          FillMode::Inference);
                        // On the compute stream, so the gather orders after
                        // the previous forward pass that reads the same buffer.
                        device_batch.upload_to_device_batch_async(device_batch,
                                                                  device::get_compute_stream());
                        network->calculate_outputs_resident(device_batch.get_inputs(),
                                                           forward_propagation, false);
                    }
                    else
                    {
                        TensorView view(static_cast<char*>(resident.data()) + i * row_bytes,
                                        Shape{batch, inputs_number}, input_type, Device::CUDA);
                        network->calculate_outputs_resident({view}, forward_propagation, false);
                    }

                    (void)forward_propagation.get_outputs();
                }

                // The clock stops when the work is done, not when it is
                // queued. The PyTorch driver ends every pass with
                // torch.cuda.synchronize(); without the same barrier here a
                // pass that no longer copies per batch reports launch
                // throughput -- 72.5M samples/s against a kernel time that
                // cannot exceed 30M.
                if (!on_cpu) device::synchronize(device::get_compute_stream());
            };

            run_pass();
            run_pass();

            vector<double> times;

            // The same marks train prints, for the same reason: energy is
            // integrated between them, so process startup and data loading
            // stay outside the window as they stay outside the timing.
            const auto unix_now = []
            {
                return chrono::duration<double>(
                    chrono::system_clock::now().time_since_epoch()).count();
            };

            cout << "TIMED_START_UNIX=" << fixed << setprecision(3)
                 << unix_now() << "\n" << defaultfloat << flush;

            for (Index r = 0; r < reps; ++r)
            {
                const auto t0 = clock_type::now();
                run_pass();
                times.push_back(chrono::duration<double>(clock_type::now() - t0).count());
            }

            cout << "TIMED_END_UNIX=" << fixed << setprecision(3)
                 << unix_now() << "\n" << defaultfloat << flush;

            (void)sink;

            cout << "batch_" << batch << "_pass_times=";
            for (size_t i = 0; i < times.size(); ++i) cout << (i ? "," : "") << times[i];
            cout << "\n";

            sort(times.begin(), times.end());
            const double median_pass_s = times[times.size() / 2];

            cout << "batch_" << batch << "_samples_per_sec=" << long(double(processed) / median_pass_s)
                 << " median_pass_s=" << median_pass_s << "\n"
                 << "batch_" << batch << "_cuda_graph="
                 << (on_cpu || !per_batch_gather ? "off"
                     : forward_propagation.cuda_graph_failed ? "failed"
                     : forward_propagation.inference_graph_exec ? "captured" : "warming")
                 << "\n" << flush;
        }
    }
    else if (mode == "capacity")
    {
        if (argc < 3) return usage();

        const Index batch = argc > 3 ? stoll(argv[3]) : 1024;
        const Options options = parse_options(argc, argv, 4);

        Configuration::instance().set(options.device, options.precision);
        cout << "baseline_rss_mib=" << resident_mb() << "\n";
        cout << "engine=opennn\nmode=capacity\ndevice="
             << (options.device == Device::CPU ? "cpu" : "cuda")
             << "\nbatch=" << batch << "\n";

        // One attempt, then exit: an out-of-memory fault leaves the CUDA
        // context unusable, so a second attempt here would measure the wreck
        // of the first. The runner re-launches and reads the exit code.
        try
        {
            cout << "dataset_train=" << filesystem::absolute(argv[2]).string() << "\n" << flush;
        TabularDataset dataset(argv[2], ",", false, false);
            dataset.set_sample_roles("Training");
            dataset.set_variable_scalers("None");

            auto network = build(dataset.get_input_shape(), dataset.get_target_shape(), options);

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

    // OPENNN_MEMORY_DEBUG=1 attributes the resident set member by member.
    if (memory_debug::enabled()) memory_debug::print(cout);

    cout << "RESULT=OK\n";

    return 0;
}
