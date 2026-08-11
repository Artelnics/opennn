//   OpenNN HIGGS dense max-batch trial, GPU.
//
//   One process = one (mode, batch, precision) attempt at the canonical HIGGS
//   dense classifier (28 -> hidden -> hidden -> 1, ReLU hidden, sigmoid
//   output, binary cross-entropy -- see docs/benchmarks/throughput/higgs/README.md), so
//   a CUDA out-of-memory fault cannot contaminate later trials. The Python
//   driver (run_higgs_maxbatch.py) does the exponential-grow + binary-search
//   by spawning this repeatedly.
//
//   Both modes run the batch MONOLITHICALLY -- one optimizer step / one
//   forward over the whole batch with activations O(batch) -- the same
//   protocol as the PyTorch and TensorFlow trials.
//
//   mode "train" runs one full-batch training step (forward + backward + Adam
//   update) with prefetch-pool depth 1 (this is a capacity benchmark; the
//   default pool of 3 holds extra device batch copies) and CUDA graph off.
//
//   mode "infer" runs forward-only on the device-resident path
//   (calculate_outputs_resident): no optimizer state, no gradients, input
//   uploaded once, output left on the GPU. `iterations` timed forwards.
//
//   The data is synthetic with the HIGGS contract shapes -- capacity depends
//   on the shapes and the training step, not on the feature values. Features
//   are uniform in [-1, 1] (the prepared HIGGS files are standardized);
//   targets are binarized to {0, 1} for the binary cross-entropy.
//
//   device "cpu" runs the same trial CPU-only (fp32; the driver caps the
//   process's memory instead of VRAM). CPU inference uses the host
//   forward_propagate path with a caller-owned ForwardPropagation, the same
//   protocol as the CPU HIGGS speed benchmark.
//
//   usage: opennn_higgs_maxbatch_trial <train|infer> <batch>
//                                      [hidden] [hidden_layers] [iterations]
//                                      [cuda|cpu]
//   env:   OPENNN_BF16=1  -> bf16 (CUDA only; else fp32)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef OPENNN_HAS_CUDA
#include <cuda_runtime.h>
#include "opennn/core/device_backend.h"
#include "opennn/core/cuda/kernel_cast.cuh"
#endif

#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/core/configuration.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/core/memory_debug.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/training_strategy/training_strategy.h"

#ifndef _WIN32
#include <sys/resource.h>
#endif

using namespace opennn;

namespace
{

constexpr Index inputs_number = 28;   // HIGGS contract: 28 features, 1 target

// Real HIGGS rows from a prepared float32 binary (rows x 29: 28 standardized
// features then the {0,1} label), selected with OPENNN_HIGGS_BIN. Rows repeat
// modulo when the requested batch exceeds the file -- the same convention as
// the ResNet-50 capacity runner. Returns false when the env var is unset, so
// the trial falls back to synthetic contract-shaped data.
bool load_higgs_rows(MatrixR& destination)
{
    const char* path = getenv("OPENNN_HIGGS_BIN");
    if (!path) return false;

    constexpr Index row_floats = inputs_number + 1;

    ifstream file(path, ios::binary | ios::ate);
    if (!file) throw runtime_error(string("cannot open OPENNN_HIGGS_BIN: ") + path);

    const Index rows_available = Index(file.tellg()) / (row_floats * Index(sizeof(float)));
    if (rows_available <= 0) throw runtime_error("OPENNN_HIGGS_BIN is empty");

    vector<float> rows(size_t(rows_available) * row_floats);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(rows.data()),
              streamsize(rows.size() * sizeof(float)));

    const Index columns = destination.cols();   // 28 (infer) or 29 (train)
    for (Index r = 0; r < destination.rows(); ++r)
        memcpy(destination.data() + r * columns,
               rows.data() + size_t(r % rows_available) * row_floats,
               size_t(columns) * sizeof(float));

    cout << "data=higgs_bin rows=" << rows_available << "\n";
    return true;
}

// Peak memory of this process, for the CPU-capped runs: RSS high-water mark
// (getrusage) and peak virtual address space (VmPeak -- what RLIMIT_AS caps,
// including mapped libraries). No-op on Windows.
void print_peak_memory()
{
#ifndef _WIN32
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) == 0)
        cout << "peak_rss_mib=" << usage.ru_maxrss / 1024 << "\n";

    ifstream status("/proc/self/status");
    string line;
    while (getline(status, line))
        if (line.rfind("VmPeak:", 0) == 0)
        {
            const long kib = stol(line.substr(7));
            cout << "vm_peak_mib=" << kib / 1024 << "\n";
            break;
        }
#endif
}

class HiggsBenchmarkNetwork final : public NeuralNetwork
{
public:
#ifdef OPENNN_HAS_CUDA
    void release_bf16_fp32_parameter_master_for_inference()
    {
        if (const char* flag = getenv("OPENNN_HIGGS_RELEASE_BF16_FP32_MASTER");
            flag && string(flag) == "0")
            return;

        if (config.training_type != Type::BF16
            || parameters.device_type != Device::CUDA
            || parameters.empty()
            || parameters_bf16_mirror.empty()
            || !parameters.owns)
            return;

        // The layer parameter views already point at parameters_bf16_mirror
        // after copy_parameters_device(). Keep the fp32-size invariant that
        // forward_propagate() validates, but stop owning a second CUDA buffer.
        const Index fp32_master_bytes = parameters.bytes;
        parameters.resize_bytes(0, Device::CUDA);
        parameters.set_view(parameters_bf16_mirror.data,
                            fp32_master_bytes,
                            Device::CUDA);
    }
#endif
};

unique_ptr<HiggsBenchmarkNetwork> make_network(Index hidden, Index hidden_layers)
{
    auto network = make_unique<HiggsBenchmarkNetwork>();
    Shape current = Shape{inputs_number};

    for (Index i = 0; i < hidden_layers; ++i)
    {
        network->add_layer(make_unique<opennn::Dense>(
            current,
            Shape{hidden},
            "ReLU",
            false,
            "higgs_dense_" + to_string(i + 1)));
        current = network->get_output_shape();
    }

    network->add_layer(make_unique<opennn::Dense>(
        current,
        Shape{1},
        "Sigmoid",
        false,
        "higgs_output"));

    network->compile();
    network->set_parameters_glorot();
    return network;
}

#ifdef OPENNN_HAS_CUDA
bool bf16_resident_input_enabled()
{
    if (const char* flag = getenv("OPENNN_HIGGS_BF16_RESIDENT_INPUT");
        flag && string(flag) == "0")
        return false;

    return true;
}

uint16_t fp32_to_bf16_bits(float value)
{
    const uint32_t bits = bit_cast<uint32_t>(value);
    const uint32_t lsb = (bits >> 16) & 1u;
    return uint16_t((bits + 0x7fffu + lsb) >> 16);
}

TensorView maybe_alias_bf16_input_cast(const TensorView& fp32_input,
                                       ForwardPropagation& propagation)
{
    if (const char* flag = getenv("OPENNN_HIGGS_ALIAS_BF16_INPUT");
        flag && string(flag) == "0")
        return fp32_input;

    if (!fp32_input.is_fp32() || !fp32_input.is_cuda())
        return fp32_input;

    // In the canonical HIGGS network, layer 0 consumes the external input and
    // layer 1's output slot is still dead. Reusing that future activation for
    // the fp32->bf16 input cast removes the persistent thread-local cast
    // workspace while preserving the same GEMM path and resident fp32 input.
    if (propagation.slots.size() < 2
        || propagation.slots[1].empty())
        return fp32_input;

    TensorView& future_activation = propagation.slots[1].back();
    if (!future_activation.is_bf16()
        || future_activation.size() < fp32_input.size())
        return fp32_input;

    cast_fp32_to_bf16(fp32_input.size(),
                      fp32_input.as<float>(),
                      future_activation.as<__nv_bfloat16>(),
                      Backend::get_compute_stream());

    memory_debug::record("forward.aliased",
                         "HIGGS bf16 input cast",
                         0,
                         "uses future activation slot");

    return TensorView(future_activation.data,
                      fp32_input.shape,
                      Type::BF16,
                      Device::CUDA);
}
#endif

}

int main(int argc, char* argv[])
{
    cout << unitbuf;
    cerr << unitbuf;

    const string mode  = argc > 1 ? argv[1] : "train";
    const Index batch       = argc > 2 ? Index(stoll(argv[2])) : 1024;
    const Index hidden      = argc > 3 ? Index(stoll(argv[3])) : 1024;
    const Index layers      = argc > 4 ? Index(stoll(argv[4])) : 2;
    const Index iterations  = argc > 5 ? max<Index>(Index(1), Index(stoll(argv[5]))) : 1;
    const string device = argc > 6 ? argv[6] : "cuda";

    try
    {
        const char* seed_env = getenv("OPENNN_BENCH_SEED");
        set_seed(seed_env && *seed_env ? stoi(seed_env) : 0);
        const bool use_cpu = device == "cpu";
#ifndef OPENNN_HAS_CUDA
        if (!use_cpu)
            throw runtime_error("built without CUDA; use device \"cpu\"");
#endif
        const bool use_bf16 = !use_cpu && getenv("OPENNN_BF16") != nullptr;
        Configuration::instance().set(use_cpu ? Device::CPU : Device::CUDA,
                                      use_bf16 ? Type::BF16 : Type::FP32);

        cout << "precision=" << (use_bf16 ? "bf16" : "fp32")
                  << " mode=" << mode
                  << " device=" << device
                  << " inputs=" << inputs_number
                  << " hidden=" << hidden << " hidden_layers=" << layers
                  << " batch=" << batch << " iterations=" << iterations << "\n";

        auto network = make_network(hidden, layers);

        cout << "parameters=" << network->get_parameters_number() << "\n";

        if (mode == "infer" && use_cpu)
        {
            // Monolithic resident inference with a caller-owned, reused
            // ForwardPropagation (one-shot calculate_outputs would re-fault
            // its arena every call). Activation memory is O(batch), the same
            // protocol as the PyTorch/TensorFlow trials.
            MatrixR inputs_host(batch, inputs_number);
            if (!load_higgs_rows(inputs_host))
            {
                inputs_host = MatrixR::Random(batch, inputs_number);
                cout << "data=synthetic\n";
            }

            ForwardPropagation propagation(batch, network.get());

            const TensorView input_view(
                const_cast<float*>(inputs_host.data()),
                Shape{batch, inputs_number}, Type::FP32);

            auto run_pass = [&]()
            {
                network->forward_propagate({input_view}, propagation, false);
            };

            run_pass();   // warmup: pages workspaces and BLAS scratch in

            const auto t0 = chrono::high_resolution_clock::now();
            for (Index i = 0; i < iterations; ++i)
                run_pass();
            const auto t1 = chrono::high_resolution_clock::now();

            const TensorView outputs = propagation.get_outputs();
            if (!isfinite(outputs.as<float>()[0]))
                throw runtime_error("non-finite outputs");

            const double wall_s = chrono::duration<double>(t1 - t0).count();

            memory_debug::print(cout);
            print_peak_memory();

            cout << "wall_s=" << wall_s << "\n";
            cout << "samples_per_sec=" << double(batch) * double(iterations) / wall_s << "\n";
            cout << "RESULT=OK\n";
            return 0;
        }

#ifdef OPENNN_HAS_CUDA
        if (mode == "infer")
        {
            // Monolithic resident inference, the GPU twin of the CPU protocol:
            // the input stays device-resident, the propagation (activations
            // O(batch)) is reused across iterations, and the output is read
            // from the propagation's own slot.
            MatrixR inputs_host(batch, inputs_number);
            if (!load_higgs_rows(inputs_host))
            {
                inputs_host = MatrixR::Random(batch, inputs_number);
                cout << "data=synthetic\n";
            }

            const bool bf16_resident_input = use_bf16 && bf16_resident_input_enabled();
            const Type input_type = bf16_resident_input ? Type::BF16 : Type::FP32;
            cout << "input_type=" << (input_type == Type::BF16 ? "bf16" : "fp32") << "\n";

            Buffer arena(Device::CUDA);
            arena.resize_bytes(get_aligned_bytes(batch * inputs_number, input_type),
                               Device::CUDA);
            char* const base = arena.as<char>();

            cudaStream_t stream = Backend::get_compute_stream();
            if (bf16_resident_input)
            {
                vector<uint16_t> inputs_bf16(size_t(batch * inputs_number));
                const float* src = inputs_host.data();
                #pragma omp parallel for if(batch * inputs_number > 4096)
                for (Index i = 0; i < batch * inputs_number; ++i)
                    inputs_bf16[size_t(i)] = fp32_to_bf16_bits(src[i]);

                device::copy_async(base, inputs_bf16.data(),
                                   batch * inputs_number * Index(sizeof(uint16_t)),
                                   device::CopyKind::HostToDevice, stream);
            }
            else
            {
                device::copy_async(base, inputs_host.data(),
                                   batch * inputs_number * Index(sizeof(float)),
                                   device::CopyKind::HostToDevice, stream);
            }

            bool parameters_uploaded = false;
            if (use_bf16)
            {
                network->copy_parameters_device();
                network->copy_states_device();
                network->release_bf16_fp32_parameter_master_for_inference();
                parameters_uploaded = true;
            }

            ForwardPropagation propagation(batch, network.get());

            Type output_type = Type::FP32;
            const void* probe_source = nullptr;

            auto run_pass = [&]()
            {
                const TensorView input_view(base, Shape{batch, inputs_number},
                                            input_type, Device::CUDA);
                const TensorView compute_view = use_bf16 && input_view.is_fp32()
                    ? maybe_alias_bf16_input_cast(input_view, propagation)
                    : input_view;

                const bool upload_parameters = !parameters_uploaded;
                const TensorView outputs = network->calculate_outputs_resident(
                    {compute_view}, propagation, upload_parameters);
                if (use_bf16 && upload_parameters)
                    network->release_bf16_fp32_parameter_master_for_inference();

                parameters_uploaded = true;
                output_type = outputs.type;
                probe_source = outputs.data;
            };

            // Warmup selects the cuDNN/cuBLAS plans, allocates the workspaces,
            // and uploads the parameters; excluded from timing.
            run_pass();
            cudaDeviceSynchronize();

            const auto t0 = chrono::high_resolution_clock::now();
            for (Index i = 0; i < iterations; ++i)
                run_pass();
            cudaDeviceSynchronize();
            const auto t1 = chrono::high_resolution_clock::now();

            float probe[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const Index probe_size = min<Index>(Index(4), batch);
            copy_device_to_host_float(probe_source, output_type, probe_size, probe, stream);
            cudaStreamSynchronize(stream);
            for (Index i = 0; i < probe_size; ++i)
                if (!isfinite(probe[i]))
                    throw runtime_error("non-finite outputs");

            const double wall_s = chrono::duration<double>(t1 - t0).count();

            memory_debug::print(cout);
            print_peak_memory();

            cout << "wall_s=" << wall_s << "\n";
            cout << "samples_per_sec=" << double(batch) * double(iterations) / wall_s << "\n";
            cout << "RESULT=OK\n";
            return 0;
        }
#endif // OPENNN_HAS_CUDA

        // train: one monolithic optimizer step over the batch -- forward,
        // backward, and Adam update with activations O(batch), the same
        // protocol as the PyTorch/TensorFlow trials.
        TabularDataset dataset(batch, Shape{inputs_number}, Shape{1});

        MatrixR data(batch, inputs_number + 1);
        if (!load_higgs_rows(data))
        {
            data = MatrixR::Random(batch, inputs_number + 1);
            data.col(inputs_number) = (data.col(inputs_number).array() > 0.0f).cast<float>();
            cout << "data=synthetic\n";
        }
        dataset.set_data(data);
        data.resize(0, 0);   // free the staging copy: the dataset owns the rows now
        dataset.set_sample_roles("Training");

        TrainingStrategy training_strategy(network.get(), &dataset);
        training_strategy.set_loss("CrossEntropy");
        training_strategy.get_loss()->set_regularization("NoRegularization");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        if (!adam) throw runtime_error("Adam optimizer not found.");

        adam->set_batch_size(batch);
        adam->set_maximum_epochs(iterations);
        adam->set_display(false);
        adam->set_gradient_clip_norm(0.0f);
        adam->set_batch_pool_size(1);   // capacity: one device batch copy, not three

        const char* target_env = getenv("OPENNN_TARGET_LOSS");
        const bool target_mode = target_env && *target_env;
        const float target = target_mode ? stof(target_env) : 0.0f;
        if (target_mode)
        {
            adam->set_maximum_epochs(iterations);
            adam->set_loss_goal(target);
        }

        const auto unix_now = []
        {
            return chrono::duration<double>(
                chrono::system_clock::now().time_since_epoch()).count();
        };
        if (target_mode)
            cout << "TRAIN_START_UNIX=" << fixed << setprecision(3)
                      << unix_now() << "\n" << defaultfloat;
        const auto t0 = chrono::high_resolution_clock::now();
        const TrainingResult result = training_strategy.train();
#ifdef OPENNN_HAS_CUDA
        if (!use_cpu) cudaDeviceSynchronize();
#endif
        const auto t1 = chrono::high_resolution_clock::now();
        if (target_mode)
            cout << "TRAIN_END_UNIX=" << fixed << setprecision(3)
                      << unix_now() << "\n" << defaultfloat;

        if (!isfinite(result.loss))
            throw runtime_error("non-finite loss");

        const double wall_s = chrono::duration<double>(t1 - t0).count();

        memory_debug::print(cout);
        print_peak_memory();

        cout << "final_loss=" << result.loss << "\n";
        if (target_mode)
        {
            const Index epochs_run = result.get_epochs_number();
            const bool reached = result.get_training_error() <= target;
            cout << "target=" << target << "\n";
            cout << "epochs_run=" << epochs_run << "\n";
            cout << "final_error=" << result.get_training_error() << "\n";
            cout << "reached_goal=" << (reached ? 1 : 0) << "\n";
            cout << "loss_history=";
            for (Index epoch = 0; epoch < result.training_error_history.size(); ++epoch)
                cout << (epoch ? "," : "") << result.training_error_history(epoch);
            cout << "\n";
        }
        cout << "wall_s=" << wall_s << "\n";
        const Index completed = target_mode ? result.get_epochs_number() : iterations;
        cout << "samples_per_sec=" << double(batch) * double(completed) / wall_s << "\n";
        cout << "RESULT=OK\n";
        return 0;
    }
    catch (const exception& e)
    {
        cout << "FAIL: " << e.what() << "\n";
        cout << "RESULT=ERROR\n";
        return 1;
    }
}
