//   OpenNN HIGGS dense max-batch trial, GPU.
//
//   One process = one (mode, batch, precision) attempt at the canonical HIGGS
//   dense classifier (28 -> hidden -> hidden -> 1, ReLU hidden, sigmoid
//   output, binary cross-entropy -- see docs/benchmarks/throughput/higgs/README.md), so
//   a CUDA out-of-memory fault cannot contaminate later trials. The Python
//   driver (run_higgs_maxbatch.py) does the exponential-grow + binary-search
//   by spawning this repeatedly.
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
//                                      [cuda|cpu] [tile_rows]
//   env:   OPENNN_BF16=1  -> bf16 (CUDA only; else fp32)
//   tile_rows: infer tile size. -1 (default) = auto: 131072 on CPU (the
//   measured MKL speed-parity point) and 65536 on CUDA (measured faster than
//   untiled in fp32, parity in bf16; 32768 loses 21% fp32 -- cuBLASLt
//   algorithm cliff). 0 = whole batch, i.e. the untiled protocol.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef OPENNN_HAS_CUDA
#include <cuda_runtime.h>
#include "opennn/device_backend.h"
#include "opennn/kernel.cuh"
#endif

#include "opennn/adaptive_moment_estimation.h"
#include "opennn/configuration.h"
#include "opennn/dense_layer.h"
#include "opennn/forward_propagation.h"
#include "opennn/memory_debug.h"
#include "opennn/neural_network.h"
#include "opennn/random_utilities.h"
#include "opennn/tabular_dataset.h"
#include "opennn/training_strategy.h"

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
    const char* path = std::getenv("OPENNN_HIGGS_BIN");
    if (!path) return false;

    constexpr Index row_floats = inputs_number + 1;

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error(std::string("cannot open OPENNN_HIGGS_BIN: ") + path);

    const Index rows_available = Index(file.tellg()) / (row_floats * Index(sizeof(float)));
    if (rows_available <= 0) throw std::runtime_error("OPENNN_HIGGS_BIN is empty");

    std::vector<float> rows(size_t(rows_available) * row_floats);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(rows.data()),
              std::streamsize(rows.size() * sizeof(float)));

    const Index columns = destination.cols();   // 28 (infer) or 29 (train)
    for (Index r = 0; r < destination.rows(); ++r)
        memcpy(destination.data() + r * columns,
               rows.data() + size_t(r % rows_available) * row_floats,
               size_t(columns) * sizeof(float));

    std::cout << "data=higgs_bin rows=" << rows_available << "\n";
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
        std::cout << "peak_rss_mib=" << usage.ru_maxrss / 1024 << "\n";

    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line))
        if (line.rfind("VmPeak:", 0) == 0)
        {
            const long kib = std::stol(line.substr(7));
            std::cout << "vm_peak_mib=" << kib / 1024 << "\n";
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
        if (const char* flag = std::getenv("OPENNN_HIGGS_RELEASE_BF16_FP32_MASTER");
            flag && std::string(flag) == "0")
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

std::unique_ptr<HiggsBenchmarkNetwork> make_network(Index hidden, Index hidden_layers)
{
    auto network = std::make_unique<HiggsBenchmarkNetwork>();
    Shape current = Shape{inputs_number};

    for (Index i = 0; i < hidden_layers; ++i)
    {
        network->add_layer(std::make_unique<opennn::Dense>(
            current,
            Shape{hidden},
            "ReLU",
            false,
            "higgs_dense_" + std::to_string(i + 1)));
        current = network->get_output_shape();
    }

    network->add_layer(std::make_unique<opennn::Dense>(
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
    if (const char* flag = std::getenv("OPENNN_HIGGS_BF16_RESIDENT_INPUT");
        flag && std::string(flag) == "0")
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
    if (const char* flag = std::getenv("OPENNN_HIGGS_ALIAS_BF16_INPUT");
        flag && std::string(flag) == "0")
        return fp32_input;

    if (!fp32_input.is_fp32() || !fp32_input.is_cuda())
        return fp32_input;

    // In the canonical HIGGS network, layer 0 consumes the external input and
    // layer 1's output slot is still dead. Reusing that future activation for
    // the fp32->bf16 input cast removes the persistent thread-local cast
    // workspace while preserving the same GEMM path and resident fp32 input.
    if (propagation.forward_slots.size() < 2
        || propagation.forward_slots[1].empty())
        return fp32_input;

    TensorView& future_activation = propagation.forward_slots[1].back();
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
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    const std::string mode  = argc > 1 ? argv[1] : "train";
    const Index batch       = argc > 2 ? Index(std::stoll(argv[2])) : 1024;
    const Index hidden      = argc > 3 ? Index(std::stoll(argv[3])) : 1024;
    const Index layers      = argc > 4 ? Index(std::stoll(argv[4])) : 2;
    const Index iterations  = argc > 5 ? std::max<Index>(Index(1), Index(std::stoll(argv[5]))) : 1;
    const std::string device = argc > 6 ? argv[6] : "cuda";
    const Index tile_raw     = argc > 7 ? Index(std::stoll(argv[7])) : Index(-1);
    const Index tile_arg     = tile_raw >= 0 ? tile_raw
                             : (device == "cpu" ? Index(131072) : Index(65536));

    try
    {
        set_seed(0);
        const bool use_cpu = device == "cpu";
#ifndef OPENNN_HAS_CUDA
        if (!use_cpu)
            throw std::runtime_error("built without CUDA; use device \"cpu\"");
#endif
        const bool use_bf16 = !use_cpu && std::getenv("OPENNN_BF16") != nullptr;
        Configuration::instance().set(use_cpu ? Device::CPU : Device::CUDA,
                                      use_bf16 ? Type::BF16 : Type::FP32);

        std::cout << "precision=" << (use_bf16 ? "bf16" : "fp32")
                  << " mode=" << mode
                  << " device=" << device
                  << " inputs=" << inputs_number
                  << " hidden=" << hidden << " hidden_layers=" << layers
                  << " batch=" << batch << " iterations=" << iterations << "\n";

        auto network = make_network(hidden, layers);

        std::cout << "parameters=" << network->get_parameters_number() << "\n";

        if (mode == "infer" && use_cpu)
        {
            // Row-tiled resident inference: caller-owned tile workspaces
            // reused across iterations (the CPU analogue of the GPU resident
            // protocol -- one-shot calculate_outputs would re-fault its
            // arena every call). Activation memory is O(tile); the memory
            // ceiling is the input + output data.
            const Index tile_rows = tile_arg > 0 ? std::min<Index>(batch, tile_arg) : batch;
            const Index tail_rows = batch % tile_rows;
            std::cout << "tile_rows=" << tile_rows << "\n";

            MatrixR inputs_host(batch, inputs_number);
            if (!load_higgs_rows(inputs_host))
            {
                inputs_host = MatrixR::Random(batch, inputs_number);
                std::cout << "data=synthetic\n";
            }
            MatrixR outputs(batch, 1);

            ForwardPropagation tile_propagation(tile_rows, network.get());
            std::unique_ptr<ForwardPropagation> tail_propagation;
            if (tail_rows > 0)
                tail_propagation = std::make_unique<ForwardPropagation>(tail_rows, network.get());

            auto run_pass = [&]()
            {
                for (Index start = 0; start < batch; start += tile_rows)
                {
                    const Index rows = std::min<Index>(tile_rows, batch - start);
                    ForwardPropagation& propagation =
                        rows == tile_rows ? tile_propagation : *tail_propagation;

                    const TensorView tile_view(
                        const_cast<float*>(inputs_host.data()) + start * inputs_number,
                        Shape{rows, inputs_number}, Type::FP32);

                    network->forward_propagate({tile_view}, propagation, false);

                    const TensorView tile_outputs = propagation.get_outputs();
                    memcpy(outputs.data() + start, tile_outputs.data,
                           size_t(rows) * sizeof(float));
                }
            };

            run_pass();   // warmup: pages workspaces and BLAS scratch in

            const auto t0 = std::chrono::high_resolution_clock::now();
            for (Index i = 0; i < iterations; ++i)
                run_pass();
            const auto t1 = std::chrono::high_resolution_clock::now();

            if (!std::isfinite(outputs(0, 0)))
                throw std::runtime_error("non-finite outputs");

            const double wall_s = std::chrono::duration<double>(t1 - t0).count();

            memory_debug::print(std::cout);
            print_peak_memory();

            std::cout << "wall_s=" << wall_s << "\n";
            std::cout << "samples_per_sec=" << double(batch) * double(iterations) / wall_s << "\n";
            std::cout << "RESULT=OK\n";
            return 0;
        }

#ifdef OPENNN_HAS_CUDA
        if (mode == "infer")
        {
            // Row-tiled resident inference, the GPU twin of the CPU protocol:
            // the input and the assembled outputs stay device-resident (the
            // honest data footprint), tile workspaces are reused across
            // iterations, and activations are O(tile) instead of O(batch).
            const Index tile_rows = tile_arg > 0 ? std::min<Index>(batch, tile_arg) : batch;
            const Index tail_rows = batch % tile_rows;
            std::cout << "tile_rows=" << tile_rows << "\n";

            MatrixR inputs_host(batch, inputs_number);
            if (!load_higgs_rows(inputs_host))
            {
                inputs_host = MatrixR::Random(batch, inputs_number);
                std::cout << "data=synthetic\n";
            }

            // The output slot is bf16 in bf16 mode: reserving it at its real
            // width (not an fp32 upper bound) is worth ~2 B/sample of batch.
            // Single-tile runs need no assembly region at all (the
            // propagation's own output slot is the result).
            const bool bf16_resident_input = use_bf16 && bf16_resident_input_enabled();
            const Type input_type = bf16_resident_input ? Type::BF16 : Type::FP32;
            std::cout << "input_type=" << (input_type == Type::BF16 ? "bf16" : "fp32") << "\n";

            const Type expected_output_type = use_bf16 ? Type::BF16 : Type::FP32;
            const Index input_bytes  = get_aligned_bytes(batch * inputs_number, input_type);
            const Index output_bytes = tile_rows >= batch
                ? Index(0) : get_aligned_bytes(batch, expected_output_type);

            Buffer arena(Device::CUDA);
            arena.resize_bytes(input_bytes + output_bytes, Device::CUDA);
            char* const base = arena.as<char>();
            char* const output_base = base + input_bytes;

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

            ForwardPropagation tile_propagation(tile_rows, network.get());
            // The tail (batch % tile) propagation OVERLAYS the tile arena
            // instead of owning a second one: the two never run concurrently
            // and the tail arena is strictly smaller, so this frees up to a
            // full tile of activations at the capacity boundary.
            std::unique_ptr<ForwardPropagation> tail_propagation;
            if (tail_rows > 0)
            {
                tail_propagation = std::make_unique<ForwardPropagation>();
                tail_propagation->set(tail_rows, network.get(), &tile_propagation.data);
            }

            // Single-tile (untiled) runs read the outputs straight from the
            // propagation slot; assembling them into the arena would hold the
            // output twice.
            const bool single_tile = tile_rows >= batch;

            Type output_type = Type::FP32;
            const void* probe_source = nullptr;

            auto run_pass = [&]()
            {
                for (Index start = 0; start < batch; start += tile_rows)
                {
                    const Index rows = std::min<Index>(tile_rows, batch - start);
                    ForwardPropagation& propagation =
                        rows == tile_rows ? tile_propagation : *tail_propagation;

                    const TensorView tile_view(base + start * inputs_number * Index(type_bytes(input_type)),
                                               Shape{rows, inputs_number}, input_type, Device::CUDA);
                    const TensorView compute_tile_view = use_bf16 && tile_view.is_fp32()
                        ? maybe_alias_bf16_input_cast(tile_view, propagation)
                        : tile_view;

                    const bool upload_parameters = !parameters_uploaded;
                    const TensorView tile_outputs = network->calculate_outputs_resident(
                        {compute_tile_view}, propagation, upload_parameters);
                    if (use_bf16 && upload_parameters)
                        network->release_bf16_fp32_parameter_master_for_inference();

                    parameters_uploaded = true;
                    output_type = tile_outputs.type;

                    if (single_tile)
                    {
                        probe_source = tile_outputs.data;
                        continue;
                    }

                    const Index element_bytes = Index(type_bytes(tile_outputs.type));
                    if (element_bytes > Index(type_bytes(expected_output_type)))
                        throw std::runtime_error("output dtype wider than reserved");
                    device::copy_async(output_base + start * element_bytes,
                                       tile_outputs.data, rows * element_bytes,
                                       device::CopyKind::DeviceToDevice, stream);
                    probe_source = output_base;
                }
            };

            // Warmup selects the cuDNN/cuBLAS plans, allocates the tile
            // workspaces, and uploads the parameters; excluded from timing.
            run_pass();
            cudaDeviceSynchronize();

            const auto t0 = std::chrono::high_resolution_clock::now();
            for (Index i = 0; i < iterations; ++i)
                run_pass();
            cudaDeviceSynchronize();
            const auto t1 = std::chrono::high_resolution_clock::now();

            float probe[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const Index probe_size = std::min<Index>(Index(4), batch);
            copy_device_to_host_float(probe_source, output_type, probe_size, probe, stream);
            cudaStreamSynchronize(stream);
            for (Index i = 0; i < probe_size; ++i)
                if (!std::isfinite(probe[i]))
                    throw std::runtime_error("non-finite outputs");

            const double wall_s = std::chrono::duration<double>(t1 - t0).count();

            memory_debug::print(std::cout);
            print_peak_memory();

            std::cout << "wall_s=" << wall_s << "\n";
            std::cout << "samples_per_sec=" << double(batch) * double(iterations) / wall_s << "\n";
            std::cout << "RESULT=OK\n";
            return 0;
        }
#endif // OPENNN_HAS_CUDA

        // train: one optimizer step over the batch. With a tile size (the
        // default) the step runs as gradient accumulation over equal tiles:
        // activation memory is O(tile) and the accumulated update equals the
        // full-batch step (sub-batch gradients are per-batch means, averaged
        // with weight 1/period). tile 0 = the monolithic untiled protocol.
        // The virtual batch rounds up to a tile multiple so tiles stay equal;
        // rows repeat modulo, and passing at the rounded batch implies the
        // requested one fits.
        const Index train_tile = tile_arg > 0 ? std::min<Index>(batch, tile_arg) : batch;
        const Index update_period = (batch + train_tile - 1) / train_tile;
        const Index samples = update_period * train_tile;
        std::cout << "tile_rows=" << train_tile
                  << " update_period=" << update_period
                  << " effective_batch=" << samples << "\n";

        TabularDataset dataset(samples, Shape{inputs_number}, Shape{1});

        MatrixR data(samples, inputs_number + 1);
        if (!load_higgs_rows(data))
        {
            data = MatrixR::Random(samples, inputs_number + 1);
            data.col(inputs_number) = (data.col(inputs_number).array() > 0.0f).cast<float>();
            std::cout << "data=synthetic\n";
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
        if (!adam) throw std::runtime_error("Adam optimizer not found.");

        adam->set_batch_size(train_tile);
        adam->set_update_period(update_period);
        adam->set_maximum_epochs(iterations);
        adam->set_display(false);
        adam->set_gradient_clip_norm(0.0f);
        adam->set_batch_pool_size(1);   // capacity: one device batch copy, not three

        const auto t0 = std::chrono::high_resolution_clock::now();
        const TrainingResult result = training_strategy.train();
#ifdef OPENNN_HAS_CUDA
        if (!use_cpu) cudaDeviceSynchronize();
#endif
        const auto t1 = std::chrono::high_resolution_clock::now();

        if (!std::isfinite(result.loss))
            throw std::runtime_error("non-finite loss");

        const double wall_s = std::chrono::duration<double>(t1 - t0).count();

        memory_debug::print(std::cout);
        print_peak_memory();

        std::cout << "final_loss=" << result.loss << "\n";
        std::cout << "wall_s=" << wall_s << "\n";
        std::cout << "samples_per_sec=" << double(samples) * double(iterations) / wall_s << "\n";
        std::cout << "RESULT=OK\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        std::cout << "RESULT=ERROR\n";
        return 1;
    }
}
