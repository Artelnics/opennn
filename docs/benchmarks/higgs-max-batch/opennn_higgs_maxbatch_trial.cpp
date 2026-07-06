//   OpenNN HIGGS dense max-batch trial, GPU.
//
//   One process = one (mode, batch, precision) attempt at the canonical HIGGS
//   dense classifier (28 -> hidden -> hidden -> 1, ReLU hidden, sigmoid
//   output, binary cross-entropy -- see docs/benchmarks/higgs/README.md), so
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
//                                      [cuda|cpu]
//   env:   OPENNN_BF16=1  -> bf16 (CUDA only; else fp32)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../../../opennn/adaptive_moment_estimation.h"
#include "../../../opennn/configuration.h"
#include "../../../opennn/dense_layer.h"
#include "../../../opennn/device_backend.h"
#include "../../../opennn/forward_propagation.h"
#include "../../../opennn/memory_debug.h"
#include "../../../opennn/neural_network.h"
#include "../../../opennn/random_utilities.h"
#include "../../../opennn/tabular_dataset.h"
#include "../../../opennn/training_strategy.h"

using namespace opennn;

namespace
{

constexpr Index inputs_number = 28;   // HIGGS contract: 28 features, 1 target

std::unique_ptr<NeuralNetwork> make_network(Index hidden, Index hidden_layers)
{
    auto network = std::make_unique<NeuralNetwork>();
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

    try
    {
        set_seed(0);
        const bool use_cpu = device == "cpu";
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
            const MatrixR inputs_host = MatrixR::Random(batch, inputs_number);
            const TensorView input_view(const_cast<float*>(inputs_host.data()),
                                        Shape{batch, inputs_number}, Type::FP32);
            const std::vector<TensorView> inputs = {input_view};

            ForwardPropagation forward_propagation(batch, network.get());

            network->forward_propagate(inputs, forward_propagation, false);   // warmup

            const auto t0 = std::chrono::high_resolution_clock::now();
            for (Index i = 0; i < iterations; ++i)
                network->forward_propagate(inputs, forward_propagation, false);
            const auto t1 = std::chrono::high_resolution_clock::now();

            const MatrixMap outputs = forward_propagation.get_outputs().as_matrix();
            if (!std::isfinite(outputs(0, 0)))
                throw std::runtime_error("non-finite outputs");

            const double wall_s = std::chrono::duration<double>(t1 - t0).count();

            std::cout << "wall_s=" << wall_s << "\n";
            std::cout << "samples_per_sec=" << double(batch) * double(iterations) / wall_s << "\n";
            std::cout << "RESULT=OK\n";
            return 0;
        }

        if (mode == "infer")
        {
            const MatrixR inputs_host = MatrixR::Random(batch, inputs_number);

            Buffer arena(Device::CUDA);
            arena.resize_bytes(get_aligned_bytes(batch * inputs_number, Type::FP32), Device::CUDA);

            TensorView input_view(arena.as<char>(), {batch, inputs_number},
                                  Type::FP32, Device::CUDA);

            cudaStream_t stream = Backend::get_compute_stream();
            device::copy_async(input_view.data, inputs_host.data(), input_view.byte_size(),
                               device::CopyKind::HostToDevice, stream);

            const std::vector<TensorView> inputs = {input_view};

            ForwardPropagation forward_propagation(batch, network.get());

            // Warmup allocates the activation workspace and uploads the
            // parameters; excluded from timing.
            network->calculate_outputs_resident(inputs, forward_propagation, true);
            cudaDeviceSynchronize();

            const auto t0 = std::chrono::high_resolution_clock::now();
            for (Index i = 0; i < iterations; ++i)
                network->calculate_outputs_resident(inputs, forward_propagation, false);
            cudaDeviceSynchronize();
            const auto t1 = std::chrono::high_resolution_clock::now();

            const TensorView output_view = forward_propagation.get_outputs();
            float probe[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const Index probe_size = std::min<Index>(Index(4), output_view.size());
            copy_device_to_host_float(output_view.data, output_view.type, probe_size, probe, stream);
            cudaStreamSynchronize(stream);
            for (Index i = 0; i < probe_size; ++i)
                if (!std::isfinite(probe[i]))
                    throw std::runtime_error("non-finite outputs");

            const double wall_s = std::chrono::duration<double>(t1 - t0).count();

            memory_debug::print(std::cout);

            std::cout << "wall_s=" << wall_s << "\n";
            std::cout << "samples_per_sec=" << double(batch) * double(iterations) / wall_s << "\n";
            std::cout << "RESULT=OK\n";
            return 0;
        }

        // train: one full-batch step per epoch (samples == batch).
        TabularDataset dataset(batch, Shape{inputs_number}, Shape{1});

        MatrixR data = MatrixR::Random(batch, inputs_number + 1);
        data.col(inputs_number) = (data.col(inputs_number).array() > 0.0f).cast<float>();
        dataset.set_data(data);
        dataset.set_sample_roles("Training");

        TrainingStrategy training_strategy(network.get(), &dataset);
        training_strategy.set_loss("CrossEntropy");
        training_strategy.get_loss()->set_regularization("NoRegularization");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        if (!adam) throw std::runtime_error("Adam optimizer not found.");

        adam->set_batch_size(batch);
        adam->set_maximum_epochs(iterations);
        adam->set_display(false);
        adam->set_gradient_clip_norm(0.0f);
        adam->set_batch_pool_size(1);   // capacity: one device batch copy, not three

        const auto t0 = std::chrono::high_resolution_clock::now();
        const TrainingResult result = training_strategy.train();
        if (!use_cpu) cudaDeviceSynchronize();
        const auto t1 = std::chrono::high_resolution_clock::now();

        if (!std::isfinite(result.loss))
            throw std::runtime_error("non-finite loss");

        const double wall_s = std::chrono::duration<double>(t1 - t0).count();

        memory_debug::print(std::cout);

        std::cout << "final_loss=" << result.loss << "\n";
        std::cout << "wall_s=" << wall_s << "\n";
        std::cout << "samples_per_sec=" << double(batch) * double(iterations) / wall_s << "\n";
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
