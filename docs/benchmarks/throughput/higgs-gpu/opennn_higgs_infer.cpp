//   OpenNN GPU HIGGS dense inference-speed benchmark.
//
//   Forward-only throughput of the canonical HIGGS dense classifier
//   (28 -> hidden -> hidden -> 1, ReLU hidden, sigmoid output -- see
//   docs/benchmarks/throughput/higgs/README.md) on the GPU. The test CSV is
//   loaded once (features then last-column label; the label is ignored for the
//   speed measurement), the inputs are made device-resident, and the network
//   forward (calculate_outputs_resident) is replayed over batches: a warmup pass
//   plus N timed passes. No optimizer state, no gradients, no per-call H2D copy.
//
//   Precision is selectable fp32 or bf16, matching the autocast / mixed_bfloat16
//   used on the PyTorch and TensorFlow sides. It is selected exactly like
//   opennn_speed.cpp: Configuration::instance().set(Device::CUDA, type).
//
//   usage:  opennn_higgs_infer <test_csv> [batch] [runs] [fp32|bf16]
//                              [hidden] [hidden_layers] [activation]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef OPENNN_HAS_CUDA
#include <cuda_runtime.h>
#include "opennn/device_backend.h"
#endif

#include "opennn/configuration.h"
#include "opennn/dense_layer.h"
#include "opennn/forward_propagation.h"
#include "opennn/neural_network.h"
#include "opennn/random_utilities.h"
#include "opennn/tabular_dataset.h"
#include "opennn/tensor_types.h"

using namespace opennn;
using clock_type = std::chrono::steady_clock;

namespace
{

std::unique_ptr<NeuralNetwork> make_network(const Shape& input_shape,
                                            const Shape& target_shape,
                                            Index hidden,
                                            Index hidden_layers,
                                            const std::string& activation)
{
    auto network = std::make_unique<NeuralNetwork>();
    Shape current = input_shape;
    const std::string hidden_activation = (activation == "relu" || activation == "ReLU")
        ? "ReLU"
        : "Tanh";

    for (Index i = 0; i < hidden_layers; ++i)
    {
        network->add_layer(std::make_unique<opennn::Dense>(
            current,
            Shape{hidden},
            hidden_activation,
            false,
            "higgs_dense_" + std::to_string(i + 1)));
        current = network->get_output_shape();
    }

    network->add_layer(std::make_unique<opennn::Dense>(
        current,
        target_shape,
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

    try
    {
        if (argc < 2)
        {
            std::cerr << "usage: opennn_higgs_infer <test_csv> [batch] [runs] [fp32|bf16]"
                         " [hidden] [hidden_layers] [activation]\n";
            return 2;
        }

        const std::string test_path = argv[1];
        const Index batch = argc > 2 ? Index(std::stoll(argv[2])) : 8192;
        const Index runs  = argc > 3 ? std::max<Index>(Index(1), Index(std::stoll(argv[3]))) : 5;
        const std::string precision = argc > 4 ? argv[4] : "fp32";
        const Index hidden = argc > 5 ? Index(std::stoll(argv[5])) : 1024;
        const Index hidden_layers = argc > 6 ? Index(std::stoll(argv[6])) : 2;
        const std::string activation = argc > 7 ? argv[7] : "relu";

        set_seed(42);
        const Type inference_type = (precision == "bf16") ? Type::BF16 : Type::FP32;
        Configuration::instance().set(Device::CUDA, inference_type);

        // Load the test CSV once (features then last-column label). The label is
        // not needed for the speed measurement; only the feature columns feed the
        // forward pass.
        TabularDataset dataset(test_path, ",", false, false);
        dataset.set_sample_roles("Testing");
        const MatrixR& all = dataset.get_data();
        const Index samples = dataset.get_samples_number();
        const Index inputs_number = dataset.get_input_shape()[0];
        const Index processed = (samples / batch) * batch;
        const MatrixR inputs = all.leftCols(inputs_number);

        std::cout << "engine=opennn\n";
        std::cout << "mode=infer\n";
        std::cout << "device=cuda\n";
        std::cout << "samples=" << processed << "\n";
        std::cout << "batch=" << batch << "\n";
        std::cout << "runs=" << runs << "\n";
        std::cout << "hidden=" << hidden << "\n";
        std::cout << "hidden_layers=" << hidden_layers << "\n";
        std::cout << "activation=" << activation << "\n";
        std::cout << "precision=" << precision << "\n";

        if (processed <= 0)
            throw std::runtime_error("batch larger than the test split");

        auto network = make_network(dataset.get_input_shape(),
                                    dataset.get_target_shape(),
                                    hidden,
                                    hidden_layers,
                                    activation);
        std::cout << "parameters=" << network->get_parameters_number() << "\n";

        const Index batches = processed / batch;

        // Device-resident inputs: upload the whole (batch-aligned) test slice
        // once, then index device batch views each pass. Matches the PyTorch /
        // TensorFlow protocol where the tensor is already on the GPU and only the
        // forward is timed.
        Buffer inputs_device(Device::CUDA);
        const Index input_bytes =
            get_aligned_bytes(processed * inputs_number, Type::FP32);
        inputs_device.resize_bytes(input_bytes, Device::CUDA);

#ifdef OPENNN_HAS_CUDA
        cudaStream_t stream = Backend::get_compute_stream();
        device::copy_async(inputs_device.data,
                           inputs.data(),
                           processed * inputs_number * Index(sizeof(float)),
                           device::CopyKind::HostToDevice,
                           stream);
        cudaStreamSynchronize(stream);
#endif

        // One persistent ForwardPropagation, reused across passes: activations
        // are allocated once, parameters uploaded once (upload_parameters=true on
        // the first call only). The forward is captured into a CUDA graph, which
        // needs a stable input pointer, so each batch is staged with a device-to-
        // device copy into one fixed buffer (~us against a launch-bound forward).
        ForwardPropagation forward_propagation(batch, network.get());
        forward_propagation.set_cuda_graph(true);

        Buffer staging_input;
        staging_input.resize_bytes(batch * inputs_number * Index(sizeof(float)), Device::CUDA);
        const TensorView staging_view(staging_input.as<float>(),
                                      Shape{batch, inputs_number}, Type::FP32, Device::CUDA);

        bool parameters_uploaded = false;
        const TensorView* last_outputs = nullptr;
        TensorView probe_view;

        auto run_pass = [&]()
        {
            cudaStream_t compute = Backend::get_compute_stream();
            for (Index b = 0; b < batches; ++b)
            {
                const Index start = b * batch;
                device::copy_async(staging_input.data,
                                   inputs_device.as<float>() + start * inputs_number,
                                   batch * inputs_number * Index(sizeof(float)),
                                   device::CopyKind::DeviceToDevice,
                                   compute);

                const bool upload_parameters = !parameters_uploaded;
                probe_view = network->calculate_outputs_resident(
                    {staging_view}, forward_propagation, upload_parameters);
                parameters_uploaded = true;
                last_outputs = &probe_view;
            }
        };

        // Warmup: selects cuBLAS/cuDNN plans, allocates the workspaces, uploads
        // parameters. Excluded from timing.
        run_pass();
#ifdef OPENNN_HAS_CUDA
        cudaDeviceSynchronize();
#endif

        std::vector<double> times;
        times.reserve(size_t(runs));
        for (Index r = 0; r < runs; ++r)
        {
            const auto t0 = clock_type::now();
            run_pass();
#ifdef OPENNN_HAS_CUDA
            cudaDeviceSynchronize();
#endif
            const auto t1 = clock_type::now();
            times.push_back(std::chrono::duration<double>(t1 - t0).count());
        }

        // Sanity probe: read a couple of outputs back so a silently-broken
        // forward is caught (mirrors the max-batch trial).
#ifdef OPENNN_HAS_CUDA
        if (last_outputs)
        {
            float probe[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const Index probe_size = std::min<Index>(Index(4), batch);
            copy_device_to_host_float(last_outputs->data, last_outputs->type,
                                      probe_size, probe, stream);
            cudaStreamSynchronize(stream);
            for (Index i = 0; i < probe_size; ++i)
                if (!std::isfinite(probe[i]))
                    throw std::runtime_error("non-finite outputs");
        }
#endif

        std::sort(times.begin(), times.end());
        const double median_pass_s = times[times.size() / 2];
        const double samples_per_sec = double(processed) / median_pass_s;
        const double ms_per_batch = median_pass_s * 1000.0 / double(batches);

        std::cout << "median_pass_s=" << median_pass_s << "\n";
        std::cout << "samples_per_sec=" << long(samples_per_sec) << "\n";
        std::cout << "ms_per_batch=" << ms_per_batch << "\n";
        std::cout << "RESULT=OK\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        std::cout << "RESULT=ERROR\n";
        return 1;
    }
}
