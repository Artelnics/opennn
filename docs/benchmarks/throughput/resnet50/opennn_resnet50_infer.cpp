//   OpenNN GPU ResNet-50 inference-speed benchmark (FORWARD ONLY).
//
//   Inference twin of opennn_resnet50_speed.cpp: same v1.5 bottleneck ResNet-50
//   (opennn::ResNet, blocks [3,4,6,3], stride on the 3x3 conv as torchvision
//   builds it), same CIFAR/ImageDataset loading, same fp32/bf16 selection. The
//   difference is that there is no optimizer, no loss and no backward pass: a
//   single GPU-resident batch is gathered from the dataset once, then a warmup
//   plus N timed forward passes run through the device-resident inference path
//   (NeuralNetwork::calculate_outputs_resident): parameters uploaded once, a
//   caller-owned ForwardPropagation (activation buffers) built once, output left
//   on the GPU. This is the zero-per-call-overhead loop, the fair counterpart to
//   PyTorch's .eval()+no_grad() / TensorFlow's training=False inference loop.
//
//   Data is kept GPU-resident (device-side gather) for the small CIFAR path
//   (image_size==0), exactly like the training benchmark; the 224px ImageNet path
//   is too large and stays host-staged. Enabled in code, not from env vars.
//
//   usage:  opennn_resnet50_infer <data_path> [batch] [runs] [fp32|bf16] [image_size] [cache_dir]

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "opennn/image_dataset.h"
#include "opennn/standard_networks.h"
#include "opennn/batch.h"
#include "opennn/forward_propagation.h"
#include "opennn/random_utilities.h"
#include "opennn/configuration.h"
#include "opennn/device_backend.h"
#include "opennn/memory_debug.h"
#include "opennn/profiler.h"

#ifdef OPENNN_HAS_CUDA
#include <cuda_runtime.h>
#endif

using namespace opennn;
using clock_type = std::chrono::steady_clock;

int main(int argc, char* argv[])
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    try
    {
        const std::string data_path = argc > 1 ? argv[1] : "cifar10/train";
        const Index batch = argc > 2 ? Index(std::stoll(argv[2])) : 128;
        const Index timed_runs = argc > 3 ? Index(std::stoll(argv[3])) : 5;
        const std::string precision = argc > 4 ? argv[4] : "fp32";
        // A negative image_size resizes to |image_size| AND keeps the set device
        // resident (fits smaller crops in VRAM so BF16/FP32 both take the gather
        // path). Positive stays host-staged (large-image ImageNet path).
        const Index image_size_arg = argc > 5 ? Index(std::stoll(argv[5])) : 0;
        const Index image_size = image_size_arg < 0 ? -image_size_arg : image_size_arg;
        const bool force_resident = image_size_arg < 0;
        const std::string cache_dir = argc > 6 ? argv[6] : "";

        memory_debug::reset();

        set_seed(42);
        const Type inference_type = (precision == "bf16") ? Type::BF16 : Type::FP32;
        Configuration::instance().set(Device::CUDA, inference_type);

        // Autotuned conv engines with the workspace cap off, matching the
        // training driver and PyTorch's cudnn.benchmark=True: the cuDNN
        // heuristic alone picks large-tile kernels that lose ~15% on CIFAR's
        // small spatial shapes.
        device::set_conv_autotune(true);
        device::set_conv_workspace_cap(0);

        // A custom image-cache directory is no longer configurable from OpenNN
        // (the setter was removed when backend toggles became code-only). The
        // cache always lands in <data_path>/.cache, which stays outside the repo
        // because datasets live under $OPENNN_BENCH_DATA.
        if (!cache_dir.empty())
            std::cerr << "note: custom cache dir ignored (OpenNN caches in "
                         "<data_path>/.cache): " << cache_dir << "\n";

        std::unique_ptr<ImageDataset> dataset_ptr =
            image_size > 0
                ? std::make_unique<ImageDataset>(data_path, Shape{image_size, image_size, 3})
                : std::make_unique<ImageDataset>(data_path);
        ImageDataset& dataset = *dataset_ptr;
        dataset.set_sample_roles("Training");

        // CIFAR (image_size==0) fits in VRAM: keep it GPU-resident so each batch is
        // a device-side gather. The 224px ImageNet path is too large, so it stays
        // host-staged. Enabled in code; there is no environment switch.
        const bool gpu_resident = (image_size == 0) || force_resident;
        if (gpu_resident)
            dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);
        else
            dataset.set_storage_mode(Dataset::StorageMode::BinaryFile);

        const Index samples = dataset.get_samples_number();
        const Index effective_batch = std::min<Index>(batch, samples);

        std::cout << "samples=" << samples << " batch=" << effective_batch
                  << " runs=" << timed_runs << " precision=" << precision
                  << " gpu_resident=" << gpu_resident;
        if (image_size > 0) std::cout << " image_size=" << image_size;
        std::cout << "\n";

        ResNet network(dataset.get_shape("Input"),
                       {3, 4, 6, 3},
                       Shape{64, 128, 256, 512},
                       dataset.get_shape("Target"),
                       /*use_bottleneck=*/true);

        std::cout << "layers=" << network.get_layers_number()
                  << " parameters=" << network.get_parameters_size() << "\n";

        // One GPU-resident batch of real CIFAR images, gathered ONCE. The dataset
        // fill path is the same one training uses; here it is done a single time so
        // the timed loop is pure forward compute with the input already on device.
        const vector<Index> input_feature_indices = dataset.get_feature_indices("Input");
        const vector<Index> decoder_feature_indices = dataset.get_feature_indices("Decoder");
        const vector<Index> target_feature_indices = dataset.get_feature_indices("Target");

        vector<Index> sample_indices(static_cast<size_t>(effective_batch));
        for (Index i = 0; i < effective_batch; ++i)
            sample_indices[size_t(i)] = i;

        Batch batch_data(effective_batch, &dataset, network.get_config());
        batch_data.fill(sample_indices,
                        input_feature_indices,
                        decoder_feature_indices,
                        target_feature_indices,
                        FillMode::Inference);

        const vector<TensorView>& inputs = batch_data.get_inputs();

        // ForwardPropagation (activation buffers) built ONCE; parameters uploaded
        // on the first resident call only. The forward is captured into a CUDA
        // graph during the two warmup calls and replayed by the timed loop.
        ForwardPropagation forward_propagation(effective_batch, &network);
        forward_propagation.set_cuda_graph(true);
        std::cout << "cuda_graph=on\n";

        network.calculate_outputs_resident(inputs, forward_propagation, /*upload=*/true);
#ifdef OPENNN_HAS_CUDA
        device::synchronize();
#endif

        // Warmup: selects the cuDNN/cuBLAS plans and pages the activation arena;
        // excluded from timing.
        network.calculate_outputs_resident(inputs, forward_propagation, /*upload=*/false);
#ifdef OPENNN_HAS_CUDA
        device::synchronize();
#endif

        std::vector<double> times;
        times.reserve(size_t(timed_runs));
        for (Index run = 0; run < timed_runs; ++run)
        {
            const auto t0 = clock_type::now();
            network.calculate_outputs_resident(inputs, forward_propagation, /*upload=*/false);
#ifdef OPENNN_HAS_CUDA
            device::synchronize();
#endif
            const auto t1 = clock_type::now();
            times.push_back(std::chrono::duration<double>(t1 - t0).count());
        }

        // Sanity check on the (device-resident) outputs.
        const TensorView outputs = forward_propagation.get_outputs();
#ifdef OPENNN_HAS_CUDA
        float probe[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const Index probe_size = std::min<Index>(Index(4), outputs.size());
        copy_device_to_host_float(outputs.data, outputs.type, probe_size,
                                  probe, Backend::get_compute_stream());
        cudaStreamSynchronize(Backend::get_compute_stream());
        for (Index i = 0; i < probe_size; ++i)
            if (!std::isfinite(probe[i]))
                throw std::runtime_error("non-finite outputs");
#endif

        // OPENNN_PROFILE=1: after the official timing, rerun the forward with the
        // library profiler on. Every PROFILE_SCOPE syncs the GPU, so the profiled
        // pass is slower than the timed one — use it for the % split, not the
        // absolute throughput.
        if (const char* profile_env = std::getenv("OPENNN_PROFILE");
            profile_env && profile_env[0] == '1')
        {
            // The per-op scopes need the eager forward (each one syncs the GPU).
            forward_propagation.set_cuda_graph(false);
            const Index profile_runs = 20;
            ::opennn::enabled() = true;
            ::opennn::global_stats().clear();
            const auto p0 = clock_type::now();
            for (Index run = 0; run < profile_runs; ++run)
                network.calculate_outputs_resident(inputs, forward_propagation, /*upload=*/false);
#ifdef OPENNN_HAS_CUDA
            device::synchronize();
#endif
            const double profile_ms =
                std::chrono::duration<double, std::milli>(clock_type::now() - p0).count();
            ::opennn::enabled() = false;
            ::opennn::global_stats().print(std::cout, "ResNet-50 inference forward breakdown",
                                           profile_ms);
            std::cout << "profile_ms_per_batch=" << profile_ms / double(profile_runs) << "\n";
        }

        std::sort(times.begin(), times.end());
        const double batch_s = times[times.size() / 2];   // median forward pass
        const double ms_per_batch = batch_s * 1000.0;

        std::cout << "ms_per_batch=" << ms_per_batch << "\n";
        std::cout << "samples_per_sec=" << long(double(effective_batch) / batch_s) << "\n";
        memory_debug::print(std::cout);
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
