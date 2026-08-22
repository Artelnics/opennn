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
//   is too large and stays host-staged (a negative image_size forces it resident
//   anyway). Enabled in code, not from env vars. OPENNN_PROFILE=1 appends the
//   library's per-scope breakdown of the forward after the timed runs.
//
//   usage:  opennn_resnet50_infer <data_path> [batch] [runs] [fp32|bf16] [image_size]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/memory_debug.h"
#include "opennn/core/profiler.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/batch.h"
#include "opennn/dataset/image_dataset.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/standard_networks.h"

#ifdef OPENNN_HAS_CUDA
#include <cuda_runtime.h>
#endif

using namespace opennn;
using clock_type = chrono::steady_clock;

int main(int argc, char* argv[])
{
    cout << unitbuf;
    cerr << unitbuf;
    try
    {
        const string data_path = argc > 1 ? argv[1] : "cifar10/train";
        const Index batch = argc > 2 ? Index(stoll(argv[2])) : 128;
        const Index timed_runs = argc > 3 ? Index(stoll(argv[3])) : 5;
        const string precision = argc > 4 ? argv[4] : "fp32";
        const Index image_size_arg = argc > 5 ? Index(stoll(argv[5])) : 0;
        const Index image_size = image_size_arg < 0 ? -image_size_arg : image_size_arg;

        memory_debug::reset();

        set_seed(42);
        Configuration::instance().set(Device::CUDA, precision == "bf16" ? Type::BF16 : Type::FP32);

        // Autotune within the library's auto workspace budget. Unbounded (cap 0)
        // lets cuDNN's first heuristic choice take a scratch that grows with the
        // batch and is slower than the budgeted plan; see opennn_resnet50_speed.cpp.
        device::set_conv_autotune(true);
        device::set_conv_workspace_cap(-1);

        const unique_ptr<ImageDataset> dataset_ptr = image_size > 0
            ? make_unique<ImageDataset>(data_path, Shape{image_size, image_size, 3})
            : make_unique<ImageDataset>(data_path);
        ImageDataset& dataset = *dataset_ptr;
        dataset.set_sample_roles("Training");

        const bool gpu_resident = image_size == 0 || image_size_arg < 0;
        dataset.set_storage_mode(gpu_resident ? Dataset::StorageMode::GPUPersistantData
                                              : Dataset::StorageMode::BinaryFile);

        const Index samples = dataset.get_samples_number();
        const Index effective_batch = min<Index>(batch, samples);

        cout << "samples=" << samples << " batch=" << effective_batch
             << " runs=" << timed_runs << " precision=" << precision
             << " gpu_resident=" << gpu_resident;
        if (image_size > 0) cout << " image_size=" << image_size;
        cout << "\n";

        ResNet network(dataset.get_shape("Input"),
                       {3, 4, 6, 3},
                       Shape{64, 128, 256, 512},
                       dataset.get_shape("Target"),
                       true);

        cout << "layers=" << network.get_layers_number()
             << " parameters=" << network.get_parameters_buffer_size() << "\n";

        // One batch - the first effective_batch samples - gathered once and
        // held on the GPU for every pass.
        const vector<Index> input_feature_indices = dataset.get_feature_indices("Input");
        const vector<Index> decoder_feature_indices = dataset.get_feature_indices("Decoder");
        const vector<Index> target_feature_indices = dataset.get_feature_indices("Target");

        vector<Index> sample_indices(size_t(effective_batch));
        for (Index i = 0; i < effective_batch; ++i)
            sample_indices[size_t(i)] = i;

        Batch batch_data(effective_batch, &dataset, network.get_config());
        batch_data.fill(sample_indices,
                        input_feature_indices,
                        decoder_feature_indices,
                        target_feature_indices,
                        FillMode::Inference);

        const vector<TensorView>& inputs = batch_data.get_inputs();

        ForwardPropagation forward_propagation(effective_batch, &network, ForwardPropagationMode::Inference);
        forward_propagation.set_cuda_graph(true);
        cout << "cuda_graph=on\n";

        // Two warmup passes before the timed ones.
        network.calculate_outputs_resident(inputs, forward_propagation, true);
#ifdef OPENNN_HAS_CUDA
        device::synchronize();
#endif
        network.calculate_outputs_resident(inputs, forward_propagation, false);
#ifdef OPENNN_HAS_CUDA
        device::synchronize();
#endif

        vector<double> times;
        for (Index run = 0; run < timed_runs; ++run)
        {
            const auto t0 = clock_type::now();
            network.calculate_outputs_resident(inputs, forward_propagation, false);
#ifdef OPENNN_HAS_CUDA
            device::synchronize();
#endif
            times.push_back(chrono::duration<double>(clock_type::now() - t0).count());
        }

#ifdef OPENNN_HAS_CUDA
        // A few output values back on the host: the passes must have produced
        // finite numbers.
        const TensorView outputs = forward_propagation.get_outputs();
        float probe[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const Index probe_size = min<Index>(Index(4), outputs.size());
        copy_device_to_host_float(outputs.get_data(), outputs.get_type(), probe_size,
                                  probe, device::get_compute_stream());
        cudaStreamSynchronize(device::get_compute_stream());
        for (Index i = 0; i < probe_size; ++i)
            if (!isfinite(probe[i]))
                throw runtime_error("non-finite outputs");
#endif

        if (const char* profile = getenv("OPENNN_PROFILE"); profile && profile[0] == '1')
        {
            // Per-scope breakdown of the eager (non-graphed) forward. Diagnostic
            // only, after the timed runs.
            forward_propagation.set_cuda_graph(false);
            const Index profile_runs = 20;
            profiler::set_enabled(true);
            profiler::stats().clear();
            const auto p0 = clock_type::now();
            for (Index run = 0; run < profile_runs; ++run)
                network.calculate_outputs_resident(inputs, forward_propagation, false);
#ifdef OPENNN_HAS_CUDA
            device::synchronize();
#endif
            const double profile_ms = chrono::duration<double, milli>(clock_type::now() - p0).count();
            profiler::set_enabled(false);
            profiler::stats().print(cout, "ResNet-50 inference forward breakdown", profile_ms);
            cout << "profile_ms_per_batch=" << profile_ms / double(profile_runs) << "\n";
        }

        sort(times.begin(), times.end());
        const double batch_s = times[times.size() / 2];

        cout << "ms_per_batch=" << batch_s * 1000.0 << "\n";
        cout << "samples_per_sec=" << long(double(effective_batch) / batch_s) << "\n";
        memory_debug::print(cout);
        cout << "RESULT=OK\n";
        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << "\n";
        cout << "RESULT=ERROR\n";
        return 1;
    }
}
