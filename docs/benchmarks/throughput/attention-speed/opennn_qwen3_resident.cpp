//   OpenNN Qwen3 (grouped-query attention) inference throughput, DEVICE-RESIDENT
//   path. Mirrors opennn_transformer_resident.cpp so the two are comparable: token
//   inputs live on the GPU, parameters are uploaded once, the ForwardPropagation is
//   built once, and the loop calls calculate_outputs_resident.
//
//   This exists to baseline the grouped-query attention path specifically.
//   MultiHeadAttention runs cuDNN's fused SDPA, but GroupedQueryAttention runs its
//   own implementation: batched cuBLAS GEMMs with CUBLAS_DEFAULT_MATH (tensor cores
//   off) around a hand-written softmax, materializing the attention matrix. This
//   measures what that costs before any change is made to it.
//
//   usage: opennn_qwen3_resident [seq] [hidden] [q_heads] [kv_heads] [head_dim]
//                                [intermediate] [layers] [vocab] [batch] [iters]
//          OPENNN_BF16=1 selects bf16; OPENNN_PROFILE=1 prints the op breakdown.

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "opennn/neural_network/standard_networks.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/profiler.h"

using namespace opennn;

int main(int argc, char* argv[])
{
    cout << unitbuf;

    const Index seq          = argc >  1 ? Index(stoll(argv[1]))  : 512;
    const Index hidden       = argc >  2 ? Index(stoll(argv[2]))  : 1024;
    const Index query_heads  = argc >  3 ? Index(stoll(argv[3]))  : 16;
    const Index kv_heads     = argc >  4 ? Index(stoll(argv[4]))  : 4;
    const Index head_dim     = argc >  5 ? Index(stoll(argv[5]))  : 64;
    const Index intermediate = argc >  6 ? Index(stoll(argv[6]))  : 2816;
    const Index layers       = argc >  7 ? Index(stoll(argv[7]))  : 4;
    const Index vocab        = argc >  8 ? Index(stoll(argv[8]))  : 32000;
    const Index batch        = argc >  9 ? Index(stoll(argv[9]))  : 8;
    const Index iters        = argc > 10 ? Index(stoll(argv[10])) : 50;

    try
    {
        set_seed(0);
        const bool use_bf16 = getenv("OPENNN_BF16") != nullptr;
        Configuration::instance().set(Device::CUDA, use_bf16 ? Type::BF16 : Type::FP32);
        cout << "precision=" << (use_bf16 ? "bf16" : "fp32") << "\n";

        Qwen3 qwen(seq, vocab, hidden, layers, query_heads, kv_heads, head_dim, intermediate);

        cout << "config seq=" << seq << " hidden=" << hidden
             << " q_heads=" << query_heads << " kv_heads=" << kv_heads
             << " head_dim=" << head_dim << " intermediate=" << intermediate
             << " layers=" << layers << " vocab=" << vocab << " batch=" << batch << "\n";
        cout << "gqa_ratio=" << double(query_heads) / double(kv_heads) << "\n";
        cout << "parameters=" << qwen.get_parameters_buffer_size() << "\n";

        Tensor3 host_in(batch, seq, 1);
        for (Index b = 0; b < batch; ++b)
            for (Index s = 0; s < seq; ++s)
                host_in(b, s, 0) = float((b * seq + s) % vocab);

        const Index in_bytes = batch * seq * Index(sizeof(float));
        Buffer in_gpu;
        in_gpu.resize_bytes(in_bytes, Device::CUDA);
        device::copy_async(in_gpu.data, host_in.data(), in_bytes, device::CopyKind::HostToDevice);
        device::synchronize();

        const vector<TensorView> gpu_inputs{
            TensorView(in_gpu.as<float>(), {batch, seq, 1}, Type::FP32, Device::CUDA)};

        ForwardPropagation forward_propagation(
            batch, &qwen, ForwardPropagationMode::Inference);

        qwen.calculate_outputs_resident(gpu_inputs, forward_propagation, true);
        device::synchronize();

        if (getenv("OPENNN_PROFILE"))
        {
            ::opennn::enabled() = true;
            ::opennn::global_stats().clear();
            const auto p0 = chrono::steady_clock::now();
            for (Index it = 0; it < 10; ++it)
                qwen.calculate_outputs_resident(gpu_inputs, forward_propagation, false);
            device::synchronize();
            const double prof_ms =
                chrono::duration<double, milli>(chrono::steady_clock::now() - p0).count();
            ::opennn::global_stats().print(cout, "Qwen3 forward op breakdown", prof_ms);
            ::opennn::enabled() = false;
            ::opennn::global_stats().clear();
        }

        for (Index it = 0; it < 5; ++it)
            qwen.calculate_outputs_resident(gpu_inputs, forward_propagation, false);
        device::synchronize();

        const auto start = chrono::steady_clock::now();
        for (Index it = 0; it < iters; ++it)
            qwen.calculate_outputs_resident(gpu_inputs, forward_propagation, false);
        device::synchronize();
        const double elapsed_s =
            chrono::duration<double>(chrono::steady_clock::now() - start).count();

        const double tokens = double(batch) * double(seq) * double(iters);
        cout << "ms_per_iter=" << (elapsed_s * 1000.0 / double(iters)) << "\n";
        cout << "tokens_per_sec=" << (tokens / elapsed_s) << "\n";

        // Output checksum, so a faster attention path can be shown to compute the
        // same thing rather than merely finishing sooner.
        const TensorView outputs = forward_propagation.get_outputs();
        vector<float> host(size_t(outputs.size()));
        device::copy_async(host.data(), outputs.data, outputs.size(),
                           Device::CPU, Device::CUDA, device::get_compute_stream());
        device::synchronize();

        double sum = 0.0, absolute_sum = 0.0;
        for (const float value : host) { sum += value; absolute_sum += fabs(double(value)); }
        cout << "checksum_sum=" << sum << "\n";
        cout << "checksum_abs=" << absolute_sum << "\n";

        cout << "RESULT=OK\n";
    }
    catch (const exception& e)
    {
        cerr << "RESULT=ERROR " << e.what() << "\n";
        return 1;
    }

    return 0;
}
