//   OpenNN Transformer inference throughput, DEVICE-RESIDENT path. The fair
//   counterpart to PyTorch's inference loop: both token inputs live on the GPU,
//   parameters are uploaded once, the ForwardPropagation (activation buffers) is
//   built once, and the loop calls calculate_outputs_resident -- no per-call
//   parameter re-upload, no input H2D, no output D2H, no buffer reallocation.
//
//   Forward correctness is validated by opennn_attention_validate.cpp.
//
//   usage: opennn_transformer_resident [seq] [d_model] [heads] [ff] [layers] [vocab] [batch] [iters]

#include <chrono>
#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include "opennn/standard_networks.h"
#include "opennn/forward_propagation.h"
#include "opennn/device_backend.h"
#include "opennn/tensor_types.h"
#include "opennn/configuration.h"
#include "opennn/random_utilities.h"
#include "opennn/profiler.h"

using namespace opennn;

int main(int argc, char* argv[])
{
    cout << unitbuf;

    const Index seq     = argc > 1 ? Index(stoll(argv[1])) : 64;
    const Index d_model = argc > 2 ? Index(stoll(argv[2])) : 512;
    const Index heads   = argc > 3 ? Index(stoll(argv[3])) : 8;
    const Index ff      = argc > 4 ? Index(stoll(argv[4])) : 2048;
    const Index layers  = argc > 5 ? Index(stoll(argv[5])) : 6;
    const Index vocab   = argc > 6 ? Index(stoll(argv[6])) : 10000;
    const Index batch   = argc > 7 ? Index(stoll(argv[7])) : 8;
    const Index iters   = argc > 8 ? Index(stoll(argv[8])) : 50;

    try
    {
        set_seed(0);
        const bool use_bf16 = getenv("OPENNN_BF16") != nullptr;
        Configuration::instance().set(Device::CUDA, use_bf16 ? Type::BF16 : Type::FP32);
        cout << "precision=" << (use_bf16 ? "bf16" : "fp32") << "\n";

        Transformer transformer(seq, seq, vocab, vocab, d_model, heads, ff, layers);

        if (const char* e = getenv("OPENNN_SDPA_MIN"))
            transformer.set_attention_sdpa_min_sequence_length(Index(stoll(e)));

        cout << "config seq=" << seq << " d_model=" << d_model << " heads=" << heads
                  << " ff=" << ff << " layers=" << layers << " vocab=" << vocab
                  << " batch=" << batch << "\n";
        cout << "parameters=" << transformer.get_parameters_size() << "\n";

        Tensor3 host_in(batch, seq, 1), host_ctx(batch, seq, 1);
        for (Index b = 0; b < batch; ++b)
            for (Index s = 0; s < seq; ++s)
            {
                host_in(b, s, 0)  = float((b * seq + s) % vocab);
                host_ctx(b, s, 0) = float((b * seq + s + 1) % vocab);
            }

        const Index in_bytes = batch * seq * Index(sizeof(float));
        Buffer in_gpu, ctx_gpu;
        in_gpu.resize_bytes(in_bytes, Device::CUDA);
        ctx_gpu.resize_bytes(in_bytes, Device::CUDA);
        device::copy_async(in_gpu.data,  host_in.data(),  in_bytes, device::CopyKind::HostToDevice);
        device::copy_async(ctx_gpu.data, host_ctx.data(), in_bytes, device::CopyKind::HostToDevice);
        device::synchronize();

        const vector<TensorView> gpu_inputs{
            TensorView(in_gpu.as<float>(),  {batch, seq, 1}, Type::FP32, Device::CUDA),
            TensorView(ctx_gpu.as<float>(), {batch, seq, 1}, Type::FP32, Device::CUDA)};

        ForwardPropagation forward_propagation(batch, &transformer);

        transformer.calculate_outputs_resident(gpu_inputs, forward_propagation,            true);
        device::synchronize();

        if (getenv("OPENNN_PROFILE"))
        {
            ::opennn::enabled() = true;
            ::opennn::global_stats().clear();
            const auto p0 = chrono::steady_clock::now();
            const Index prof_iters = 10;
            for (Index it = 0; it < prof_iters; ++it)
                transformer.calculate_outputs_resident(gpu_inputs, forward_propagation,            false);
            device::synchronize();
            const double prof_ms =
                chrono::duration<double, milli>(chrono::steady_clock::now() - p0).count();
            ::opennn::global_stats().print(cout, "Transformer forward op breakdown", prof_ms);
            ::opennn::enabled() = false;
            ::opennn::global_stats().clear();
        }

        forward_propagation.set_cuda_graph(true);
        for (Index it = 0; it < 2; ++it)
            transformer.calculate_outputs_resident(gpu_inputs, forward_propagation,            false);
        device::synchronize();
        cout << "cuda_graph=on\n";

        cudaStream_t stream = device::get_compute_stream();
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);

        const auto t0 = chrono::steady_clock::now();
        cudaEventRecord(ev0, stream);
        for (Index it = 0; it < iters; ++it)
            transformer.calculate_outputs_resident(gpu_inputs, forward_propagation,            false);
        cudaEventRecord(ev1, stream);
        device::synchronize();
        const auto t1 = chrono::steady_clock::now();

        float gpu_ms = 0.0f;
        cudaEventElapsedTime(&gpu_ms, ev0, ev1);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);

        const double per = chrono::duration<double>(t1 - t0).count() / double(iters);
        const double gpu_per = double(gpu_ms) / 1000.0 / double(iters);
        const double tokens = double(batch) * double(seq);
        cout << "step_s=" << per << "\n";
        cout << "gpu_step_s=" << gpu_per << "\n";
        cout << "host_overhead_s=" << (per - gpu_per)
                  << " (" << long((per - gpu_per) / per * 100) << "% of step)\n";
        cout << "tokens_per_sec=" << long(tokens / per) << "\n";
        cout << "gpu_bound_tokens_per_sec=" << long(tokens / gpu_per) << "\n";
        cout << "sequences_per_sec=" << long(double(batch) / per) << "\n";
        cout << "RESULT=OK\n";
        return 0;
    }
    catch (const exception& e)
    {
        cerr << "FAIL: " << e.what() << "\n";
        cout << "RESULT=ERROR\n";
        return 1;
    }
}
