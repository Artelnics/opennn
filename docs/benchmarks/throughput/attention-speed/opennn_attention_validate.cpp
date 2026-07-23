//   Correctness probe for OpenNN's multi-head attention / Transformer forward
//   path on GPU. The library's MHA unit tests only check construction (the
//   forward/backward bodies are commented out), so before benchmarking we
//   validate that the GPU attention compute agrees with OpenNN's own CPU path:
//   build one small Transformer, set identical constant parameters, feed the
//   same token inputs, run calculate_outputs on CPU and on CUDA, and compare.
//
//   Agreement (max abs diff ~ 0, all finite) means the GPU attention forward is
//   internally consistent and safe to benchmark. Divergence or NaNs means it is
//   not benchmark-ready.
//
//   usage: opennn_attention_validate [seq_len] [d_model] [heads] [ff] [layers] [vocab] [batch]

#include <cmath>
#include <iostream>
#include <vector>

#include "opennn/standard_networks.h"
#include "opennn/configuration.h"
#include "opennn/random_utilities.h"

using namespace opennn;

static Tensor3 run_on(Device device,
                      Index seq, Index d_model, Index heads, Index ff, Index layers,
                      Index vocab, Index batch, Type training_type = Type::FP32)
{
    Configuration::instance().set(device, training_type);
    set_seed(0);

    Transformer transformer(seq, seq, vocab, vocab, d_model, heads, ff, layers);

    VectorR params(transformer.get_parameters_size());
    params.setConstant(0.02f);
    transformer.set_parameters(params);

    Tensor3 inputs(batch, seq, 1);
    Tensor3 context(batch, seq, 1);
    for (Index b = 0; b < batch; ++b)
        for (Index s = 0; s < seq; ++s)
        {
            inputs(b, s, 0)  = float((b * seq + s) % vocab);
            context(b, s, 0) = float((b * seq + s + 1) % vocab);
        }

    return transformer.calculate_outputs(inputs, context);
}

int main(int argc, char* argv[])
{
    cout << unitbuf;

    const Index seq     = argc > 1 ? Index(stoll(argv[1])) : 16;
    const Index d_model = argc > 2 ? Index(stoll(argv[2])) : 64;
    const Index heads   = argc > 3 ? Index(stoll(argv[3])) : 4;
    const Index ff      = argc > 4 ? Index(stoll(argv[4])) : 128;
    const Index layers  = argc > 5 ? Index(stoll(argv[5])) : 2;
    const Index vocab   = argc > 6 ? Index(stoll(argv[6])) : 100;
    const Index batch   = argc > 7 ? Index(stoll(argv[7])) : 4;

    try
    {
        cout << "config seq=" << seq << " d_model=" << d_model << " heads=" << heads
                  << " ff=" << ff << " layers=" << layers << " vocab=" << vocab
                  << " batch=" << batch << "\n";

        const bool gpu_bf16 = getenv("OPENNN_BF16") != nullptr;
        const Tensor3 cpu = run_on(Device::CPU,  seq, d_model, heads, ff, layers, vocab, batch);
        const Tensor3 gpu = run_on(Device::CUDA, seq, d_model, heads, ff, layers, vocab, batch,
                                   gpu_bf16 ? Type::BF16 : Type::FP32);

        cout << "cpu_shape=" << cpu.dimension(0) << "x" << cpu.dimension(1) << "x" << cpu.dimension(2)
                  << "  gpu_shape=" << gpu.dimension(0) << "x" << gpu.dimension(1) << "x" << gpu.dimension(2) << "\n";

        if (cpu.dimension(0) != gpu.dimension(0)
            || cpu.dimension(1) != gpu.dimension(1)
            || cpu.dimension(2) != gpu.dimension(2))
        {
            cout << "RESULT=SHAPE_MISMATCH\n";
            return 1;
        }

        double max_abs_diff = 0.0, max_abs_val = 0.0;
        bool cpu_nan = false, gpu_nan = false;
        for (Index i = 0; i < cpu.size(); ++i)
        {
            const double c = cpu.data()[i], g = gpu.data()[i];
            if (isnan(c) || isinf(c)) cpu_nan = true;
            if (isnan(g) || isinf(g)) gpu_nan = true;
            if (!isnan(c) && !isnan(g) && !isinf(c) && !isinf(g))
                max_abs_diff = max(max_abs_diff, abs(c - g));
            if (!isnan(c) && !isinf(c))
                max_abs_val = max(max_abs_val, abs(c));
        }
        const bool any_nan = cpu_nan || gpu_nan;

        cout << "max_abs_value=" << max_abs_val << "\n";
        cout << "max_abs_diff_cpu_vs_gpu=" << max_abs_diff << "\n";
        cout << "cpu_nan=" << (cpu_nan ? "YES" : "no")
                  << " gpu_nan=" << (gpu_nan ? "YES" : "no") << "\n";

        const double tol = gpu_bf16 ? 0.05 * max_abs_val + 1e-3 : 1e-3;
        const bool ok = !any_nan && max_abs_diff < tol;
        cout << "tolerance=" << tol << "\n";
        cout << "RESULT=" << (ok ? "MATCH" : "DIVERGE") << "\n";
        return ok ? 0 : 2;
    }
    catch (const exception& e)
    {
        cerr << "EXCEPTION: " << e.what() << "\n";
        cout << "RESULT=CRASH\n";
        return 3;
    }
}
