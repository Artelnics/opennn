//   OpenNN Transformer inference throughput benchmark ("Attention Is All You
//   Need" forward pass). Builds an encoder-decoder Transformer (the opennn::
//   Transformer standard network) and times the steady-state forward pass on the
//   GPU, after a warmup. Token inputs are generated on-host once; the forward
//   (calculate_outputs) is repeated and timed. Reports tokens/sec.
//
//   The forward path is CPU-vs-GPU validated by opennn_attention_validate.cpp.
//
//   usage: opennn_transformer_infer [seq] [d_model] [heads] [ff] [layers] [vocab] [batch] [iters] [fp32|bf16] [percall|reuse]

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

#include "opennn/neural_network/standard_networks.h"
#include "opennn/neural_network/layers/scaling_layer.h"
#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "docs/benchmarks/transformer_benchmark.h"

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
    const bool use_bf16 = argc > 9 ? string(argv[9]) == "bf16" : false;
    const bool reuse_outputs = argc > 10 ? string(argv[10]) == "reuse" : false;

    try
    {
        set_seed(0);
        Configuration::instance().set(Device::CUDA, use_bf16 ? Type::BF16 : Type::FP32);

        Transformer transformer(seq, seq, vocab, vocab, d_model, heads, ff, layers);

        const Index sdpa_min_sequence_length =
            benchmark::configure_transformer_sdpa(transformer);

        cout << "config seq=" << seq << " d_model=" << d_model << " heads=" << heads
                  << " ff=" << ff << " layers=" << layers << " vocab=" << vocab
                  << " batch=" << batch
                  << " sdpa_min=" << sdpa_min_sequence_length << " precision=" << (use_bf16 ? "bf16" : "fp32") << "\n";
        cout << "parameters=" << transformer.get_parameters_buffer_size() << "\n";

        Tensor3 inputs(batch, seq, 1);
        Tensor3 context(batch, seq, 1);
        for (Index b = 0; b < batch; ++b)
            for (Index s = 0; s < seq; ++s)
            {
                inputs(b, s, 0)  = float((b * seq + s) % vocab);
                context(b, s, 0) = float((b * seq + s + 1) % vocab);
            }

        auto checksum = [](const float* values, Index count)
        {
            double total = 0.0;
            for (Index i = 0; i < count; ++i) total += double(values[i]);
            return total;
        };

        const vector<TensorView> input_views = {
            TensorView(const_cast<float*>(inputs.data()),
                       {{inputs.dimension(0), inputs.dimension(1), inputs.dimension(2)}}),
            TensorView(const_cast<float*>(context.data()),
                       {{context.dimension(0), context.dimension(1), context.dimension(2)}})};

        chrono::steady_clock::time_point t0, t1;
        double result_checksum = 0.0;

        if (reuse_outputs)
        {
            MatrixR outputs;
            transformer.calculate_outputs(input_views, outputs);

            t0 = chrono::steady_clock::now();
            for (Index it = 0; it < iters; ++it)
                transformer.calculate_outputs(input_views, outputs);
            t1 = chrono::steady_clock::now();

            result_checksum = checksum(outputs.data(), Index(outputs.size()));
        }
        else
        {
            Tensor3 outputs = transformer.calculate_outputs(inputs, context);

            t0 = chrono::steady_clock::now();
            for (Index it = 0; it < iters; ++it)
                outputs = transformer.calculate_outputs(inputs, context);
            t1 = chrono::steady_clock::now();

            result_checksum = checksum(outputs.data(), Index(outputs.size()));
        }

        cout << "mode=" << (reuse_outputs ? "reuse" : "percall") << "\n";
        cout << "checksum=" << result_checksum << "\n";

        const double per = chrono::duration<double>(t1 - t0).count() / double(iters);
        const double tokens = double(batch) * double(seq);
        cout << "step_s=" << per << "\n";
        cout << "tokens_per_sec=" << long(tokens / per) << "\n";
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
