//   OpenNN Transformer inference throughput benchmark ("Attention Is All You
//   Need" forward pass). Builds an encoder-decoder Transformer (the opennn::
//   Transformer standard network) and times the steady-state forward pass on the
//   GPU, after a warmup. Token inputs are generated on-host once; the forward
//   (calculate_outputs) is repeated and timed. Reports tokens/sec.
//
//   The forward path is CPU-vs-GPU validated by opennn_attention_validate.cpp.
//
//   usage: opennn_transformer_infer [seq] [d_model] [heads] [ff] [layers] [vocab] [batch] [iters]

#include <chrono>
#include <iostream>
#include <string>

#include "opennn/standard_networks.h"
#include "opennn/scaling_layer.h"
#include "opennn/configuration.h"
#include "opennn/random_utilities.h"

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
        Configuration::instance().set(Device::CUDA, Type::FP32);

        Transformer transformer(seq, seq, vocab, vocab, d_model, heads, ff, layers);

        cout << "config seq=" << seq << " d_model=" << d_model << " heads=" << heads
                  << " ff=" << ff << " layers=" << layers << " vocab=" << vocab
                  << " batch=" << batch << "\n";
        cout << "parameters=" << transformer.get_parameters_size() << "\n";

        Tensor3 inputs(batch, seq, 1);
        Tensor3 context(batch, seq, 1);
        for (Index b = 0; b < batch; ++b)
            for (Index s = 0; s < seq; ++s)
            {
                inputs(b, s, 0)  = float((b * seq + s) % vocab);
                context(b, s, 0) = float((b * seq + s + 1) % vocab);
            }

        transformer.calculate_outputs(inputs, context);

        const auto t0 = chrono::steady_clock::now();
        for (Index it = 0; it < iters; ++it)
            transformer.calculate_outputs(inputs, context);
        const auto t1 = chrono::steady_clock::now();

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
