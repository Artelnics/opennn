//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G P T - 2   T E X T   G E N E R A T I O N   E X A M P L E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

//   Generates text with a pretrained GPT-2 small imported into OpenNN.
//   The architecture is a standard OpenNN TextGenerationNetwork configured as GPT-2;
//   only the weights (.bin) and the byte-pair tokenizer (vocab.json + merges.txt) are
//   downloaded (GitHub release assets)
//
//   usage: gpt2 [prompt] [max_new] [temperature] [top_k]
//     prompt      text to continue in one shot; omit it (or pass --interactive) for a REPL
//     max_new     number of tokens to generate (default 40)
//     temperature sampling temperature; <= 0 = greedy (default 0.8)
//     top_k       keep only the top-k most likely tokens when sampling (default 40)

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "opennn/io_utilities.h"
#include "opennn/standard_networks.h"
#include "opennn/tokenizer_operator.h"
#include "opennn/configuration.h"

using namespace opennn;
using namespace std;

namespace
{

const string BASE_URL =
    "https://github.com/Artelnics/opennn/releases/download/gpt2-weights-v1/";
const string WEIGHTS_URL = BASE_URL + "gpt2-small-seq256.bin";
const string VOCAB_URL   = BASE_URL + "vocab.json";
const string MERGES_URL  = BASE_URL + "merges.txt";

constexpr Index VOCABULARY_SIZE = 50258;      // 50257 + 1 ([PAD] = 0)
constexpr Index HIDDEN_SIZE     = 768;
constexpr Index HEADS_NUMBER    = 12;
constexpr Index INTERMEDIATE    = 3072;
constexpr Index LAYERS_NUMBER   = 12;
constexpr Index SEQUENCE_LENGTH = 256;

}


int main(int argc, char* argv[])
{
    try
    {
        cout << "OpenNN. GPT-2 text generation example." << endl;

        const string prompt         = argc > 1 ? argv[1] : "";   // no prompt => interactive REPL
        const Index  max_new_tokens = argc > 2 ? Index(stol(argv[2])) : 40;
        const float  temperature    = argc > 3 ? stof(argv[3]) : 0.8f;
        const Index  top_k          = argc > 4 ? Index(stol(argv[4])) : 40;

        const string weights_path = "../data/gpt2/gpt2-small-seq256.bin";
        const string vocab_path   = "../data/gpt2/vocab.json";
        const string merges_path  = "../data/gpt2/merges.txt";

        Configuration::instance().set(Device::CUDA, Type::FP32);   // weights .bin is FP32

        download_if_missing(weights_path, WEIGHTS_URL);
        download_if_missing(vocab_path, VOCAB_URL);
        download_if_missing(merges_path, MERGES_URL);

        auto tokenizer = make_unique<BytePairTokenizer>();
        tokenizer->load(vocab_path, merges_path);

        // Neural network: the GPT-2 small architecture

        TextGenerationNetwork model(SEQUENCE_LENGTH, VOCABULARY_SIZE, HIDDEN_SIZE,
                                    HEADS_NUMBER, INTERMEDIATE, LAYERS_NUMBER,
                                    /*pre_normalization*/ true, /*scale_embedding*/ false,
                                    /*learned_positional*/ true, /*feed_forward_activation*/ "GELUTanh");
        model.set_tokenizer(move(tokenizer));

        if (model.get_parameters_size() != Index(filesystem::file_size(weights_path) / sizeof(float)))
            throw runtime_error("Weights size mismatch: the .bin was exported for a different seq. "
                                "Use the seq=256 weights or adjust SEQUENCE_LENGTH.");

        cout << "Loading pretrained weights..." << endl;
        model.load_parameters_binary(weights_path);

        SamplingConfig sampling;
        sampling.maximum_tokens = max_new_tokens;
        sampling.temperature = temperature;
        sampling.top_k = top_k;

        // Interactive by default; a prompt argument switches to one-shot generation.
        const bool interactive = prompt.empty() || prompt == "--interactive" || prompt == "-i";

        if (interactive)
            model.chat(sampling);
        else
        {
            cout << model.generate(prompt, sampling) << endl;
            cout << "Good bye!" << endl;
        }
        return 0;
    }
    catch (const exception& e)
    {
        cout << e.what() << endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
