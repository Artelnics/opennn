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

#include "opennn/chat.h"
#include "opennn/io_utilities.h"
#include "opennn/standard_networks.h"
#include "opennn/tokenizer_operator.h"
#include "opennn/configuration.h"

using namespace opennn;
using namespace std;

int main(int argc, char* argv[])
{
    try
    {
        constexpr string_view base_url =
            "https://github.com/Artelnics/opennn/releases/download/gpt2-weights-v1/";
        const vector<string_view> data_files = {
            "gpt2-small-seq256.bin", "vocab.json", "merges.txt"
        };

        constexpr Index vocabulary_size = 50258; // 50257 + 1 ([PAD] = 0)
        constexpr Index hidden_size = 768;
        constexpr Index heads_number = 12;
        constexpr Index intermediate = 3072;
        constexpr Index layers_number = 12;
        constexpr Index sequence_length = 256;

        cout << "OpenNN. GPT-2 text generation example." << endl;

        const string prompt         = argc > 1 ? argv[1] : "";   // no prompt => interactive REPL
        const Index  max_new_tokens = argc > 2 ? Index(stol(argv[2])) : 40;
        const float  temperature    = argc > 3 ? stof(argv[3]) : 0.8f;
        const Index  top_k          = argc > 4 ? Index(stol(argv[4])) : 40;

        const filesystem::path data_directory = "../data/gpt2";
        const filesystem::path weights_path = data_directory / "gpt2-small-seq256.bin";

        Configuration::instance().set(Device::CUDA, Type::FP32);   // weights .bin is FP32

        download_files_if_missing(data_directory, base_url, data_files);

        auto tokenizer = make_unique<BytePairTokenizer>(
            data_directory / "vocab.json", data_directory / "merges.txt");

        // Neural network: the GPT-2 small architecture

        TextGenerationNetwork model(sequence_length, vocabulary_size, hidden_size,
                                    heads_number, intermediate, layers_number,
                                    /*pre_normalization*/ true, /*scale_embedding*/ false,
                                    /*learned_positional*/ true, /*feed_forward_activation*/ "GELUTanh");
        model.set_tokenizer(move(tokenizer));

        cout << "Loading pretrained weights..." << endl;
        model.load_parameters_binary(weights_path);

        SamplingConfig sampling;
        sampling.maximum_tokens = max_new_tokens;
        sampling.temperature = temperature;
        sampling.top_k = top_k;

        ChatSession session(model);
        ChatOptions options;
        options.sampling = sampling;

        // Interactive by default; a prompt argument switches to one-shot generation.
        const bool interactive = prompt.empty() || prompt == "--interactive" || prompt == "-i";

        if (interactive)
            session.chat(options);
        else
        {
            cout << session.send(prompt, options).content << endl;
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
