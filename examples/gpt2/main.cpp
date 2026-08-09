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
//   usage: gpt2 [--int8] [prompt] [max_new] [temperature] [top_k]
//     --int8      weight-only INT8 inference (CUDA); weights are quantized after loading
//     prompt      text to continue in one shot; omit it (or pass --interactive) for a REPL
//     max_new     number of tokens to generate (default 40)
//     temperature sampling temperature; <= 0 = greedy (default 0.8)
//     top_k       keep only the top-k most likely tokens when sampling (default 40)

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "opennn/neural_network/chat.h"
#include "opennn/core/io_utilities.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/neural_network/operators/tokenizer_operator.h"
#include "opennn/core/configuration.h"

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

        constexpr Index vocabulary_size = 50258;
        constexpr Index hidden_size = 768;
        constexpr Index heads_number = 12;
        constexpr Index intermediate = 3072;
        constexpr Index layers_number = 12;
        constexpr Index sequence_length = 256;

        cout << "OpenNN. GPT-2 text generation example." << endl;

        vector<string> args;
        bool want_int8 = false;
        for (int i = 1; i < argc; ++i)
        {
            const string argument = argv[i];
            if (argument == "--int8") want_int8 = true;
            else args.push_back(argument);
        }

        const string prompt         = args.size() > 0 ? args[0] : "";
        const Index  max_new_tokens = args.size() > 1 ? Index(stol(args[1])) : 40;
        const float  temperature    = args.size() > 2 ? stof(args[2]) : 0.8f;
        const Index  top_k          = args.size() > 3 ? Index(stol(args[3])) : 40;

        const filesystem::path data_directory = "../data/gpt2";
        const filesystem::path weights_path = data_directory / "gpt2-small-seq256.bin";

        Configuration::instance().set(Device::CUDA, want_int8 ? Type::INT8 : Type::FP32);

        download_files_if_missing(data_directory, base_url, data_files);

        auto tokenizer = make_unique<BytePairTokenizer>(
            data_directory / "vocab.json", data_directory / "merges.txt");

        TextGenerationNetwork model(sequence_length, vocabulary_size, hidden_size,
                                    heads_number, intermediate, layers_number,
 true,  false,
 true,  "GELUTanh");
        model.set_tokenizer(move(tokenizer));

        cout << "Loading pretrained weights..." << endl;
        model.load_parameters_binary(weights_path);
        if (want_int8) model.upload_parameters_int8_inference();

        SamplingConfig sampling;
        sampling.maximum_tokens = max_new_tokens;
        sampling.temperature = temperature;
        sampling.top_k = top_k;

        ChatSession session(model);
        ChatOptions options;
        options.sampling = sampling;

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
