//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G P T - 2   T E X T   G E N E R A T I O N   E X A M P L E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

//   Downloads GPT-2 small and generates a continuation for a prompt. Requires CUDA.

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "opennn/neural_network/chat.h"
#include "opennn/core/io_utilities.h"
#include "opennn/models/models.h"
#include "opennn/neural_network/operators/tokenizer_operator.h"
#include "opennn/core/configuration.h"

using namespace opennn;

int main(int argc, char* argv[])
{
    try
    {
        constexpr string_view base_url =
            "https://github.com/Artelnics/opennn/releases/download/gpt2-weights-v1/";
        const vector<string_view> data_files = {
            "gpt2-small-seq256.bin", "vocab.json", "merges.txt"
        };

        cout << "OpenNN. GPT-2 text generation example." << endl;

        const filesystem::path data_directory = "../data/gpt2";
        const filesystem::path weights_path = data_directory / "gpt2-small-seq256.bin";

        Configuration::instance().set(Device::CUDA, Type::FP32);

        TextGenerationNetwork model(256, 50258, 768, 12, 3072, 12,
                                    true, false, true, "GELUTanh");

        download_files_if_missing(data_directory, base_url, data_files);

        model.set_tokenizer(make_unique<BytePairTokenizer>(
            data_directory / "vocab.json", data_directory / "merges.txt"));

        model.load_parameters_binary(weights_path);

        ChatOptions options;
        options.sampling = SamplingConfig{
            .temperature = 0.8f, .top_k = 40, .maximum_tokens = 40};

        const string prompt = argc > 1 ? argv[1] : "Artificial intelligence";
        ChatSession session(model);

        cout << session.send(prompt, options).content << endl;

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
