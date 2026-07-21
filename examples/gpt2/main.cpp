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
//   downloaded (GitHub release assets) and loaded with load_parameters_binary().
//
//   usage: gpt2 [prompt] [max_new] [temperature] [top_k]
//     prompt      text to continue in one shot; omit it (or pass --interactive) for a REPL
//     max_new     number of tokens to generate (default 40)
//     temperature sampling temperature; <= 0 = greedy (default 0.8)
//     top_k       keep only the top-k most likely tokens when sampling (default 40)

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "../../opennn/standard_networks.h"
#include "../../opennn/forward_propagation.h"
#include "../../opennn/tensor_types.h"
#include "../../opennn/tokenizer_operator.h"
#include "../../opennn/configuration.h"

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
constexpr Index SEQUENCE_LENGTH = 256;        // must match the downloaded weights

void download_if_missing(const string& path, const string& url)
{
    if (filesystem::exists(path)) return;

    cout << "Downloading " << url << " -> " << path << " ..." << endl;
    const string command = "curl -L --fail -o \"" + path + "\" \"" + url + "\"";
    if (system(command.c_str()) != 0 || !filesystem::exists(path))
        throw runtime_error("Download failed. Get it manually from:\n  " + url);
}

Index sample_next_token(const float* probabilities, Index vocabulary_size,
                        float temperature, Index top_k, mt19937& rng)
{
    vector<Index> ids(size_t(vocabulary_size - 1));
    iota(ids.begin(), ids.end(), Index(1));   // id 0 = [PAD], never generated

    if (top_k > 0 && top_k < Index(ids.size()))
    {
        ranges::nth_element(ids, ids.begin() + top_k,
                            [&](Index a, Index b) { return probabilities[a] > probabilities[b]; });
        ids.resize(size_t(top_k));
    }

    if (temperature <= 0.0f)
        return *ranges::max_element(ids, [&](Index a, Index b) {
            return probabilities[a] < probabilities[b];
        });

    vector<double> weights(ids.size());
    double total = 0.0;

    const double inv_temperature = 1.0 / double(temperature);
    for (size_t i = 0; i < ids.size(); ++i)
    {
        const double p = max(double(probabilities[ids[i]]), 1.0e-30);
        weights[i] = pow(p, inv_temperature);
        total += weights[i];
    }

    if (total <= 0.0 || !isfinite(total))
        return ids.front();

    discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    return ids[distribution(rng)];
}

vector<float> make_input_window(const vector<Index>& context, Index sequence_length)
{
    vector<float> input(size_t(sequence_length), 0.0f);

    const Index available = Index(context.size());
    const Index used = min(sequence_length, available);
    const Index start = available - used;

    for (Index i = 0; i < used; ++i)
        input[size_t(i)] = float(context[size_t(start + i)]);

    return input;
}

string generate_text(TextGenerationNetwork& network, BytePairTokenizer& tokenizer,
                     const string& prompt, Index max_new_tokens, Index sequence_length,
                     float temperature, Index top_k, mt19937& rng)
{
    vector<Index> context = tokenizer.encode(prompt);
    if (context.empty())
        context.push_back(VOCABULARY_SIZE - 1);

    ForwardPropagation forward(1, &network);

    for (Index step = 0; step < max_new_tokens; ++step)
    {
        vector<float> input = make_input_window(context, sequence_length);
        const Index used = min(sequence_length, Index(context.size()));
        const Index position = max(Index(0), used - 1);

        vector<TensorView> inputs = {TensorView(input.data(), {1, sequence_length})};
        network.forward_propagate(inputs, forward, false);

        const TensorView output = forward.get_outputs();
        const float* probabilities = output.as<float>() + size_t(position) * VOCABULARY_SIZE;

        context.push_back(sample_next_token(probabilities, VOCABULARY_SIZE, temperature, top_k, rng));
    }

    return tokenizer.decode(context);
}
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

        Configuration::instance().set(Device::CPU, Type::FP32);   // weights .bin is FP32

        filesystem::create_directories("../data/gpt2");
        download_if_missing(weights_path, WEIGHTS_URL);
        download_if_missing(vocab_path, VOCAB_URL);
        download_if_missing(merges_path, MERGES_URL);

        // Tokenizer: byte-pair encoding. vocab.json = token<->id table,
        // merges.txt = ordered BPE merge rules (both are needed to split text).

        BytePairTokenizer tokenizer;
        tokenizer.load(vocab_path, merges_path);

        // Neural network: the GPT-2 small architecture, weights from the .bin.

        TextGenerationNetwork model(SEQUENCE_LENGTH, VOCABULARY_SIZE, HIDDEN_SIZE,
                                    HEADS_NUMBER, INTERMEDIATE, LAYERS_NUMBER,
                                    /*pre_normalization*/ true, /*scale_embedding*/ false,
                                    /*learned_positional*/ true, /*feed_forward_activation*/ "GELUTanh");

        if (model.get_parameters_size() != Index(filesystem::file_size(weights_path) / sizeof(float)))
            throw runtime_error("Weights size mismatch: the .bin was exported for a different seq. "
                                "Use the seq=256 weights or adjust SEQUENCE_LENGTH.");

        cout << "Loading pretrained weights..." << endl;
        model.load_parameters_binary(weights_path);

        mt19937 rng(random_device{}());

        // Interactive by default; a prompt argument switches to one-shot generation.
        const bool interactive = prompt.empty() || prompt == "--interactive" || prompt == "-i";

        if (interactive)
        {
            cout << "Interactive mode. Type a prompt; empty line, 'exit' or 'quit' finishes." << endl;
            string line;
            while (true)
            {
                cout << "\n> " << flush;
                if (!getline(cin, line)) break;
                if (line.empty() || line == "exit" || line == "quit") break;

                cout << generate_text(model, tokenizer, line, max_new_tokens,
                                      SEQUENCE_LENGTH, temperature, top_k, rng) << endl;
            }
        }
        else
        {
            cout << generate_text(model, tokenizer, prompt, max_new_tokens,
                                  SEQUENCE_LENGTH, temperature, top_k, rng) << endl;
        }

        cout << "Good bye!" << endl;
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
