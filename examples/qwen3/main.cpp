//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Q W E N 3   C H A T   E X A M P L E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

//   A local conversational chatbot running a pretrained Qwen3 decoder-only model
//   imported into OpenNN. The architecture is a standard OpenNN Qwen3 network
//   (grouped-query attention with RoPE + QK-Norm, SwiGLU MLP, RMSNorm, tied output
//   projection); it uses the native Qwen3Tokenizer (byte-level BPE + ChatML) and the
//   weights from a single .bin. No Python at runtime.
//
//   The weights (.bin) and the tokenizer files are downloaded from the Hugging Face
//   Hub on first run (cached in the data directory afterwards), so you can just run
//   the binary and chat. To use local files instead, drop them in the data directory
//   and the download is skipped.
//
//   usage: qwen3 [data_dir] [cpu|gpu]
//     data_dir   where to cache the .bin + tokenizer files (default ../data)
//     cpu | gpu  compute device (default: gpu when built with CUDA, else cpu)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "../../opennn/standard_networks.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/forward_propagation.h"
#include "../../opennn/tensor_types.h"
#include "../../opennn/tokenizer_operator.h"
#include "../../opennn/configuration.h"
#ifdef OPENNN_HAS_CUDA
#include "../../opennn/device_backend.h"
#endif

using namespace opennn;
using namespace std;

namespace
{

// Context window: fixed at build time (the network is compiled for this
// sequence length); it bounds the KV cache and the kept conversation.
constexpr Index CONTEXT_LENGTH = 1024;

// Defaults (overridable on the command line). temperature <= 0 => greedy; the
// top-k / top-p values are Qwen3's recommended sampling settings.
constexpr Index DEFAULT_MAX_NEW    = 640;
constexpr float DEFAULT_TEMPERATURE = 0.7f;
constexpr Index SAMPLING_TOP_K     = 20;
constexpr float SAMPLING_TOP_P     = 0.8f;

// Weights (.bin) and tokenizer files, downloaded on first run and cached in the
// data directory. qwen3_meta.txt carries the config, so the network adapts to
// whatever repo this points at.
const string HF_BASE = "https://huggingface.co/Artelnics/qwen3-4b-opennn/resolve/main/";
const char* const DATA_FILES[] = { "qwen3_meta.txt", "vocab.json", "merges.txt",
                                   "qwen3_special.tsv", "qwen3.bin" };


void download_if_missing(const string& path, const string& url)
{
    if (std::filesystem::exists(path)) return;

    cout << "Downloading " << url << " ..." << endl;
    const string command = "curl -L --fail -o \"" + path + "\" \"" + url + "\"";
    if (system(command.c_str()) != 0 || !std::filesystem::exists(path))
        throw runtime_error("Download failed. Get it manually from:\n  " + url);
}


// An explicit argument or $QWEN3_DATA win; otherwise a few common relative
// paths are probed so the binary "just runs" from the build dir or repo root.
string find_data_dir(const string& override_dir)
{
    if (!override_dir.empty()) return override_dir;
    if (const char* env = getenv("QWEN3_DATA")) return env;

    for (const string& candidate : { string("data"), string("../data"),
                                     string("../../examples/qwen3/data"),
                                     string("examples/qwen3/data") })
        if (ifstream(candidate + "/qwen3.bin").good()) return candidate;

    return "../data";   // fall back; a clear error is raised later if it is missing
}


struct Qwen3Config
{
    Index hidden = 0, layers = 0, query_heads = 0, key_value_heads = 0,
          head_dim = 0, intermediate = 0, vocabulary = 0;
    float rope_theta = 1000000.0f, rms_epsilon = 1.0e-6f;

    // Reads the first 9 fields of qwen3_meta.txt (any trailing fields are ignored).
    void load(const string& path)
    {
        ifstream file(path);
        if (!file) throw runtime_error("Cannot open config: " + path);
        file >> hidden >> layers >> query_heads >> key_value_heads >> head_dim
             >> intermediate >> vocabulary >> rope_theta >> rms_epsilon;
        if (!file) throw runtime_error("Malformed config: " + path);
    }
};


// Sample the next token from a decoded logits row. Column 0 ([PAD]) is never
// chosen. temperature <= 0 gives greedy; otherwise temperature scaling, then top-k,
// then top-p (nucleus). Qwen3's recommended top-k / top-p are used as constants.
Index sample_token(const vector<float>& logits, float temperature, mt19937& rng)
{
    const Index vocabulary = Index(logits.size());

    if (temperature <= 0.0f)   // greedy
    {
        Index best = 1; float best_value = logits[1];
        for (Index i = 2; i < vocabulary; ++i) if (logits[size_t(i)] > best_value) { best_value = logits[size_t(i)]; best = i; }
        return best;
    }

    vector<pair<float, Index>> candidates;
    candidates.reserve(size_t(vocabulary - 1));
    for (Index i = 1; i < vocabulary; ++i) candidates.push_back({ logits[size_t(i)] / temperature, i });

    if (SAMPLING_TOP_K > 0 && SAMPLING_TOP_K < Index(candidates.size()))
    {
        nth_element(candidates.begin(), candidates.begin() + SAMPLING_TOP_K, candidates.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
        candidates.resize(size_t(SAMPLING_TOP_K));
    }

    float maximum = -1.0e30f;
    for (const auto& p : candidates) maximum = max(maximum, p.first);
    double sum = 0.0;
    for (auto& p : candidates) { p.first = float(exp(double(p.first - maximum))); sum += p.first; }
    for (auto& p : candidates) p.first = float(p.first / sum);

    if (SAMPLING_TOP_P > 0.0f && SAMPLING_TOP_P < 1.0f)
    {
        sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        double cumulative = 0.0; size_t keep = 0;
        for (size_t i = 0; i < candidates.size(); ++i) { cumulative += candidates[i].first; keep = i + 1; if (cumulative >= SAMPLING_TOP_P) break; }
        candidates.resize(keep);
    }

    vector<double> weights; weights.reserve(candidates.size());
    for (const auto& p : candidates) weights.push_back(double(p.first));
    discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    return candidates[distribution(rng)].second;
}


// Sample a token from the logits row at sequence position `pos` (output may be BF16).
Index sample_row(const ForwardPropagation& forward_propagation, Index pos, Index open_vocabulary,
                 float temperature, mt19937& rng)
{
    const TensorView output = forward_propagation.get_outputs();
    const Index elem = Index(type_bytes(output.type));
    const char* row = static_cast<const char*>(output.data) + size_t(pos) * open_vocabulary * elem;
    vector<char> host_row(size_t(open_vocabulary) * size_t(elem));

#ifdef OPENNN_HAS_CUDA
    if (output.is_cuda())
    {
        cudaStream_t stream = device::get_compute_stream();
        device::copy_async(host_row.data(), row, Index(host_row.size()), Device::CUDA, Device::CPU, stream);
        device::synchronize(stream);
    }
    else
#endif
        std::memcpy(host_row.data(), row, host_row.size());

    vector<float> logits(size_t(open_vocabulary), 0.0f);
    if (output.is_fp32())
        std::memcpy(logits.data(), host_row.data(), size_t(open_vocabulary) * sizeof(float));
    else   // BF16 -> FP32 (bf16 is the high 16 bits of the float)
    {
        const uint16_t* bf16 = reinterpret_cast<const uint16_t*>(host_row.data());
        for (Index i = 0; i < open_vocabulary; ++i)
        {
            const uint32_t bits = uint32_t(bf16[size_t(i)]) << 16;
            float value; std::memcpy(&value, &bits, sizeof(float));
            logits[size_t(i)] = value;
        }
    }

    return sample_token(logits, temperature, rng);
}


// Generate a reply with a KV cache: one prefill over the whole context, then a
// single-token decode pass per token. The <think> block is a transient generation
// prompt (not stored). The reply is streamed and returned for the caller to append.
vector<Index> generate_reply(NeuralNetwork& network, const Qwen3Tokenizer& tokenizer,
                             const vector<Index>& conversation, const vector<Index>& think_prompt,
                             Index open_vocabulary, Index max_new, float temperature, mt19937& rng,
                             ForwardPropagation& forward_propagation, vector<float>& window)
{
    const Index im_end = tokenizer.get_im_end_id();
    const Index eos    = tokenizer.get_endoftext_id();

    vector<Index> context = conversation;
    context.insert(context.end(), think_prompt.begin(), think_prompt.end());

    // Forward over `count` tokens in window[0..count-1] at positions past..past+count-1.
    const auto run = [&](Index count, Index past)
    {
        forward_propagation.past_length = past;
        forward_propagation.set_active_sequence_length(count);
        vector<TensorView> inputs = { TensorView(window.data(), {1, count}) };
        network.forward_propagate(inputs, forward_propagation, false);
    };

    using clock_type = chrono::steady_clock;

    // Prefill: the window keeps the last CONTEXT_LENGTH tokens (oldest dropped).
    const Index prompt = min(CONTEXT_LENGTH, Index(context.size()));
    const Index start  = Index(context.size()) - prompt;
    for (Index i = 0; i < prompt; ++i) window[size_t(i)] = float(context[size_t(start + i)]);
    const auto prefill_start = clock_type::now();
    run(prompt, 0);
    Index next      = sample_row(forward_propagation, prompt - 1, open_vocabulary, temperature, rng);
    const double prefill_ms = chrono::duration<double, milli>(clock_type::now() - prefill_start).count();
    Index cache_len = prompt;

    vector<Index> reply;
    string printed;
    const auto decode_start = clock_type::now();

    for (Index step = 0; step < max_new; ++step)
    {
        if (next == im_end || next == eos) break;
        reply.push_back(next);

        const string text = tokenizer.decode(reply);
        if (text.size() > printed.size()) { cout << text.substr(printed.size()) << flush; printed = text; }

        if (cache_len >= CONTEXT_LENGTH) break;   // window full

        window[0] = float(next);
        run(1, cache_len);
        ++cache_len;
        next = sample_row(forward_propagation, 0, open_vocabulary, temperature, rng);
    }

    cout << endl;

    const double decode_ms = chrono::duration<double, milli>(clock_type::now() - decode_start).count();
    const Index  decoded   = Index(reply.size());
    cerr << "[" << prompt << " prompt tok, prefill " << fixed << setprecision(0) << prefill_ms << " ms | "
         << decoded << " gen tok, " << setprecision(1)
         << (decoded > 1 && decode_ms > 0 ? (decoded - 1) * 1000.0 / decode_ms : 0.0) << " tok/s]" << endl;

    return reply;
}

}


int main(int argc, char* argv[])
{
    try
    {
        // Options (all optional, so a plain "qwen3" just works):
        //   --temp T   sampling temperature (0 = greedy)     default 0.7
        //   --max  N   maximum tokens per reply              default 640
        //   --cpu | --gpu   compute device                   default gpu (if CUDA)
        //   --data DIR (or a bare path)   data directory
#ifdef OPENNN_HAS_CUDA
        bool want_gpu = true;
#else
        bool want_gpu = false;
#endif
        Index max_new     = DEFAULT_MAX_NEW;
        float temperature = DEFAULT_TEMPERATURE;
        string data_arg;
        for (int i = 1; i < argc; ++i)
        {
            const string a = argv[i];
            if      (a == "--cpu" || a == "cpu")            want_gpu = false;
            else if (a == "--gpu" || a == "gpu")            want_gpu = true;
            else if (a == "--max"  && i + 1 < argc)         max_new = Index(stol(argv[++i]));
            else if (a == "--temp" && i + 1 < argc)         temperature = stof(argv[++i]);
            else if (a == "--data" && i + 1 < argc)         data_arg = argv[++i];
            else if (a.rfind("--", 0) != 0)                 data_arg = a;   // bare argument = data dir
        }
        const string data_dir = find_data_dir(data_arg);

#ifdef OPENNN_HAS_CUDA
        // BF16 on GPU: tensor cores + half the weight VRAM (the .bin stays FP32; the
        // network builds a bf16 mirror on the device).
        Configuration::instance().set(want_gpu ? Device::CUDA : Device::CPU,
                                      want_gpu ? Type::BF16 : Type::FP32);
#else
        Configuration::instance().set(Device::CPU, Type::FP32);
#endif

        cout << "OpenNN. Qwen3 chat." << endl;

        std::filesystem::create_directories(data_dir);
        for (const char* file : DATA_FILES)
            download_if_missing(data_dir + "/" + file, HF_BASE + file);

        cout << "Loading..." << flush;

        // Tokenizer: native byte-level BPE with the Qwen pre-tokenizer and ChatML.
        Qwen3Tokenizer tokenizer;
        tokenizer.load(data_dir + "/vocab.json", data_dir + "/merges.txt", data_dir + "/qwen3_special.tsv");

        // Neural network: the Qwen3 architecture, weights from the .bin.
        Qwen3Config config;
        config.load(data_dir + "/qwen3_meta.txt");
        Qwen3 model(CONTEXT_LENGTH, config.vocabulary, config.hidden, config.layers,
                    config.query_heads, config.key_value_heads, config.head_dim, config.intermediate,
                    config.rope_theta, config.rms_epsilon);
        model.load_parameters_binary(data_dir + "/qwen3.bin");

#ifdef OPENNN_HAS_CUDA
        // Build the device bf16 mirror tensor-by-tensor: the full fp32 master
        // never hits the device.
        if (want_gpu) model.upload_parameters_bf16_inference();
#endif

        cout << "\rDevice: " << (want_gpu ? "GPU (CUDA, BF16)" : "CPU (FP32)")
             << "  |  temperature " << temperature << "  max " << max_new << " tokens.\n"
             << "Type a message; empty line, 'exit' or 'quit' to leave.\n" << endl;

        // The history grows every turn; the sliding window keeps the last
        // CONTEXT_LENGTH tokens. ForwardPropagation and the window are reused.
        vector<Index> conversation;
        const vector<Index> think_prompt = tokenizer.encode("<think>\n\n</think>\n\n");
        mt19937 rng(random_device{}());
        ForwardPropagation forward_propagation(1, &model);
        vector<float> window(size_t(CONTEXT_LENGTH), 0.0f);

        string line;
        while (true)
        {
            cout << "You:  " << flush;
            if (!getline(cin, line)) break;
            if (line == "exit" || line == "quit") break;
            if (line.empty()) continue;

            // Append the user turn in ChatML. The <think> generation prompt is added
            // transiently inside generate_reply, so it is not kept in the history.
            const string user_turn = "<|im_start|>user\n" + line + "<|im_end|>\n<|im_start|>assistant\n";
            const vector<Index> user_ids = tokenizer.encode(user_turn);
            conversation.insert(conversation.end(), user_ids.begin(), user_ids.end());

            cout << "Qwen: " << flush;
            const vector<Index> reply = generate_reply(model, tokenizer, conversation, think_prompt,
                                                       config.vocabulary + 1, max_new, temperature, rng,
                                                       forward_propagation, window);

            // Store the reply and close the assistant turn (clean history: no <think>).
            conversation.insert(conversation.end(), reply.begin(), reply.end());
            const vector<Index> end_ids = tokenizer.encode("<|im_end|>\n");
            conversation.insert(conversation.end(), end_ids.begin(), end_ids.end());
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
