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
//   usage: qwen3 [data_dir] [cpu|gpu] [--auto|--think|--no-think]
//     data_dir   where to cache the .bin + tokenizer files (default ../data)
//     cpu | gpu  compute device (default: gpu when built with CUDA, else cpu)

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "opennn/chat.h"
#include "opennn/io_utilities.h"
#include "opennn/standard_networks.h"
#include "opennn/neural_network.h"
#include "opennn/tokenizer_operator.h"
#include "opennn/configuration.h"

using namespace opennn;
using namespace std;

namespace
{


constexpr Index CONTEXT_LENGTH = 1024;
constexpr Index DEFAULT_MAX_NEW = 640;

const string HF_BASE = "https://huggingface.co/Artelnics/qwen3-4b-opennn/resolve/main/";
const char* const DATA_FILES[] = { "qwen3_meta.txt", "vocab.json", "merges.txt",
                                   "qwen3_special.tsv", "qwen3.bin" };

string find_data_dir(const string& override_dir)
{
    if (!override_dir.empty()) return override_dir;
    if (const char* env = getenv("QWEN3_DATA")) return env;

    for (const string& candidate : { string("data"), string("../data"),
                                     string("../../examples/qwen3/data"),
                                     string("examples/qwen3/data") })
        if (ifstream(candidate + "/qwen3.bin").good()) return candidate;

    return "../data";
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

string reasoning_mode_name(const ReasoningMode mode)
{
    switch (mode)
    {
    case ReasoningMode::Automatic: return "auto";
    case ReasoningMode::Enabled:   return "thinking";
    case ReasoningMode::Disabled:  return "non-thinking";
    }
    return "unknown";
}

}


int main(int argc, char* argv[])
{
    try
    {
        // Options (all optional, so a plain "qwen3" just works):
        //   --auto | --think | --no-think   reasoning mode   default auto
        //   --show-thinking | --hide-thinking                default show
        //   --temp T | --top-k K | --top-p P                 model defaults
        //   --max N   maximum total reasoning + answer tokens default 640
        //   --cpu | --gpu   compute device                   default gpu (if CUDA)
        //   --data DIR (or a bare path)   data directory
#ifdef OPENNN_HAS_CUDA
        bool want_gpu = true;
#else
        bool want_gpu = false;
#endif
        Index max_new = DEFAULT_MAX_NEW;
        bool use_draft = false;
        Index draft_tokens = 4;
        ReasoningMode reasoning_mode = ReasoningMode::Automatic;
        bool show_thinking = true;
        optional<float> temperature_override;
        optional<Index> top_k_override;
        optional<float> top_p_override;
        string data_arg;
        for (int i = 1; i < argc; ++i)
        {
            const string a = argv[i];
            if      (a == "--cpu" || a == "cpu")            want_gpu = false;
            else if (a == "--gpu" || a == "gpu")            want_gpu = true;
            else if (a == "--auto")                          reasoning_mode = ReasoningMode::Automatic;
            else if (a == "--think")                         reasoning_mode = ReasoningMode::Enabled;
            else if (a == "--no-think")                      reasoning_mode = ReasoningMode::Disabled;
            else if (a == "--show-thinking")                 show_thinking = true;
            else if (a == "--hide-thinking")                 show_thinking = false;
            else if (a == "--max"  && i + 1 < argc)         max_new = Index(stol(argv[++i]));
            else if (a == "--temp" && i + 1 < argc)         temperature_override = stof(argv[++i]);
            else if (a == "--top-k" && i + 1 < argc)        top_k_override = Index(stol(argv[++i]));
            else if (a == "--top-p" && i + 1 < argc)        top_p_override = stof(argv[++i]);
            else if (a == "--data" && i + 1 < argc)         data_arg = argv[++i];
            else if (a == "--draft")                        use_draft = true;
            else if (a == "--draft-k" && i + 1 < argc)      { use_draft = true; draft_tokens = Index(stol(argv[++i])); }
            else if (a.rfind("--", 0) != 0)                 data_arg = a;   // bare argument = data dir
            else throw runtime_error("Unknown option: " + a);
        }
        throw_if(max_new <= 0, "--max must be greater than zero.");
        const string data_dir = find_data_dir(data_arg);

#ifdef OPENNN_HAS_CUDA
        Configuration::instance().set(want_gpu ? Device::CUDA : Device::CPU,
                                      want_gpu ? Type::BF16 : Type::FP32);
#else
        Configuration::instance().set(Device::CPU, Type::FP32);
#endif

        cout << "OpenNN. Qwen3 chat." << endl;

        for (const char* file : DATA_FILES)
            download_if_missing(data_dir + "/" + file, HF_BASE + file);

        cout << "Loading..." << flush;

        Qwen3Tokenizer tokenizer;
        tokenizer.load(data_dir + "/vocab.json", data_dir + "/merges.txt", data_dir + "/qwen3_special.tsv");

        Qwen3Config config;
        config.load(data_dir + "/qwen3_meta.txt");
        Qwen3 model(CONTEXT_LENGTH, config.vocabulary, config.hidden, config.layers,
                    config.query_heads, config.key_value_heads, config.head_dim, config.intermediate,
                    config.rope_theta, config.rms_epsilon);
        model.load_parameters_binary(data_dir + "/qwen3.bin");

#ifdef OPENNN_HAS_CUDA
        if (want_gpu) model.upload_parameters_bf16_inference();
#endif

        unique_ptr<Qwen3> draft_model;
        ChatSession session(
            model, tokenizer, make_unique<Qwen3ChatTemplate>());

        if (use_draft)
        {
            throw_if(!want_gpu, "--draft requires the GPU build.");
            for (const char* file : {"qwen3_draft_meta.txt", "qwen3_draft.bin"})
                download_if_missing(data_dir + "/" + file, HF_BASE + file);
            Qwen3Config draft_config;
            draft_config.load(data_dir + "/qwen3_draft_meta.txt");
            draft_model = make_unique<Qwen3>(
                CONTEXT_LENGTH, draft_config.vocabulary, draft_config.hidden,
                draft_config.layers, draft_config.query_heads, draft_config.key_value_heads,
                draft_config.head_dim, draft_config.intermediate,
                draft_config.rope_theta, draft_config.rms_epsilon);
            draft_model->load_parameters_binary(data_dir + "/qwen3_draft.bin");
            draft_model->upload_parameters_bf16_inference();
            session.attach_draft_model(*draft_model, draft_tokens);
            cout << "Draft: qwen3_draft.bin (K=" << draft_tokens
                 << ", greedy only)" << endl;
        }

        const auto make_chat_options = [&]()
        {
            SamplingConfig sampling =
                session.default_sampling(reasoning_mode);
            if (temperature_override)
                sampling.temperature = *temperature_override;
            if (top_k_override)
                sampling.top_k = *top_k_override;
            if (top_p_override)
                sampling.top_p = *top_p_override;
            sampling.maximum_tokens = max_new;

            ChatOptions options;
            options.reasoning_mode = reasoning_mode;
            options.sampling = sampling;
            return options;
        };

        const auto print_settings = [&]()
        {
            const ChatOptions options = make_chat_options();
            const SamplingConfig& sampling = *options.sampling;
            cout << "Mode: " << reasoning_mode_name(reasoning_mode)
                 << " (resolved "
                 << reasoning_mode_name(
                        session.resolve_reasoning_mode(reasoning_mode))
                 << "), temperature " << sampling.temperature
                 << ", top-k " << sampling.top_k
                 << ", top-p " << sampling.top_p
                 << ", max " << sampling.maximum_tokens << " tokens."
                 << endl;
        };

        cout << "\rDevice: "
             << (want_gpu ? "GPU (CUDA, BF16)" : "CPU (FP32)") << endl;
        print_settings();

        cout << "Commands: :auto, :think, :no-think, :clear. "
                "Empty line, 'exit' or 'quit' leaves.\n"
             << endl;

        string line;
        while (true)
        {
            cout << "You:  " << flush;
            if (!getline(cin, line)) break;
            if (line == "exit" || line == "quit") break;
            if (line.empty()) continue;

            if (line == ":auto" || line == ":think"
                || line == ":no-think")
            {
                reasoning_mode = line == ":auto"
                    ? ReasoningMode::Automatic
                    : line == ":think"
                        ? ReasoningMode::Enabled
                        : ReasoningMode::Disabled;
                print_settings();
                continue;
            }
            if (line == ":clear")
            {
                session.clear();
                cout << "Conversation cleared." << endl;
                continue;
            }

            bool reasoning_started = false;
            bool content_started = false;
            const ChatCallback stream = [&](const ChatDelta& delta)
            {
                if (delta.channel == GenerationChannel::Reasoning)
                {
                    if (!show_thinking) return;
                    if (!reasoning_started)
                    {
                        cout << "Thinking: " << flush;
                        reasoning_started = true;
                    }
                    cout << delta.text << flush;
                    return;
                }

                if (!content_started)
                {
                    if (reasoning_started) cout << "\n";
                    cout << "Qwen: " << flush;
                    content_started = true;
                }
                cout << delta.text << flush;
            };

            const ChatResponse response =
                session.send(line, make_chat_options(), stream);
            if (!content_started)
            {
                if (reasoning_started) cout << "\n";
                cout << "Qwen: " << response.content;
            }
            cout << endl;

            const double tokens_per_second =
                response.generated_tokens > 1
                    && response.decode_milliseconds > 0.0
                ? (response.generated_tokens - 1) * 1000.0
                    / response.decode_milliseconds
                : 0.0;
            cerr << "[" << response.prefill_tokens << "/"
                 << response.prompt_tokens << " prompt tok, prefill "
                 << fixed << setprecision(0)
                 << response.prefill_milliseconds << " ms | "
                 << response.generated_tokens << " generated ("
                 << response.reasoning_tokens << " reasoning, "
                 << response.content_tokens << " content), "
                 << setprecision(1) << tokens_per_second << " tok/s]"
                 << endl;
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
