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
//   BF16 weights streamed directly from a single .bin. No Python at runtime.
//
//   The weights (.bin) and the tokenizer files are downloaded from the Hugging Face
//   Hub on first run (cached in the data directory afterwards), so you can just run
//   the binary and chat. To use local files instead, drop them in the data directory
//   and the download is skipped.
//
//   usage: qwen3 [data_dir] [cpu|gpu] [--context N]
//                [--auto|--think|--no-think]
//     data_dir   where to cache the .bin + tokenizer files (default ../data)
//     cpu | gpu  compute device (default: gpu when built with CUDA, else cpu)

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
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

constexpr Index MODEL_MAX_CONTEXT = 32768;
constexpr Index DEFAULT_CONTEXT_LENGTH = MODEL_MAX_CONTEXT;
constexpr Index DEFAULT_MAX_NEW = 640;

const string MAIN_HF_BASE =
    "https://huggingface.co/Artelnics/qwen3-4b-opennn/resolve/main/";
const string DRAFT_HF_BASE =
    "https://huggingface.co/Artelnics/qwen3-0.6b-opennn/resolve/main/";
constexpr const char* MAIN_BF16_WEIGHTS = "qwen3_bf16.bin";
constexpr const char* DRAFT_BF16_WEIGHTS = "qwen3_draft_bf16.bin";
const char* const DATA_FILES[] = { "qwen3_meta.txt", "vocab.json", "merges.txt",
                                   "qwen3_special.tsv" };

string find_data_dir(const string& override_dir)
{
    if (!override_dir.empty()) return override_dir;
    if (const char* env = getenv("QWEN3_DATA")) return env;

    for (const string& candidate : { string("data"), string("../data"),
                                     string("../../examples/qwen3/data"),
                                     string("examples/qwen3/data") })
        if (ifstream(candidate + "/qwen3_meta.txt").good()) return candidate;

    return "../data";
}

struct Qwen3Config
{
    Index hidden = 0, layers = 0, query_heads = 0, key_value_heads = 0,
          head_dim = 0, intermediate = 0, vocabulary = 0;
    float rope_theta = 1000000.0f, rms_epsilon = 1.0e-6f;

    void load(const string& path)
    {
        ifstream file(path);
        if (!file) throw runtime_error("Cannot open config: " + path);
        file >> hidden >> layers >> query_heads >> key_value_heads >> head_dim
             >> intermediate >> vocabulary >> rope_theta >> rms_epsilon;
        if (!file) throw runtime_error("Malformed config: " + path);
    }
};

}

int main(int argc, char* argv[])
{
    try
    {

#ifdef OPENNN_HAS_CUDA
        bool want_gpu = true;
#else
        bool want_gpu = false;
#endif
        Index max_new = DEFAULT_MAX_NEW;
        Index context_length = DEFAULT_CONTEXT_LENGTH;
        bool want_int8 = false;
        bool use_draft = false;
        Index draft_tokens = 4;
        ReasoningMode reasoning_mode = ReasoningMode::Automatic;
        optional<float> temperature_override;
        optional<Index> top_k_override;
        optional<float> top_p_override;
        string data_arg;
        for (int i = 1; i < argc; ++i)
        {
            const string a = argv[i];
            if      (a == "--cpu" || a == "cpu")            want_gpu = false;
            else if (a == "--gpu" || a == "gpu")            want_gpu = true;
            else if (a == "--int8")                         want_int8 = true;
            else if (a == "--bf16")                         want_int8 = false;
            else if (a == "--auto")                         reasoning_mode = ReasoningMode::Automatic;
            else if (a == "--think")                        reasoning_mode = ReasoningMode::Enabled;
            else if (a == "--no-think")                     reasoning_mode = ReasoningMode::Disabled;
            else if (a == "--max"  && i + 1 < argc)         max_new = Index(stol(argv[++i]));
            else if (a == "--context" && i + 1 < argc)      context_length = Index(stol(argv[++i]));
            else if (a == "--temp" && i + 1 < argc)         temperature_override = stof(argv[++i]);
            else if (a == "--top-k" && i + 1 < argc)        top_k_override = Index(stol(argv[++i]));
            else if (a == "--top-p" && i + 1 < argc)        top_p_override = stof(argv[++i]);
            else if (a == "--data" && i + 1 < argc)         data_arg = argv[++i];
            else if (a == "--draft")                        use_draft = true;
            else if (a == "--draft-k" && i + 1 < argc)      { use_draft = true; draft_tokens = Index(stol(argv[++i])); }
            else if (a.rfind("--", 0) != 0)                 data_arg = a;
            else throw runtime_error("Unknown option: " + a);
        }
        throw_if(max_new <= 0, "--max must be greater than zero.");
        throw_if(context_length < 1 || context_length > MODEL_MAX_CONTEXT,
                 "--context must be between 1 and {} tokens.",
                 MODEL_MAX_CONTEXT);
        const string data_dir = find_data_dir(data_arg);
        throw_if(want_int8 && !want_gpu, "--int8 requires the GPU (CUDA).");

#ifdef OPENNN_HAS_CUDA
        Configuration::instance().set(want_gpu ? Device::CUDA : Device::CPU,
                                      !want_gpu ? Type::FP32
                                      : want_int8 ? Type::INT8 : Type::BF16);
#else
        Configuration::instance().set(Device::CPU, Type::FP32);
#endif

        cout << "OpenNN. Qwen3 chat." << endl;

        for (const char* file : DATA_FILES)
            download_if_missing(data_dir + "/" + file, MAIN_HF_BASE + file);
        download_if_missing(data_dir + "/" + MAIN_BF16_WEIGHTS,
                            MAIN_HF_BASE + MAIN_BF16_WEIGHTS);

        cout << "Loading..." << flush;

        Qwen3Tokenizer tokenizer;
        tokenizer.load(data_dir + "/vocab.json", data_dir + "/merges.txt", data_dir + "/qwen3_special.tsv");

        Qwen3Config config;
        config.load(data_dir + "/qwen3_meta.txt");
        Qwen3 model(context_length, config.vocabulary, config.hidden, config.layers,
                    config.query_heads, config.key_value_heads, config.head_dim, config.intermediate,
                    config.rope_theta, config.rms_epsilon);
        model.load_parameters_bf16_inference_binary(
            data_dir + "/" + MAIN_BF16_WEIGHTS);

        unique_ptr<Qwen3> draft_model;
        ChatSession session(
            model, tokenizer, make_unique<Qwen3ChatTemplate>());

        if (use_draft)
        {
            throw_if(!want_gpu, "--draft requires the GPU build.");
            download_if_missing(data_dir + "/qwen3_draft_meta.txt",
                                DRAFT_HF_BASE + "qwen3_meta.txt");
            Qwen3Config draft_config;
            draft_config.load(data_dir + "/qwen3_draft_meta.txt");

            download_if_missing(data_dir + "/" + DRAFT_BF16_WEIGHTS,
                                DRAFT_HF_BASE + "qwen3_bf16.bin");
            draft_model = make_unique<Qwen3>(
                context_length, draft_config.vocabulary, draft_config.hidden,
                draft_config.layers, draft_config.query_heads, draft_config.key_value_heads,
                draft_config.head_dim, draft_config.intermediate,
                draft_config.rope_theta, draft_config.rms_epsilon);
            draft_model->load_parameters_bf16_inference_binary(
                data_dir + "/" + DRAFT_BF16_WEIGHTS);
            session.attach_draft_model(*draft_model, draft_tokens);
            cout << "Draft: " << DRAFT_BF16_WEIGHTS << " (K="
                 << draft_tokens
                 << ", greedy only)" << endl;
        }

        SamplingConfig sampling = session.default_sampling(reasoning_mode);
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

        cout << "\rDevice: "
             << (!want_gpu ? "CPU (FP32)"
                 : want_int8 ? "GPU (CUDA, INT8)" : "GPU (CUDA, BF16)") << endl;
        cout << "Context: " << context_length
             << " tokens; prefill block: "
             << min(context_length, ChatSession::PREFILL_BLOCK_SIZE)
             << " tokens." << endl;

        session.chat(options);
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
