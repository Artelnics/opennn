// Qwen3-4B benchmark driver.
//
// The core mode measures only model forward passes.  The runtime mode follows
// the public ChatSession path and timestamps its streaming callback.  Both
// modes emit one JSON document so the Python runner never has to infer a
// metric from human-readable output.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/tensor_types.h"
#include "opennn/models/models.h"
#include "opennn/neural_network/chat.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/neural_network/operators/tokenizer_operator.h"

using namespace opennn;
using namespace std;

namespace
{

using Clock = chrono::steady_clock;
constexpr Index PREFILL_BLOCK = 1024;
constexpr unsigned long long SEED = 42;
constexpr long long LOGICAL_PARAMETERS = 4022468096LL;

struct ModelConfig
{
    Index hidden = 0;
    Index layers = 0;
    Index query_heads = 0;
    Index key_value_heads = 0;
    Index head_dim = 0;
    Index intermediate = 0;
    Index vocabulary = 0;
    float rope_theta = 1000000.0f;
    float rms_epsilon = 1.0e-6f;
};

struct CoreSample
{
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    uint64_t prompt_hash = 0;
    uint64_t decode_hash = 0;
};

struct RuntimeSample
{
    double total_ms = 0.0;
    double ttft_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    Index prompt_tokens = 0;
    Index generated_tokens = 0;
    Index recovered_tokens = 0;
    string finish_reason;
    string output_text;
    uint64_t output_hash = 0;
};

double milliseconds(const Clock::time_point begin, const Clock::time_point end)
{
    return chrono::duration<double, milli>(end - begin).count();
}

void synchronize()
{
#ifdef OPENNN_HAS_CUDA
    device::synchronize(device::get_compute_stream());
#endif
}

uint64_t fnv1a(const void* data, const size_t bytes)
{
    const auto* input = static_cast<const unsigned char*>(data);
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i)
    {
        hash ^= input[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t token_hash(const vector<Index>& ids)
{
    return fnv1a(ids.data(), ids.size() * sizeof(Index));
}

string hex_hash(const uint64_t hash)
{
    ostringstream out;
    out << hex << setw(16) << setfill('0') << hash;
    return out.str();
}

string json_string(const string_view text)
{
    ostringstream out;
    out << '"';
    for (const unsigned char c : text)
    {
        switch (c)
        {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20)
                out << "\\u" << hex << setw(4) << setfill('0') << int(c) << dec;
            else
                out << char(c);
        }
    }
    out << '"';
    return out.str();
}

string read_text(const filesystem::path& path)
{
    ifstream input(path, ios::binary);
    if (!input) throw runtime_error("Cannot open " + path.string());
    return string(istreambuf_iterator<char>(input), istreambuf_iterator<char>());
}

ModelConfig read_config(const filesystem::path& directory)
{
    ModelConfig config;
    ifstream input(directory / "qwen3_meta.txt");
    if (!input) throw runtime_error("Cannot open qwen3_meta.txt in " + directory.string());
    input >> config.hidden >> config.layers >> config.query_heads
          >> config.key_value_heads >> config.head_dim >> config.intermediate
          >> config.vocabulary >> config.rope_theta >> config.rms_epsilon;
    if (!input) throw runtime_error("Malformed qwen3_meta.txt");
    return config;
}

Qwen3Tokenizer read_tokenizer(const filesystem::path& directory)
{
    Qwen3Tokenizer tokenizer;
    tokenizer.load(directory / "vocab.json", directory / "merges.txt",
                   directory / "qwen3_special.tsv");
    return tokenizer;
}

unique_ptr<Qwen3> load_model(const filesystem::path& directory,
                             const Index context,
                             double& load_ms)
{
    const ModelConfig config = read_config(directory);
    const auto begin = Clock::now();
    auto model = make_unique<Qwen3>(
        context, config.vocabulary, config.hidden, config.layers,
        config.query_heads, config.key_value_heads, config.head_dim,
        config.intermediate, config.rope_theta, config.rms_epsilon);
    model->load_parameters_bf16_inference_binary(directory / "qwen3_bf16.bin");
    synchronize();
    load_ms = milliseconds(begin, Clock::now());
    return model;
}

class LlamaBenchRandom
{
public:
    Index next(const Index vocabulary)
    {
        // llama-bench uses std::rand() without calling std::srand().  Both
        // executables are built with the same pinned MSVC CRT; spell out its
        // default seed=1 recurrence so the sequence remains auditable.  GGUF
        // IDs are zero based, while OpenNN reserves ID zero as its sentinel.
        state = state * 214013U + 2531011U;
        return 1 + Index((state >> 16U) & 0x7fffU) % vocabulary;
    }

    vector<Index> take(const Index count, const Index vocabulary)
    {
        vector<Index> ids(static_cast<size_t>(count));
        generate(ids.begin(), ids.end(), [&] { return next(vocabulary); });
        return ids;
    }

private:
    uint32_t state = 1U;
};

void run_prefill(Qwen3& model,
                 ForwardPropagation& propagation,
                 const vector<Index>& ids)
{
    vector<float> input(size_t(min(PREFILL_BLOCK, Index(ids.size()))));
    for (Index offset = 0; offset < Index(ids.size()); offset += PREFILL_BLOCK)
    {
        const Index count = min(PREFILL_BLOCK, Index(ids.size()) - offset);
        for (Index i = 0; i < count; ++i)
            input[size_t(i)] = float(ids[size_t(offset + i)]);
        propagation.past_length = offset;
        propagation.set_active_sequence_length(count);
        propagation.set_output_sequence_window(count - 1, 1);
        const vector<TensorView> inputs = {
            TensorView(input.data(), {1, count}, Type::FP32, Device::CPU)
        };
        model.forward_propagate(inputs, propagation,
                                ForwardPropagationMode::Inference);
    }
}

void stage_token(Buffer& destination, const Index id)
{
    const float value = float(id);
    device::copy_async(destination.data(), &value, Index(sizeof(float)),
                       device::CopyKind::HostToDevice,
                       device::get_compute_stream());
    device::synchronize(device::get_compute_stream());
}

CoreSample core_iteration(Qwen3& model,
                          ForwardPropagation& prefill,
                          ForwardPropagation& decode,
                          Buffer& token_device,
                          const vector<TensorView>& decode_inputs,
                          const vector<Index>& prompt,
                          const vector<Index>& generated)
{
    synchronize();
    const auto prefill_begin = Clock::now();
    run_prefill(model, prefill, prompt);
    synchronize();
    const auto prefill_end = Clock::now();

    // llama-bench's tg test clears its KV memory and starts at position zero.
    // The allocation still has the full per-cell context capacity.
    Index past = 0;
    const auto decode_begin = Clock::now();
    for (const Index id : generated)
    {
        stage_token(token_device, id);
        decode.past_length = past++;
        model.calculate_outputs_resident(decode_inputs, decode, false);
    }
    synchronize();
    const auto decode_end = Clock::now();
    return {milliseconds(prefill_begin, prefill_end),
            milliseconds(decode_begin, decode_end),
            token_hash(prompt), token_hash(generated)};
}

string finish_name(const FinishReason reason)
{
    switch (reason)
    {
    case FinishReason::Stop: return "stop";
    case FinishReason::MaximumTokens: return "maximum_tokens";
    case FinishReason::ContextLimit: return "context_limit";
    }
    return "unknown";
}

int tokens_mode(const filesystem::path& directory,
                const filesystem::path& content_path)
{
    Qwen3Tokenizer tokenizer = read_tokenizer(directory);
    Qwen3ChatTemplate chat_template;
    const vector<Index> ids = chat_template.render(
        {{ChatRole::User, read_text(content_path)}},
        ReasoningMode::Disabled, tokenizer);

    cout << "{\"schema_version\":1,\"mode\":\"tokens\",\"prompt_tokens\":"
         << ids.size() << ",\"token_hash\":\"" << hex_hash(token_hash(ids))
         << "\",\"token_ids\":[";
    for (size_t i = 0; i < ids.size(); ++i)
    {
        if (i) cout << ',';
        cout << ids[i];
    }
    cout << "]}\n";
    return 0;
}

int core_mode(const filesystem::path& directory, const Index prompt_tokens,
              const Index generated_tokens, const Index repeats,
              const Index context)
{
    double load_ms = 0.0;
    unique_ptr<Qwen3> model = load_model(directory, context, load_ms);
    const ModelConfig config = read_config(directory);
    LlamaBenchRandom random;

    // Match llama-bench's exact random-number consumption: a complete pp
    // warm-up, all timed pp repetitions, then a one-token tg warm-up (which
    // consumes the following, unused token) and all timed tg repetitions.
    const vector<Index> warm_prompt = random.take(prompt_tokens, config.vocabulary);
    vector<vector<Index>> prompts;
    prompts.reserve(size_t(repeats));
    for (Index i = 0; i < repeats; ++i)
        prompts.push_back(random.take(prompt_tokens, config.vocabulary));
    const vector<Index> warm_generated = random.take(1, config.vocabulary);
    random.next(config.vocabulary);
    vector<vector<Index>> generated;
    generated.reserve(size_t(repeats));
    for (Index i = 0; i < repeats; ++i)
    {
        generated.push_back(random.take(generated_tokens, config.vocabulary));
        random.next(config.vocabulary);
    }

    ForwardPropagation prefill(
        1, model.get(), ForwardPropagationMode::Inference,
        {.sequence_capacity = min(context, PREFILL_BLOCK),
         .final_output_capacity = 1,
         .retained_output_layers = {}});
    ForwardPropagation decode;
    decode.set(1, model.get(), &prefill.arena,
               ForwardPropagationMode::Inference,
               {.sequence_capacity = 1,
                .final_output_capacity = 1,
                .retained_output_layers = {}});
    decode.share_session_state_from(prefill);
    decode.set_active_sequence_length(1);
    decode.set_cuda_graph(true);

    Buffer token_device{Device::CUDA};
    token_device.resize_bytes(Index(sizeof(float)), Device::CUDA);
    const vector<TensorView> decode_inputs = {
        TensorView(token_device.data(), {1, 1}, Type::FP32, Device::CUDA)
    };

    // Capture and stabilize the graph outside every reported interval.
    core_iteration(*model, prefill, decode, token_device, decode_inputs,
                   warm_prompt, warm_generated);

    vector<CoreSample> samples;
    samples.reserve(size_t(repeats));
    const double timed_start = chrono::duration<double>(
        chrono::system_clock::now().time_since_epoch()).count();
    for (Index i = 0; i < repeats; ++i)
        samples.push_back(core_iteration(*model, prefill, decode, token_device,
                                         decode_inputs, prompts[size_t(i)],
                                         generated[size_t(i)]));
    const double timed_end = chrono::duration<double>(
        chrono::system_clock::now().time_since_epoch()).count();

    cout << setprecision(12)
         << "{\"schema_version\":1,\"engine\":\"opennn\",\"track\":\"core\""
         << ",\"precision\":\"bf16\",\"kv_precision\":\"bf16\""
         << ",\"cuda_graph\":" << (decode.inference_graph_exec ? "true" : "false")
         << ",\"prompt_tokens\":" << prompt_tokens
         << ",\"generated_tokens\":" << generated_tokens
         << ",\"context_tokens\":" << context
         << ",\"batch\":1,\"logical_parameters\":" << LOGICAL_PARAMETERS
         << ",\"serialized_elements\":" << model->get_parameters_buffer_size()
         << ",\"load_ms\":" << load_ms
         << ",\"synthetic_sequence\":\"llama-bench-msvc-rand-seed-1-plus-openNN-sentinel\""
         << ",\"timed_start_unix\":" << timed_start
         << ",\"timed_end_unix\":" << timed_end
         << ",\"samples\":[";
    for (size_t i = 0; i < samples.size(); ++i)
    {
        if (i) cout << ',';
        const CoreSample& sample = samples[i];
        cout << "{\"prefill_ms\":" << sample.prefill_ms
             << ",\"decode_ms\":" << sample.decode_ms
             << ",\"prefill_tokens_per_second\":"
             << (1000.0 * double(prompt_tokens) / sample.prefill_ms)
             << ",\"decode_tokens_per_second\":"
             << (1000.0 * double(generated_tokens) / sample.decode_ms)
             << ",\"prompt_token_hash\":\"" << hex_hash(sample.prompt_hash) << "\""
             << ",\"decode_token_hash\":\"" << hex_hash(sample.decode_hash) << "\"}";
    }
    cout << "]}\n";
    return 0;
}

int runtime_mode(const filesystem::path& directory,
                 const filesystem::path& content_path,
                 const Index generated_tokens, const Index repeats,
                 const Index context)
{
    Qwen3Tokenizer tokenizer = read_tokenizer(directory);
    const string content = read_text(content_path);
    double load_ms = 0.0;
    unique_ptr<Qwen3> model = load_model(directory, context, load_ms);

    const auto ready_begin = Clock::now();
    ChatSession session(*model, tokenizer, make_unique<Qwen3ChatTemplate>(), SEED);
    synchronize();
    const double ready_ms = milliseconds(ready_begin, Clock::now());

    SamplingConfig sampling;
    sampling.temperature = 0.0f;
    sampling.top_k = 0;
    sampling.top_p = 1.0f;
    sampling.repetition_penalty = 1.0f;
    sampling.maximum_tokens = generated_tokens;
    ChatOptions options;
    options.reasoning_mode = ReasoningMode::Disabled;
    options.sampling = sampling;

    // One unreported request stabilizes allocations and graph capture.
    session.send(content, options);
    session.clear();

    vector<RuntimeSample> samples;
    samples.reserve(size_t(repeats));
    const double timed_start = chrono::duration<double>(
        chrono::system_clock::now().time_since_epoch()).count();
    for (Index i = 0; i < repeats; ++i)
    {
        session.clear();
        RuntimeSample sample;
        bool first = true;
        Clock::time_point first_delta;
        Clock::time_point last_delta;
        const auto begin = Clock::now();
        const ChatResponse response = session.send(
            content, options,
            [&](const ChatDelta& delta)
            {
                if (delta.text.empty()) return;
                const auto now = Clock::now();
                if (first)
                {
                    first_delta = now;
                    first = false;
                }
                last_delta = now;
            });
        synchronize();
        const auto end = Clock::now();

        const vector<Index> recovered = tokenizer.encode(response.content);
        sample.total_ms = milliseconds(begin, end);
        sample.ttft_ms = first ? response.prefill_milliseconds
                               : milliseconds(begin, first_delta);
        sample.prefill_ms = response.prefill_milliseconds;
        sample.decode_ms = response.decode_milliseconds;
        sample.prompt_tokens = response.prompt_tokens;
        sample.generated_tokens = response.generated_tokens;
        sample.recovered_tokens = Index(recovered.size());
        sample.finish_reason = finish_name(response.finish_reason);
        sample.output_text = response.content;
        sample.output_hash = token_hash(recovered);
        samples.push_back(sample);
    }
    const double timed_end = chrono::duration<double>(
        chrono::system_clock::now().time_since_epoch()).count();

    cout << setprecision(12)
         << "{\"schema_version\":1,\"engine\":\"opennn\",\"track\":\"runtime\""
         << ",\"precision\":\"bf16\",\"kv_precision\":\"bf16\""
         << ",\"cuda_graph\":"
         << (session.get_decode_propagation().inference_graph_exec ? "true" : "false")
         << ",\"context_tokens\":" << context
         << ",\"requested_generated_tokens\":" << generated_tokens
         << ",\"batch\":1,\"logical_parameters\":" << LOGICAL_PARAMETERS
         << ",\"serialized_elements\":" << model->get_parameters_buffer_size()
         << ",\"model_load_ms\":" << load_ms
         << ",\"runtime_ready_ms\":" << ready_ms
         << ",\"timed_start_unix\":" << timed_start
         << ",\"timed_end_unix\":" << timed_end
         << ",\"samples\":[";
    for (size_t i = 0; i < samples.size(); ++i)
    {
        if (i) cout << ',';
        const RuntimeSample& sample = samples[i];
        const Index decode_steps = max(Index(0), sample.generated_tokens - 1);
        cout << "{\"total_ms\":" << sample.total_ms
             << ",\"ttft_ms\":" << sample.ttft_ms
             << ",\"prefill_ms\":" << sample.prefill_ms
             << ",\"decode_ms\":" << sample.decode_ms
             << ",\"prompt_tokens\":" << sample.prompt_tokens
             << ",\"generated_tokens\":" << sample.generated_tokens
             << ",\"recovered_tokens\":" << sample.recovered_tokens
             << ",\"finish_reason\":" << json_string(sample.finish_reason)
             << ",\"output_text\":" << json_string(sample.output_text)
             << ",\"output_token_hash\":\"" << hex_hash(sample.output_hash) << "\""
             << ",\"prefill_tokens_per_second\":"
             << (1000.0 * double(sample.prompt_tokens) / max(sample.prefill_ms, 1.0e-9))
             << ",\"decode_tokens_per_second\":"
             << (1000.0 * double(decode_steps) / max(sample.decode_ms, 1.0e-9))
             << ",\"end_to_end_tokens_per_second\":"
             << (1000.0 * double(sample.generated_tokens) / max(sample.total_ms, 1.0e-9))
             << '}';
    }
    cout << "]}\n";
    return 0;
}

Index number(const char* value, const char* label)
{
    try
    {
        const long long parsed = stoll(value);
        if (parsed <= 0) throw invalid_argument("non-positive");
        return Index(parsed);
    }
    catch (const exception&)
    {
        throw runtime_error(string(label) + " must be a positive integer");
    }
}

}

int main(int argc, char** argv)
{
    try
    {
        if (argc < 4)
            throw runtime_error(
                "usage: qwen_opennn tokens <data-dir> <content-file> | "
                "core <data-dir> <prompt> <generated> <repeats> <context> | "
                "runtime <data-dir> <content-file> <generated> <repeats> <context>");

        const string mode = argv[1];
        const filesystem::path directory = argv[2];
        if (mode == "tokens") return tokens_mode(directory, argv[3]);

#ifndef OPENNN_HAS_CUDA
        throw runtime_error("qwen_opennn core/runtime requires a CUDA build");
#else
        Configuration::instance().set(Device::CUDA, Type::BF16);
        if (mode == "core" && argc == 7)
            return core_mode(directory, number(argv[3], "prompt"),
                             number(argv[4], "generated"),
                             number(argv[5], "repeats"),
                             number(argv[6], "context"));
        if (mode == "runtime" && argc == 7)
            return runtime_mode(directory, argv[3],
                                number(argv[4], "generated"),
                                number(argv[5], "repeats"),
                                number(argv[6], "context"));
        throw runtime_error("invalid qwen_opennn arguments");
#endif
    }
    catch (const exception& error)
    {
        cerr << "qwen_opennn: " << error.what() << '\n';
        return 2;
    }
}
