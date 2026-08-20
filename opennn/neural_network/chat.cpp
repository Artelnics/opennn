//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C H A T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/chat.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <utility>

#include "opennn/core/device_backend.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/core/parallel_algorithms.h"
#include "opennn/core/random_utilities.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/core/statistics.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/operators/tokenizer_operator.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/core/cuda/kernel_attention.cuh"
#endif

namespace opennn
{

namespace
{

#ifdef OPENNN_HAS_CUDA

// Draws one token from a logits row entirely on the device, so the sampled id
// never has to round-trip through the host. Only the CUDA path uses it; the
// callers below are guarded, so there is no host fallback to keep in sync.
void sample_logits_row(const TensorView& logits_row, float temperature, Index top_k, float top_p,
                       unsigned long long seed, unsigned long long step,
                       void* candidates_scratch, int* id_device, float* token_device)
{
    throw_if(!logits_row.is_cuda() || !candidates_scratch || !id_device,
             "sample_logits_row: a GPU logits row, device scratch and a device id are required.");

    logits_row.dispatch([&]<typename T>() {
        sample_logits_row_cuda<T>(to_int(logits_row.size()), temperature, to_int(top_k), top_p,
                                  seed, step, logits_row.as<T>(),
                                  static_cast<float2*>(candidates_scratch), id_device, token_device);
    });
}

Index sample_logits_scratch_floats()
{
    return Index(LOGITS_SAMPLE_BLOCKS) * 32 * 2;
}

#endif

bool is_prefix(const vector<Index>& candidate, const vector<Index>& sequence)
{
    return candidate.size() <= sequence.size()
        && equal(candidate.begin(), candidate.end(), sequence.begin());
}

size_t complete_utf8_prefix_size(string_view text)
{
    size_t i = 0;

    while (i < text.size())
    {
        size_t length =
            utf8_sequence_length(static_cast<unsigned char>(text[i]));

        if (i + length > text.size()) break;

        bool valid = length > 1;
        for (size_t j = 1; j < length; ++j)
            valid = valid && is_utf8_continuation(
                static_cast<unsigned char>(text[i + j]));

        if (length > 1 && !valid) length = 1;

        i += length;
    }

    return i;
}

vector<Index> encode_required(const TokenizerOperator& tokenizer,
                              string_view text,
                              const char* label)
{
    const vector<Index> ids = tokenizer.encode(text);
    throw_if(ids.empty(),
             format("Qwen3ChatTemplate: '{}' produced no tokens.", label));
    return ids;
}

string_view role_name(ChatRole role)
{
    switch (role)
    {
    case ChatRole::System:    return "system";
    case ChatRole::User:      return "user";
    case ChatRole::Assistant: return "assistant";
    }

    throw runtime_error("Qwen3ChatTemplate: unknown chat role.");
}

SamplingConfig clamp_sampling(const SamplingConfig& sampling_config)
{
    SamplingConfig config = sampling_config;
    config.temperature = max(config.temperature, 0.0f);
    config.top_k = max(config.top_k, Index(0));
    config.top_p = clamp(config.top_p, 0.0f, 1.0f);
    if (config.repetition_penalty <= 0.0f)
        config.repetition_penalty = 1.0f;
    return config;
}

bool descending_first(const pair<float, Index>& left,
                      const pair<float, Index>& right)
{
    return left.first > right.first;
}

void top_k_partition(vector<pair<float, Index>>& values, Index top_k)
{
    nth_element_parallel_if_large(values.begin(),
                                  values.begin() + top_k,
                                  values.end(),
                                  descending_first);
}

struct SamplingWorkspace
{
    VectorR original;
    vector<pair<float, Index>> ranked;
    vector<char> keep;
};

Index sample_token_with_workspace(VectorR& probabilities,
                                  const SamplingConfig& sampling_config,
                                  const vector<Index>& history,
                                  SamplingWorkspace& workspace)
{
    const Index vocabulary_size = probabilities.size();
    throw_if(vocabulary_size == 0,
             "sample_token: probability distribution is empty.");

    const SamplingConfig config = clamp_sampling(sampling_config);

    if (config.temperature == 0.0f)
        return maximal_index(probabilities);

    VectorR& original = workspace.original;
    vector<pair<float, Index>>& ranked = workspace.ranked;
    vector<char>& keep = workspace.keep;

    original = probabilities;
    if (config.repetition_penalty != 1.0f)
        for (const Index token : history)
            if (token >= 0 && token < vocabulary_size)
                probabilities(token) /= config.repetition_penalty;

    if (config.temperature != 1.0f)
    {
        const float inverse_temperature = 1.0f / config.temperature;
        probabilities = probabilities.array().max(0.0f).pow(inverse_temperature).matrix();
    }

    const bool top_k_applied =
        config.top_k > 0 && config.top_k < vocabulary_size;

    if (top_k_applied)
    {
        ranked.resize(size_t(vocabulary_size));
        for (Index i = 0; i < vocabulary_size; ++i)
            ranked[size_t(i)] = {probabilities(i), i};
        top_k_partition(ranked, config.top_k);
        ranked.resize(size_t(config.top_k));

        keep.assign(size_t(vocabulary_size), 0);
        for (Index i = 0; i < config.top_k; ++i)
            keep[size_t(ranked[size_t(i)].second)] = 1;
        for (Index i = 0; i < vocabulary_size; ++i)
            if (!keep[size_t(i)]) probabilities(i) = 0.0f;
    }

    if (config.top_p > 0.0f && config.top_p < 1.0f)
    {
        if (!top_k_applied)
        {
            ranked.resize(size_t(vocabulary_size));
            for (Index i = 0; i < vocabulary_size; ++i)
                ranked[size_t(i)] = {probabilities(i), i};
        }

        const float total = probabilities.sum();

        if (total > 0.0f)
        {
            sort_parallel_if_large(ranked.begin(), ranked.end(), descending_first);
            float cumulative = 0.0f;
            keep.assign(size_t(vocabulary_size), 0);
            for (const auto& [probability, token] : ranked)
            {
                cumulative += probability / total;
                keep[size_t(token)] = 1;
                if (cumulative >= config.top_p) break;
            }
            for (Index i = 0; i < vocabulary_size; ++i)
                if (!keep[size_t(i)]) probabilities(i) = 0.0f;
        }
    }

    const float total = probabilities.sum();
    if (total <= 0.0f) return maximal_index(original);

    const float threshold = random_uniform(0.0f, total);
    float cumulative = 0.0f;
    for (Index i = 0; i < vocabulary_size; ++i)
    {
        cumulative += probabilities(i);
        if (cumulative >= threshold) return i;
    }
    return vocabulary_size - 1;
}

}

Index sample_token(VectorR& probabilities,
                   const SamplingConfig& sampling_config,
                   const vector<Index>& history)
{
    SamplingWorkspace workspace;
    return sample_token_with_workspace(probabilities, sampling_config, history, workspace);
}

ReasoningMode ChatTemplate::resolve_reasoning_mode(
    const ReasoningMode requested) const
{
    if (requested == ReasoningMode::Automatic)
        return supports_reasoning()
            ? default_reasoning_mode()
            : ReasoningMode::Disabled;

    throw_if(requested == ReasoningMode::Enabled && !supports_reasoning(),
             "ChatTemplate: this model does not support reasoning.");
    return requested;
}

SamplingConfig Qwen3ChatTemplate::default_sampling(
    const ReasoningMode requested) const
{
    const ReasoningMode mode = resolve_reasoning_mode(requested);

    SamplingConfig config;
    config.temperature = mode == ReasoningMode::Enabled ? 0.6f : 0.7f;
    config.top_k = 20;
    config.top_p = mode == ReasoningMode::Enabled ? 0.95f : 0.8f;
    config.repetition_penalty = 1.0f;
    return config;
}

vector<Index> Qwen3ChatTemplate::render(
    const vector<ChatMessage>& messages,
    const ReasoningMode requested,
    const TokenizerOperator& tokenizer) const
{
    const ReasoningMode mode = resolve_reasoning_mode(requested);

    string prompt;
    for (const ChatMessage& message : messages)
    {
        prompt += "<|im_start|>";
        prompt += role_name(message.role);
        prompt += "\n";
        prompt += message.content;
        prompt += "<|im_end|>\n";
    }

    prompt += "<|im_start|>assistant\n";
    prompt += mode == ReasoningMode::Enabled
        ? "<think>\n"
        : "<think>\n\n</think>\n\n";

    return encode_required(tokenizer, prompt, "chat prompt");
}

GenerationParserSpec Qwen3ChatTemplate::parser_spec(
    const ReasoningMode requested,
    const TokenizerOperator& tokenizer) const
{
    const ReasoningMode mode = resolve_reasoning_mode(requested);

    GenerationParserSpec parser;
    parser.initial_channel = mode == ReasoningMode::Enabled
        ? GenerationChannel::Reasoning
        : GenerationChannel::Content;
    parser.reasoning_start = encode_required(tokenizer, "<think>", "<think>");
    parser.reasoning_end = encode_required(tokenizer, "</think>", "</think>");
    parser.stop_sequences = {
        encode_required(tokenizer, "<|im_end|>", "<|im_end|>"),
        encode_required(tokenizer, "<|endoftext|>", "<|endoftext|>")
    };
    return parser;
}

GenerationParser::GenerationParser(
    const TokenizerOperator& new_tokenizer,
    const GenerationParserSpec& new_spec)
    : tokenizer(&new_tokenizer),
      spec(new_spec),
      channel(new_spec.initial_channel),
      incremental(new_tokenizer.supports_incremental_decode())
{
    const auto validate = [](const vector<Index>& sequence, const char* label)
    {
        throw_if(ranges::find(sequence, Index(0)) != sequence.end(),
                 format("GenerationParser: {} contains the padding token.", label));
    };

    validate(spec.reasoning_start, "reasoning_start");
    validate(spec.reasoning_end, "reasoning_end");
    for (const vector<Index>& stop : spec.stop_sequences)
    {
        throw_if(stop.empty(), "GenerationParser: stop sequences cannot be empty.");
        validate(stop, "stop sequence");
    }
}

bool GenerationParser::push(const Index token_id,
                            const ChatCallback& callback)
{
    if (stopped) return true;

    pending.push_back(token_id);
    return process_pending(callback, false);
}

void GenerationParser::finish(const ChatCallback& callback)
{
    if (!stopped) process_pending(callback, true);
    emit_stable_delta(GenerationChannel::Reasoning, callback);
    emit_stable_delta(GenerationChannel::Content, callback);
}

bool GenerationParser::process_pending(const ChatCallback& callback,
                                       const bool flush)
{
    while (!pending.empty())
    {
        if (const auto stop = ranges::find(spec.stop_sequences, pending);
            stop != spec.stop_sequences.end())
        {
            control_tokens += Index(stop->size());
            pending.clear();
            stopped = true;
            return true;
        }

        if (!spec.reasoning_start.empty() && pending == spec.reasoning_start)
        {
            control_tokens += Index(pending.size());
            pending.clear();
            channel = GenerationChannel::Reasoning;
            continue;
        }

        if (!spec.reasoning_end.empty() && pending == spec.reasoning_end)
        {
            control_tokens += Index(pending.size());
            pending.clear();
            channel = GenerationChannel::Content;
            continue;
        }

        const bool possible_control =
            is_prefix(pending, spec.reasoning_start)
            || is_prefix(pending, spec.reasoning_end)
            || ranges::any_of(spec.stop_sequences,
                              [&](const vector<Index>& stop)
                              { return is_prefix(pending, stop); });

        if (possible_control && !flush) return false;

        const Index data_token = pending.front();
        pending.erase(pending.begin());
        append_data_token(data_token, callback);
    }

    return stopped;
}

void GenerationParser::append_data_token(const Index token_id,
                                         const ChatCallback& callback)
{
    ChannelState& state = channel_state(channel);
    ++(channel == GenerationChannel::Reasoning
           ? reasoning_tokens
           : content_tokens);

    if (incremental)
        state.tail += tokenizer->decode_token(token_id);
    else
        state.ids.push_back(token_id);

    emit_stable_delta(channel, callback);
}

GenerationParser::ChannelState& GenerationParser::channel_state(
    const GenerationChannel output_channel) noexcept
{
    return output_channel == GenerationChannel::Reasoning
        ? reasoning_state
        : content_state;
}

void GenerationParser::emit_stable_delta(const GenerationChannel output_channel,
                                         const ChatCallback& callback)
{
    ChannelState& state = channel_state(output_channel);
    string delta;

    if (incremental)
    {

        const size_t stable_bytes = complete_utf8_prefix_size(state.tail);
        if (stable_bytes == 0) return;

        if (stable_bytes == state.tail.size())
        {
            delta = std::move(state.tail);
            state.tail.clear();
        }
        else
        {
            delta.assign(state.tail, 0, stable_bytes);
            state.tail.erase(0, stable_bytes);
        }
    }
    else
    {
        const string decoded = tokenizer->decode(state.ids);
        const size_t stable_bytes = complete_utf8_prefix_size(decoded);

        throw_if(stable_bytes < state.text.size()
                 || !decoded.starts_with(state.text),
                 "GenerationParser: tokenizer decoding is not prefix-stable.");

        if (stable_bytes <= state.text.size()) return;

        delta = decoded.substr(state.text.size(),
                               stable_bytes - state.text.size());
    }

    state.text.append(delta);
    if (callback && !delta.empty())
        callback({output_channel, delta});
}

namespace
{

class DecoderSampler
{
public:
    DecoderSampler(Index new_output_vocabulary,
                   Index new_sample_vocabulary,
                   unsigned long long new_seed,
                   Buffer* new_token_device)
        : output_vocabulary(new_output_vocabulary),
          vocabulary(new_sample_vocabulary),
          generator(new_seed),
#ifdef OPENNN_HAS_CUDA
          seed(new_seed),
          token_device(new_token_device),
#endif
          logits(size_t(new_sample_vocabulary)),
          bf16_logits(size_t(new_sample_vocabulary))
    {
        throw_if(vocabulary <= 1,
                 "ChatSession: output vocabulary must contain at least two tokens.");

#ifdef OPENNN_HAS_CUDA
        if (token_device)
        {
            pinned_id.resize_bytes(Index(sizeof(int)));
            gpu_candidates.resize_bytes(
                sample_logits_scratch_floats() * Index(sizeof(float)),
                Device::CUDA);
            gpu_id.resize_bytes(Index(sizeof(int)), Device::CUDA);
        }
#endif
    }

    Index sample_row(const ForwardPropagation& propagation,
                     Index row_index,
                     const SamplingConfig& input_config,
                     const vector<Index>& history)
    {
        const SamplingConfig config = clamp_sampling(input_config);

        const TensorView output = propagation.get_outputs();
        throw_if(output.get_shape().empty()
                 || output.get_shape().back() != output_vocabulary,
                 "ChatSession: output vocabulary does not match the sampler.");
        throw_if(row_index < 0
                 || row_index >= output.size() / output_vocabulary,
                 "ChatSession: logits row is out of range.");

        const Index element_bytes = Index(type_bytes(output.get_type()));
        char* const row = static_cast<char*>(output.get_data())
            + row_index * output_vocabulary * element_bytes;

#ifdef OPENNN_HAS_CUDA
        const bool fast_gpu = output.is_cuda()
            && config.repetition_penalty == 1.0f
            && (config.temperature == 0.0f
                || (config.top_k > 0 && config.top_k <= 32));

        if (fast_gpu)
        {
            const TensorView row_view(row, {vocabulary},
                                      output.get_type(), Device::CUDA);
            sample_logits_row(row_view,
                              config.temperature,
                              config.temperature == 0.0f ? 1 : config.top_k,
                              config.top_p,
                              seed,
                              step++,
                              gpu_candidates.data(),
                              static_cast<int*>(gpu_id.data()),
                              token_device
                                  ? static_cast<float*>(token_device->data())
                                  : nullptr);
            device::copy_async(pinned_id.data(),
                               gpu_id.data(),
                               Index(sizeof(int)),
                               device::CopyKind::DeviceToHost,
                               device::get_compute_stream());
            device::synchronize(device::get_compute_stream());
            return Index(*pinned_id.as<int>());
        }
#endif

        read_logits(TensorView(row, {vocabulary}, output.get_type(), output.get_device()));
        const Index sampled = sample_host(config, history);

#ifdef OPENNN_HAS_CUDA
        if (token_device)
        {
            const float token_value = float(sampled);
            device::copy_async(token_device->data(),
                               &token_value,
                               Index(sizeof(float)),
                               device::CopyKind::HostToDevice,
                               device::get_compute_stream());
            device::synchronize(device::get_compute_stream());
        }
#endif

        return sampled;
    }

private:
    void read_logits(const TensorView& row)
    {
        if (row.is_cuda())
        {
            if (row.is_fp32())
            {
                device::copy_async(logits.data(),
                                   row.get_data(),
                                   vocabulary * Index(sizeof(float)),
                                   device::CopyKind::DeviceToHost,
                                   device::get_compute_stream());
                return device::synchronize(device::get_compute_stream());
            }

            throw_if(!row.is_bf16(),
                     "ChatSession: unsupported logits dtype.");
            device::copy_async(bf16_logits.data(),
                               row.get_data(),
                               vocabulary * Index(sizeof(uint16_t)),
                               device::CopyKind::DeviceToHost,
                               device::get_compute_stream());
            device::synchronize(device::get_compute_stream());
        }
        else if (row.is_fp32())
        {
            memcpy(logits.data(), row.get_data(),
                   size_t(vocabulary) * sizeof(float));
            return;
        }
        else
        {
            throw_if(!row.is_bf16(),
                     "ChatSession: unsupported logits dtype.");
            memcpy(bf16_logits.data(), row.get_data(),
                   size_t(vocabulary) * sizeof(uint16_t));
        }

        for (Index i = 0; i < vocabulary; ++i)
            logits[size_t(i)] =
                bfloat16_to_float_host(bf16_logits[size_t(i)]);
    }

    Index sample_host(const SamplingConfig& config,
                      const vector<Index>& history)
    {
        adjusted = logits;
        adjusted[0] = NEG_INFINITY;

        if (config.repetition_penalty != 1.0f)
            for (const Index token : history)
                if (token > 0 && token < vocabulary)
                {
                    float& value = adjusted[size_t(token)];
                    value = value < 0.0f
                        ? value * config.repetition_penalty
                        : value / config.repetition_penalty;
                }

        if (config.temperature == 0.0f)
            return Index(distance(adjusted.begin(),
                                  max_element(adjusted.begin(),
                                              adjusted.end())));

        candidates.clear();
        candidates.reserve(size_t(vocabulary - 1));
        for (Index token = 1; token < vocabulary; ++token)
            candidates.push_back({
                adjusted[size_t(token)] / config.temperature,
                token
            });

        if (config.top_k > 0 && config.top_k < ssize(candidates))
        {
            top_k_partition(candidates, config.top_k);
            candidates.resize(size_t(config.top_k));
        }

        sort_parallel_if_large(candidates.begin(), candidates.end(), descending_first);

        const float maximum = candidates.front().first;
        double probability_sum = 0.0;
        for (auto& [value, token] : candidates)
        {
            (void)token;
            value = exp(value - maximum);
            probability_sum += value;
        }
        for (auto& [value, token] : candidates)
        {
            (void)token;
            value = float(value / probability_sum);
        }

        double kept_probability = 1.0;
        if (config.top_p > 0.0f && config.top_p < 1.0f)
        {
            double cumulative = 0.0;
            size_t keep = 0;
            for (size_t i = 0; i < candidates.size(); ++i)
            {
                cumulative += candidates[i].first;
                keep = i + 1;
                if (cumulative >= config.top_p) break;
            }
            candidates.resize(keep);
            kept_probability = cumulative;
        }

        const double draw =
            uniform_real_distribution<double>(0.0, kept_probability)(generator);
        double cumulative = 0.0;
        for (const auto& [probability, token] : candidates)
        {
            cumulative += probability;
            if (draw <= cumulative) return token;
        }
        return candidates.back().second;
    }

    Index output_vocabulary = 0;
    Index vocabulary = 0;
    mt19937_64 generator;
#ifdef OPENNN_HAS_CUDA
    unsigned long long seed = 0;
    unsigned long long step = 0;
    Buffer* token_device = nullptr;
#endif

    vector<float> logits;
    vector<uint16_t> bf16_logits;

    vector<float> adjusted;
    vector<pair<float, Index>> candidates;
#ifdef OPENNN_HAS_CUDA
    device::PinnedBuffer pinned_id;
    Buffer gpu_candidates{Device::CUDA};
    Buffer gpu_id{Device::CUDA};
#endif
};

bool valid_complete_history(const vector<ChatMessage>& messages)
{
    size_t index = 0;
    while (index < messages.size()
           && messages[index].role == ChatRole::System)
        ++index;

    while (index < messages.size())
    {
        if (messages[index].role != ChatRole::User) return false;
        ++index;
        if (index >= messages.size()
            || messages[index].role != ChatRole::Assistant)
            return false;
        ++index;
    }
    return true;
}

bool remove_oldest_turn(vector<ChatMessage>& messages)
{
    size_t first = 0;
    while (first < messages.size()
           && messages[first].role == ChatRole::System)
        ++first;

    if (first + 1 >= messages.size()) return false;
    if (messages[first].role != ChatRole::User
        || messages[first + 1].role != ChatRole::Assistant)
        return false;

    messages.erase(messages.begin() + ptrdiff_t(first),
                   messages.begin() + ptrdiff_t(first + 2));
    return true;
}

enum class ClassicSessionKind
{
    SequenceToSequence,
    DecoderOnly
};

struct ClassicGenerationState
{
    explicit ClassicGenerationState(const ClassicSessionKind new_kind)
        : kind(new_kind)
    {
    }

    ClassicSessionKind kind;
    Transformer* transformer = nullptr;
    TextGenerationNetwork* decoder = nullptr;
    const TokenizerOperator* input_tokenizer = nullptr;
    const TokenizerOperator* output_tokenizer = nullptr;

    Buffer arena{Device::CUDA};
    TensorView source_device;
    TensorView target_device;
    unique_ptr<ForwardPropagation> propagation;
    vector<TensorView> inputs;

    Tensor2 source;
    Tensor2 target;
    vector<Index> history;
    VectorR distribution;
    SamplingWorkspace sampling_workspace;
    vector<uint16_t> bf16_staging;

    Index input_length = 0;
    Index sequence_length = 0;
    Index encoder_embedding = -1;
    Index encoder_last = -1;
    Index decoder_embedding = -1;
    Index decoder_first = -1;
    Index output_projection = -1;

    vector<Index> retained_outputs;
};

void prepare_classic_network(NeuralNetwork& network)
{
    throw_if(!network.is_gpu() || !device::is_cuda_build(),
             "ChatSession: classic text generation requires CUDA.");
    network.copy_parameters_device();

    network.release_bf16_fp32_parameter_master_for_inference();
    network.link_parameters();
    network.copy_states_device();
    network.link_states();
}

void allocate_classic_buffers(ClassicGenerationState& state,
                              NeuralNetwork& network)
{
    constexpr Index batch_size = 1;
    const Index source_bytes = state.input_length > 0
        ? get_aligned_bytes(state.input_length, Type::FP32)
        : 0;
    const Index target_bytes =
        get_aligned_bytes(state.sequence_length, Type::FP32);

    state.arena.resize_bytes(source_bytes + target_bytes, Device::CUDA);
    char* const base = state.arena.as<char>();
    state.target_device = TensorView(base + source_bytes,
                                     {batch_size, state.sequence_length},
                                     Type::FP32, Device::CUDA);
    if (source_bytes > 0)
        state.source_device = TensorView(base,
                                         {batch_size, state.input_length},
                                         Type::FP32, Device::CUDA);

    InferenceShapePolicy shape_policy;
    shape_policy.retained_output_layers = state.retained_outputs;
    state.propagation = make_unique<ForwardPropagation>(
        batch_size, &network, ForwardPropagationMode::Inference, shape_policy);
    state.target = Tensor2(batch_size, state.sequence_length);
    state.history.reserve(size_t(state.sequence_length));

    const Index vocabulary =
        network.get_layers().back()->get_output_shape().back();
    state.distribution = VectorR::Zero(vocabulary);
    state.bf16_staging.assign(size_t(vocabulary), 0);
}

unique_ptr<ClassicGenerationState>
make_sequence_to_sequence_state(Transformer& network)
{
    prepare_classic_network(network);

    auto state = make_unique<ClassicGenerationState>(
        ClassicSessionKind::SequenceToSequence);
    state->transformer = &network;
    state->input_tokenizer = network.get_input_tokenizer();
    state->output_tokenizer = network.get_target_tokenizer();
    state->input_length = network.get_input_sequence_length();
    state->sequence_length = network.get_decoder_sequence_length();

    throw_if(!state->input_tokenizer
             || state->input_tokenizer->get_vocabulary().empty(),
             "ChatSession: Transformer input vocabulary is empty.");
    throw_if(!state->output_tokenizer
             || state->output_tokenizer->get_vocabulary().empty(),
             "ChatSession: Transformer target vocabulary is empty.");

    const auto& layers = network.get_layers();
    state->encoder_embedding =
        network.get_layer_index("encoder_embedding");
    state->decoder_embedding =
        network.get_layer_index("decoder_embedding");

    const auto cross_attention = ranges::find_if(layers, [](const unique_ptr<Layer>& layer)
                                                 { return layer->get_label().starts_with("cross_attention_"); });
    throw_if(cross_attention == layers.end(),
             "ChatSession: Transformer has no cross-attention layer.");

    const Index first_cross_attention = ranges::distance(layers.begin(), cross_attention);

    const vector<Index>& cross_sources =
        network.get_source_layers()[size_t(first_cross_attention)];
    throw_if(cross_sources.size() < 2 || cross_sources[1] < 0,
             "ChatSession: invalid Transformer cross-attention inputs.");
    state->encoder_last = cross_sources[1];
    state->decoder_first = state->encoder_last + 1;
    state->output_projection = ssize(layers) - 1;

    throw_if(state->decoder_first >= ssize(layers)
             || layers[size_t(state->decoder_first)]->get_label()
                    != "decoder_self_attention_1"
             || layers.back()->get_label() != "output_projection",
             "ChatSession: unsupported Transformer decoder layout.");

    const auto& source_layers = network.get_source_layers();
    for (Index i = state->decoder_first; i <= state->output_projection; ++i)
        for (const Index source : source_layers[size_t(i)])
            if (source >= state->encoder_embedding
                && source <= state->encoder_last
                && ranges::find(state->retained_outputs, source)
                       == state->retained_outputs.end())
                state->retained_outputs.push_back(source);

    allocate_classic_buffers(*state, network);
    state->source = Tensor2(1, state->input_length);
    state->inputs = {state->target_device, state->source_device};
    return state;
}

unique_ptr<ClassicGenerationState>
make_decoder_only_state(TextGenerationNetwork& network)
{
    prepare_classic_network(network);

    auto state = make_unique<ClassicGenerationState>(
        ClassicSessionKind::DecoderOnly);
    state->decoder = &network;
    state->output_tokenizer = network.get_tokenizer();
    state->sequence_length = network.get_sequence_length();

    throw_if(!state->output_tokenizer
             || state->output_tokenizer->get_vocabulary().empty(),
             "ChatSession: text-generation vocabulary is empty.");
    network.get_layer_index("embedding");
    throw_if(network.get_layers().back()->get_label()
                 != "output_projection",
             "ChatSession: unsupported decoder-only layout.");

    allocate_classic_buffers(*state, network);
    state->inputs = {state->target_device};
    return state;
}

void copy_classic_input(const TensorView& destination,
                        const float* source)
{
    device::copy_async(destination.get_data(),
                       source,
                       destination.byte_size(),
                       device::CopyKind::HostToDevice,
                       device::get_compute_stream());
}

void read_classic_distribution(ClassicGenerationState& state,
                               const Index position)
{
    const TensorView output = state.propagation->get_outputs();
    const Index vocabulary = output.get_shape().back();
    const Index offset = position * vocabulary;
    const cudaStream_t stream = device::get_compute_stream();

    if (output.is_bf16())
    {
        device::copy_async(state.bf16_staging.data(),
                           output.as<bfloat16>() + offset,
                           vocabulary * Index(sizeof(uint16_t)),
                           device::CopyKind::DeviceToHost,
                           stream);
        device::synchronize(stream);
        ranges::transform(state.bf16_staging | views::take(vocabulary),
                          state.distribution.data(), bfloat16_to_float_host);
        return;
    }

    throw_if(!output.is_fp32(),
             "ChatSession: unsupported text output dtype.");
    device::copy_async(state.distribution.data(),
                       output.as<float>() + offset,
                       vocabulary * Index(sizeof(float)),
                       device::CopyKind::DeviceToHost,
                       stream);
    device::synchronize(stream);
}

}

struct ChatSession::Impl
{
    explicit Impl(Transformer& new_network)
        : network(&new_network),
          tokenizer(new_network.get_target_tokenizer()),
          gpu(true),
          context_length(new_network.get_decoder_sequence_length())
    {
        classic = make_sequence_to_sequence_state(new_network);
        vocabulary = classic->distribution.size();
    }

    explicit Impl(TextGenerationNetwork& new_network)
        : network(&new_network),
          tokenizer(new_network.get_tokenizer()),
          gpu(true),
          context_length(new_network.get_sequence_length())
    {
        classic = make_decoder_only_state(new_network);
        vocabulary = classic->distribution.size();
    }

    Impl(NeuralNetwork& new_network,
         const TokenizerOperator& new_tokenizer,
         unique_ptr<ChatTemplate> new_template,
         unsigned long long new_seed)
        : network(&new_network),
          tokenizer(&new_tokenizer),
          chat_template(std::move(new_template)),
          gpu(new_network.is_gpu() && device::is_cuda_build()),
          context_length(new_network.get_input_shape().empty()
                             ? Index(0)
                             : new_network.get_input_shape()[0]),
          vocabulary(new_network.get_output_shape().empty()
                         ? Index(0)
                         : new_network.get_output_shape().back()),
          prefill(1, &new_network, ForwardPropagationMode::Inference,
                  {.sequence_capacity =
                       min(context_length, ChatSession::PREFILL_BLOCK_SIZE),
                   .final_output_capacity = 1,
                   .retained_output_layers = {}})
    {
        throw_if(!chat_template,
                 "ChatSession: chat template is not set.");
        throw_if(context_length <= 0,
                 "ChatSession: network has no token sequence input.");
        throw_if(vocabulary <= 1,
                 "ChatSession: network has no vocabulary output.");
        throw_if(tokenizer->get_vocabulary_size() > vocabulary,
                 format("ChatSession: tokenizer vocabulary ({}) exceeds "
                        "network output ({}).",
                        tokenizer->get_vocabulary_size(), vocabulary));

        token_window.assign(size_t(context_length), 0.0f);
        cached_tokens.reserve(size_t(context_length));
        prefill_inputs.resize(1);

#ifdef OPENNN_HAS_CUDA
        if (gpu)
        {
            token_device.resize_bytes(Index(sizeof(float)), Device::CUDA);
            initialize_cuda_input(prefill);
            decode.set(1, network, &prefill.arena,
                       ForwardPropagationMode::Inference,
                       {.sequence_capacity = 1,
                        .final_output_capacity = 1,
                        .retained_output_layers = {}});
            decode.share_session_state_from(prefill);
            decode.set_active_sequence_length(1);
            decode.set_cuda_graph(true);
            const cudaStream_t stream = device::get_compute_stream();
            prefill.stage_position(stream);
            decode.stage_position(stream);
            decode_inputs = {
                TensorView(token_device.data(), {1, 1},
                           Type::FP32, Device::CUDA)
            };
        }
#endif

        const unsigned long long effective_seed = new_seed == 0
            ? (static_cast<unsigned long long>(random_device{}()) << 32)
                ^ static_cast<unsigned long long>(random_device{}())
            : new_seed;
        sampler = make_unique<DecoderSampler>(
            vocabulary, tokenizer->get_vocabulary_size(),
            effective_seed, gpu ? &token_device : nullptr);

#ifdef OPENNN_HAS_CUDA
        if (gpu)
        {

            const Index warmup_tokens = prefill.get_sequence_capacity();
            fill_n(token_window.begin(), size_t(warmup_tokens), 0.0f);
            run_prefill(prefill, prefill_inputs, *network, warmup_tokens, 0);
            stage_token(token_device, 0);
            run_decode(0, 0);
            run_decode(0, 0);
            device::synchronize(device::get_compute_stream());
        }
#endif
    }

    vector<Index> render_fitting_prompt(vector<ChatMessage>& candidate,
                                        ReasoningMode mode) const
    {
        vector<Index> prompt =
            chat_template->render(candidate, mode, *tokenizer);

        while (ssize(prompt) > context_length)
        {
            throw_if(!remove_oldest_turn(candidate),
                     format("ChatSession: system and current user prompt "
                            "need {} tokens, exceeding the {}-token context.",
                            prompt.size(), context_length));
            prompt = chat_template->render(candidate, mode, *tokenizer);
        }

        throw_if(prompt.empty(),
                 "ChatSession: chat template produced no tokens.");
        return prompt;
    }

    struct SpeculativeDraft
    {
        NeuralNetwork* network = nullptr;
        Index propose_count = 4;
        ForwardPropagation prefill;
        ForwardPropagation decode;
        ForwardPropagation target_verify;
        Buffer token_device{Device::CUDA};
        vector<TensorView> prefill_inputs;
        vector<TensorView> decode_inputs;
        vector<TensorView> target_verify_inputs;
        unique_ptr<DecoderSampler> sampler;
        vector<Index> proposals;
    };

    void run_prefill(ForwardPropagation& propagation,
                     vector<TensorView>& inputs,
                     NeuralNetwork& propagated,
                     Index count, Index past)
    {
        throw_if(count < 1 || past < 0 || past + count > context_length,
                 "ChatSession: prefill [{}, {}) exceeds the {}-token context.",
                 past, past + count, context_length);

        const Index block_capacity = propagation.get_sequence_capacity();
        for (Index offset = 0; offset < count; offset += block_capacity)
        {
            const Index block = min(block_capacity, count - offset);
            propagation.past_length = past + offset;
            propagation.set_active_sequence_length(block);
            propagation.set_output_sequence_window(block - 1, 1);
            inputs[0] =
                TensorView(token_window.data() + offset, {1, block});
            propagated.forward_propagate(inputs, propagation, false);
        }
    }

    static void stage_token(Buffer& destination, Index token)
    {
        const float value = float(token);
        device::copy_async(destination.data(), &value, Index(sizeof(float)),
                           device::CopyKind::HostToDevice,
                           device::get_compute_stream());
        device::synchronize(device::get_compute_stream());
    }

    static void initialize_cuda_input(ForwardPropagation& propagation)
    {
        propagation.staged_input_storage.resize(1);
        propagation.staged_inputs.resize(1);
        propagation.staged_input_storage[0].resize_bytes(
            propagation.get_sequence_capacity() * Index(sizeof(float)),
            Device::CUDA);
    }

    ForwardPropagation& run_target_verify(Index count, Index past)
    {
        throw_if(!draft || count < 1
                 || count > draft->target_verify.get_sequence_capacity(),
                 "ChatSession: invalid speculative verification width {}.",
                 count);
        throw_if(past < 0 || past + count > context_length,
                 "ChatSession: speculative verification [{}, {}) exceeds "
                 "the {}-token context.",
                 past, past + count, context_length);

        draft->target_verify.past_length = past;
        draft->target_verify.set_active_sequence_length(count);
        draft->target_verify.set_output_sequence_window(0, count);
        draft->target_verify_inputs[0] =
            TensorView(token_window.data(), {1, count});
        network->forward_propagate(
            draft->target_verify_inputs, draft->target_verify, false);
        return draft->target_verify;
    }

    void run_draft_decode(Index past)
    {
        draft->decode.past_length = past;
        draft->network->calculate_outputs_resident(
            draft->decode_inputs, draft->decode, false);
    }

    ForwardPropagation& run_decode(Index token, Index past)
    {
#ifdef OPENNN_HAS_CUDA
        if (gpu)
        {
            decode.past_length = past;
            network->calculate_outputs_resident(
                decode_inputs, decode, false);
            return decode;
        }
#endif

        token_window[0] = float(token);
        run_prefill(prefill, prefill_inputs, *network, 1, past);
        return prefill;
    }

    NeuralNetwork* network = nullptr;
    const TokenizerOperator* tokenizer = nullptr;
    unique_ptr<ChatTemplate> chat_template;
    unique_ptr<ClassicGenerationState> classic;
    bool gpu = false;
    Index context_length = 0;
    Index vocabulary = 0;

    vector<ChatMessage> messages;
    vector<Index> cached_tokens;
    vector<float> token_window;

    ForwardPropagation prefill;
    ForwardPropagation decode;
    Buffer token_device{Device::CUDA};
    vector<TensorView> prefill_inputs;
    vector<TensorView> decode_inputs;
    unique_ptr<DecoderSampler> sampler;
    unique_ptr<SpeculativeDraft> draft;
};

ChatSession::ChatSession(Transformer& network)
    : impl(make_unique<Impl>(network))
{
}

ChatSession::ChatSession(TextGenerationNetwork& network)
    : impl(make_unique<Impl>(network))
{
}

ChatSession::ChatSession(
    NeuralNetwork& network,
    const TokenizerOperator& tokenizer,
    unique_ptr<ChatTemplate> chat_template,
    const unsigned long long seed)
    : impl(make_unique<Impl>(network, tokenizer,
                             std::move(chat_template), seed))
{
}

ChatSession::~ChatSession() = default;

void ChatSession::attach_draft_model(NeuralNetwork& draft_network, Index draft_tokens)
{
    throw_if(impl->classic != nullptr,
             "ChatSession::attach_draft_model: unsupported session type.");
    throw_if(!impl->gpu,
             "ChatSession::attach_draft_model: speculative decoding requires the GPU session.");
    throw_if(draft_tokens < 1,
             "ChatSession::attach_draft_model: draft_tokens must be at least 1.");
    throw_if(draft_network.get_output_shape().empty()
             || draft_network.get_output_shape().back() != impl->vocabulary,
             "ChatSession::attach_draft_model: draft vocabulary does not match the main network.");
    throw_if(draft_network.get_input_shape().empty()
             || draft_network.get_input_shape()[0] < impl->context_length,
             "ChatSession::attach_draft_model: draft context is shorter than the session context.");
    throw_if(!draft_network.is_gpu(),
             "ChatSession::attach_draft_model: the draft network is not compiled for CUDA.");
    throw_if(draft_network.get_training_type() != impl->network->get_training_type(),
             "ChatSession::attach_draft_model: draft compute dtype does not match the main network.");

#ifdef OPENNN_HAS_CUDA
    auto draft = make_unique<Impl::SpeculativeDraft>();
    draft->network = &draft_network;
    draft->propose_count = draft_tokens;

    draft->prefill.set(1, &draft_network, nullptr,
                       ForwardPropagationMode::Inference,
                       {.sequence_capacity =
                            min(impl->context_length,
                                ChatSession::PREFILL_BLOCK_SIZE),
                        .final_output_capacity = 1,
                        .retained_output_layers = {}});
    draft->token_device.resize_bytes(Index(sizeof(float)), Device::CUDA);
    draft->prefill_inputs.resize(1);
    Impl::initialize_cuda_input(draft->prefill);
    draft->decode.set(1, &draft_network, &draft->prefill.arena,
                      ForwardPropagationMode::Inference,
                      {.sequence_capacity = 1,
                       .final_output_capacity = 1,
                       .retained_output_layers = {}});
    draft->decode.share_session_state_from(draft->prefill);
    draft->decode.set_active_sequence_length(1);
    draft->decode.set_cuda_graph(true);

    const Index verify_capacity = draft_tokens + 1;
    draft->target_verify.set(
        1, impl->network, nullptr, ForwardPropagationMode::Inference,
        {.sequence_capacity = verify_capacity,
         .final_output_capacity = verify_capacity,
         .retained_output_layers = {}});
    draft->target_verify.share_session_state_from(impl->prefill);
    draft->target_verify_inputs.resize(1);
    Impl::initialize_cuda_input(draft->target_verify);

    const cudaStream_t stream = device::get_compute_stream();
    draft->prefill.stage_position(stream);
    draft->decode.stage_position(stream);
    draft->target_verify.stage_position(stream);
    draft->decode_inputs = {
        TensorView(draft->token_device.data(), {1, 1},
                   Type::FP32, Device::CUDA)
    };
    draft->sampler = make_unique<DecoderSampler>(
        impl->vocabulary, impl->tokenizer->get_vocabulary_size(),
        1ull, &draft->token_device);
    draft->proposals.reserve(size_t(draft_tokens));

    impl->draft = std::move(draft);

    const Index warmup_tokens =
        impl->draft->prefill.get_sequence_capacity();
    fill_n(impl->token_window.begin(), size_t(warmup_tokens), 0.0f);
    impl->run_prefill(impl->draft->prefill, impl->draft->prefill_inputs,
                      *impl->draft->network, warmup_tokens, 0);
    Impl::stage_token(impl->draft->token_device, 0);
    impl->run_draft_decode(0);
    impl->run_draft_decode(0);
    fill_n(impl->token_window.begin(), size_t(verify_capacity), 0.0f);
    for (Index width = 2; width <= verify_capacity; ++width)
        impl->run_target_verify(width, 0);
    device::synchronize(device::get_compute_stream());
#endif
}

namespace
{

struct ClassicDecodeLoop
{
    ClassicDecodeLoop(ClassicGenerationState& new_state,
                      const SamplingConfig& new_sampling,
                      const ChatCallback& new_callback)
        : state(new_state),
          sampling(new_sampling),
          callback(new_callback),
          parser(*new_state.output_tokenizer, {})
    {
    }

    Index token_budget() const
    {
        return sampling.maximum_tokens > 0
            ? sampling.maximum_tokens
            : state.sequence_length;
    }

    Index sample_at(Index position)
    {
        read_classic_distribution(state, position);
        const Index next =
            sample_token_with_workspace(state.distribution, sampling, state.history,
                                        state.sampling_workspace);
        ++response.generated_tokens;
        state.history.push_back(next);
        return next;
    }

    void dispatch(Index next)
    {
        constexpr Index pad = 0;
        if (next == pad
            || next < 0
            || next >= state.output_tokenizer->get_vocabulary_size())
            ++response.control_tokens;
        else
            parser.push(next, callback);
    }

    ChatResponse finish()
    {
        parser.finish(callback);
        response.content = parser.get_content();
        response.content_tokens = parser.get_content_tokens();
        response.control_tokens += parser.get_control_tokens();
        return response;
    }

    ClassicGenerationState& state;
    const SamplingConfig& sampling;
    const ChatCallback& callback;
    GenerationParser parser;
    ChatResponse response;
};

ChatResponse send_sequence_to_sequence(
    ClassicGenerationState& state,
    const string_view source,
    const SamplingConfig& sampling,
    const ChatCallback& callback)
{
    constexpr Index pad = 0;
    constexpr Index start = TokenizerOperator::START_INDEX;
    constexpr Index end = TokenizerOperator::END_INDEX;
    using Clock = chrono::steady_clock;

    state.target.setConstant(pad);
    state.target(0, 0) = float(start);
    state.history.clear();
    copy_classic_input(state.target_device, state.target.data());

    state.source.setConstant(pad);
    const vector<Index> source_ids =
        state.input_tokenizer->encode_sequence(source, state.input_length);
    for (Index i = 0; i < ssize(source_ids); ++i)
        state.source(0, i) = float(source_ids[size_t(i)]);
    copy_classic_input(state.source_device, state.source.data());

    ClassicDecodeLoop loop(state, sampling, callback);
    ChatResponse& response = loop.response;
    response.prompt_tokens = ssize(source_ids);
    response.prefill_tokens = ssize(source_ids);

    const auto prefill_start = Clock::now();
    state.transformer->forward_propagate(
        state.inputs, *state.propagation, false,
        state.encoder_embedding, state.encoder_last);
    device::synchronize(device::get_compute_stream());
    const auto prefill_end = Clock::now();
    response.prefill_milliseconds =
        chrono::duration<double, milli>(prefill_end - prefill_start).count();

    const Index limit =
        min(loop.token_budget() + 1, state.sequence_length);

    response.finish_reason =
        sampling.maximum_tokens > 0
        && sampling.maximum_tokens <= state.sequence_length - 1
            ? FinishReason::MaximumTokens
            : FinishReason::ContextLimit;

    const auto decode_start = Clock::now();
    for (Index position = 1; position < limit; ++position)
    {
        state.transformer->forward_propagate(
            state.inputs, *state.propagation, false,
            state.decoder_embedding, state.decoder_embedding);
        state.transformer->forward_propagate(
            state.inputs, *state.propagation, false,
            state.decoder_first, state.output_projection);

        const Index next = loop.sample_at(position - 1);
        state.target(0, position) = float(next);
        device::copy_async(state.target_device.as<float>() + position,
                           &state.target(0, position),
                           Index(sizeof(float)),
                           device::CopyKind::HostToDevice,
                           device::get_compute_stream());

        if (next == end)
        {
            ++response.control_tokens;
            response.finish_reason = FinishReason::Stop;
            break;
        }

        loop.dispatch(next);
    }
    const auto decode_end = Clock::now();
    response.decode_milliseconds =
        chrono::duration<double, milli>(decode_end - decode_start).count();
    return loop.finish();
}

ChatResponse send_classic_decoder(
    ClassicGenerationState& state,
    const string_view prompt,
    const SamplingConfig& sampling,
    const ChatCallback& callback)
{
    constexpr Index pad = 0;
    using Clock = chrono::steady_clock;

    vector<Index> context = state.output_tokenizer->encode(prompt);
    throw_if(context.empty(),
             "ChatSession: prompt produced no tokens.");
    state.history = context;

    ClassicDecodeLoop loop(state, sampling, callback);
    ChatResponse& response = loop.response;
    response.prompt_tokens = ssize(context);
    response.prefill_tokens =
        min(ssize(context), state.sequence_length);
    response.finish_reason = FinishReason::MaximumTokens;

    const Index maximum_tokens = loop.token_budget();

    double decode_milliseconds = 0.0;
    for (Index step = 0; step < maximum_tokens; ++step)
    {
        const Index window_length =
            min(ssize(context), state.sequence_length);
        const Index window_start = ssize(context) - window_length;

        state.target.setConstant(pad);
        for (Index i = 0; i < window_length; ++i)
            state.target(0, i) =
                float(context[size_t(window_start + i)]);
        copy_classic_input(state.target_device, state.target.data());

        const auto step_start = Clock::now();
        state.decoder->forward_propagate(
            state.inputs, *state.propagation, false);
        const Index next = loop.sample_at(window_length - 1);
        const auto step_end = Clock::now();
        const double elapsed =
            chrono::duration<double, milli>(step_end - step_start).count();
        if (step == 0)
            response.prefill_milliseconds = elapsed;
        else
            decode_milliseconds += elapsed;

        context.push_back(next);
        loop.dispatch(next);
    }

    response.decode_milliseconds = decode_milliseconds;
    return loop.finish();
}

}

ChatResponse ChatSession::send(
    const string_view user_message,
    const ChatOptions& options,
    const ChatCallback& callback)
{
    throw_if(user_message.empty(),
             "ChatSession::send: user message cannot be empty.");

    const ReasoningMode mode =
        resolve_reasoning_mode(options.reasoning_mode);
    const SamplingConfig sampling =
        options.sampling.value_or(default_sampling(mode));

    if (impl->classic)
        return impl->classic->kind == ClassicSessionKind::SequenceToSequence
            ? send_sequence_to_sequence(
                  *impl->classic, user_message, sampling, callback)
            : send_classic_decoder(
                  *impl->classic, user_message, sampling, callback);

    const Index maximum_tokens = sampling.maximum_tokens > 0
        ? sampling.maximum_tokens
        : impl->context_length;

    vector<ChatMessage> candidate = impl->messages;
    candidate.push_back({ChatRole::User, string(user_message)});
    vector<Index> prompt = impl->render_fitting_prompt(candidate, mode);

    Index prefix = 0;
    const Index reusable =
        min(ssize(impl->cached_tokens), ssize(prompt));
    while (prefix < reusable
           && impl->cached_tokens[size_t(prefix)]
                  == prompt[size_t(prefix)])
        ++prefix;

    const Index past = min(prefix, ssize(prompt) - 1);
    const Index count = ssize(prompt) - past;
    for (Index i = 0; i < count; ++i)
        impl->token_window[size_t(i)] =
            float(prompt[size_t(past + i)]);

    const bool speculative = impl->draft
        && sampling.temperature == 0.0f
        && sampling.repetition_penalty == 1.0f;

    using Clock = chrono::steady_clock;
    const auto prefill_start = Clock::now();
    vector<Index> sampling_history = prompt;
    sampling_history.reserve(size_t(min(impl->context_length,
                                        ssize(prompt) + maximum_tokens)));
    Index next = -1;
    {
        impl->run_prefill(impl->prefill, impl->prefill_inputs,
                          *impl->network, count, past);
        next = impl->sampler->sample_row(
            impl->prefill, 0, sampling, sampling_history);

        if (speculative)
        {
            for (Index i = 0; i < ssize(prompt); ++i)
                impl->token_window[size_t(i)] = float(prompt[size_t(i)]);
            impl->run_prefill(impl->draft->prefill, impl->draft->prefill_inputs,
                              *impl->draft->network, ssize(prompt), 0);
        }
    }
    const auto prefill_end = Clock::now();

    impl->cached_tokens = prompt;
    Index cache_length = ssize(prompt);

    GenerationParser parser(
        *impl->tokenizer,
        impl->chat_template->parser_spec(mode, *impl->tokenizer));

    ChatResponse response;
    response.prompt_tokens = ssize(prompt);
    response.prefill_tokens = count;
    response.prefill_milliseconds =
        chrono::duration<double, milli>(prefill_end - prefill_start).count();

    const auto decode_start = Clock::now();

    const auto emit = [&](Index token)
    {
        ++response.generated_tokens;
        sampling_history.push_back(token);

        if (parser.push(token, callback))
        {
            response.finish_reason = FinishReason::Stop;
            return false;
        }
        if (response.generated_tokens >= maximum_tokens)
        {
            response.finish_reason = FinishReason::MaximumTokens;
            return false;
        }
        return true;
    };

    if (speculative)
    {
        Index draft_cache = ssize(prompt);
        vector<Index>& proposals = impl->draft->proposals;

        while (true)
        {
            if (!emit(next)) break;

            if (cache_length >= impl->context_length)
            {
                response.finish_reason = FinishReason::ContextLimit;
                break;
            }

            const Index propose = min({impl->draft->propose_count,
                                       impl->context_length - cache_length - 1,
                                       maximum_tokens - response.generated_tokens});

            if (propose < 1)
            {
                Impl::stage_token(impl->token_device, next);
                const ForwardPropagation& decoded =
                    impl->run_decode(next, cache_length);
                impl->cached_tokens.push_back(next);
                ++cache_length;
                next = impl->sampler->sample_row(
                    decoded, 0, sampling, sampling_history);
                continue;
            }

            {

                proposals.clear();
                for (Index i = draft_cache; i < cache_length; ++i)
                {
                    Impl::stage_token(impl->draft->token_device,
                                      impl->cached_tokens[size_t(i)]);
                    impl->run_draft_decode(draft_cache);
                    ++draft_cache;
                }
                Impl::stage_token(impl->draft->token_device, next);
                impl->run_draft_decode(draft_cache);
                ++draft_cache;

                while (ssize(proposals) < propose)
                {
                    if (!proposals.empty())
                    {
                        impl->run_draft_decode(draft_cache);
                        ++draft_cache;
                    }
                    proposals.push_back(impl->draft->sampler->sample_row(
                        impl->draft->decode, 0, sampling, sampling_history));
                }
            }

            impl->token_window[0] = float(next);
            for (Index j = 0; j < propose; ++j)
                impl->token_window[size_t(1 + j)] = float(proposals[size_t(j)]);
            ForwardPropagation& verification =
                impl->run_target_verify(propose + 1, cache_length);

            Index accepted = 0;
            Index correction = -1;
            for (Index j = 0; j <= propose; ++j)
            {
                const Index choice = impl->sampler->sample_row(
                    verification, j, sampling, sampling_history);
                if (j < propose && choice == proposals[size_t(j)])
                {
                    ++accepted;
                    continue;
                }
                correction = choice;
                break;
            }

            impl->cached_tokens.push_back(next);
            ++cache_length;
            for (Index j = 0; j < accepted; ++j)
            {
                impl->cached_tokens.push_back(proposals[size_t(j)]);
                ++cache_length;
            }
            draft_cache = min(draft_cache, ssize(impl->cached_tokens));

            bool stopped = false;
            for (Index j = 0; j < accepted && !stopped; ++j)
                stopped = !emit(proposals[size_t(j)]);
            if (stopped) break;

            next = correction;
        }
    }
    else
    {
        while (true)
        {
            if (!emit(next)) break;

            if (cache_length >= impl->context_length)
            {
                response.finish_reason = FinishReason::ContextLimit;
                break;
            }

            const ForwardPropagation& decoded =
                impl->run_decode(next, cache_length);
            impl->cached_tokens.push_back(next);
            ++cache_length;
            next = impl->sampler->sample_row(
                decoded, 0, sampling, sampling_history);
        }
    }
    const auto decode_end = Clock::now();

    parser.finish(callback);

    response.reasoning = parser.get_reasoning();
    response.content = parser.get_content();
    response.reasoning_tokens = parser.get_reasoning_tokens();
    response.content_tokens = parser.get_content_tokens();
    response.control_tokens = parser.get_control_tokens();
    response.decode_milliseconds =
        chrono::duration<double, milli>(decode_end - decode_start).count();

    impl->messages = std::move(candidate);
    impl->messages.push_back({
        ChatRole::Assistant,
        response.content
    });

    return response;
}

void ChatSession::chat(const ChatOptions& options)
{
    cout << "Enter prompts. Empty line, 'exit' or 'quit' finishes.\n";

    string prompt;
    while (true)
    {
        cout << "\n> " << flush;
        if (!getline(cin, prompt) || contains({"", "exit", "quit"}, prompt))
            break;

        bool reasoning_started = false;
        bool content_started = false;
        const ChatResponse response = send(
            prompt, options,
            [&](const ChatDelta& delta)
            {
                if (delta.channel == GenerationChannel::Reasoning)
                {
                    if (!reasoning_started)
                    {
                        cout << "Thinking: ";
                        reasoning_started = true;
                    }
                }
                else if (!content_started)
                {
                    if (reasoning_started) cout << "\n";
                    cout << "Response: ";
                    content_started = true;
                }
                cout << delta.text << flush;
            });

        if (!content_started)
        {
            if (reasoning_started) cout << "\n";
            cout << "Response: " << response.content;
        }
        cout << "\n";
    }
    cout << "Bye!\n";
}

void ChatSession::set_messages(
    const vector<ChatMessage>& messages)
{
    throw_if(impl->classic != nullptr,
             "ChatSession::set_messages: this session has no "
             "semantic conversation history.");
    throw_if(!valid_complete_history(messages),
             "ChatSession::set_messages: expected leading system "
             "messages followed by complete user/assistant turns.");
    impl->messages = messages;
    impl->cached_tokens.clear();
}

const vector<ChatMessage>& ChatSession::get_messages() const noexcept
{
    return impl->messages;
}

void ChatSession::clear()
{
    impl->messages.clear();
    impl->cached_tokens.clear();
    if (impl->classic)
    {
        impl->classic->history.clear();
        return;
    }
    impl->prefill.past_length = 0;
    impl->decode.past_length = 0;
}

ReasoningMode ChatSession::resolve_reasoning_mode(
    const ReasoningMode mode) const
{
    if (impl->chat_template)
        return impl->chat_template->resolve_reasoning_mode(mode);

    throw_if(mode == ReasoningMode::Enabled,
             "ChatSession: this model does not support reasoning.");
    return ReasoningMode::Disabled;
}

SamplingConfig ChatSession::default_sampling(
    const ReasoningMode mode) const
{
    const ReasoningMode resolved = resolve_reasoning_mode(mode);
    if (impl->chat_template)
        return impl->chat_template->default_sampling(resolved);
    return {.temperature = 0.0f};
}

const ForwardPropagation&
ChatSession::get_decode_propagation() const
{
    if (impl->classic) return *impl->classic->propagation;
    return impl->gpu ? impl->decode : impl->prefill;
}

}
