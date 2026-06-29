//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A N S F O R M E R   D E C O D E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "transformer_decoder.h"
#include "device_backend.h"
#include "random_utilities.h"
#include "string_utilities.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace opennn
{

namespace
{

constexpr Index pad_token_id     = 0;
constexpr Index unknown_token_id = 1;
constexpr Index start_token_id   = 2;
constexpr Index end_token_id     = 3;

bool is_printable_token(Index token_id)
{
    return token_id != pad_token_id
        && token_id != start_token_id
        && token_id != end_token_id;
}

TransformerDecoder::SamplingConfig greedy_config()
{
    TransformerDecoder::SamplingConfig config;
    config.temperature = 0.0f;
    return config;
}

Index sample_token(VectorR& probabilities,
                   const TransformerDecoder::SamplingConfig& sampling_config,
                   const vector<Index>& history)
{
    const Index vocabulary_size = probabilities.size();

    TransformerDecoder::SamplingConfig config = sampling_config;
    config.temperature = max(config.temperature, 0.0f);
    if (config.repetition_penalty <= 0.0f) config.repetition_penalty = 1.0f;
    config.top_k = max(config.top_k, Index(0));
    config.top_p = clamp(config.top_p, 0.0f, 1.0f);

    if (config.temperature == 0.0f)
    {
        Index best;
        probabilities.maxCoeff(&best);
        return best;
    }

    const VectorR original = probabilities;

    if (config.repetition_penalty != 1.0f)
        for (Index token_id : history)
            if (token_id >= 0 && token_id < vocabulary_size)
                probabilities(token_id) /= config.repetition_penalty;

    if (config.temperature != 1.0f)
    {
        const float inverse_temperature = 1.0f / config.temperature;
        for (Index i = 0; i < vocabulary_size; ++i)
            probabilities(i) = pow(max(probabilities(i), 0.0f), inverse_temperature);
    }

    if (config.top_k > 0 && config.top_k < vocabulary_size)
    {
        vector<pair<float, Index>> indexed(vocabulary_size);
        for (Index i = 0; i < vocabulary_size; ++i) indexed[i] = {probabilities(i), i};
        nth_element(indexed.begin(),
                    indexed.begin() + config.top_k,
                    indexed.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
        vector<bool> keep(vocabulary_size, false);
        for (Index i = 0; i < config.top_k; ++i) keep[indexed[i].second] = true;
        for (Index i = 0; i < vocabulary_size; ++i) if (!keep[i]) probabilities(i) = 0.0f;
    }

    if (config.top_p < 1.0f && config.top_p > 0.0f)
    {
        vector<pair<float, Index>> sorted_probabilities(vocabulary_size);
        float total = 0.0f;
        for (Index i = 0; i < vocabulary_size; ++i)
        {
            sorted_probabilities[i] = {probabilities(i), i};
            total += probabilities(i);
        }
        if (total > 0.0f)
        {
            ranges::sort(sorted_probabilities,
                         [](const auto& a, const auto& b) { return a.first > b.first; });
            float cumulative_probability = 0.0f;
            vector<bool> keep(vocabulary_size, false);
            for (const auto& [probability, token_id] : sorted_probabilities)
            {
                cumulative_probability += probability / total;
                keep[token_id] = true;
                if (cumulative_probability >= config.top_p) break;
            }
            for (Index i = 0; i < vocabulary_size; ++i) if (!keep[i]) probabilities(i) = 0.0f;
        }
    }

    const float sum = probabilities.sum();
    if (sum <= 0.0f)
    {
        Index best;
        original.maxCoeff(&best);
        return best;
    }

    const float sample_threshold = random_uniform(0.0f, sum);
    float cumulative_probability = 0.0f;
    for (Index i = 0; i < vocabulary_size; ++i)
    {
        cumulative_probability += probabilities(i);
        if (cumulative_probability >= sample_threshold) return i;
    }
    return vocabulary_size - 1;
}

}

TransformerDecoder::TransformerDecoder(Transformer& new_transformer,
                                       const LanguageDataset& new_language_dataset)
    : transformer(new_transformer),
      language_dataset(new_language_dataset)
{
    throw_if(!transformer.is_gpu() || !device::is_cuda_build(),
             "TransformerDecoder requires GPU configuration.");

    const Index input_sequence_length   = transformer.get_input_sequence_length();
    const Index decoder_sequence_length = transformer.get_decoder_sequence_length();

    throw_if(input_sequence_length != language_dataset.get_maximum_input_sequence_length(),
             format("TransformerDecoder: input sequence length mismatch (transformer={}, dataset={}).",
                    input_sequence_length, language_dataset.get_maximum_input_sequence_length()));
    throw_if(decoder_sequence_length != language_dataset.get_maximum_target_sequence_length(),
             format("TransformerDecoder: decoder sequence length mismatch (transformer={}, dataset={}).",
                    decoder_sequence_length, language_dataset.get_maximum_target_sequence_length()));

    throw_if(language_dataset.get_input_vocabulary_map().empty(),
             "TransformerDecoder: dataset input vocabulary is empty.");
    throw_if(language_dataset.get_target_vocabulary().empty(),
             "TransformerDecoder: dataset target vocabulary is empty.");

    transformer.copy_parameters_device();
    transformer.link_parameters();
    transformer.copy_states_device();
    transformer.link_states();

    identify_layer_ranges();

    constexpr Index batch_size = 1;

    const Index source_bytes = get_aligned_bytes(batch_size * input_sequence_length, Type::FP32);
    const Index target_bytes = get_aligned_bytes(batch_size * decoder_sequence_length, Type::FP32);

    arena.resize_bytes(source_bytes + target_bytes, Device::CUDA);

    char* const base = arena.as<char>();
    source_ids_device = TensorView(base,
                                   {batch_size, input_sequence_length},
                                   Type::FP32,
                                   Device::CUDA);
    target_ids_device = TensorView(base + source_bytes,
                                   {batch_size, decoder_sequence_length},
                                   Type::FP32,
                                   Device::CUDA);

    forward_propagation = make_unique<ForwardPropagation>(batch_size, &transformer);

    source_ids = Tensor2(batch_size, input_sequence_length);
    target_ids = Tensor2(batch_size, decoder_sequence_length);
    history.reserve(decoder_sequence_length);

    inputs = {target_ids_device, source_ids_device};

    const Index vocabulary_size = transformer.get_layers().back()->get_output_shape().back();
    distribution = VectorR::Zero(vocabulary_size);
    bf16_staging.assign(static_cast<size_t>(vocabulary_size), 0);
}

void TransformerDecoder::identify_layer_ranges()
{
    const auto& layers = transformer.get_layers();
    const Index layers_number = static_cast<Index>(layers.size());

    throw_if(layers_number < 4,
             format("TransformerDecoder: unexpected layer count ({}). Transformer must have at least decoder_embedding + encoder_embedding + cross_attention + output_projection.",
                    layers_number));

    throw_if(layers[0]->get_label() != "decoder_embedding",
             format("TransformerDecoder: layer 0 expected to be 'decoder_embedding', found '{}'.", layers[0]->get_label()));
    decoder_embedding_index = 0;

    throw_if(layers[1]->get_label() != "encoder_embedding",
             format("TransformerDecoder: layer 1 expected to be 'encoder_embedding', found '{}'.", layers[1]->get_label()));
    encoder_embedding_index = 1;

    Index first_cross_attention_index = -1;
    for (Index i = 0; i < layers_number; ++i)
    {
        if (layers[i]->get_label().starts_with("cross_attention_"))
        {
            first_cross_attention_index = i;
            break;
        }
    }
    throw_if(first_cross_attention_index < 0,
             "TransformerDecoder: no 'cross_attention_*' layer found.");

    const vector<Index>& cross_sources = transformer.get_source_layers()[first_cross_attention_index];
    throw_if(cross_sources.size() < 2 || cross_sources[1] < 0,
             "TransformerDecoder: first cross_attention layer must have 2 valid inputs (decoder, encoder).");

    encoder_last_index = cross_sources[1];

    decoder_first_index = encoder_last_index + 1;
    throw_if(decoder_first_index >= layers_number,
             "TransformerDecoder: decoder stack first index out of range.");
    throw_if(layers[decoder_first_index]->get_label() != "decoder_self_attention_1",
             format("TransformerDecoder: layer after encoder expected to be 'decoder_self_attention_1', found '{}'.",
                    layers[decoder_first_index]->get_label()));

    output_projection_index = layers_number - 1;
    throw_if(layers[output_projection_index]->get_label() != "output_projection",
             format("TransformerDecoder: last layer expected to be 'output_projection', found '{}'.",
                    layers[output_projection_index]->get_label()));
}

void TransformerDecoder::reset_per_prompt_state()
{
    target_ids.setConstant(pad_token_id);
    target_ids(0, 0) = start_token_id;
    history.clear();

    cudaStream_t stream = Backend::get_compute_stream();
    device::copy_async(target_ids_device.data,
                       target_ids.data(),
                       target_ids_device.byte_size(),
                       device::CopyKind::HostToDevice,
                       stream);
}

void TransformerDecoder::encode_source(const string& source)
{
    const auto& input_vocabulary_map = language_dataset.get_input_vocabulary_map();

    const Index input_sequence_length = transformer.get_input_sequence_length();

    source_ids.setConstant(pad_token_id);
    source_ids(0, 0) = start_token_id;

    const vector<string> source_tokens = tokenize(source);
    Index write_index = 1;
    for (const string& token : source_tokens)
    {
        if (write_index >= input_sequence_length) break;

        const auto it = input_vocabulary_map.find(token);
        source_ids(0, write_index) = (it != input_vocabulary_map.end())
                                         ? static_cast<float>(it->second)
                                         : unknown_token_id;
        ++write_index;
    }
    if (write_index < input_sequence_length)
        source_ids(0, write_index) = end_token_id;

    cudaStream_t stream = Backend::get_compute_stream();
    device::copy_async(source_ids_device.data,
                       source_ids.data(),
                       source_ids_device.byte_size(),
                       device::CopyKind::HostToDevice,
                       stream);
    transformer.forward_propagate(inputs, *forward_propagation, false,
                                  encoder_embedding_index,
                                  encoder_last_index);
}

Index TransformerDecoder::decode_step([[maybe_unused]] Index step_index,
                                       const SamplingConfig& config)
{
    cudaStream_t stream = Backend::get_compute_stream();

    transformer.forward_propagate(inputs, *forward_propagation, false,
                                  decoder_embedding_index,
                                  decoder_embedding_index);

    transformer.forward_propagate(inputs, *forward_propagation, false,
                                  decoder_first_index,
                                  output_projection_index);

    const TensorView output_view = forward_propagation->get_outputs();
    const Index vocabulary_size = output_view.shape[2];
    const Index slice_offset = (step_index - 1) * vocabulary_size;
    if (output_view.is_bf16())
    {
        device::copy_async(bf16_staging.data(),
                           output_view.as<bfloat16>() + slice_offset,
                           vocabulary_size * Index(sizeof(uint16_t)),
                           device::CopyKind::DeviceToHost,
                           stream);
        device::synchronize(stream);
        for (Index i = 0; i < vocabulary_size; ++i)
        {
            const uint32_t bits = static_cast<uint32_t>(bf16_staging[size_t(i)]) << 16;
            memcpy(&distribution(i), &bits, sizeof(float));
        }
    }
    else if (output_view.is_fp32())
    {
        device::copy_async(distribution.data(),
                           output_view.as<float>() + slice_offset,
                           vocabulary_size * Index(sizeof(float)),
                           device::CopyKind::DeviceToHost,
                           stream);
        device::synchronize(stream);
    }
    else
    {
        throw runtime_error("TransformerDecoder: unsupported output dtype.");
    }

    return sample_token(distribution, config, history);
}

string TransformerDecoder::assemble_output_string() const
{
    const vector<string>& output_vocabulary = language_dataset.get_target_vocabulary();
    const Index decoder_sequence_length = transformer.get_decoder_sequence_length();

    string result;
    for (Index i = 1; i < decoder_sequence_length; ++i)
    {
        const Index token_id = static_cast<Index>(target_ids(0, i));
        if (token_id == end_token_id || token_id == pad_token_id) break;

        if (token_id < 0 || token_id >= ssize(output_vocabulary)) continue;

        if (!result.empty()) result += " ";
        result += output_vocabulary[size_t(token_id)];
    }
    return result;
}

string TransformerDecoder::decode(const string& source)
{
    return decode(source, greedy_config(), TokenCallback{});
}

string TransformerDecoder::decode(const string& source, const SamplingConfig& config)
{
    return decode(source, config, TokenCallback{});
}

string TransformerDecoder::decode(const string& source, const TokenCallback& on_token)
{
    return decode(source, greedy_config(), on_token);
}

string TransformerDecoder::decode(const string& source,
                                   const SamplingConfig& config,
                                   const TokenCallback& on_token)
{
    reset_per_prompt_state();
    encode_source(source);

    const Index decoder_sequence_length = transformer.get_decoder_sequence_length();
    const Index generation_limit = (config.maximum_tokens > 0)
        ? min(config.maximum_tokens + Index(1), decoder_sequence_length)
        : decoder_sequence_length;

    const vector<string>& output_vocabulary = language_dataset.get_target_vocabulary();

    for (Index i = 1; i < generation_limit; ++i)
    {
        const Index next_token_id = decode_step(i, config);

        target_ids(0, i) = static_cast<float>(next_token_id);
        history.push_back(next_token_id);

        cudaStream_t stream = Backend::get_compute_stream();
        device::copy_async(target_ids_device.as<float>() + i,
                           &target_ids(0, i),
                           Index(sizeof(float)),
                           device::CopyKind::HostToDevice,
                           stream);

        if (next_token_id == end_token_id)
            break;

        if (on_token && is_printable_token(next_token_id)
            && next_token_id >= 0 && next_token_id < ssize(output_vocabulary))
            on_token(output_vocabulary[size_t(next_token_id)]);
    }

    return assemble_output_string();
}

string TransformerDecoder::decode_to_stream(const string& source, ostream& out)
{
    return decode_to_stream(source, greedy_config(), out);
}

string TransformerDecoder::decode_to_stream(const string& source,
                                             const SamplingConfig& config,
                                             ostream& out)
{
    bool first_token = true;

    return decode(source, config, [&](const string& token)
    {
        const bool is_punctuation = token.size() == 1
            && string_view(",.!?;:").find(token[0]) != string_view::npos;

        if (!first_token && !is_punctuation) out << ' ';
        out << token << flush;
        first_token = false;
    });
}


void TransformerDecoder::chat()
{
    chat(greedy_config());
}


void TransformerDecoder::chat(const SamplingConfig& config)
{
    cout << "Enter prompts. Empty line or Ctrl+D to exit.\n";

    string prompt_line;
    while (true)
    {
        cout << "\n> " << flush;
        if (!getline(cin, prompt_line) || prompt_line.empty()) break;

        decode_to_stream(prompt_line, config, cout);
        cout << "\n";
    }
    cout << "Bye!\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
