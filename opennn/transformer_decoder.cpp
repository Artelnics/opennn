//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A N S F O R M E R   D E C O D E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "transformer_decoder.h"
#include "random_utilities.h"
#include "string_utilities.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace opennn
{

namespace
{

constexpr float pad_token_id     = 0.0f;
constexpr float unknown_token_id = 1.0f;
constexpr float start_token_id   = 2.0f;
constexpr float end_token_id     = 3.0f;

bool is_printable_token(Index token_id)
{
    return token_id != static_cast<Index>(pad_token_id)
        && token_id != static_cast<Index>(start_token_id)
        && token_id != static_cast<Index>(end_token_id);
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
    if (config.temperature        < 0.0f)  config.temperature        = 0.0f;
    if (config.repetition_penalty <= 0.0f) config.repetition_penalty = 1.0f;
    if (config.top_k              < 0)     config.top_k              = 0;
    if (config.top_p              < 0.0f)  config.top_p              = 0.0f;
    if (config.top_p              > 1.0f)  config.top_p              = 1.0f;

    if (config.temperature == 0.0f)
    {
        Index best = 0;
        for (Index i = 1; i < vocabulary_size; ++i)
            if (probabilities(i) > probabilities(best)) best = i;
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
            sort(sorted_probabilities.begin(), sorted_probabilities.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            float cumulative_probability = 0.0f;
            vector<bool> keep(vocabulary_size, false);
            for (size_t i = 0; i < sorted_probabilities.size(); ++i)
            {
                cumulative_probability += sorted_probabilities[i].first / total;
                keep[sorted_probabilities[i].second] = true;
                if (cumulative_probability >= config.top_p) break;
            }
            for (Index i = 0; i < vocabulary_size; ++i) if (!keep[i]) probabilities(i) = 0.0f;
        }
    }

    float sum = 0.0f;
    for (Index i = 0; i < vocabulary_size; ++i) sum += probabilities(i);
    if (sum <= 0.0f)
    {
        Index best = 0;
        for (Index i = 1; i < vocabulary_size; ++i)
            if (original(i) > original(best)) best = i;
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
    if (!Configuration::instance().is_gpu())
        throw runtime_error("TransformerDecoder requires GPU configuration.");

    const Index input_sequence_length   = transformer.get_input_sequence_length();
    const Index decoder_sequence_length = transformer.get_decoder_sequence_length();

    if (input_sequence_length != language_dataset.get_maximum_input_sequence_length())
        throw runtime_error("TransformerDecoder: input sequence length mismatch (transformer="
                            + to_string(input_sequence_length) + ", dataset="
                            + to_string(language_dataset.get_maximum_input_sequence_length()) + ").");
    if (decoder_sequence_length != language_dataset.get_maximum_target_sequence_length())
        throw runtime_error("TransformerDecoder: decoder sequence length mismatch (transformer="
                            + to_string(decoder_sequence_length) + ", dataset="
                            + to_string(language_dataset.get_maximum_target_sequence_length()) + ").");

    if (language_dataset.get_input_vocabulary_map().empty())
        throw runtime_error("TransformerDecoder: dataset input vocabulary is empty.");
    if (language_dataset.get_target_inverse_vocabulary_map().empty())
        throw runtime_error("TransformerDecoder: dataset target vocabulary is empty.");

#ifdef OPENNN_HAS_CUDA
    transformer.copy_parameters_device();
    transformer.link_parameters();
    transformer.copy_states_device();
    transformer.link_states();
#endif

    identify_layer_ranges();

    constexpr Index batch_size = 1;

    const Index source_bytes = get_aligned_bytes(batch_size * input_sequence_length, Type::FP32);
    const Index target_bytes = get_aligned_bytes(batch_size * decoder_sequence_length, Type::FP32);
    const Index total_bytes  = source_bytes + target_bytes;

    arena.resize_bytes(total_bytes, Device::CUDA);

    char* base = arena.as<char>();
    source_ids_device = TensorView(base,
                                   {batch_size, input_sequence_length},
                                   Type::FP32);
    target_ids_device = TensorView(base + source_bytes,
                                   {batch_size, decoder_sequence_length},
                                   Type::FP32);

    forward_propagation = make_unique<ForwardPropagation>(batch_size, &transformer);

    source_ids = Tensor2(batch_size, input_sequence_length);
    target_ids = Tensor2(batch_size, decoder_sequence_length);
    history.reserve(decoder_sequence_length);

    inputs = {target_ids_device, source_ids_device};

    const auto& last_layer = transformer.get_layers().back();
    const Index vocabulary_size = last_layer->get_output_shape().back();
    distribution = VectorR::Zero(vocabulary_size);
    bf16_staging.assign(static_cast<size_t>(vocabulary_size), 0);
}

void TransformerDecoder::identify_layer_ranges()
{
    const auto& layers = transformer.get_layers();
    const Index layers_number = static_cast<Index>(layers.size());

    if (layers_number < 4)
        throw runtime_error("TransformerDecoder: unexpected layer count (" +
                            to_string(layers_number) + "). Transformer must have at least decoder_embedding + encoder_embedding + cross_attention + output_projection.");

    if (layers[0]->get_label() != "decoder_embedding")
        throw runtime_error("TransformerDecoder: layer 0 expected to be 'decoder_embedding', found '" + layers[0]->get_label() + "'.");
    decoder_embedding_layer_index = 0;

    if (layers[1]->get_label() != "encoder_embedding")
        throw runtime_error("TransformerDecoder: layer 1 expected to be 'encoder_embedding', found '" + layers[1]->get_label() + "'.");
    encoder_embedding_layer_index = 1;

    const auto& layer_input_indices = transformer.get_layer_input_indices();
    Index first_cross_attention_index = -1;
    for (Index i = 0; i < layers_number; ++i)
    {
        const string& name = layers[i]->get_label();
        if (name.compare(0, 16, "cross_attention_") == 0)
        {
            first_cross_attention_index = i;
            break;
        }
    }
    if (first_cross_attention_index < 0)
        throw runtime_error("TransformerDecoder: no 'cross_attention_*' layer found.");

    const vector<Index>& cross_inputs = layer_input_indices[first_cross_attention_index];
    if (cross_inputs.size() < 2 || cross_inputs[1] < 0)
        throw runtime_error("TransformerDecoder: first cross_attention layer must have 2 valid inputs (decoder, encoder).");

    encoder_last_layer_index = cross_inputs[1];

    decoder_stack_first_layer_index = encoder_last_layer_index + 1;
    if (decoder_stack_first_layer_index >= layers_number)
        throw runtime_error("TransformerDecoder: decoder stack first index out of range.");
    if (layers[decoder_stack_first_layer_index]->get_label() != "decoder_self_attention_1")
        throw runtime_error("TransformerDecoder: layer after encoder expected to be 'decoder_self_attention_1', found '" +
                            layers[decoder_stack_first_layer_index]->get_label() + "'.");

    output_projection_layer_index = layers_number - 1;
    if (layers[output_projection_layer_index]->get_label() != "output_projection")
        throw runtime_error("TransformerDecoder: last layer expected to be 'output_projection', found '" +
                            layers[output_projection_layer_index]->get_label() + "'.");
}

void TransformerDecoder::reset_per_prompt_state()
{
    target_ids.setConstant(pad_token_id);
    target_ids(0, 0) = start_token_id;
    history.clear();

#ifdef OPENNN_HAS_CUDA
    // One full H2D per prompt; subsequent steps update only the new token slot.
    const Index decoder_sequence_length = transformer.get_decoder_sequence_length();
    constexpr Index batch_size = 1;
    cudaStream_t stream = Backend::get_compute_stream();
    CHECK_CUDA(cudaMemcpyAsync(target_ids_device.data,
                               target_ids.data(),
                               batch_size * decoder_sequence_length * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
#endif
}

void TransformerDecoder::encode_source(const string& source)
{
    const auto& input_vocabulary_map = language_dataset.get_input_vocabulary_map();

    const Index input_sequence_length = transformer.get_input_sequence_length();

    source_ids.setConstant(pad_token_id);
    source_ids(0, 0) = start_token_id;

    const vector<string> source_tokens = tokenize(source);
    Index write_index = 1;
    for (size_t i = 0; i < source_tokens.size() && write_index < input_sequence_length; ++i, ++write_index)
    {
        const auto it = input_vocabulary_map.find(source_tokens[i]);
        source_ids(0, write_index) = (it != input_vocabulary_map.end())
                                         ? static_cast<float>(it->second)
                                         : unknown_token_id;
    }
    if (write_index < input_sequence_length)
        source_ids(0, write_index) = end_token_id;

#ifdef OPENNN_HAS_CUDA
    constexpr Index batch_size = 1;
    cudaStream_t stream = Backend::get_compute_stream();
    CHECK_CUDA(cudaMemcpyAsync(source_ids_device.data,
                               source_ids.data(),
                               batch_size * input_sequence_length * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
#endif
    transformer.forward_propagate(inputs, *forward_propagation, false,
                                  encoder_embedding_layer_index,
                                  encoder_last_layer_index);
}

Index TransformerDecoder::decode_step([[maybe_unused]] Index step_index,
                                       const SamplingConfig& config)
{
#ifdef OPENNN_HAS_CUDA
    cudaStream_t stream = Backend::get_compute_stream();
#endif

    transformer.forward_propagate(inputs, *forward_propagation, false,
                                  decoder_embedding_layer_index,
                                  decoder_embedding_layer_index);

    transformer.forward_propagate(inputs, *forward_propagation, false,
                                  decoder_stack_first_layer_index,
                                  output_projection_layer_index);

#ifdef OPENNN_HAS_CUDA
    const TensorView output_view = forward_propagation->get_outputs();
    const Index vocabulary_size = output_view.shape[2];
    const Index slice_offset = (step_index - 1) * vocabulary_size;
    if (output_view.type == Type::BF16)
    {
        CHECK_CUDA(cudaMemcpyAsync(bf16_staging.data(),
                                   output_view.as<bfloat16>() + slice_offset,
                                   vocabulary_size * sizeof(uint16_t),
                                   cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        for (Index k = 0; k < vocabulary_size; ++k)
        {
            const uint32_t bits = static_cast<uint32_t>(bf16_staging[size_t(k)]) << 16;
            memcpy(&distribution(k), &bits, sizeof(float));
        }
    }
    else if (output_view.type == Type::FP32)
    {
        CHECK_CUDA(cudaMemcpyAsync(distribution.data(),
                                   output_view.as<float>() + slice_offset,
                                   vocabulary_size * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    else
    {
        throw runtime_error("TransformerDecoder: unsupported output dtype.");
    }
#endif

    return sample_token(distribution, config, history);
}

string TransformerDecoder::assemble_output_string() const
{
    const auto& output_inverse_vocabulary_map = language_dataset.get_target_inverse_vocabulary_map();
    const Index decoder_sequence_length = transformer.get_decoder_sequence_length();

    string result;
    for (Index i = 1; i < decoder_sequence_length; ++i)
    {
        const Index token_id = static_cast<Index>(target_ids(0, i));
        if (token_id == static_cast<Index>(end_token_id) ||
            token_id == static_cast<Index>(pad_token_id)) break;

        const auto it = output_inverse_vocabulary_map.find(token_id);
        if (it == output_inverse_vocabulary_map.end()) continue;

        if (!result.empty()) result += " ";
        result += it->second;
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

    const auto& output_inverse_vocabulary_map = language_dataset.get_target_inverse_vocabulary_map();

    for (Index i = 1; i < generation_limit; ++i)
    {
        const Index next_token_id = decode_step(i, config);

        target_ids(0, i) = static_cast<float>(next_token_id);
        history.push_back(next_token_id);

#ifdef OPENNN_HAS_CUDA
        cudaStream_t stream = Backend::get_compute_stream();
        CHECK_CUDA(cudaMemcpyAsync(target_ids_device.as<float>() + i,
                                   &target_ids(0, i),
                                   sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
#endif

        if (next_token_id == static_cast<Index>(end_token_id))
            break;

        if (on_token && is_printable_token(next_token_id))
        {
            const auto it = output_inverse_vocabulary_map.find(next_token_id);
            if (it != output_inverse_vocabulary_map.end())
                on_token(it->second);
        }
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
        // Single-char punctuation never gets a leading space.
        const bool is_punctuation = token.size() == 1
            && string(",.!?;:").find(token[0]) != string::npos;

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
        if (!getline(cin, prompt_line)) break;
        if (prompt_line.empty()) break;

        decode_to_stream(prompt_line, config, cout);
        cout << "\n";
    }
    cout << "Bye!\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
