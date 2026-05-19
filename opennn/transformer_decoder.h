//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A N S F O R M E R   D E C O D E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <functional>

#include "forward_propagation.h"
#include "language_dataset.h"
#include "standard_networks.h"

namespace opennn
{

/// @brief Drives token-by-token inference of a Transformer model with configurable sampling strategies.
class TransformerDecoder
{
public:

    /// @brief Sampling parameters that control how the next token is drawn from the model output distribution.
    struct SamplingConfig
    {
        float temperature = 1.0f;
        Index top_k = 0;
        float top_p = 1.0f;
        float repetition_penalty = 1.0f;
        Index maximum_tokens = 0;
    };

    /// @brief Callback invoked for each token emitted during streaming decoding.
    using TokenCallback = function<void(const string& token)>;

    /// @brief Builds the decoder bound to a Transformer model and the language dataset providing its vocabulary.
    TransformerDecoder(Transformer&, const LanguageDataset&);
    TransformerDecoder(const TransformerDecoder&) = delete;
    TransformerDecoder& operator=(const TransformerDecoder&) = delete;
    ~TransformerDecoder() = default;

    /// @brief Generates a completion for the given source using the default sampling configuration.
    string decode(const string& source);

    /// @brief Generates a completion for the given source using the supplied sampling configuration.
    string decode(const string& source, const SamplingConfig& config);

    /// @brief Generates a completion for the given source and invokes the callback for each emitted token.
    string decode(const string& source, const TokenCallback& on_token);

    /// @brief Generates a completion using the given sampling configuration and streams tokens via the callback.
    string decode(const string& source, const SamplingConfig& config, const TokenCallback& on_token);

    /// @brief Generates a completion for the given source and writes each emitted token to the output stream.
    string decode_to_stream(const string& source, ostream& out);

    /// @brief Generates a completion using the given sampling configuration and writes each token to the stream.
    string decode_to_stream(const string& source, const SamplingConfig& config, ostream& out);

    /// @brief Runs an interactive REPL: reads prompts from cin, streams predictions to cout, exits on empty line / Ctrl+D.
    void chat();

    /// @brief Runs the interactive REPL with the supplied sampling configuration.
    void chat(const SamplingConfig& config);

private:

    Transformer& transformer;
    const LanguageDataset& language_dataset;

    Buffer arena{Device::CUDA};
    TensorView source_ids_device;
    TensorView target_ids_device;

    unique_ptr<ForwardPropagation> forward_propagation;

    vector<TensorView> inputs;

    Tensor2 source_ids;
    Tensor2 target_ids;
    vector<Index> history;
    VectorR distribution;
    vector<uint16_t> bf16_staging;

    Index decoder_embedding_layer_index   = -1;
    Index encoder_embedding_layer_index   = -1;
    Index encoder_last_layer_index        = -1;
    Index decoder_stack_first_layer_index = -1;
    Index output_projection_layer_index   = -1;

    void identify_layer_ranges();
    void encode_source(const string& source);
    Index decode_step(Index step_index, const SamplingConfig& config);
    void reset_per_prompt_state();
    string assemble_output_string() const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
