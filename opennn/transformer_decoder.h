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

class TransformerDecoder
{
public:

    struct SamplingConfig
    {
        float temperature = 1.0f;
        Index top_k = 0;
        float top_p = 1.0f;
        float repetition_penalty = 1.0f;
        Index maximum_tokens = 0;
    };

    using TokenCallback = function<void(const string& token)>;

    TransformerDecoder(Transformer&, const LanguageDataset&);
    TransformerDecoder(const TransformerDecoder&) = delete;
    TransformerDecoder& operator=(const TransformerDecoder&) = delete;
    ~TransformerDecoder() = default;

    string decode(const string& source);
    string decode(const string& source, const SamplingConfig& config);
    string decode(const string& source, const TokenCallback& on_token);
    string decode(const string& source, const SamplingConfig& config, const TokenCallback& on_token);

    string decode_to_stream(const string& source, ostream& out);
    string decode_to_stream(const string& source, const SamplingConfig& config, ostream& out);

    // Interactive REPL: reads prompts from cin, streams predictions to
    // cout. Empty line or Ctrl+D exits.
    void chat();
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
