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
#include "text_generation_dataset.h"
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

    using TokenCallback = function<void(const string&)>;

    TransformerDecoder(Transformer&, const LanguageDataset&);
    TransformerDecoder(TextGenerationNetwork&, const TextGenerationDataset&);
    TransformerDecoder(const TransformerDecoder&) = delete;
    TransformerDecoder& operator=(const TransformerDecoder&) = delete;
    ~TransformerDecoder() = default;

    string decode(const string&);
    string decode(const string&, const SamplingConfig&);
    string decode(const string&, const TokenCallback&);
    string decode(const string&, const SamplingConfig&, const TokenCallback&);

    string decode_to_stream(const string&, ostream&);
    string decode_to_stream(const string&, const SamplingConfig&, ostream&);

    string generate(const string&);
    string generate(const string&, const SamplingConfig&);
    string generate(const string&, const TokenCallback&);
    string generate(const string&, const SamplingConfig&, const TokenCallback&);

    string generate_to_stream(const string&, ostream&);
    string generate_to_stream(const string&, const SamplingConfig&, ostream&);

    void chat();
    void chat(const SamplingConfig&);

private:

    NeuralNetwork& network;
    const LanguageDataset* language_dataset = nullptr;
    const TextGenerationDataset* generation_dataset = nullptr;

    bool decoder_only = false;

    Index input_sequence_length = 0;
    Index sequence_length = 0;

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

    Index decoder_embedding_index = -1;
    Index encoder_embedding_index = -1;
    Index encoder_last_index      = -1;
    Index decoder_first_index     = -1;
    Index output_projection_index = -1;

    void identify_layer_ranges();
    void identify_layer_ranges_decoder_only();
    void encode_source(const string&);
    void read_distribution(Index);
    Index decode_step(Index, const SamplingConfig&);
    Index generate_step(const vector<Index>&, const SamplingConfig&);
    void reset_per_prompt_state();
    string assemble_output_string() const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
