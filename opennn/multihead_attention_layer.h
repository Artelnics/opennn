//  OpenNN: Open Neural Networks Library
//  www.opennn.net
//
//  M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//  artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"
#include "tensor_operations.h"

namespace opennn
{

class MultiHeadAttention final : public Layer
{
public:

    MultiHeadAttention(const Shape& = Shape({0, 0}),
                       Index = 0,
                       const string& = {});

    MultiHeadAttention(const Shape&,
                       const Shape&,
                       Index = 0,
                       const string& = {});

    Shape get_input_shape() const noexcept override;
    Shape get_output_shape() const override;

    Index get_query_sequence_length() const noexcept { return query_sequence_length; }
    Index get_source_sequence_length() const noexcept { return source_sequence_length; }
    Index get_embedding_dimension() const noexcept { return embedding_dimension; }
    Index get_heads_number() const noexcept { return heads_number; }
    Index get_head_dimension() const noexcept
    {
        return (heads_number == 0) ? 0 : Index(embedding_dimension / heads_number);
    }
    vector<TensorSpec> get_forward_specs(Index) const override;
    vector<TensorSpec> get_backward_specs(Index) const override;

    bool is_forward_slot_transient(size_t spec) const override
    {
        return spec == size_t(TransposeScratch) - 1;
    }

    void set(Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             bool = false,
             const string& = "multihead_attention_layer");

    void set_input_shape(const Shape&) override;
    void on_compute_dtype_changed() override;

    void set_dropout_rate(float new_dropout_rate) { attention.dropout.set_rate(new_dropout_rate); }

    void set_zero_padded_queries(bool);
    bool get_zero_padded_queries() const noexcept { return attention.zero_padded_queries; }

    static constexpr Index default_sdpa_min_sequence_length = 192;

    void set_sdpa_auto(bool);
    void set_sdpa_min_sequence_length(Index);

    bool  get_sdpa_auto() const noexcept { return sdpa_auto; }
    Index get_sdpa_min_sequence_length() const noexcept { return sdpa_min_sequence_length; }

    bool should_use_sdpa() const;

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    Index embedding_dimension = 0;
    Index heads_number = 0;
    Index query_sequence_length = 0;
    Index source_sequence_length = 0;

    bool  sdpa_auto = true;
    Index sdpa_min_sequence_length = default_sdpa_min_sequence_length;

    MultiHeadProjectionOperator query_projection;
    MultiHeadProjectionOperator key_projection;
    MultiHeadProjectionOperator value_projection;
    CombinationOperator         output_projection;
    AttentionOperator           attention;
    MergeOperator               merge;

    enum Forward {Input, Query, Key, AttentionWeights, AttentionWeightsDropped,
                  ConcatenatedAttentionOutputs, Value, TransposeScratch, Output};
    enum Backward {
        OutputDelta,
        InputQueryDelta,
        InputSourceDelta,
        AttentionWeightDelta,
        ValueHeadDelta,
        ConcatenatedOutputDelta,
        QueryHeadDelta,
        KeyHeadDelta
    };
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
