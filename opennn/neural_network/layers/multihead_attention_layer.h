//  OpenNN: Open Neural Networks Library
//  www.opennn.net
//
//  M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//  artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/layers/layer.h"
#include "opennn/neural_network/operators/attention_operator.h"
#include "opennn/neural_network/operators/combination_operator.h"
#include "opennn/neural_network/operators/multihead_projection_operator.h"

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
    Index get_sources_number() const noexcept override { return cross_attention ? 2 : 1; }
    vector<TensorSpec> get_forward_specs(Index) const override;
    vector<TensorSpec> get_backward_specs(Index) const override;

    ForwardSlotKind get_forward_slot_kind(size_t spec) const override
    {
        if (spec == size_t(TransposeScratch) - 1 || spec == size_t(SdpaQkvPack) - 1)
            return ForwardSlotKind::Transient;
        if (spec == size_t(AttentionWeightsDropped) - 1)
            return ForwardSlotKind::TrainingOnly;
        return ForwardSlotKind::Pooled;
    }
    bool backward_uses_forward_output() const noexcept override { return false; }
    bool preserves_output_delta_during_backward() const noexcept override { return true; }

    void set(Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             bool = false,
             const string& = "multihead_attention_layer");

    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 2); }

    void apply_input_shape(const Shape&) override;
    void on_compute_dtype_changed() override;

    void set_dropout_rate(float new_dropout_rate) { attention.dropout.set_rate(new_dropout_rate); }

    void set_zero_padded_queries(bool);

    static constexpr Index default_sdpa_min_sequence_length = 192;

    void set_sdpa_auto(bool);
    void set_sdpa_min_sequence_length(Index);

    bool should_use_sdpa() const;

    // Applies should_use_sdpa() to the attention and to the three projections
    // (the head layout follows the attention path).
    void apply_sdpa_choice();

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    Index embedding_dimension = 0;
    Index heads_number = 0;
    Index query_sequence_length = 0;
    Index source_sequence_length = 0;

    bool  sdpa_auto = true;
    Index sdpa_min_sequence_length = default_sdpa_min_sequence_length;
    bool  cross_attention = false;

    MultiHeadProjectionOperator query_projection;
    MultiHeadProjectionOperator key_projection;
    MultiHeadProjectionOperator value_projection;
    CombinationOperator         output_projection;
    AttentionOperator           attention;

    enum Forward {Input, Query, Key, AttentionWeights, AttentionWeightsDropped,
                  ConcatenatedAttentionOutputs, Value, TransposeScratch, SdpaQkvPack, Output};
    enum Backward {
        OutputDelta,
        InputQueryDelta,
        InputSourceDelta,
        AttentionWeightDelta,
        ValueHeadDelta,
        ConcatenatedOutputDelta,
        QueryHeadDelta,
        KeyHeadDelta,
        SdpaOutputGradBF16,
        SdpaQueryGradBF16,
        SdpaKeyGradBF16,
        SdpaValueGradBF16,
        SdpaQueryRematBF16,
        SdpaKeyRematBF16,
        SdpaValueRematBF16,
        SdpaOutputRematBF16
    };
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
