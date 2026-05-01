//  OpenNN: Open Neural Networks Library
//  www.opennn.net
//
//  M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S   H E A D E R
//
//  Artificial Intelligence Techniques SL
//  artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"
#include "math_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

class MultiHeadAttention final : public Layer
{

public:

    MultiHeadAttention(const Shape& = Shape({0,0}),
                       Index = 0,
                       const string& = string());

    MultiHeadAttention(const Shape&,
                       const Shape&,
                       Index = 0,
                       const string& = string());

    Index get_query_sequence_length() const { return query_sequence_length; }
    Index get_source_sequence_length() const { return source_sequence_length; }
    Index get_embedding_dimension() const { return embedding_dimension; }
    Index get_heads_number() const { return heads_number; }
    Index get_head_dimension() const;

    Shape heads_shape(Index batch_size) const
    {
        return {batch_size, heads_number, query_sequence_length, get_head_dimension()};
    }

    Shape concat_shape(Index batch_size) const
    {
        return {batch_size, query_sequence_length, heads_number, get_head_dimension()};
    }

    float get_scaling_factor() const;

    Shape get_input_shape() const override;

    Shape get_output_shape() const override;

    vector<pair<Shape, Type>> get_parameter_specs() const override;

    vector<pair<Shape, Type>> get_forward_specs(const Index batch_size) const override
    {
        const Index head_dimension = get_head_dimension();
        const Index max_seq = max(query_sequence_length, source_sequence_length);
        const Type act = activation_dtype;

        const Shape attn_drop_shape = dropout.active()
            ? Shape{batch_size, heads_number, query_sequence_length, source_sequence_length}
            : Shape{};

        return {
            /*Query*/                        {{batch_size, heads_number, query_sequence_length, head_dimension},         act},
            /*Key*/                          {{batch_size, heads_number, source_sequence_length, head_dimension},        act},
            /*AttentionWeights*/             {{batch_size, heads_number, query_sequence_length, source_sequence_length}, act},
            /*AttentionWeightsDropped*/      {attn_drop_shape,                                                           act},
            /*ConcatenatedAttentionOutputs*/ {{batch_size, query_sequence_length, embedding_dimension},                  act},
            /*Value*/                        {{batch_size, heads_number, source_sequence_length, head_dimension},        act},
            /*PaddingMask*/                  {{batch_size, source_sequence_length},                                      act},
            /*TransposeScratch*/             {{batch_size, max_seq, embedding_dimension},                                act},
            /*AttentionOutputTransposed*/    {{batch_size, heads_number, query_sequence_length, head_dimension},         act},
            /*Output*/                       {{batch_size, query_sequence_length, embedding_dimension},                  act},
        };
    }

    vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const override
    {
        const Index head_dimension = get_head_dimension();
        const Type act = activation_dtype;

        return {
            /*InputQueryDelta*/        {{batch_size, query_sequence_length, embedding_dimension},                  act},
            /*InputSourceDelta*/       {{batch_size, source_sequence_length, embedding_dimension},                 act},
            /*AttentionWeightDelta*/   {{batch_size, heads_number, query_sequence_length, source_sequence_length}, act},
            /*ConcatenatedOutputDelta*/{{batch_size, query_sequence_length, embedding_dimension},                  act},
            /*QueryDelta (transposed)*/{{batch_size, heads_number, query_sequence_length, head_dimension},         act},
            /*KeyDelta (transposed)*/  {{batch_size, heads_number, source_sequence_length, head_dimension},        act},
            /*ValueDelta (transposed)*/{{batch_size, heads_number, source_sequence_length, head_dimension},        act},
        };
    }

    void set_input_shape(const Shape& new_input_shape) override
    {
        query_sequence_length = new_input_shape[0];
        embedding_dimension = new_input_shape[1];
    }

    void set(Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             bool = false,
             const string& = "multihead_attention_layer");

    void set_dropout_rate(float new_dropout_rate) { dropout.set_rate(new_dropout_rate); }

    void set_parameters_random() override;

    float* link_parameters(float* pointer) override;

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    void to_JSON(JsonWriter&) const override;
    void from_JSON(const JsonDocument&) override;

private:

    Index embedding_dimension;
    Index heads_number = 0;
    Index query_sequence_length = 0;
    Index source_sequence_length = 0;

    enum Parameters {QueryWeight, QueryBias, KeyWeight, KeyBias, ValueWeight, ValueBias,
                     ProjectionWeight, ProjectionBias};
    enum Forward {Input, Query, Key, AttentionWeights, AttentionWeightsDropped,
                  ConcatenatedAttentionOutputs, Value,
                  PaddingMask, TransposeScratch, AttentionOutputTransposed};
    enum Backward {OutputDelta, InputQueryDelta, InputSourceDelta,
                   AttentionWeightDelta, ConcatenatedOutputDelta,
                   QueryDelta, KeyDelta, ValueDelta};

    static bool is_self_attention(const vector<vector<TensorView>>& forward_views)
    {
        return forward_views[Input].size() == 1;
    }

    static const TensorView& get_query_input(const vector<vector<TensorView>>& forward_views)
    {
        return forward_views[Input][0];
    }

    static const TensorView& get_source_input(const vector<vector<TensorView>>& forward_views)
    {
        return is_self_attention(forward_views) ? forward_views[Input][0] : forward_views[Input][1];
    }

    bool use_causal_mask = false;

    MatrixR causal_mask;
    MatrixB key_mask;

    Combination query_projection;
    Combination key_projection;
    Combination value_projection;
    Combination output_projection;
    Dropout     dropout;
};

} 

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
