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

    MultiHeadAttention(const Shape& = Shape({0, 0}),
                       Index = 0,
                       const string& = string());

    MultiHeadAttention(const Shape& new_query_dimensions,
                       const Shape& new_source_dimensions,
                       Index = 0,
                       const string& = string());

    Shape get_input_shape() const override;
    Shape get_output_shape() const override;

    Index get_query_sequence_length() const { return query_sequence_length; }
    Index get_source_sequence_length() const { return source_sequence_length; }
    Index get_embedding_dimension() const { return embedding_dimension; }
    Index get_heads_number() const { return heads_number; }
    Index get_head_dimension() const
    {
        return (heads_number == 0) ? 0 : Index(embedding_dimension / heads_number);
    }
    Shape get_heads_shape(Index batch_size) const
    {
        return {batch_size, heads_number, query_sequence_length, get_head_dimension()};
    }

    Shape get_concat_shape(Index batch_size) const
    {
        return {batch_size, query_sequence_length, heads_number, get_head_dimension()};
    }

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

    vector<Operator*> get_operators() override;
    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;
    vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const override;

    void set(Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             bool = false,
             const string& = "multihead_attention_layer");

    void set_input_shape(const Shape& new_input_shape) override
    {
        if (new_input_shape.rank != 2)
            throw runtime_error("MultiHeadAttention input shape must have rank 2.");

        query_sequence_length  = new_input_shape[0];
        source_sequence_length = new_input_shape[0];
        embedding_dimension    = new_input_shape[1];
    }

    void on_compute_dtype_changed() override
    {
        query_projection .compute_dtype           = compute_dtype;
        query_projection .combination.weight_type = compute_dtype;
        key_projection   .compute_dtype           = compute_dtype;
        key_projection   .combination.weight_type = compute_dtype;
        value_projection .compute_dtype           = compute_dtype;
        value_projection .combination.weight_type = compute_dtype;
        output_projection.weight_type             = compute_dtype;
        attention.compute_dtype                   = compute_dtype;
    }

    void set_dropout_rate(float new_dropout_rate) { attention.set_dropout_rate(new_dropout_rate); }

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    Index embedding_dimension = 0;
    Index heads_number = 0;
    Index query_sequence_length = 0;
    Index source_sequence_length = 0;

    MultiHeadProjection query_projection;
    MultiHeadProjection key_projection;
    MultiHeadProjection value_projection;
    Combination         output_projection;
    Attention           attention;

    enum Forward {Input, Query, Key, AttentionWeights, AttentionWeightsDropped,
                  ConcatenatedAttentionOutputs, Value, TransposeScratch};
    enum Backward {OutputDelta, InputQueryDelta, InputSourceDelta,
                   AttentionWeightDelta, ValueDelta};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
