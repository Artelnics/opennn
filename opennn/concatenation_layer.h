//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N C A T E N A T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operator.h"

namespace opennn
{

struct ConcatenationOperator : Operator
{
    vector<Index> input_channels;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;
};

class Concatenation final : public Layer
{
public:

    Concatenation(const Shape& = {},
                  const vector<Index>& per_input_channels = {},
                  const string& = "concatenation_layer");

    Shape get_output_shape() const override;

    Index get_sources_number() const noexcept { return ssize(concatenation.input_channels); }

    vector<TensorSpec> get_backward_specs(Index) const override;

    void set(const Shape&, const vector<Index>&, const string&);
    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 3); }

    void apply_input_shape(const Shape&) override;

    void from_JSON(const JsonDocument&) override;
    void load_state_from_JSON(const JsonDocument&) override;

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    const Json* legacy_body(const JsonDocument&) const;

    ConcatenationOperator concatenation;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
