//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/layers/layer.h"
#include "opennn/neural_network/operators/operator.h"

namespace opennn
{

struct AdditionOperator : Operator
{
    void forward_propagate(ForwardPropagation&, size_t, ForwardPropagationMode) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;
};

class Addition final : public Layer
{
public:

    Addition(const Shape& = {}, const string& = "", Index num_inputs = 2);

    Shape get_output_shape() const noexcept override { return input_shape; }

    vector<TensorSpec> get_backward_specs(Index) const override;

    void set(const Shape&, const string&, Index);
    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 2, 3); }

    void apply_input_shape(const Shape& shape) override { set(shape, label, inputs_number); }

    Index get_sources_number() const noexcept override { return inputs_number; }

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    AdditionOperator add;

    Index inputs_number = 2;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
