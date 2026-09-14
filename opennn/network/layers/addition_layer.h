// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/network/layers/layer.h"
#include "opennn/network/operators/operator.h"

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
