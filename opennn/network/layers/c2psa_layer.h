// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#pragma once

#include "opennn/network/layers/layer.h"
#include "opennn/network/operators/c2psa_operator.h"

namespace opennn
{

class C2PSA final : public Layer
{
public:

    C2PSA(const Shape& = {}, const string& = "c2psa_layer");

    Shape get_output_shape() const override { return input_shape; }

    void set(const Shape&, const string&);
    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 3); }

    void apply_input_shape(const Shape& new_input_shape) override { set(new_input_shape, label); }

    vector<TensorSpec> get_forward_specs(Index) const override;
    vector<TensorSpec> get_backward_specs(Index) const override;

    ForwardSlotKind get_forward_slot_kind(size_t slot) const override
    {
        return slot == C2PSAOperator::ForwardScratch
            ? ForwardSlotKind::Transient
            : ForwardSlotKind::Pooled;
    }

    void on_compute_dtype_changed() override { c2psa.compute_dtype = get_compute_dtype(); }

private:

    C2PSAOperator c2psa;

    using enum C2PSAOperator::Slot;

    enum Backward {OutputDelta, InputDelta, BackwardScratch};

    void configure_operator();
};

}
