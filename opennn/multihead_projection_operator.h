//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   P R O J E C T I O N   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"
#include "combination_operator.h"

namespace opennn
{

struct MultiHeadProjectionOperator : Operator
{
    CombinationOperator combination;

    Index input_features = 0;

    size_t input_view_index = 0;

    vector<size_t> scratch_slots;

    vector<size_t> input_delta_slots_self;
    vector<size_t> input_delta_slots_cross;
    bool accumulate_input_delta_self  = false;
    bool accumulate_input_delta_cross = false;

    void set(Index, Index, Index, Type);

    vector<TensorSpec> parameter_specs() const override { return combination.parameter_specs(); }
    void link_parameters(span<const TensorView> views) override { combination.link_parameters(views); }
    void link_gradients (span<const TensorView> views) override { combination.link_gradients(views); }

    void set_parameters_random() override { combination.set_parameters_random(); }
    void set_parameters_glorot() override { combination.set_parameters_glorot(); }

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
