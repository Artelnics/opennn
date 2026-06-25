//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   N O R M   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

struct LayerNormOperator : Operator
{
    Index sequence_length     = 0;
    Index embedding_dimension = 0;

    // When, the op takes a second input (the residual) and fuses the
    // residual-add into the layer-norm kernel, replacing a separate Addition
    // layer (mirrors the BatchNorm). The post-add sum is written to the
    // NormalizedInput output slot so the backward can read the residual stream.
    bool fuse_add = false;
    size_t residual_delta_slot = 0;

    TensorView gamma;
    TensorView beta;

    TensorView gamma_gradient;
    TensorView beta_gradient;

    void set(Index, Index);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView>) override;
    void link_gradients (span<const TensorView>) override;

    void set_parameters_random() override { init_defaults(); }
    void set_parameters_glorot() override { init_defaults(); }

    void init_defaults();

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
