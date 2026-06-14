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

struct LayerNormOp : Operator
{
    Index sequence_length     = 0;
    Index embedding_dimension = 0;

    // When set, the op takes a second input (the residual) and fuses the
    // residual-add into the layer-norm kernel, replacing a separate Addition
    // layer (mirrors the BatchNorm fuse_add). The post-add sum is written to the
    // NormalizedInput output slot so the backward can read the residual stream.
    bool fuse_add = false;
    size_t residual_delta_slot = 0;

    TensorView gamma;
    TensorView beta;

    TensorView gamma_gradient;
    TensorView beta_gradient;

    void set(Index sequence_length, Index embedding_dimension);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView> views) override;
    void link_gradients (span<const TensorView> views) override;

    void set_parameters_random() override { init_defaults(); }
    void set_parameters_glorot() override { init_defaults(); }

    void init_defaults();

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
