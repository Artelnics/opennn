//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L O O K U P   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

struct EmbeddingLookupOp : Operator
{
    Index vocabulary_size     = 0;
    Index sequence_length     = 0;
    Index embedding_dimension = 0;

    bool  scale_embedding         = false;
    bool  add_positional_encoding = false;

    TensorView weights;
    TensorView positional_encoding;

    TensorView weight_gradient;

    void set(Index new_vocabulary_size, Index new_sequence_length, Index new_embedding_dimension);

    vector<TensorSpec> parameter_specs() const override;
    vector<TensorSpec> state_specs()     const override;
    void link_parameters(span<const TensorView> views) override;
    void link_gradients (span<const TensorView> views) override;
    void link_states    (span<const TensorView> views) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void init_positional_encoding();

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

    void load_state_from_JSON(const Json* parent) override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
