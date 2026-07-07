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

struct EmbeddingLookupOperator : Operator
{
    Index vocabulary_size     = 0;
    Index sequence_length     = 0;
    Index embedding_dimension = 0;

    bool scale_embedding         = false;
    bool add_positional_encoding = false;
    bool  positional_trainable    = false;
    bool  export_valid_lengths    = false;

    TensorView weights;
    TensorView positional_encoding;

    TensorView weight_gradient;
    TensorView positional_gradient;

    void set(Index, Index, Index);

    vector<TensorSpec> parameter_specs() const override;
    vector<TensorSpec> state_specs()     const override;
    void link_parameters(span<const TensorView>) override;
    void link_gradients (span<const TensorView>) override;
    void link_states    (span<const TensorView>) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void init_positional_encoding();
    void init_trainable_positional();
    void compute_valid_lengths(const TensorView& indices, vector<Index>& valid_lengths) const;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    void load_state_from_JSON(const Json*) override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
