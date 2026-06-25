//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O M B I N A T I O N   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

struct CombinationOperator : Operator
{
    Index input_features  = 0;
    Index output_features = 0;
    Type  weight_type     = Type::FP32;

    bool  fuse_relu       = false;

    TensorView weights;
    TensorView bias;

    TensorView weight_gradient;
    TensorView bias_gradient;

    void set(Index, Index, Type new_weight_type = Type::FP32);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView>) override;
    void link_gradients (span<const TensorView>) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    void apply(const TensorView&, TensorView&, cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS);

    void apply_delta(const TensorView&,
                     const TensorView&,
                     TensorView&,
                     bool accumulate_input_delta = false) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
