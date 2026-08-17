//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   N O R M   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/operators/operator.h"

namespace opennn
{

enum class NormalizationMethod { LayerNorm, RMS };

// The normalization kernels. LayerNormalizationOperator is their only
// caller, so they live beside it rather than in core.
void layer_normalization_forward(const TensorView&, const TensorView&, const TensorView&,
                        TensorView&, TensorView&,
                        TensorView&, TensorView&, float);
void layer_normalization_add_forward(const TensorView&, const TensorView&,
                            const TensorView&, const TensorView&,
                            TensorView&, TensorView&,
                            TensorView&, TensorView&, TensorView&, float);
// The trailing view, when non-null, receives a second copy of the input delta
// (the residual branch of a fused add + norm) without a separate copy pass.
void layer_normalization_backward(const TensorView&, const TensorView&,
                         const TensorView&, const TensorView&,
                         const TensorView&, const TensorView&,
                         const TensorView&, const TensorView&,
                         TensorView&, TensorView* = nullptr);
void rms_normalization_forward(const TensorView&, const TensorView&,
                      TensorView&, TensorView&, TensorView&, float);
void rms_normalization_backward(const TensorView&, const TensorView&,
                       const TensorView&, const TensorView&, const TensorView&,
                       const TensorView&, TensorView&);

struct LayerNormalizationOperator : Operator
{
    Index sequence_length     = 0;
    Index embedding_dimension = 0;

    NormalizationMethod method = NormalizationMethod::LayerNorm;

    float epsilon = 1.0e-6f;

    bool fuse_add = false;

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
