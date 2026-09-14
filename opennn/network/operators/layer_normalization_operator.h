// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/network/operators/operator.h"

namespace opennn
{

enum class NormalizationMethod { LayerNorm, RMS };

void layer_normalization_forward(const TensorView&, const TensorView&, const TensorView&,
                        TensorView&, TensorView&,
                        TensorView&, TensorView&, float);
void layer_normalization_add_forward(const TensorView&, const TensorView&,
                            const TensorView&, const TensorView&,
                            TensorView&, TensorView&,
                            TensorView&, TensorView&, TensorView&, float);
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
    vector<ParameterSlot> parameter_slots() override;

    void set_parameters_random() override { init_defaults(); }
    void set_parameters_glorot() override { init_defaults(); }

    void init_defaults();

    void forward_propagate(ForwardPropagation&, size_t, ForwardPropagationMode) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;
};

}
