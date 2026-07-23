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

// Which per-row statistic the normalization divides by.
// LayerNorm: (x - mean) / std * gamma + beta, using the global EPSILON.
// RMS (Zhang & Sennrich, 2019; LLaMA / Qwen3): x / sqrt(mean(x^2) + epsilon)
// * gamma, with no mean subtraction and no beta.
enum class NormalizationMethod { LayerNorm, RMS };

struct LayerNormalizationOperator : Operator
{
    Index sequence_length     = 0;
    Index embedding_dimension = 0;

    // Changes the parameter layout (RMS drops beta), so it must be chosen
    // before the network is compiled.
    NormalizationMethod method = NormalizationMethod::LayerNorm;

    // RMS only (Qwen3's rms_norm_eps); the LayerNorm kernels use the global
    // EPSILON (~1.19e-7), which differs.
    float epsilon = 1.0e-6f;

    // When set, the op takes a second input (the residual) and fuses the
    // residual-add into the layer-norm kernel, replacing a separate Addition
    // layer (mirrors the BatchNorm). The post-add sum is written to the
    // NormalizedInput output slot so the backward can read the residual stream.
    // LayerNorm method only.
    bool fuse_add = false;
    size_t residual_delta_slot = 0;

    TensorView gamma;   // RMS: the single scale (Qwen's ".weight")
    TensorView beta;    // LayerNorm only

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
