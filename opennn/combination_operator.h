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

    // Activation folded into the GEMM epilogue on CUDA (Identity = none).
    // ReLU: RELU_BIAS, in place on the output slot. GELUTanh: GELU_AUX_BIAS,
    // activated result goes to output_slots[1] while output_slots[0] keeps
    // the pre-activation the backward pass needs.
    ActivationFunction fused_activation = ActivationFunction::Identity;
    // Bias-free linear (LLaMA / Qwen3 projections have no bias). When false the
    // operator exposes only the weight parameter and skips the bias add/gradient.
    bool  use_bias        = true;

    // Add into the input-delta buffer instead of overwriting it (for several
    // projections sharing one input, e.g. the gated Dense).
    bool  accumulate_input_delta = false;

    // The weight view aliases another layer's table stored TRANSPOSED
    // ([out, in], forward computes y = x @ Wᵀ — e.g. an lm_head tied to the
    // embedding). Inference-only: backward throws, and the inits leave the
    // aliased source untouched.
    bool  tied_transposed = false;

    TensorView weights;
    TensorView bias;

    TensorView weight_gradient;
    TensorView bias_gradient;

    void set(Index, Index, Type new_compute_dtype = Type::FP32);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView>) override;
    void link_gradients (span<const TensorView>) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;
    void set_parameters_pytorch() override;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;


};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
