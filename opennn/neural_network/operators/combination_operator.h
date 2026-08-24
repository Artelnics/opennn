//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O M B I N A T I O N   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/operators/operator.h"

namespace opennn
{

struct CombinationOperator : Operator
{
    Index input_features  = 0;
    Index output_features = 0;

    ActivationFunction fused_activation = ActivationFunction::Identity;

    bool  use_bias        = true;

    bool  accumulate_input_delta = false;

    bool  folds_input_delta_addend = false;

    bool  tied_transposed = false;

    bool  transposed_inference_preferred = false;
    bool  transposed_inference_active    = false;

    bool fuse_input_relu = false;
    Index input_relu_source_layer = -1;

    bool emit_relu_mask = false;
    mutable bool relu_mask_fusion_disabled = false;
    size_t relu_mask_slot = SIZE_MAX;
    const CombinationOperator* drelu_source = nullptr;
    Index drelu_source_layer = -1;

    TensorView weights;
    TensorView bias;
    TensorView weight_scale;

    TensorView weight_gradient;
    TensorView bias_gradient;

    void set(Index, Index, Type new_compute_dtype = Type::FP32);

    vector<TensorSpec> parameter_specs() const override;
    vector<SlotQuantization> parameter_quantization() const override;
    void link_parameters(span<const TensorView>) override;
    void link_gradients (span<const TensorView>) override;
    void link_parameter_scales(span<const TensorView>) override;

    bool owns_initializable_weights() const noexcept
    {
        return !weights.empty() && !tied_transposed;
    }

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
