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

    // Set by a layer whose input 0 this combination consumes directly (Dense):
    // the backward then folds BackPropagation::input_delta_addend(layer, 0)
    // into the input delta it writes. Combinations inside other layers (an
    // attention layer's output projection) leave it off.
    bool  folds_input_delta_addend = false;

    bool  tied_transposed = false;

    bool  transposed_inference_preferred = false;
    bool  transposed_inference_active    = false;

    // Set by the layer when this combination's input is the output of a ReLU
    // whose backward it can absorb (Dense::try_wire_single_output_relu_fusion).
    // input_relu_fused_active says whether the last backward actually did it,
    // and the producing activation operator reads that to decide whether to run.
    bool fuse_input_relu = false;
    mutable bool input_relu_fused_active = false;

    mutable bool emit_relu_mask = false;
    mutable bool relu_mask_fused_active = false;
    Buffer relu_mask{Device::CUDA};
    TensorView relu_mask_view;
    const CombinationOperator* drelu_source = nullptr;

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

    // A tied projection borrows its source layer's weights, so it has nothing
    // of its own to initialise.
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
