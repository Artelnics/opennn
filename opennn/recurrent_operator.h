//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"
#include "activation_operator.h"

namespace opennn
{

struct RecurrentOp : Operator
{
    enum BackwardSlot
    {
        OutputDeltaSlot = 0,
        InputDeltaSlot,
        StepInputScratchSlot,
        StepPrevHScratchSlot,
        DeltaScratchSlot,
        NextCarryScratchSlot,
        StepInDeltaScratchSlot
    };

    Index input_features  = 0;
    Index time_steps      = 0;
    Index output_features = 0;
    Type  weight_type     = Type::FP32;
    ActivationOp::Function activation = ActivationOp::Function::Tanh;

    bool return_sequences = false;

    TensorView bias;
    TensorView input_weights;
    TensorView recurrent_weights;

    TensorView bias_gradient;
    TensorView input_weight_gradient;
    TensorView recurrent_weight_gradient;

    void set(Index new_input_features,
             Index new_time_steps,
             Index new_output_features,
             ActivationOp::Function = ActivationOp::Function::Tanh,
             Type = Type::FP32);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView> views) override;
    void link_gradients (span<const TensorView> views) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

private:
    void apply(const TensorView& input,
               TensorView& hidden_states,
               TensorView& activation_derivatives,
               TensorView& output,
               bool is_training);
    void apply_gpu(const TensorView& input,
                   TensorView& hidden_states,
                   TensorView& activation_derivatives,
                   TensorView& output,
                   bool is_training);

    void apply_delta(const TensorView& input,
                     const TensorView& hidden_states,
                     const TensorView& activation_derivatives,
                     const TensorView& output_delta,
                     TensorView& input_delta) const;
    void apply_delta_gpu(const TensorView& input,
                         const TensorView& hidden_states,
                         const TensorView& activation_derivatives,
                         const TensorView& output_delta,
                         TensorView& input_delta,
                         TensorView& step_input_scratch,
                         TensorView& step_prev_h_scratch,
                         TensorView& delta_scratch,
                         TensorView& next_carry_scratch,
                         TensorView& step_in_delta_scratch) const;
    mutable Buffer step_input_buf      {Device::CUDA};
    mutable Buffer step_hidden_buf     {Device::CUDA};
    mutable Buffer prev_hidden_buf     {Device::CUDA};
    mutable Buffer step_derivs_buf     {Device::CUDA};
    mutable Buffer step_seq_delta_buf  {Device::CUDA};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
