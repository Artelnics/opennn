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

struct RecurrentOperator : Operator
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
    ActivationFunction activation = ActivationFunction::Tanh;

    bool return_sequences = false;

    TensorView bias;
    TensorView input_weights;
    TensorView recurrent_weights;

    TensorView bias_gradient;
    TensorView input_weight_gradient;
    TensorView recurrent_weight_gradient;

    void set(Index,
             Index,
             Index,
             ActivationFunction = ActivationFunction::Tanh,
             Type = Type::FP32);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView>) override;
    void link_gradients (span<const TensorView>) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;
    void set_parameters_pytorch() override;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

private:
    void apply(const TensorView&,
               TensorView&,
               TensorView&,
               TensorView&,
               bool);
    void apply_gpu(const TensorView&,
                   TensorView&,
                   TensorView&,
                   TensorView&,
                   bool);

    void apply_delta(const TensorView&,
                     const TensorView&,
                     const TensorView&,
                     const TensorView&,
                     TensorView&) const;
    void apply_delta_gpu(const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         TensorView&,
                         TensorView&,
                         TensorView&,
                         TensorView&,
                         TensorView&,
                         TensorView&) const;
    mutable Buffer step_input_buf     {Device::CUDA};
    mutable Buffer step_hidden_buf    {Device::CUDA};
    mutable Buffer prev_hidden_buf    {Device::CUDA};
    mutable Buffer step_derivs_buf    {Device::CUDA};
    mutable Buffer step_seq_delta_buf {Device::CUDA};

    bool cudnn_rnn_eligible_(const TensorView&) const;
    void ensure_cudnn_setup_(Index, bool) const;
    void ensure_cudnn_setup_attempt_(Index, bool) const;
    void pack_weights_to_cudnn_() const;
    void unpack_gradients_from_cudnn_() const;
    void apply_gpu_cudnn_(const TensorView&, TensorView&, TensorView&, bool);
    void apply_delta_gpu_cudnn_(const TensorView&, const TensorView&,
                                const TensorView&, TensorView&) const;

    mutable Buffer weight_space_buf  {Device::CUDA};
    mutable Buffer dweight_space_buf {Device::CUDA};
    mutable Buffer workspace_buf     {Device::CUDA};
    mutable Buffer reserve_space_buf {Device::CUDA};
    mutable Buffer dy_buf            {Device::CUDA};
    mutable Buffer dx_scratch_buf    {Device::CUDA};

    mutable CudnnDescriptor<cudnnRNNDescriptor_t>     rnn_desc;
    mutable CudnnDescriptor<cudnnDropoutDescriptor_t> dropout_desc;
    mutable Buffer dropout_states_buf{Device::CUDA};

    mutable CudnnRnnShapeSlot shape_slots_[RNN_SHAPE_SLOTS];
    mutable int active_shape_ = -1;
    mutable int shape_stamp_  = 0;
    CudnnRnnShapeSlot& active_shape() const { return shape_slots_[active_shape_]; }

    mutable Index cached_input_features  = -1;
    mutable Index cached_output_features = -1;

    mutable float* cudnn_w_ptrs_[2]  = {};
    mutable float* cudnn_b_ptrs_[2]  = {};
    mutable float* cudnn_gw_ptrs_[2] = {};
    mutable float* cudnn_gb_ptrs_[2] = {};

    mutable bool persist_algo_failed_ = false;
    mutable bool persist_algo_active_ = false;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
