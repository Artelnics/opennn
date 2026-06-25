//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"
#include "activation_operator.h"

namespace opennn
{

// Force the scalar LSTM kernel path (default off uses the vectorized path for
// hidden size >= 64). Set from code; there is no environment variable.
void set_lstm_scalar(bool);
bool lstm_scalar_enabled();

struct LongShortTermMemoryOp : Operator
{
    enum ForwardSlot
    {
        InputSlot = 0,
        ForgetGateSlot,
        InputGateSlot,
        CandidateGateSlot,
        OutputGateSlot,
        CellStateSlot,
        HiddenStateSlot,
        CellActivationSlot,
        OutputSlot
    };

    enum BackwardSlot
    {
        OutputDeltaSlot = 0,
        InputDeltaSlot,
        HiddenDeltaScratchSlot,
        CellDeltaScratchSlot,
        ForgetDeltaScratchSlot,
        InputDeltaScratchSlot,
        CandidateDeltaScratchSlot,
        OutputDeltaScratchSlot
    };

    Index input_features  = 0;
    Index output_features = 0;
    Index time_steps      = 0;

    bool return_sequences = false;

    ActivationOp::Function activation_function = ActivationOp::Function::Tanh;
    ActivationOp::Function recurrent_activation_function = ActivationOp::Function::Sigmoid;

    TensorView forget_bias;
    TensorView input_bias;
    TensorView candidate_bias;
    TensorView output_bias;

    TensorView forget_weights;
    TensorView input_weights;
    TensorView candidate_weights;
    TensorView output_weights;

    TensorView forget_recurrent_weights;
    TensorView input_recurrent_weights;
    TensorView candidate_recurrent_weights;
    TensorView output_recurrent_weights;

    TensorView forget_bias_gradient;
    TensorView input_bias_gradient;
    TensorView candidate_bias_gradient;
    TensorView output_bias_gradient;

    TensorView forget_weight_gradient;
    TensorView input_weight_gradient;
    TensorView candidate_weight_gradient;
    TensorView output_weight_gradient;

    TensorView forget_recurrent_weight_gradient;
    TensorView input_recurrent_weight_gradient;
    TensorView candidate_recurrent_weight_gradient;
    TensorView output_recurrent_weight_gradient;

    void set(Index new_input_features,
             Index new_output_features,
             Index new_time_steps,
             ActivationOp::Function new_activation_function = ActivationOp::Function::Tanh,
             ActivationOp::Function new_recurrent_activation_function = ActivationOp::Function::Sigmoid);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView> views) override;
    void link_gradients (span<const TensorView> views) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

private:
    void apply(const TensorView& input,
               TensorView& output,
               TensorView& forget_gate,
               TensorView& input_gate,
               TensorView& candidate_gate,
               TensorView& output_gate,
               TensorView& cell_state,
               TensorView& hidden_state,
               TensorView& cell_activation) const;

    void apply_delta(const TensorView& input,
                     const TensorView& output_delta,
                     TensorView& input_delta,
                     TensorView& hidden_delta_scratch,
                     TensorView& cell_delta_scratch,
                     TensorView& forget_delta_scratch,
                     TensorView& input_delta_scratch,
                     TensorView& candidate_delta_scratch,
                     TensorView& output_delta_scratch,
                     const TensorView& forget_gate,
                     const TensorView& input_gate,
                     const TensorView& candidate_gate,
                     const TensorView& output_gate,
                     const TensorView& cell_state,
                     const TensorView& hidden_state,
                     const TensorView& cell_activation) const;

    void apply_gpu(const TensorView& input,
                   TensorView& output,
                   bool return_seq) const;

    void apply_delta_gpu(const TensorView& input,
                         const TensorView& output_delta,
                         TensorView& input_delta,
                         bool return_seq) const;

    void ensure_cudnn_setup_(Index batch_size) const;
    void pack_weights_to_cudnn_() const;
    void unpack_gradients_from_cudnn_() const;

    mutable Buffer weight_space_buf    {Device::CUDA};
    mutable Buffer dweight_space_buf   {Device::CUDA};
    mutable Buffer workspace_buf       {Device::CUDA};
    mutable Buffer reserve_space_buf   {Device::CUDA};
    mutable Buffer y_buf               {Device::CUDA};   // (B, T, H) rank-3 y from cuDNN
    mutable Buffer dy_buf              {Device::CUDA};   // (B, T, H) rank-3 dy for cuDNN
    mutable Buffer dx_scratch_buf      {Device::CUDA};   // (B, T, F) dx sink when input_delta is unused
    mutable Buffer seq_lengths_host_buf{Device::CPU};    // int32[batch], all equal to T
    mutable Buffer seq_lengths_dev_buf {Device::CUDA};

    mutable CudnnDescriptor<cudnnRNNDescriptor_t>     rnn_desc;
    mutable CudnnDescriptor<cudnnRNNDataDescriptor_t> x_data_desc;
    mutable CudnnDescriptor<cudnnRNNDataDescriptor_t> y_data_desc;
    mutable CudnnDescriptor<cudnnTensorDescriptor_t>  h_desc;
    mutable CudnnDescriptor<cudnnTensorDescriptor_t>  c_desc;
    mutable CudnnDescriptor<cudnnDropoutDescriptor_t> dropout_desc;
    mutable Buffer dropout_states_buf{Device::CUDA};

    mutable Index cached_batch_size = -1;
    mutable Index cached_time_steps = -1;
    mutable Index cached_input_features  = -1;
    mutable Index cached_output_features = -1;

    mutable vector<float> grad_tls_buf_;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
