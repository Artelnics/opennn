//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/layers/layer.h"
#include "opennn/neural_network/operators/operator.h"
#include "opennn/neural_network/operators/activation_operator.h"
#include "opennn/neural_network/operators/cudnn_rnn.h"

namespace opennn
{

struct RecurrentOperator : Operator, CudnnRnnState
{
    enum ForwardScratchSlot
    {
        StepInputForwardSlot = 3,
        StepHiddenForwardSlot,
        PreviousHiddenForwardSlot,
        StepDerivativesForwardSlot,
        CudnnInputSequenceForwardSlot,
        CudnnOutputSequenceForwardSlot
    };

    enum BackwardSlot
    {
        OutputDeltaSlot = 0,
        InputDeltaSlot,
        StepInputScratchSlot,
        StepPrevHScratchSlot,
        DeltaScratchSlot,
        NextCarryScratchSlot,
        StepInDeltaScratchSlot,
        SequenceDeltaScratchSlot,
        CudnnInputDeltaScratchSlot
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
               bool) const;
    void apply_gpu(const TensorView&,
                   TensorView&,
                   TensorView&,
                   TensorView&,
                   TensorView&,
                   TensorView&,
                   TensorView&,
                   TensorView&,
                   TensorView&,
                   TensorView&,
                   Buffer&,
                   bool) const;

    void apply_delta(const TensorView&,
                     const TensorView&,
                     const TensorView&,
                     const TensorView&,
                     TensorView&,
                     TensorView&,
                     Buffer&) const;
    void apply_delta_gpu(const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         TensorView&,
                         TensorView&,
                         TensorView&,
                         TensorView&,
                         TensorView&,
                         TensorView&,
                         TensorView&,
                         TensorView&,
                         const TensorView&,
                         const TensorView&,
                         const Buffer&,
                         Buffer&) const;

    bool cudnn_rnn_eligible_(const TensorView&) const;
    CudnnRnnShapeSlot& ensure_cudnn_setup_(Index, bool) const;
    void pack_weights_to_cudnn_(Buffer&) const;
    void unpack_gradients_from_cudnn_(Buffer&) const;
    void apply_gpu_cudnn_(const TensorView&, TensorView&, TensorView&,
                          TensorView&, TensorView&, Buffer&, bool) const;
    void apply_delta_gpu_cudnn_(const TensorView&, const TensorView&,
                                const TensorView&, const TensorView&,
                                const TensorView&, TensorView&, TensorView&,
                                TensorView&, const Buffer&, Buffer&) const;
};

class Recurrent final : public Layer
{
public:

    Recurrent(const Shape& = {0, 0},
              const Shape& = {0},
              const string& = "Tanh",
              const string& = "recurrent_layer");

    Shape get_input_shape() const noexcept override { return input_shape; }
    Shape get_output_shape() const override
    {
        return return_sequences ? Shape{input_shape[0], output_features}
                                : Shape{output_features};
    }

    void set_return_sequences(bool);

    string get_activation_function() const { return ActivationOperator::to_string(recurrent_op.activation); }
    ActivationFunction get_output_activation() const override { return recurrent_op.activation; }

    vector<TensorSpec> get_forward_specs(Index) const override;
    vector<TensorSpec> get_backward_specs(Index) const override;
    ForwardSlotKind get_forward_slot_kind(size_t spec) const override
    {
        if (spec == 1) return ForwardSlotKind::TrainingOnly;
        if (spec >= 2 && spec <= 5) return ForwardSlotKind::Transient;
        return ForwardSlotKind::Pooled;
    }

    void set(const Shape& = {},
             const Shape& = {},
             const string& = "Tanh",
             const string& = "recurrent_layer");

    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 2); }
    bool is_recurrent() const noexcept override { return true; }

    void apply_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;
    void on_compute_dtype_changed() override { configure_operators(); }

    void set_activation_function(const string&);

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

    string write_expression(const vector<string>&,
                            const vector<string>&) const override;

private:

    enum Forward
    {
        Input,
        HiddenStates,
        ActivationDerivatives,
        StepInputScratch,
        StepHiddenScratch,
        PreviousHiddenScratch,
        StepDerivativesScratch,
        CudnnInputSequence,
        CudnnOutputSequence,
        Output
    };

    Index output_features = 0;
    bool  return_sequences = false;

    RecurrentOperator recurrent_op;

    void configure_operators();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
