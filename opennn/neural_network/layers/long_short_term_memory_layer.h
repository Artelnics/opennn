//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   L A Y E R   H E A D E R
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

struct LongShortTermMemoryOperator : Operator, CudnnRnnState
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
        OutputDeltaScratchSlot,
        CudnnOutputDeltaScratchSlot,
        CudnnInputDeltaScratchSlot
    };

    Index input_features  = 0;
    Index output_features = 0;
    Index time_steps      = 0;

    bool return_sequences = false;

    ActivationFunction activation_function = ActivationFunction::Tanh;
    ActivationFunction recurrent_activation_function = ActivationFunction::Sigmoid;

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

    void set(Index,
             Index,
             Index,
             ActivationFunction new_activation_function = ActivationFunction::Tanh,
             ActivationFunction new_recurrent_activation_function = ActivationFunction::Sigmoid);

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
               TensorView&,
               TensorView&,
               TensorView&,
               TensorView&,
               TensorView&) const;

    void apply_delta(const TensorView&,
                     const TensorView&,
                     TensorView&,
                     TensorView&,
                     TensorView&,
                     TensorView&,
                     TensorView&,
                     TensorView&,
                     TensorView&,
                     const TensorView&,
                     const TensorView&,
                     const TensorView&,
                     const TensorView&,
                     const TensorView&,
                     const TensorView&,
                     const TensorView&) const;

    void apply_gpu(const TensorView&,
                   TensorView&,
                   TensorView&,
                   Buffer&,
                   bool,
                   bool) const;

    void apply_delta_gpu(const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         TensorView&,
                         TensorView&,
                         TensorView&,
                         const Buffer&,
                         Buffer&,
                         bool) const;

    CudnnRnnShapeSlot& ensure_cudnn_setup_(Index, bool) const;
    void pack_weights_to_cudnn_(Buffer&) const;
    void unpack_gradients_from_cudnn_(Buffer&) const;

};

class LongShortTermMemory final : public Layer
{
public:

    LongShortTermMemory(const Shape& = {},
                        const Shape& = {},
                        const string& = "Tanh",
                        const string& = "Sigmoid",
                        const string& = "long_short_term_memory_layer");

    Shape get_input_shape()  const noexcept override { return input_shape; }
    Shape get_output_shape() const override
    {
        return return_sequences ? Shape{get_time_steps(), output_features}
                                : Shape{output_features};
    }

    Index get_time_steps()      const noexcept { return input_shape.get_rank() == 2 ? input_shape[0] : Index(0); }
    Index get_input_features()  const noexcept { return input_shape.get_rank() == 2 ? input_shape[1] : Index(0); }
    Index get_output_features() const noexcept { return output_features; }

    bool get_return_sequences() const noexcept { return return_sequences; }
    void set_return_sequences(bool);

    const TensorView& get_forget_bias()    const noexcept { return lstm_op.forget_bias; }

    const ActivationFunction& get_activation_function() const noexcept { return lstm_op.activation_function; }
    const ActivationFunction& get_recurrent_activation_function() const noexcept { return lstm_op.recurrent_activation_function; }
    ActivationFunction get_output_activation() const noexcept override { return lstm_op.activation_function; }

    vector<TensorSpec> get_forward_specs(Index)  const override;
    vector<TensorSpec> get_backward_specs(Index) const override;

    void set(const Shape& = {},
             const Shape& = {},
             const string& = "Tanh",
             const string& = "Sigmoid",
             const string& = "long_short_term_memory_layer");

    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 2); }
    bool is_recurrent() const noexcept override { return true; }

    void apply_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;
    void on_compute_dtype_changed() override { configure_operators(); }

    void set_activation_function(const string&);
    void set_recurrent_activation_function(const string&);

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

    string write_expression(const vector<string>&,
                            const vector<string>&) const override;

private:

    Index output_features = 0;
    bool  return_sequences = false;

    LongShortTermMemoryOperator lstm_op;

    void configure_operators();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
