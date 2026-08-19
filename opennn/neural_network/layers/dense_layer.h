//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/layers/layer.h"
#include "opennn/neural_network/operators/activation_operator.h"
#include "opennn/neural_network/operators/batch_norm_operator.h"
#include "opennn/neural_network/operators/combination_operator.h"
#include "opennn/neural_network/operators/dropout_operator.h"
#include "opennn/neural_network/operators/swiglu_operator.h"

namespace opennn
{

class Dense final : public Layer
{
public:

    Dense(const Shape& = {},
          const Shape& = {},
          const string& = "Tanh",
          bool = false,
          const string& = "dense_layer");

    Shape get_output_shape() const override;

    Index get_input_features() const { return input_shape.empty() ? 0 : input_shape.back(); }

    const ActivationFunction& get_activation_function() const { return activation_operator.activation_function; }
    ActivationFunction get_output_activation() const override { return activation_operator.activation_function; }

    bool get_batch_normalization() const { return batch_norm.active(); }

    vector<TensorSpec> get_forward_specs(Index) const override;
    vector<TensorSpec> get_backward_specs(Index) const override;
    bool backward_uses_forward_output() const noexcept override { return gated || batch_norm.active() || activation_operator.activation_function != ActivationFunction::Identity; }
    bool preserves_output_delta_during_backward() const noexcept override { return !backward_uses_forward_output() && !dropout.active(); }
    bool folds_input_delta_addend(size_t input) const noexcept override
    {
        return input == 0 && combination.folds_input_delta_addend && !gated && !tied_source
            && !combination.accumulate_input_delta && !combination.drelu_source;
    }

    ForwardSlotKind get_forward_slot_kind(size_t spec) const override
    {
        return !gated && spec == size_t(ActivationView) - 1
            ? ForwardSlotKind::TrainingOnly
            : ForwardSlotKind::Pooled;
    }

    void set(const Shape& = {},
             const Shape& = {},
             const string& = "Tanh",
             bool = false,
             const string& = "dense_layer");

    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 1, 2); }

    void apply_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;
    void on_compute_dtype_changed() override { configure_operators(); }

    void set_activation_function(const string&);
    void set_batch_normalization(bool);

    void set_use_bias(bool use_bias) { combination.use_bias = use_bias; up_combination.use_bias = use_bias; }
    bool get_use_bias() const { return combination.use_bias; }

    void set_gated(bool);
    bool get_gated() const { return gated; }

    void set_transposed_inference(bool v) { combination.transposed_inference_preferred = v; }
    bool get_transposed_inference() const { return combination.transposed_inference_preferred; }

    void set_tied_weight_source(const Layer*);
    void set_tied_weight(const TiedWeight&) override;
    TiedWeight get_tied_weight() const override
    {
        return tied_source ? TiedWeight{tied_source, 0, 0} : TiedWeight{};
    }
    void set_dropout_rate(float new_rate)
    {
        const bool was_active = dropout.active();
        dropout.set_rate(new_rate);
        if (was_active != dropout.active())
            configure_operators();
    }
    void set_momentum(float);

    bool try_wire_drelu_fusion(Dense& producer);
    void reset_drelu_fusion();
    bool drelu_fusion_wired() const { return combination.drelu_source != nullptr; }
    bool drelu_fusion_ran() const
    {
        return combination.drelu_source && combination.drelu_source->relu_mask_fused_active;
    }

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;
    void on_loaded() override { configure_operators(); }

    string write_expression(const vector<string>&,
                            const vector<string>&) const override;

private:

    Index output_features = 0;

    bool gated = false;

    const Layer* tied_source = nullptr;

    CombinationOperator combination;
    CombinationOperator up_combination;
    SwiGLUOperator      swiglu;
    ActivationOperator  activation_operator;
    BatchNormalizationOperator   batch_norm;
    DropoutOperator     dropout;

    enum Forward {Input, CombinationView, BatchNormMean, BatchNormInverseVariance, ActivationView, Output};

    void configure_operators();
    bool saves_pre_dropout_activation() const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
