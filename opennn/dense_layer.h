//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "activation_operator.h"
#include "batch_norm_operator.h"
#include "combination_operator.h"
#include "dropout_operator.h"
#include "swiglu_operator.h"

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

    void set(const Shape& = {},
             const Shape& = {},
             const string& = "Tanh",
             bool = false,
             const string& = "dense_layer");

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;
    void on_compute_dtype_changed() override { configure_operators(); }

    void set_activation_function(const string&);
    void set_batch_normalization(bool);

    // Bias-free linear (Qwen3 / LLaMA projections). Must be set before compile()
    // so the parameter buffer omits the bias.
    void set_use_bias(bool use_bias) { combination.use_bias = use_bias; up_combination.use_bias = use_bias; }
    bool get_use_bias() const { return combination.use_bias; }

    // Gated (SwiGLU) feed-forward: output = silu(x·Wg) * (x·Wu). Two weight
    // matrices, gate first then up (the parameter layout of a separate gate/up
    // Dense pair). Identity activation, no batch norm; set before compile().
    void set_gated(bool);
    bool get_gated() const { return gated; }
    void set_dropout_rate(float new_rate)
    {
        const bool was_active = dropout.active();
        dropout.set_rate(new_rate);
        if (was_active != dropout.active())
            configure_operators();
    }
    void set_momentum(float);

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;
    void from_JSON(const JsonDocument&) override;

    string write_expression(const vector<string>&,
                            const vector<string>&) const override;

private:

    Index output_features = 0;

    bool gated = false;

    CombinationOperator combination;
    CombinationOperator up_combination;   // gated mode only
    SwiGLUOperator      swiglu;           // gated mode only
    ActivationOperator  activation_operator;
    BatchNormalizationOperator   batch_norm;
    DropoutOperator     dropout;

    // Gated mode: CombinationView holds the gate projection, ActivationView the up.
    enum Forward {Input, CombinationView, BatchNormMean, BatchNormInverseVariance, ActivationView, Output};

    void configure_operators();
    bool saves_pre_dropout_activation() const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
