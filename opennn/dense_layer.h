//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

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

    CombinationOperator combination;
    ActivationOperator  activation_operator;
    BatchNormalizationOperator   batch_norm;
    DropoutOperator     dropout;

    enum Forward {Input, CombinationView, BatchNormMean, BatchNormInverseVariance, ActivationView, Output};

    void configure_operators();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
