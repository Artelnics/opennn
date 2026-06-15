//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

class Recurrent final : public Layer
{
public:


    Recurrent(const Shape& = {0, 0},
              const Shape& = {0},
              const string& = "Tanh",
              const string& = "recurrent_layer");

    Shape get_input_shape() const override { return {time_steps, input_features}; }
    Shape get_output_shape() const override
    {
        return return_sequences ? Shape{time_steps, output_features}
                                : Shape{output_features};
    }

    void set_return_sequences(bool value);

    string get_activation_function() const { return ActivationOp::to_string(recurrent_op.activation); }
    ActivationOp::Function get_output_activation() const override { return recurrent_op.activation; }

    vector<TensorSpec> get_forward_specs(Index batch_size) const override;
    vector<TensorSpec> get_backward_specs(Index batch_size) const override;

    void set(const Shape& = {},
             const Shape& = {},
             const string& = "Tanh",
             const string& = "recurrent_layer");

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;
    void on_compute_dtype_changed() override { configure_operators(); }

    void set_activation_function(const string&);

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

    string write_expression(const vector<string>& input_names,
                            const vector<string>& output_names) const override;

private:

    enum Forward {Input, HiddenStates, ActivationDerivatives, Output};

    Index time_steps      = 0;
    Index input_features  = 0;
    Index output_features = 0;
    bool  return_sequences = false;

    RecurrentOp recurrent_op;

    void configure_operators();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
