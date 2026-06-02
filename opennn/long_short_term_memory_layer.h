//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   L A Y E R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

class LongShortTermMemory final : public Layer
{
public:

    LongShortTermMemory(const Shape& = {},
                        const Shape& = {},
                        const string& = "Tanh",
                        const string& = "Sigmoid",
                        const string& = "long_short_term_memory_layer");

    Shape get_input_shape()  const override { return input_shape; }
    Shape get_output_shape() const override
    {
        return return_sequences ? Shape{get_time_steps(), output_features}
                                : Shape{output_features};
    }

    Index get_time_steps()      const { return input_shape.rank == 2 ? input_shape[0] : Index(0); }
    Index get_input_features()  const { return input_shape.rank == 2 ? input_shape[1] : Index(0); }
    Index get_output_features() const { return output_features; }

    bool get_return_sequences() const { return return_sequences; }
    void set_return_sequences(bool value);

    const TensorView& get_forget_bias()    const { return lstm_op.forget_bias; }
    const TensorView& get_input_bias()     const { return lstm_op.input_bias; }
    const TensorView& get_candidate_bias() const { return lstm_op.candidate_bias; }
    const TensorView& get_output_bias()    const { return lstm_op.output_bias; }

    const ActivationOp::Function& get_activation_function() const { return lstm_op.activation_function; }
    const ActivationOp::Function& get_recurrent_activation_function() const { return lstm_op.recurrent_activation_function; }
    ActivationOp::Function get_output_activation() const override { return lstm_op.activation_function; }

    vector<TensorSpec> get_forward_specs(Index batch_size)  const override;
    vector<TensorSpec> get_backward_specs(Index batch_size) const override;

    void set(const Shape& = {},
             const Shape& = {},
             const string& = "Tanh",
             const string& = "Sigmoid",
             const string& = "long_short_term_memory_layer");

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;
    void on_compute_dtype_changed() override { configure_operators(); }

    void set_activation_function(const string&);
    void set_recurrent_activation_function(const string&);

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

    string write_expression(const vector<string>& input_names,
                            const vector<string>& output_names) const override;

private:

    Index output_features = 0;
    bool  return_sequences = false;

    LongShortTermMemoryOp lstm_op;

    void configure_operators();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
