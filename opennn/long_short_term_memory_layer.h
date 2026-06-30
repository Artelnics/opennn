//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   L A Y E R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "long_short_term_memory_operator.h"

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

    Shape get_input_shape()  const noexcept override { return input_shape; }
    Shape get_output_shape() const override
    {
        return return_sequences ? Shape{get_time_steps(), output_features}
                                : Shape{output_features};
    }

    Index get_time_steps()      const noexcept { return input_shape.rank == 2 ? input_shape[0] : Index(0); }
    Index get_input_features()  const noexcept { return input_shape.rank == 2 ? input_shape[1] : Index(0); }
    Index get_output_features() const noexcept { return output_features; }

    bool get_return_sequences() const noexcept { return return_sequences; }
    void set_return_sequences(bool);

    const TensorView& get_forget_bias()    const noexcept { return lstm_op.forget_bias; }

    const ActivationFunction& get_activation_function() const noexcept { return lstm_op.activation_function; }
    ActivationFunction get_output_activation() const noexcept override { return lstm_op.activation_function; }

    vector<TensorSpec> get_forward_specs(Index)  const override;
    vector<TensorSpec> get_backward_specs(Index) const override;

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
