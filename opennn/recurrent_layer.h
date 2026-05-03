//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"

namespace opennn
{

class Recurrent final : public Layer
{

public:

    Recurrent(const Shape& = {0, 0}, const Shape& = {0});

    Shape get_output_shape() const override;

    vector<pair<Shape, Type>> get_parameter_specs() const override;

    Shape get_input_shape() const override { return {time_steps, input_features}; }

    vector<pair<Shape, Type>> get_forward_specs(const Index batch_size) const override
    {
        const Index outputs_number = get_outputs_number();
        const Type act = activation_dtype;

        return {
            {{batch_size, outputs_number},             act},
            {{batch_size, time_steps, outputs_number}, act},
            {{batch_size, time_steps, outputs_number}, act},
        };
    }

    vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const override
    {
        return {{{batch_size, time_steps, input_features}, activation_dtype}};
    }

    void set(const Shape& = {}, const Shape& = {});

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;

    void set_activation_function(const string&);

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    const TensorView& get_biases() const { return biases; }
    const TensorView& get_input_weights() const { return input_weights; }
    const TensorView& get_recurrent_weights() const { return recurrent_weights; }
    const string& get_activation_function() const { return activation_function; }

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

private:

    Index time_steps = 0;
    Index input_features = 0;

    TensorView biases;
    TensorView input_weights;
    TensorView recurrent_weights;

    Tensor2 empty_2;  // sentinel passed to calculate_activations<2>(...) when no derivative output is wanted

    string activation_function = "Tanh";
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
