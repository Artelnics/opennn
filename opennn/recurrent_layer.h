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

    vector<Shape> get_parameter_shapes() const override;

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        const Index outputs_number = get_outputs_number();
        const Index time_steps = input_shape[0];

        return {{ batch_size, outputs_number },            // Final layer outputs
            { batch_size, time_steps, outputs_number },    // All hidden_states
            { batch_size, time_steps, outputs_number }};   // All activation_derivatives

    }

    vector<Shape> get_backward_shapes(Index batch_size) const override
    {
        const Index time_steps = input_shape[0];
        const Index input_size = input_shape[1];

        // Input Sequence Gradients (dX): {batch, time_steps, input_size}
        return {{ batch_size, time_steps, input_size}};
    }


    void set(const Shape& = {}, const Shape& = {});

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;

    void set_activation_function(const string&);

    void forward_propagate(ForwardPropagation&, size_t, bool) override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    string get_expression(const vector<string>& = vector<string>(),
                          const vector<string>& = vector<string>()) const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

private:

    TensorView biases;
    TensorView input_weights;
    TensorView recurrent_weights;

    string activation_function = "HyperbolicTangent";
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
