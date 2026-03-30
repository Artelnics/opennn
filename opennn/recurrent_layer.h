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

    Shape get_input_shape() const override;
    Shape get_output_shape() const override;

    vector<Shape> get_parameter_shapes() const override;

    vector<Shape> get_forward_shapes() const override
    {
        /*
    if(layer == nullptr)
        throw runtime_error("Recurrrent layer is nullptr");

    const Index batch = batch_size;
    const Index outputs_num = layer->get_outputs_number();
    const Index steps = layer->get_input_shape()[0];

    outputs.shape = {batch, outputs_num};
    hidden_states.shape = {batch, steps, outputs_num};
    activation_derivatives.shape = {batch, steps, outputs_num};

*/
        return {};
    }

    vector<Shape> get_backward_shapes() const override
    {
        /*
    const Index outputs_number = layer->get_outputs_number();
    const Index inputs_number = layer->get_input_shape()[1];
    const Index time_steps = layer->get_input_shape()[0];

    bias_gradients.shape = {outputs_number};
    input_weight_gradients.shape = {inputs_number, outputs_number};
    recurrent_weight_gradients.shape = {outputs_number, outputs_number};

    input_gradients = {{nullptr, { batch_size, time_steps, inputs_number }}};

*/
        return {};
    }


    string get_activation_function() const;

    void set(const Shape& = {}, const Shape& = {});

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;

    void set_activation_function(const string&);

    void forward_propagate(ForwardPropagation&, size_t, bool) override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    string get_expression(const vector<string>& = vector<string>(),
                          const vector<string>& = vector<string>()) const override;

    void print() const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

private:

    Shape input_shape;

    TensorView biases;
    TensorView input_weights;
    TensorView recurrent_weights;

    string activation_function = "HyperbolicTangent";

#ifdef OPENNN_CUDA
    // @todo
#endif

};


struct RecurrentForwardPropagation final : LayerForwardPropagation
{
    TensorView hidden_states;
    TensorView activation_derivatives;
};


struct RecurrentBackPropagation final : LayerBackPropagation
{
    TensorView bias_gradients;
    TensorView input_weight_gradients;
    TensorView recurrent_weight_gradients;
};

#ifdef OPENNN_CUDA
// @todo
#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
