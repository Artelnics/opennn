//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A C T I V A T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "activation_layer.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

Activation::Activation(const Shape& new_input_shape,
                       const string& new_activation,
                       const string& new_name)
    : Layer(LayerType::Activation)
{
    operators = {&activation};

    activation.input_slots         = {Input};
    activation.output_slots        = {Output};
    activation.input_delta_slots   = {InputDelta};
    activation.output_delta_slots  = {OutputDelta};

    set(new_input_shape, new_activation, new_name);
}

void Activation::set(const Shape& new_input_shape,
                     const string& new_activation,
                     const string& new_label)
{
    check_rank(new_input_shape, {1, 2, 3}, "Activation", "input");

    input_shape = new_input_shape;
    activation.set_function(new_activation);

    set_label(new_label);
}

REGISTER(Layer, Activation, "Activation")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
