//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A C T I V A T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "activation_layer.h"

namespace opennn
{

Activation::Activation(const Shape& new_input_shape,
                       const string& new_activation,
                       const string& new_name)
    : Layer(LayerType::Activation)
{
    operators = {&activation_operator};

    set(new_input_shape, new_activation, new_name);
}

void Activation::set(const Shape& new_input_shape,
                     const string& new_activation,
                     const string& new_label)
{
    set_input_shape(new_input_shape);
    activation_operator.set_activation_function(new_activation);

    set_label(new_label);
}

void Activation::set_input_shape(const Shape& new_input_shape)
{
    check_rank(new_input_shape, {1, 2, 3}, "Activation", "input");

    input_shape = new_input_shape;
}

REGISTER(Layer, Activation, "Activation")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
