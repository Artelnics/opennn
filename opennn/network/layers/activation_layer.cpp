// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/network/layers/activation_layer.h"
#include "opennn/registry.h"

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

}
