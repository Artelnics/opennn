// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/network/layers/flatten_layer.h"
#include "opennn/registry.h"

namespace opennn
{

Flatten::Flatten(const Shape& new_input_shape, const string& new_label)
    : Layer(LayerType::Flatten)
{
    set(new_input_shape, new_label);
}

void Flatten::set(const Shape& new_input_shape, const string& new_label)
{
    check_rank(new_input_shape, {1, 2, 3}, "Flatten", "input");

    input_shape = new_input_shape;

    set_label(new_label);
}

}
