// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, S.L. and Roberto Lopez

#include "opennn/network/layers/flatten_layer.h"
#include "opennn/network/layers/layer_registry.h"

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
