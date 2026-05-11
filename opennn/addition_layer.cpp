//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "addition_layer.h"
#include "math_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

Addition::Addition(const Shape& new_input_shape, const string& new_name)
    : Layer("Addition", LayerType::Addition)
{
    operators = {&add};

    set(new_input_shape, new_name);

    add.input_slots  = {Input};
    add.output_slots = {Output};

    add.output_delta_slots = {OutputDelta};
    add.input_delta_slots  = {InputDelta0, InputDelta1};
}

vector<pair<Shape, Type>> Addition::get_backward_specs(Index batch_size) const
{
    return vector<pair<Shape, Type>>(2, {Shape{batch_size}.append(input_shape), compute_dtype});
}

void Addition::set(const Shape& new_input_shape, const string& new_label)
{
    check_rank(new_input_shape, {2, 3}, "Addition", "input");

    input_shape = new_input_shape;

    set_label(new_label);
}

REGISTER(Layer, Addition, "Addition")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
