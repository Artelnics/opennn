//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "flatten_layer.h"
#include "math_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

Flatten::Flatten(const Shape& new_input_shape)
    : Layer("Flatten", LayerType::Flatten)
{
    operators = {&flat};

    set(new_input_shape);

    flat.input_slots  = {Input};
    flat.output_slots = {Output};

    flat.output_delta_slots = {OutputDelta};
    flat.input_delta_slots  = {InputDelta};
}
void Flatten::set(const Shape& new_input_shape)
{
    check_rank(new_input_shape, {1, 2, 3}, "Flatten", "input");

    input_shape = new_input_shape;

    set_label("flatten_layer");
}
REGISTER(Layer, Flatten, "Flatten")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
