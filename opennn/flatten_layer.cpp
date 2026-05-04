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
{
    set(new_input_shape);
}

void Flatten::set(const Shape& new_input_shape)
{
    input_shape = new_input_shape;
    set_label("flatten_layer");
    name = "Flatten";
    layer_type = LayerType::Flatten;

    if (!input_shape.empty()
        && input_shape.rank != 1 && input_shape.rank != 2 && input_shape.rank != 3)
        throw runtime_error("Flatten layer supports input rank 1, 2 or 3 (got "
                            + to_string(input_shape.rank) + ").");
}

vector<pair<Shape, Type>> Flatten::get_forward_specs(Index batch_size) const
{
    return {{Shape{batch_size}.append(get_output_shape()), activation_dtype}};
}

vector<pair<Shape, Type>> Flatten::get_backward_specs(Index batch_size) const
{
    return {{Shape{batch_size}.append(input_shape), activation_dtype}};
}

void Flatten::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool) noexcept
{
    auto& forward_views = forward_propagation.views[layer];

    copy(forward_views[Input][0], forward_views[Output][0]);
}

void Flatten::back_propagate(ForwardPropagation&,
                             BackPropagation& back_propagation,
                             size_t layer) const noexcept
{
    auto& delta_views = back_propagation.delta_views[layer];

    copy(delta_views[OutputDelta][0], delta_views[InputDelta][0]);
}

void Flatten::from_JSON(const JsonDocument& document)
{
    const Json* element = document.first_child("Flatten");
    if (!element) throw runtime_error(name + " element is nullptr.");

    set(string_to_shape(read_json_string(element, "InputDimensions")));
}

void Flatten::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Flatten");

    write_json(printer, {
        {"InputDimensions", shape_to_string(input_shape)}
    });

    printer.close_element();
}

REGISTER(Layer, Flatten, "Flatten")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
