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

Addition::Addition(const Shape& new_input_shape, const string& new_name) : Layer()
{
    name = "Addition";
    layer_type = LayerType::Addition;

    set(new_input_shape, new_name);
}

// Getters

vector<pair<Shape, Type>> Addition::get_forward_specs(Index batch_size) const
{
    return {{Shape{batch_size}.append(input_shape), compute_dtype}};
}

vector<pair<Shape, Type>> Addition::get_backward_specs(Index batch_size) const
{
    return {
        {Shape{batch_size}.append(input_shape), compute_dtype}, // InputDelta0
        {Shape{batch_size}.append(input_shape), compute_dtype}, // InputDelta1
    };
}

// Setters

void Addition::set(const Shape& new_input_shape, const string& new_label)
{
    if (!new_input_shape.empty() && new_input_shape.rank != 2 && new_input_shape.rank != 3)
        throw runtime_error("Addition layer supports input rank 2 or 3 (got "
                            + to_string(new_input_shape.rank) + ").");

    input_shape = new_input_shape;

    set_label(new_label);
}

// Forward / back propagation

void Addition::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool) noexcept
{
    auto& forward_views = forward_propagation.views[layer];

    addition(forward_views[Input][0], forward_views[Input][1], forward_views[Output][0]);
}

void Addition::back_propagate(ForwardPropagation&,
                              BackPropagation& back_propagation,
                              size_t layer) const noexcept
{
    auto& delta_views = back_propagation.delta_views[layer];

    copy(delta_views[OutputDelta][0], delta_views[InputDelta0][0]);
    copy(delta_views[OutputDelta][0], delta_views[InputDelta1][0]);
}

// Serialization

void Addition::from_JSON(const JsonDocument& document)
{
    const Json* element = get_json_root(document, "Addition");

    const string new_label = read_json_string(element, "Label");
    const Shape new_input_shape = string_to_shape(read_json_string(element, "InputDimensions"));

    set(new_input_shape, new_label);
}

void Addition::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Addition");

    write_json(printer, {
        {"Label", label},
        {"InputDimensions", shape_to_string(input_shape)}
    });

    printer.close_element();
}

REGISTER(Layer, Addition, "Addition")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
