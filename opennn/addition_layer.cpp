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

Addition::Addition(const Shape& new_input_shape, const string& new_name, Index new_inputs_number)
    : Layer(LayerType::Addition)
{
    operators = {&add};

    set(new_input_shape, new_name, new_inputs_number);
}

vector<TensorSpec> Addition::get_backward_specs(Index batch_size) const
{
    return vector<TensorSpec>(inputs_number, {Shape{batch_size}.append(input_shape), compute_dtype});
}

void Addition::set(const Shape& new_input_shape, const string& new_label, Index new_inputs_number)
{
    check_rank(new_input_shape, {2, 3}, "Addition", "input");

    throw_if(new_inputs_number < 2, "Addition: inputs_number must be >= 2.");

    input_shape = new_input_shape;
    inputs_number = new_inputs_number;

    add.input_delta_slots.resize(size_t(inputs_number));
    iota(add.input_delta_slots.begin(), add.input_delta_slots.end(), size_t(1));

    set_label(new_label);
}

void Addition::read_JSON_body(const Json* addition_layer_element)
{
    const Index new_inputs_number = read_json_index(addition_layer_element, "InputsNumber");

    if (new_inputs_number >= 2)
        set(input_shape, label, new_inputs_number);
}

void Addition::write_JSON_body(JsonWriter& printer) const
{
    add_json_field(printer, "InputsNumber", to_string(inputs_number));
}

REGISTER(Layer, Addition, "Addition")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
