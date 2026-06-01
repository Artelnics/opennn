//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U P S A M P L E   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "upsample_layer.h"
#include "json.h"

namespace opennn
{

Upsample::Upsample(const Shape& new_input_shape,
                   Index new_scale_factor,
                   const string& new_label)
    : Layer(LayerType::Upsample, /*trainable=*/false)
{
    operators = {&upsample};
    set(new_input_shape, new_scale_factor, new_label);
}

Shape Upsample::get_output_shape() const
{
    if (input_shape.empty()) return {};
    return { input_shape[0] * upsample.scale_factor,
             input_shape[1] * upsample.scale_factor,
             input_shape[2] };
}

void Upsample::set(const Shape& new_input_shape,
                   Index new_scale_factor,
                   const string& new_label)
{
    if (!new_input_shape.empty())
        check_rank(new_input_shape, {3}, "Upsample", "input");

    input_shape = new_input_shape;
    upsample.scale_factor = new_scale_factor;
    set_label(new_label);
    configure_operator();
}

void Upsample::set_input_shape(const Shape& new_input_shape)
{
    check_rank(new_input_shape, {3}, "Upsample", "input");
    input_shape = new_input_shape;
    configure_operator();
}

void Upsample::set_scale_factor(Index new_scale_factor)
{
    upsample.scale_factor = new_scale_factor;
    configure_operator();
}

void Upsample::configure_operator()
{
    if (input_shape.empty()) return;
    upsample.set(input_shape[0], input_shape[1], input_shape[2], upsample.scale_factor);
    is_trainable = true;
}

void Upsample::read_JSON_body(const Json* root)
{
    set_scale_factor(read_json_index(root, "ScaleFactor"));
}

void Upsample::write_JSON_body(JsonWriter& writer) const
{
    add_json_field(writer, "ScaleFactor", to_string(upsample.scale_factor));
}

REGISTER(Layer, Upsample, "Upsample")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
