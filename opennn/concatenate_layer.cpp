//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N C A T E N A T E   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "concatenate_layer.h"
#include "json.h"

namespace opennn
{

Concatenate::Concatenate(const Shape& new_input_shape,
                         const vector<Index>& per_input_channels,
                         const string& new_label)
    : Layer(LayerType::Concatenate)
{
    operators = {&concatenate};
    set(new_input_shape, per_input_channels, new_label);
}

Shape Concatenate::get_output_shape() const
{
    if (input_shape.empty()) return {};
    const Index total_channels = accumulate(concatenate.input_channels.begin(), concatenate.input_channels.end(), Index(0));
    return { input_shape[0], input_shape[1], total_channels };
}

vector<TensorSpec> Concatenate::get_backward_specs(Index batch_size) const
{
    const size_t n = static_cast<size_t>(ssize(concatenate.input_channels));
    vector<TensorSpec> specs;
    specs.reserve(n);
    for (size_t i = 0; i < n; ++i)
        specs.push_back({ Shape{batch_size, input_shape[0], input_shape[1], concatenate.input_channels[i]},
                          compute_dtype });
    return specs;
}

void Concatenate::set(const Shape& new_input_shape,
                      const vector<Index>& per_input_channels,
                      const string& new_label)
{
    if (!new_input_shape.empty())
        check_rank(new_input_shape, {3}, "Concatenate", "input");

    input_shape = new_input_shape;
    concatenate.input_channels = per_input_channels;
    set_label(new_label);

    concatenate.input_delta_slots.resize(per_input_channels.size());
    iota(concatenate.input_delta_slots.begin(), concatenate.input_delta_slots.end(), size_t(1));

    configure_operator();
}

void Concatenate::set_input_shape(const Shape& new_input_shape)
{
    check_rank(new_input_shape, {3}, "Concatenate", "input");
    input_shape = new_input_shape;
    configure_operator();
}

void Concatenate::configure_operator()
{
    if (input_shape.empty()) return;
    concatenate.set(input_shape[0], input_shape[1], concatenate.input_channels);
}

void Concatenate::read_JSON_body(const Json* root)
{
    istringstream stream(read_json_string(root, "InputChannels"));
    vector<Index> channels;
    for (Index value; stream >> value; )
        channels.push_back(value);

    set(input_shape, channels, label);
}

void Concatenate::write_JSON_body(JsonWriter& writer) const
{
    add_json_field(writer, "InputChannels", vector_to_string(concatenate.input_channels, " "));
}

REGISTER(Layer, Concatenate, "Concatenate")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
