//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N C A T E N A T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "concatenation_layer.h"
#include "json.h"

namespace opennn
{

Concatenation::Concatenation(const Shape& new_input_shape,
                             const vector<Index>& per_input_channels,
                             const string& new_label)
    : Layer(LayerType::Concatenation)
{
    operators = {&concatenation};
    set(new_input_shape, per_input_channels, new_label);
}

Shape Concatenation::get_output_shape() const
{
    if (input_shape.empty()) return {};
    const Index total_channels = accumulate(concatenation.input_channels.begin(), concatenation.input_channels.end(), Index(0));
    return { input_shape[0], input_shape[1], total_channels };
}

vector<TensorSpec> Concatenation::get_backward_specs(Index batch_size) const
{
    const size_t n = static_cast<size_t>(ssize(concatenation.input_channels));
    vector<TensorSpec> specs;
    specs.reserve(n);
    for (size_t i = 0; i < n; ++i)
        specs.push_back({ Shape{batch_size, input_shape[0], input_shape[1], concatenation.input_channels[i]},
                          compute_dtype });
    return specs;
}

void Concatenation::set(const Shape& new_input_shape,
                        const vector<Index>& per_input_channels,
                        const string& new_label)
{
    if (!new_input_shape.empty())
        check_rank(new_input_shape, {3}, "Concatenation", "input");

    input_shape = new_input_shape;
    concatenation.input_channels = per_input_channels;
    set_label(new_label);

    concatenation.input_delta_slots.resize(per_input_channels.size());
    iota(concatenation.input_delta_slots.begin(), concatenation.input_delta_slots.end(), size_t(1));

    configure_operator();
}

void Concatenation::set_input_shape(const Shape& new_input_shape)
{
    check_rank(new_input_shape, {3}, "Concatenation", "input");
    input_shape = new_input_shape;
    configure_operator();
}

void Concatenation::configure_operator()
{
    if (input_shape.empty()) return;
    concatenation.set(input_shape[0], input_shape[1], concatenation.input_channels);
}

void Concatenation::read_JSON_body(const Json* root)
{
    istringstream stream(read_json_string(root, "InputChannels"));
    vector<Index> channels;
    for (Index value; stream >> value; )
        channels.push_back(value);

    set(input_shape, channels, label);
}

void Concatenation::write_JSON_body(JsonWriter& writer) const
{
    add_json_field(writer, "InputChannels", vector_to_string(concatenation.input_channels, " "));
}

REGISTER(Layer, Concatenation, "Concatenation")

namespace
{
    const bool Concatenation_legacy_registered = []
    {
        Registry<Layer>::instance().register_component("Concatenate", []
        {
            return make_unique<Concatenation>();
        });
        return true;
    }();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
