//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "layer.h"
#include "operators.h"

namespace opennn
{

vector<TensorSpec> Layer::get_parameter_specs() const
{
    vector<TensorSpec> result;
    for (Operator* op : get_operators())
    {
        auto specs = op->parameter_specs();
        result.insert(result.end(),
                      make_move_iterator(specs.begin()),
                      make_move_iterator(specs.end()));
    }

    return result;
}

vector<TensorSpec> Layer::get_state_specs() const
{
    vector<TensorSpec> result;
    for (Operator* op : get_operators())
    {
        auto specs = op->state_specs();
        result.insert(result.end(),
                      make_move_iterator(specs.begin()),
                      make_move_iterator(specs.end()));
    }

    return result;
}

void Layer::redistribute_parameters_to_operators()
{
    size_t offset = 0;
    for (Operator* op : get_operators())
    {
        const size_t n = op->parameter_specs().size();
        if (n == 0) continue;
        if (offset + n > parameters.size()) break;
        op->link_parameters(span(parameters).subspan(offset, n));
        offset += n;
    }
}

Index Layer::get_parameters_number() const
{
    Index count = 0;

    for (Operator* op : get_operators())
        for (const auto& [shape, _] : op->parameter_specs())
            count += shape.size();

    return count;
}

float* Layer::link_views_to_operators(vector<TensorView>& views, float* pointer,
                                      vector<TensorSpec> (Operator::*specs_fn)() const,
                                      void (Operator::*link_fn)(span<const TensorView>),
                                      Device device)
{
    views.clear();

    for (Operator* op : get_operators())
    {
        const auto specs = (op->*specs_fn)();
        if (specs.empty()) continue;

        const size_t start = views.size();

        for (const auto& [shape, _] : specs)
        {
            if (shape.empty()) { views.emplace_back(); continue; }

            throw_if(!is_aligned(pointer),
                     format("Layer::link_views_to_operators: unaligned memory in layer \"{}\"", get_name()));

            views.emplace_back(pointer, shape, Type::FP32, device);
            pointer += get_aligned_size(shape.size());
        }

        (op->*link_fn)(span(views).subspan(start));
    }

    return pointer;
}

float* Layer::link_states(float* pointer, Device device)
{
    return link_views_to_operators(states, pointer,
                                   &Operator::state_specs,
                                   &Operator::link_states,
                                   device);
}

float* Layer::link_gradients(float* pointer, vector<TensorView>& gradient_views, Device device)
{
    return link_views_to_operators(gradient_views, pointer,
                                   &Operator::parameter_specs,
                                   &Operator::link_gradients,
                                   device);
}

void Layer::set_input_shape(const Shape&)
{
}

void Layer::set_output_shape(const Shape&)
{
}

void Layer::from_JSON(const JsonDocument& document)
{
    const Json* root = get_json_root(document, get_name());
    if (!root) return;

    const string json_label = read_json_string(root, "Label");

    set_input_shape(string_to_shape(read_json_string(root, "InputDimensions")));
    set_output_shape(string_to_shape(read_json_string(root, "OutputDimensions")));
    set_label(json_label);

    read_JSON_body(root);
    for (Operator* op : get_operators())
        op->from_JSON(root);
}

void Layer::load_state_from_JSON(const JsonDocument& document)
{
    if (const Json* root = get_json_root(document, get_name()))
        for (Operator* op : get_operators())
            op->load_state_from_JSON(root);
}

void Layer::to_JSON(JsonWriter& writer) const
{
    writer.open_element(get_name());

    add_json_field(writer, "Label", label);
    add_json_field(writer, "InputDimensions", shape_to_string(get_input_shape()));
    add_json_field(writer, "OutputDimensions", shape_to_string(get_output_shape()));

    write_JSON_body(writer);

    for (Operator* op : get_operators())
        op->to_JSON(writer);

    writer.close_element();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
