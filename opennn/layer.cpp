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

vector<pair<Shape, Type>> Layer::get_parameter_specs() const
{
    vector<pair<Shape, Type>> result;
    for (Operator* op : get_operators())
    {
        const auto specs = op->parameter_specs();
        result.insert(result.end(), specs.begin(), specs.end());
    }
    return result;
}

vector<pair<Shape, Type>> Layer::get_state_specs() const
{
    vector<pair<Shape, Type>> result;
    for (Operator* op : get_operators())
    {
        const auto specs = op->state_specs();
        result.insert(result.end(), specs.begin(), specs.end());
    }
    return result;
}

void Layer::distribute_to_operators(
    vector<TensorView>& views,
    void (Operator::*link)(const vector<TensorView>&),
    size_t (Operator::*count)() const)
{
    size_t offset = 0;
    for (Operator* op : get_operators())
    {
        const size_t n = (op->*count)();
        if (n == 0) continue;
        if (offset + n > views.size()) break;
        (op->*link)(vector<TensorView>(views.begin() + offset,
                                       views.begin() + offset + n));
        offset += n;
    }
}

Index Layer::get_parameters_number() const
{
    Index total = 0;
    for (const Shape& shape : get_parameter_shapes())
        total += shape.size();
    return total;
}

float* Layer::link_views_to_operators(vector<TensorView>& views, float* pointer,
                                       vector<pair<Shape, Type>> (Operator::*specs_fn)() const,
                                       void (Operator::*link_fn)(const vector<TensorView>&))
{
    views.clear();

    for (Operator* op : get_operators())
    {
        const auto specs = (op->*specs_fn)();
        if (specs.empty()) continue;

        const size_t start = views.size();

        for (const auto& [shape, dtype] : specs)
        {
            if (shape.empty())
            {
                views.emplace_back();
                continue;
            }

            if (!is_aligned(pointer))
                throw runtime_error("Layer::link_views_to_operators: unaligned memory in layer \"" + get_name() + "\"");
            views.emplace_back(pointer, shape, Type::FP32);
            pointer += get_aligned_size(shape.size());
        }

        (op->*link_fn)(vector<TensorView>(views.begin() + start, views.end()));
    }

    return pointer;
}

float* Layer::link_parameters(float* pointer)
{
    return link_views_to_operators(parameters, pointer,
                                   &Operator::parameter_specs, &Operator::link_parameters);
}

float* Layer::link_states(float* pointer)
{
    return link_views_to_operators(states, pointer,
                                   &Operator::state_specs, &Operator::link_states);
}

void Layer::set_input_shape(const Shape&)
{
    // Default no-op: layers override to update geometry when input changes.
}

void Layer::set_output_shape(const Shape& shape)
{
    set_input_shape(shape);
}

void Layer::from_JSON(const JsonDocument& document)
{
    if (const Json* root = get_json_root(document, name))
    {
        set_label(read_json_string(root, "Label"));
        set_input_shape(string_to_shape(read_json_string(root, "InputDimensions")));
        set_output_shape(string_to_shape(read_json_string(root, "OutputDimensions")));
        read_JSON_body(root);
        for (Operator* op : get_operators())
            op->from_JSON(root);
    }
}

void Layer::load_state_from_JSON(const JsonDocument& document)
{
    if (const Json* root = get_json_root(document, name))
        for (Operator* op : get_operators())
            op->load_state_from_JSON(root);
}

void Layer::to_JSON(JsonWriter& writer) const
{
    writer.open_element(name);

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
