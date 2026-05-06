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
    auto* self = const_cast<Layer*>(this);
    for (Operator* op : self->get_operators())
    {
        const auto specs = op->parameter_specs();
        result.insert(result.end(), specs.begin(), specs.end());
    }
    return result;
}

vector<pair<Shape, Type>> Layer::get_state_specs() const
{
    vector<pair<Shape, Type>> result;
    auto* self = const_cast<Layer*>(this);
    for (Operator* op : self->get_operators())
    {
        const auto specs = op->state_specs();
        result.insert(result.end(), specs.begin(), specs.end());
    }
    return result;
}

void Layer::distribute_to_operators(
    vector<TensorView>& views,
    void (Operator::*link)(const vector<TensorView>&),
    vector<pair<Shape, Type>> (Operator::*specs)() const)
{
    size_t offset = 0;
    for (Operator* op : get_operators())
    {
        const size_t count = (op->*specs)().size();
        if (count == 0) continue;
        if (offset + count > views.size()) break;
        const vector<TensorView> slice(views.begin() + offset,
                                       views.begin() + offset + count);
        (op->*link)(slice);
        offset += count;
    }
}

Index Layer::get_parameters_number() const
{
    Index total = 0;
    for (const Shape& shape : get_parameter_shapes())
        total += shape.size();
    return total;
}


float* Layer::link_views(float* pointer,
                         const vector<Shape>& shapes,
                         vector<TensorView>& views,
                         const char* tag) const
{
    views.resize(shapes.size());

    for (size_t i = 0; i < shapes.size(); ++i)
    {
        if (shapes[i].empty()) continue;

        if (!is_aligned(pointer))
            throw runtime_error(string("Layer::") + tag + ": unaligned memory in layer \"" + name + "\"");

        views[i] = TensorView(pointer, shapes[i], Type::FP32);

        pointer += get_aligned_size(shapes[i].size());
    }

    return pointer;
}

float* Layer::link_parameters(float* pointer)
{
    pointer = link_views(pointer, get_parameter_shapes(), parameters, "link_parameters");
    distribute_to_operators(parameters, &Operator::link_parameters, &Operator::parameter_specs);
    return pointer;
}

float* Layer::link_states(float* pointer)
{
    pointer = link_views(pointer, get_state_shapes(), states, "link_states");
    distribute_to_operators(states, &Operator::link_states, &Operator::state_specs);
    return pointer;
}

void Layer::set_input_shape(const Shape&)
{
    // Default no-op: layers override to update geometry when input changes.
}

void Layer::set_output_shape(const Shape&)
{
    // Default no-op: layers whose output is derived from input + config (Conv, Pool,
    // MultiHead, etc.) don't need to do anything here. Layers whose output is a
    // primary input (Dense, Bounding, Recurrent, ...) override.
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

    for (Operator* op : const_cast<Layer*>(this)->get_operators())
        op->to_JSON(writer);

    writer.close_element();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
