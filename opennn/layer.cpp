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

void Layer::set_parameters_random()
{
    for (auto& param : parameters)
    {
        if (param.empty()) continue;
        set_random_uniform(param.as_vector());
    }
}

void Layer::set_parameters_glorot()
{
    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    const float limit = (inputs_number + outputs_number > 0)
        ? sqrt(6.0 / (inputs_number + outputs_number))
        : 0.05f;

    for (auto& param : parameters)
    {
        if (param.empty()) continue;
        set_random_uniform(param.as_vector(), -limit, limit);
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
    throw runtime_error("This method is not implemented in the layer type (" + name + ").\n");
}

void Layer::set_output_shape(const Shape&)
{
    throw runtime_error("This method is not implemented in the layer type (" + name + ").\n");
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
