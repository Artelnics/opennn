//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "layer.h"

namespace opennn
{

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

    const type limit = (inputs_number + outputs_number > 0)
        ? sqrt(6.0 / (inputs_number + outputs_number))
        : type(0.05);

    for(auto& param : parameters)
    {
        if(param.empty()) continue;
        set_random_uniform(VectorMap(param.data, param.size()), -limit, limit);
    }
}

Index Layer::get_parameters_number() const
{
    return get_size(get_parameter_shapes());
}

string Layer::get_expression(const vector<string>&, const vector<string>&) const
{
    return string();
}

vector<string> Layer::get_default_feature_names() const
{
    const Index inputs_number = get_inputs_number();

    vector<string> input_names(inputs_number);

    for(Index i = 0; i < inputs_number; i++)
        input_names[i] = "input_" + to_string(i);

    return input_names;
}

vector<string> Layer::get_default_output_names() const
{
    const Index outputs_number = get_outputs_number();

    vector<string> output_names(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
        output_names[i] = "output_" + to_string(i);

    return output_names;
}

type *Layer::link_parameters(type *pointer)
{
    const vector<Shape> shapes = get_parameter_shapes();
    parameters.resize(shapes.size());

    for (size_t i = 0; i < shapes.size(); ++i)
    {
        if (shapes[i].empty()) continue;

        assert(is_aligned(pointer));

        parameters[i] = TensorView(pointer, shapes[i]);

        pointer += get_aligned_size(shapes[i].size());
    }
    return pointer;
}

void Layer::add_gradients(const vector<TensorView>& output_gradient_views) const
{
    VectorMap output_gradients = output_gradient_views[0].as_vector();

    for(size_t i = 1; i < output_gradient_views.size(); i++)
        output_gradients.noalias() += output_gradient_views[i].as_vector();
}

void Layer::forward_propagate(ForwardPropagation&, size_t, bool)
{
    throw runtime_error("This method is not implemented in the layer type (" + name + ").\n");
}

void Layer::set_input_shape(const Shape&)
{
    throw runtime_error("This method is not implemented in the layer type (" + name + ").\n");
}

void Layer::set_output_shape(const Shape&)
{
    throw runtime_error("This method is not implemented in the layer type (" + name + ").\n");
}

// @todo CUDA add_gradients using device pointers
} 

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
