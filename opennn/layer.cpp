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
    Index total = 0;
    for(const Shape& s : get_parameter_shapes())
        total += s.size();
    return total;
}

vector<string> Layer::get_default_feature_names() const
{
    const Index inputs_number = get_inputs_number();

    vector<string> input_names(inputs_number);

    for(Index i = 0; i < inputs_number; ++i)
        input_names[i] = "input_" + to_string(i);

    return input_names;
}

vector<string> Layer::get_default_output_names() const
{
    const Index outputs_number = get_outputs_number();

    vector<string> output_names(outputs_number);

    for(Index i = 0; i < outputs_number; ++i)
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

        if(!is_aligned(pointer))
            throw runtime_error("Layer::link_parameters: unaligned memory in layer \"" + name + "\"");

        parameters[i] = TensorView(pointer, shapes[i], CUDNN_DATA_FLOAT);

        pointer += get_aligned_size(shapes[i].size());
    }
    return pointer;
}

type *Layer::link_states(type *pointer)
{
    const vector<Shape> shapes = get_state_shapes();
    states.resize(shapes.size());

    for (size_t i = 0; i < shapes.size(); ++i)
    {
        if (shapes[i].empty()) continue;

        if(!is_aligned(pointer))
            throw runtime_error("Layer::link_states: unaligned memory in layer \"" + name + "\"");

        states[i] = TensorView(pointer, shapes[i], CUDNN_DATA_FLOAT);

        pointer += get_aligned_size(shapes[i].size());
    }
    return pointer;
}

void Layer::add_gradients(const vector<TensorView>& output_delta_views) const
{
    if(output_delta_views.size() <= 1) return;

#ifndef OPENNN_WITH_CUDA
    VectorMap output_deltas = output_delta_views[0].as_vector();

    for(size_t i = 1; i < output_delta_views.size(); ++i)
        output_deltas.noalias() += output_delta_views[i].as_vector();
#else
    const size_t n = output_delta_views[0].size();

    for(size_t i = 1; i < output_delta_views.size(); ++i)
        if(output_delta_views[i].data)
            addition_cuda(n, output_delta_views[0].data, output_delta_views[i].data, output_delta_views[0].data);
#endif
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
