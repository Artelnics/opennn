//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <vector>

#include "layer.h"

namespace opennn
{

const string& Layer::get_label() const
{
    return label;
}


const string& Layer::get_name() const
{
    return name;
}


void Layer::set_label(const string& new_label)
{
    label = new_label;
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
/*
    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    const type limit = sqrt(6.0 / (inputs_number + outputs_number));

    const vector<TensorView*> parameter_views = get_parameter_views();

    for(const TensorView* view : parameter_views)
        set_random_uniform(VectorMap(view->data, view->size()), -limit, limit);
*/
}


Index Layer::get_parameters_number()
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


bool Layer::get_is_trainable() const
{
    return is_trainable;
}



type *Layer::link_parameters(type *pointer)
{
    const vector<Shape> shapes = get_parameter_shapes();
    parameters.resize(shapes.size());

    for (size_t i = 0; i < shapes.size(); ++i)
    {
        if (shapes[i].count() == 0) continue;

        assert(is_aligned(pointer));

        parameters[i] = TensorView(pointer, shapes[i]);

        pointer += get_aligned_size(shapes[i].count());
    }
    return pointer;
}


void Layer::add_gradients(const vector<TensorView>& output_gradient_views) const
{
    TensorMap3 output_gradients = tensor_map<3>(output_gradient_views[0]);

    for(Index i = 1; i < Index(output_gradient_views.size()); i++)
        output_gradients.device(get_device()) += tensor_map<3>(output_gradient_views[i]);
}


Index Layer::get_inputs_number() const
{
    return get_input_shape().count();
}


Index Layer::get_outputs_number() const
{
    return get_output_shape().count();
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

#ifdef CUDA

void Layer::add_gradients(const vector<TensorViewCuda>& output_gradient_views) const
{
    if (output_gradient_views.size() <= 1) return;
    if (!output_gradient_views[0].data) return;

    const size_t n = output_gradient_views[0].size();

    for (size_t i = 1; i < output_gradient_views.size(); i++)
        if (output_gradient_views[i].data)
            addition_cuda(n, output_gradient_views[0].data, output_gradient_views[i].data, output_gradient_views[0].data);
}

#endif
} 


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
