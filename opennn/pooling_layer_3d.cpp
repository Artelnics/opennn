//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
#include "pooling_layer_3d.h"
#include "loss.h"

namespace opennn
{

Pooling3d::Pooling3d(const Shape& new_input_shape,
                     const PoolingMethod& new_pooling_method,
                     const string& new_name) : Layer()
{
    set(new_input_shape, new_pooling_method, new_name);
}


Shape Pooling3d::get_output_shape() const
{
    return {input_shape[1]};
}


string Pooling3d::write_pooling_method() const
{
    return pooling_method == PoolingMethod::MaxPooling ? "MaxPooling" : "AveragePooling";
}


void Pooling3d::set(const Shape& new_input_shape, const PoolingMethod& new_pooling_method, const string& new_label)
{
    name = "Pooling3d";
    input_shape = new_input_shape;
    pooling_method = new_pooling_method;
    set_label(new_label);
}


void Pooling3d::set_pooling_method(const string& new_pooling_method)
{
    if (new_pooling_method == "MaxPooling") pooling_method = PoolingMethod::MaxPooling;
    else if (new_pooling_method == "AveragePooling") pooling_method = PoolingMethod::AveragePooling;
    else throw runtime_error("Unknown pooling type: " + new_pooling_method);
}


void Pooling3d::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training)
{
    const TensorView& input = forward_propagation.views[layer][Inputs][0];
    TensorView& output = forward_propagation.views[layer].back()[0];

    if(pooling_method == PoolingMethod::MaxPooling)
    {
        TensorView& maximal_indices = forward_propagation.views[layer][MaximalIndices][0];
        max_pooling_3d_forward(input, output, maximal_indices, is_training);
    }
    else
        average_pooling_3d_forward(input, output);
}


void Pooling3d::back_propagate(ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation,
                               size_t layer) const
{
    const TensorView& input = forward_propagation.views[layer][Inputs][0];
    TensorView& output_gradient = back_propagation.backward_views[layer][OutputGradients][0];
    TensorView& input_gradient = back_propagation.backward_views[layer][InputGradients][0];

    if(pooling_method == PoolingMethod::MaxPooling)
    {
        const TensorView& maximal_indices = forward_propagation.views[layer][MaximalIndices][0];
        max_pooling_3d_backward(maximal_indices, output_gradient, input_gradient);
    }
    else
        average_pooling_3d_backward(input, output_gradient, input_gradient);
}


void Pooling3d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Pooling3d");
    add_xml_element(printer, "InputDimensions", shape_to_string(get_input_shape()));
    add_xml_element(printer, "PoolingMethod", write_pooling_method());
    printer.CloseElement();
}


void Pooling3d::from_XML(const XMLDocument& document)
{
    const XMLElement* element = get_xml_root(document, "Pooling3d");

    set_input_shape(string_to_shape(read_xml_string(element, "InputDimensions")));
    set_pooling_method(read_xml_string(element, "PoolingMethod"));
}

#ifdef CUDA
    // @todo CUDA path
#endif

REGISTER(Layer, Pooling3d, "Pooling3d")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
