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
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

Pooling3d::Pooling3d(const Shape& new_input_shape,
                     const PoolingMethod& new_pooling_method,
                     const string& new_name) : Layer()
{
    set(new_input_shape, new_pooling_method, new_name);
}

// Getters

Shape Pooling3d::get_output_shape() const
{
    return {input_shape[1]};
}

string Pooling3d::write_pooling_method() const
{
    return pooling_method_to_string(pooling_method);
}

// Setters

void Pooling3d::set(const Shape& new_input_shape, const PoolingMethod& new_pooling_method, const string& new_label)
{
    name = "Pooling3d";
    layer_type = LayerType::Pooling3d;
    input_shape = new_input_shape;
    pooling_method = new_pooling_method;
    set_label(new_label);
}

void Pooling3d::set_pooling_method(const string& new_pooling_method)
{
    pooling_method = string_to_pooling_method(new_pooling_method);
}

// Forward / back propagation

void Pooling3d::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training) noexcept
{
    auto& forward_views = forward_propagation.views[layer];

    const TensorView& input = forward_views[Inputs][0];
    TensorView& output = forward_views.back()[0];

    if(pooling_method == PoolingMethod::MaxPooling)
        max_pooling_3d_forward(input, output, forward_views[MaximalIndices][0], is_training);
    else
        average_pooling_3d_forward(input, output);
}

void Pooling3d::back_propagate(ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation,
                               size_t layer) const noexcept
{
    auto& forward_views = forward_propagation.views[layer];
    auto& backward_views = back_propagation.backward_views[layer];

    const TensorView& input = forward_views[Inputs][0];
    TensorView& output_gradient = backward_views[OutputGradients][0];
    TensorView& input_gradient = backward_views[InputGradients][0];

    if(pooling_method == PoolingMethod::MaxPooling)
        max_pooling_3d_backward(forward_views[MaximalIndices][0], output_gradient, input_gradient);
    else
        average_pooling_3d_backward(input, output_gradient, input_gradient);
}

// Serialization

void Pooling3d::from_XML(const XmlDocument& document)
{
    const XmlElement* element = get_xml_root(document, "Pooling3d");

    set_input_shape(string_to_shape(read_xml_string(element, "InputDimensions")));
    set_pooling_method(read_xml_string(element, "PoolingMethod"));
}

void Pooling3d::to_XML(XmlPrinter& printer) const
{
    printer.open_element("Pooling3d");
    write_xml_properties(printer, {
        {"InputDimensions", shape_to_string(get_input_shape())},
        {"PoolingMethod", write_pooling_method()}
    });
    printer.close_element();
}

REGISTER(Layer, Pooling3d, "Pooling3d")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
