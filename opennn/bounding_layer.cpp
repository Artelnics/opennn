//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
#include "bounding_layer.h"
#include "neural_network.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

Bounding::Bounding(const Shape& output_shape, const string& new_name) : Layer()
{
    set(output_shape, new_name);
}

const Bounding::BoundingMethod& Bounding::get_bounding_method() const
{
    return bounding_method;
}

Shape Bounding::get_output_shape() const
{
    return output_shape;
}

const VectorR& Bounding::get_lower_bounds() const
{
    return lower_bounds;
}

const VectorR& Bounding::get_upper_bounds() const
{
    return upper_bounds;
}

void Bounding::set(const Shape& new_output_shape, const string& new_label)
{
    set_output_shape(new_output_shape);

    label = new_label;

    bounding_method = BoundingMethod::Bounding;

    name = "Bounding";
    layer_type = LayerType::Bounding;

    is_trainable = false;
}

void Bounding::set_bounding_method(const BoundingMethod& new_method)
{
    bounding_method = new_method;
}

void Bounding::set_bounding_method(const string& new_method_string)
{
    if(new_method_string == "NoBounding" || new_method_string == "No bounding")
        bounding_method = BoundingMethod::NoBounding;
    else if(new_method_string == "Positive outputs" || new_method_string == "Data range" || new_method_string == "Bounding")
        bounding_method = BoundingMethod::Bounding;
    else
        throw runtime_error("Unknown bounding method: " + new_method_string + ".\n");
}

void Bounding::set_input_shape(const Shape& new_input_shape)
{
    if(!output_shape.empty() && new_input_shape != output_shape)
        throw runtime_error("Bounding: input shape mismatch with output shape.");
}

void Bounding::set_output_shape(const Shape& new_output_shape)
{
    output_shape = new_output_shape;

    const Index n = new_output_shape[0];

    lower_bounds.resize(n);
    lower_bounds.setConstant(-numeric_limits<type>::max());

    upper_bounds.resize(n);
    upper_bounds.setConstant(numeric_limits<type>::max());
}

void Bounding::set_lower_bound(const Index index, type new_lower_bound)
{
    const Shape output_shape = get_output_shape();

    if(lower_bounds.size() != output_shape[0])
    {
        lower_bounds.resize(output_shape[0]);
        lower_bounds.setConstant(-MAX);
    }

    lower_bounds[index] = new_lower_bound;
}

void Bounding::set_lower_bounds(const VectorR& new_lower_bounds)
{
    lower_bounds = new_lower_bounds;
}

void Bounding::set_upper_bounds(const VectorR& new_upper_bounds)
{
    upper_bounds = new_upper_bounds;
}

void Bounding::set_upper_bound(const Index index, type new_upper_bound)
{
    const Shape output_shape = get_output_shape();

    if(upper_bounds.size() != output_shape[0])
    {
        upper_bounds.resize(output_shape[0]);
        upper_bounds.setConstant(MAX);
    }

    upper_bounds[index] = new_upper_bound;
}

type* Bounding::link_states(type* pointer)
{
    type* next = Layer::link_states(pointer);

    if(bounding_method == BoundingMethod::NoBounding) return next;

    if(lower_bounds.size() == states[Lower].size() && states[Lower].data)
        VectorMap(states[Lower].data, states[Lower].size()) = lower_bounds;

    if(upper_bounds.size() == states[Upper].size() && states[Upper].data)
        VectorMap(states[Upper].data, states[Upper].size()) = upper_bounds;

    return next;
}

void Bounding::forward_propagate(ForwardPropagation& forward_propagation, size_t layer_index, bool) noexcept
{
    auto& forward_views = forward_propagation.views[layer_index];

    if(bounding_method == BoundingMethod::NoBounding)
    {
        copy(forward_views[Input][0], forward_views[Output][0]);
        return;
    }

    bounding(forward_views[Input][0],
             states[Lower],
             states[Upper],
             forward_views[Output][0]);
}

void Bounding::to_XML(XmlPrinter& printer) const
{
    printer.open_element("Bounding");

    const Shape output_shape = get_output_shape();

    add_xml_element(printer, "NeuronsNumber", to_string(output_shape[0]));

    add_xml_element(printer, "LowerBounds", vector_to_string(lower_bounds));
    add_xml_element(printer, "UpperBounds", vector_to_string(upper_bounds));

    add_xml_element(printer, "BoundingMethod",
                     bounding_method == BoundingMethod::Bounding ? "Bounding" : "NoBounding");

    printer.close_element();
}

void Bounding::from_XML(const XmlDocument& document)
{
    const XmlElement* root_element = get_xml_root(document, "Bounding");

    const Index neurons_number = read_xml_index(root_element, "NeuronsNumber");

    set({ neurons_number });

    string_to_vector(read_xml_string(root_element, "LowerBounds"), lower_bounds);
    string_to_vector(read_xml_string(root_element, "UpperBounds"), upper_bounds);

    set_bounding_method(read_xml_string(root_element, "BoundingMethod"));
}

REGISTER(Layer, Bounding, "Bounding")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
