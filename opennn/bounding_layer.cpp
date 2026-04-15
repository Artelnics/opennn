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
    return input_shape;
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
}

void Bounding::set_output_shape(const Shape& new_output_shape)
{
    input_shape = new_output_shape;
}

void Bounding::set_lower_bound(const Index index, type new_lower_bound)
{
    const Shape output_shape = get_output_shape();

    if(lower_bounds.size() != output_shape[0])
    {
        lower_bounds.resize(output_shape[0]);
        lower_bounds.setConstant(-numeric_limits<type>::max());
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
        upper_bounds.setConstant(numeric_limits<type>::max());
    }

    upper_bounds[index] = new_upper_bound;
}

void Bounding::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    const TensorView& input = forward_propagation.views[layer][Inputs][0];
    TensorView& output = forward_propagation.views[layer][Outputs][0];

    if(bounding_method == BoundingMethod::NoBounding)
        copy(input, output);
    else
    {
        const TensorView lb(lower_bounds.data(), {lower_bounds.size()});
        const TensorView ub(upper_bounds.data(), {upper_bounds.size()});
        bounding(input, lb, ub, output);
    }
}

void Bounding::to_XML(XmlPrinter& printer) const
{
    printer.open_element("Bounding");

    const Shape output_shape = get_input_shape();

    add_xml_element(printer, "NeuronsNumber", to_string(output_shape[0]));

    for(Index i = 0; i < output_shape[0]; i++)
    {
        printer.open_element("Item");
        printer.push_attribute("Index", unsigned(i + 1));
/*
        add_xml_element(printer, "LowerBound", to_string(lower_bounds[i]));
        add_xml_element(printer, "UpperBound", to_string(upper_bounds[i]));
*/
        printer.close_element();
    }

    add_xml_element(printer, "BoundingMethod",
                     bounding_method == BoundingMethod::Bounding ? "Bounding" : "NoBounding");

    printer.close_element();
}

void Bounding::from_XML(const XmlDocument& document)
{
    const XmlElement* root_element = get_xml_root(document, "Bounding");

    const Index neurons_number = read_xml_index(root_element, "NeuronsNumber");

    set({ neurons_number });

    const auto* item_element = root_element->first_child_element("Item");

    for(Index i = 0; i < neurons_number && item_element; i++)
    {
        unsigned index = 0;
        item_element->query_unsigned_attribute("Index", &index);

        if (index != i + 1)
            throw runtime_error("Index " + to_string(index) + " is incorrect.\n");
/*
        lower_bounds[index - 1] = read_xml_type(item_element, "LowerBound");
        upper_bounds[index - 1] = read_xml_type(item_element, "UpperBound");
*/
        item_element = item_element->next_sibling_element("Item");
    }

    set_bounding_method(read_xml_string(root_element, "BoundingMethod"));
}

REGISTER(Layer, Bounding, "Bounding")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
