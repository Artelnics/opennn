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

/*
type Bounding::get_lower_bound(const Index i) const
{
    return lower_bounds[i];
}

const VectorR& Bounding::get_lower_bounds() const
{
    return lower_bounds;
}

type Bounding::get_upper_bound(const Index i) const
{
    return upper_bounds(i);
}

const VectorR& Bounding::get_upper_bounds() const
{
    return upper_bounds;
}
*/

void Bounding::set(const Shape& new_output_shape, const string& new_label)
{
    set_output_shape(new_output_shape);

    label = new_label;

    bounding_method = BoundingMethod::Bounding;

    name = "Bounding";

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

/*
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

void Bounding::set_output_shape(const Shape& new_output_shape)
{
    lower_bounds.resize(new_output_shape[0]);
    upper_bounds.resize(new_output_shape[0]);

    lower_bounds.setConstant(-MAX);
    upper_bounds.setConstant(MAX);
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
*/

void Bounding::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    const TensorView& input = forward_propagation.views[layer][Inputs][0];
    TensorView& output = forward_propagation.views[layer][Outputs][0];

    if(bounding_method == BoundingMethod::NoBounding)
        copy(input, output);
    else
    {
        TensorView lb(lower_bounds.data(), {lower_bounds.size()});
        TensorView ub(upper_bounds.data(), {upper_bounds.size()});
        bounding(input, lb, ub, output);
    }
}

string Bounding::get_expression(const vector<string>& new_input_names, const vector<string>& new_output_names) const
{
    const vector<string> input_names = new_input_names.empty()
                                           ? get_default_feature_names()
                                           : new_input_names;

    const vector<string> output_names = new_output_names.empty()
                                            ? get_default_output_names()
                                            : new_output_names;

    if (bounding_method == BoundingMethod::NoBounding)
        return string();

    ostringstream buffer;

    buffer.precision(10);

    const Shape output_shape = get_output_shape();
/*
    for(Index i = 0; i < output_shape[0]; i++)
        buffer << output_names[i] << " = max(" << lower_bounds[i] << ", " << input_names[i] << ")\n"
               << output_names[i] << " = min(" << upper_bounds[i] << ", " << output_names[i] << ")\n";
*/
    return buffer.str();
}

void Bounding::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Bounding");

    const Shape output_shape = get_input_shape();

    add_xml_element(printer, "NeuronsNumber", to_string(output_shape[0]));

    for(Index i = 0; i < output_shape[0]; i++)
    {
        printer.OpenElement("Item");
        printer.PushAttribute("Index", unsigned(i + 1));
/*
        add_xml_element(printer, "LowerBound", to_string(lower_bounds[i]));
        add_xml_element(printer, "UpperBound", to_string(upper_bounds[i]));
*/
        printer.CloseElement();
    }

    add_xml_element(printer, "BoundingMethod",
                     bounding_method == BoundingMethod::Bounding ? "Bounding" : "NoBounding");

    printer.CloseElement();
}

void Bounding::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = get_xml_root(document, "Bounding");

    const Index neurons_number = read_xml_index(root_element, "NeuronsNumber");

    set({ neurons_number });

    const auto* item_element = root_element->FirstChildElement("Item");

    for(Index i = 0; i < neurons_number && item_element; i++)
    {
        unsigned index = 0;
        item_element->QueryUnsignedAttribute("Index", &index);

        if (index != i + 1)
            throw runtime_error("Index " + to_string(index) + " is incorrect.\n");
/*
        lower_bounds[index - 1] = read_xml_type(item_element, "LowerBound");
        upper_bounds[index - 1] = read_xml_type(item_element, "UpperBound");
*/
        item_element = item_element->NextSiblingElement("Item");
    }

    set_bounding_method(read_xml_string(root_element, "BoundingMethod"));
}

REGISTER(Layer, Bounding, "Bounding")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
