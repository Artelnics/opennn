//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "bounding_layer.h"

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


Shape Bounding::get_input_shape() const
{
    return { lower_bounds.rows() };
}


type Bounding::get_lower_bound(const Index i) const
{
    return lower_bounds[i];
}


const VectorR& Bounding::get_lower_bounds() const
{
    return lower_bounds;
}


Shape Bounding::get_output_shape() const
{
    return { lower_bounds.rows() };
}


type Bounding::get_upper_bound(const Index i) const
{
    return upper_bounds(i);
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
    lower_bounds.resize(new_input_shape[0]);
    upper_bounds.resize(new_input_shape[0]);
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


void Bounding::set_output_shape(const Shape& new_output_shape)
{
    lower_bounds.resize(new_output_shape[0]);
    upper_bounds.resize(new_output_shape[0]);

    lower_bounds.setConstant(-numeric_limits<type>::max());
    upper_bounds.setConstant(numeric_limits<type>::max());
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


void Bounding::forward_propagate(const vector<TensorView>& input_views,
                                 unique_ptr<LayerForwardPropagation>& forward_propagation,
                                 bool)
{
    const MatrixMap inputs = matrix_map(input_views[0]);
    MatrixMap outputs = matrix_map(forward_propagation->outputs);

    if(bounding_method == BoundingMethod::NoBounding)
    {
        outputs = inputs;
        return;
    }

    const Index columns_number = inputs.cols();

    for(Index j = 0; j < columns_number; j++)
        outputs.col(j).array() = inputs.col(j).array().max(lower_bounds(j)).min(upper_bounds(j));
}


string Bounding::get_bounding_method_string() const
{
    if(bounding_method == BoundingMethod::Bounding)
        return "Bounding";
    else if(bounding_method == BoundingMethod::NoBounding)
        return "NoBounding";
    else
        throw runtime_error("Unknown bounding method.\n");
}


string Bounding::get_expression(const vector<string>& new_feature_names, const vector<string>& new_output_names) const
{
    const vector<string> input_names = new_feature_names.empty()
                                           ? get_default_feature_names()
                                           : new_feature_names;

    const vector<string> output_names = new_output_names.empty()
                                            ? get_default_output_names()
                                            : new_output_names;


    if (bounding_method == BoundingMethod::NoBounding)
        return string();

    ostringstream buffer;

    buffer.precision(10);

    const Shape output_shape = get_output_shape();

    for(Index i = 0; i < output_shape[0]; i++)
        buffer << output_names[i] << " = max(" << lower_bounds[i] << ", " << input_names[i] << ")\n"
               << output_names[i] << " = min(" << upper_bounds[i] << ", " << output_names[i] << ")\n";

    return buffer.str();
}


void Bounding::print() const
{
    cout << "Bounding layer" << endl
         << "Lower bounds: " << lower_bounds << endl
         << "Upper bounds: " << upper_bounds << endl;
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

        add_xml_element(printer, "LowerBound", to_string(lower_bounds[i]));
        add_xml_element(printer, "UpperBound", to_string(upper_bounds[i]));

        printer.CloseElement();
    }

    add_xml_element(printer, "BoundingMethod", get_bounding_method_string());

    printer.CloseElement();
}


void Bounding::from_XML(const XMLDocument& document)
{
    const auto* root_element = document.FirstChildElement("Bounding");

    if(!root_element)
        throw runtime_error("Bounding element is nullptr.\n");

    const Index neurons_number = read_xml_index(root_element, "NeuronsNumber");

    set({ neurons_number });

    const auto* item_element = root_element->FirstChildElement("Item");

    for(Index i = 0; i < neurons_number && item_element; i++)
    {
        unsigned index = 0;
        item_element->QueryUnsignedAttribute("Index", &index);

        if (index != i + 1)
            throw runtime_error("Index " + to_string(index) + " is incorrect.\n");

        lower_bounds[index - 1] = read_xml_type(item_element, "LowerBound");
        upper_bounds[index - 1] = read_xml_type(item_element, "UpperBound");

        item_element = item_element->NextSiblingElement("Item");
    }

    set_bounding_method(read_xml_string(root_element, "BoundingMethod"));
}


BoundingForwardPropagation::BoundingForwardPropagation(const Index new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


void BoundingForwardPropagation::initialize()
{
    const Index neurons_number = static_cast<Bounding*>(layer)->get_output_shape()[0];

    outputs.shape = {batch_size, neurons_number};
}


void BoundingForwardPropagation::print() const
{
/*
    cout << "Outputs:" << endl
         << outputs.shape << endl;
*/
}

REGISTER(Layer, Bounding, "Bounding")
REGISTER(LayerForwardPropagation, BoundingForwardPropagation, "Bounding")


#ifdef OPENNN_CUDA

void Bounding::forward_propagate(const vector<TensorViewCuda>& inputs,
                                      unique_ptr<LayerForwardPropagationCuda>& forward_propagation,
                                      bool)
{
    // @todo Implement bounding in CUDA
}


BoundingForwardPropagationCuda::BoundingForwardPropagationCuda(const Index new_batch_size, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void BoundingForwardPropagationCuda::initialize()
{
    // @todo
}


void BoundingForwardPropagationCuda::print() const
{
    const Index outputs_number = layer->get_outputs_number();

    cout << "Bounding CUDA Outputs (pass-through):" << endl
        << matrix_from_device(outputs.data, batch_size, outputs_number) << endl;
}

REGISTER(LayerForwardPropagationCuda, BoundingForwardPropagationCuda, "Bounding")

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
