//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "bounding_layer.h"
#include "tensors.h"

namespace opennn
{


BoundingLayer::BoundingLayer(const dimensions& neurons_number) : Layer()
{
    set(neurons_number[0]);

    set_default();
}


const BoundingLayer::BoundingMethod& BoundingLayer::get_bounding_method() const
{
    return bounding_method;
}


Index BoundingLayer::get_inputs_number() const
{
    return lower_bounds.dimension(0);
}


type BoundingLayer::get_lower_bound(const Index& i) const
{
    return lower_bounds[i];
}


const Tensor<type, 1>& BoundingLayer::get_lower_bounds() const
{
    return lower_bounds;
}


Index BoundingLayer::get_neurons_number() const
{
    return lower_bounds.dimension(0);
}


dimensions BoundingLayer::get_output_dimensions() const
{
    return { get_neurons_number() };
}


type BoundingLayer::get_upper_bound(const Index& i) const
{
    return upper_bounds(i);
}


const Tensor<type, 1>& BoundingLayer::get_upper_bounds() const
{
    return upper_bounds;
}


bool BoundingLayer::is_empty() const
{
    return get_neurons_number() == 0;
}


void BoundingLayer::set(const Index& new_neurons_number)
{
    set_neurons_number(new_neurons_number);

    set_default();
}


void BoundingLayer::set_bounding_method(const BoundingMethod& new_method)
{
    bounding_method = new_method;
}


void BoundingLayer::set_bounding_method(const string& new_method_string)
{
    if(new_method_string == "NoBounding")
        bounding_method = BoundingMethod::NoBounding;
    else if(new_method_string == "BoundingLayer")
        bounding_method = BoundingMethod::Bounding;
    else
        throw runtime_error("Unknown bounding method: " + new_method_string + ".\n");
}


void BoundingLayer::set_default()
{
    name = "bounding_layer";

    bounding_method = BoundingMethod::Bounding;

    layer_type = Layer::Type::Bounding;
}


void BoundingLayer::set_display(const bool& new_display)
{
    display = new_display;
}


void BoundingLayer::set_inputs_number(const Index& new_inputs_number)
{
    lower_bounds.resize(new_inputs_number);
    upper_bounds.resize(new_inputs_number);
}


void BoundingLayer::set_lower_bound(const Index& index, const type& new_lower_bound)
{
    const Index neurons_number = get_neurons_number();

    if(lower_bounds.size() != neurons_number)
    {
        lower_bounds.resize(neurons_number);
        lower_bounds.setConstant(-numeric_limits<type>::max());
    }

    lower_bounds[index] = new_lower_bound;
}


void BoundingLayer::set_lower_bounds(const Tensor<type, 1>& new_lower_bounds)
{
    lower_bounds = new_lower_bounds;
}


void BoundingLayer::set_neurons_number(const Index& new_neurons_number)
{
    lower_bounds.resize(new_neurons_number);
    upper_bounds.resize(new_neurons_number);

    lower_bounds.setConstant(-numeric_limits<type>::max());
    upper_bounds.setConstant(numeric_limits<type>::max());
}


void BoundingLayer::set_upper_bounds(const Tensor<type, 1>& new_upper_bounds)
{
    upper_bounds = new_upper_bounds;
}


void BoundingLayer::set_upper_bound(const Index& index, const type& new_upper_bound)
{
    const Index neurons_number = get_neurons_number();

    if(upper_bounds.size() != neurons_number)
    {
        upper_bounds.resize(neurons_number);
        upper_bounds.setConstant(numeric_limits<type>::max());
    }

    upper_bounds[index] = new_upper_bound;
}


void BoundingLayer::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                      unique_ptr<LayerForwardPropagation>& forward_propagation,
                                      const bool& is_training)
{
    const TensorMap<Tensor<type,2>> inputs = tensor_map_2(input_pairs[0]);

    BoundingLayerForwardPropagation* bounding_layer_forward_propagation =
        static_cast<BoundingLayerForwardPropagation*>(forward_propagation.get());

    Tensor<type,2>& outputs = bounding_layer_forward_propagation->outputs;

    if(bounding_method == BoundingMethod::Bounding)
    {
        const Index rows_number = inputs.dimension(0);
        const Index columns_number = inputs.dimension(1);

        #pragma omp parallel for
        for (Index j = 0; j < columns_number; j++)
        {
            const type& lower_bound = lower_bounds(j);
            const type& upper_bound = upper_bounds(j);

            for (Index i = 0; i < rows_number; i++)
                outputs(i, j) = min(max(inputs(i, j), lower_bound), upper_bound);
        }
    }
    else
    {
        outputs.device(*thread_pool_device) = inputs;
    }
}


string BoundingLayer::get_bounding_method_string() const
{
    if(bounding_method == BoundingMethod::Bounding)
        return "BoundingLayer";
    else if(bounding_method == BoundingMethod::NoBounding)
        return "NoBounding";
    else
        throw runtime_error("Unknown bounding method.\n");
}


string BoundingLayer::write_expression(const Tensor<string, 1>& input_names, const Tensor<string, 1>& output_names) const
{
    ostringstream buffer;

    buffer.precision(10);

    if(bounding_method == BoundingMethod::Bounding)
    {
        const Index neurons_number = get_neurons_number();

        for(Index i = 0; i < neurons_number; i++)
            buffer << output_names[i] << " = max(" << lower_bounds[i] << ", " << input_names[i] << ")\n"
                   << output_names[i] << " = min(" << upper_bounds[i] << ", " << output_names[i] << ")\n";
    }

    return buffer.str();
}


void BoundingLayer::print() const
{
    cout << "Bounding layer" << endl
         << "Lower bounds: " << lower_bounds << endl
         << "Upper bounds: " << upper_bounds << endl;
}


void BoundingLayer::to_XML(tinyxml2::XMLPrinter& printer) const
{
    printer.OpenElement("BoundingLayer");

    add_xml_element(printer, "BoundingNeuronsNumber", to_string(get_neurons_number()));

    const Index neurons_number = get_neurons_number();

    for (Index i = 0; i < neurons_number; i++) 
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


void BoundingLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    const auto* root_element = document.FirstChildElement("BoundingLayer");
    
    if (!root_element)
        throw runtime_error("BoundingLayer element is nullptr.\n");

    const Index neurons_number = read_xml_index(root_element, "BoundingNeuronsNumber");

    set(neurons_number);

    const auto* item_element = root_element->FirstChildElement("Item");

    for (Index i = 0; i < neurons_number && item_element; i++) 
    {
        unsigned index = 0;
        item_element->QueryUnsignedAttribute("Index", &index);

        if (index != i + 1) 
            throw runtime_error("Index " + std::to_string(index) + " is incorrect.\n");
        
        lower_bounds[index - 1] = read_xml_type(item_element, "LowerBound");
        upper_bounds[index - 1] = read_xml_type(item_element, "UpperBound");

        item_element = item_element->NextSiblingElement("Item");
    }

    set_bounding_method(read_xml_string(root_element, "BoundingMethod"));
}


pair<type*, dimensions> BoundingLayerForwardPropagation::get_outputs_pair() const
{
    const Index neurons_number = layer->get_neurons_number();

    return { (type*)outputs.data(), { batch_samples_number, neurons_number } };
}


void BoundingLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    const Index neurons_number = static_cast<BoundingLayer*>(layer)->get_neurons_number();

    batch_samples_number = new_batch_samples_number;

    outputs.resize(batch_samples_number, neurons_number);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
