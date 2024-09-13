//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "bounding_layer.h"

namespace opennn
{

BoundingLayer::BoundingLayer() : Layer()
{
    set();

    set_default();
}


BoundingLayer::BoundingLayer(const Index& neurons_number) : Layer()
{
    set(neurons_number);

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
#ifdef OPENNN_DEBUG

    const Index neurons_number = get_neurons_number();

    if(i >= neurons_number)
        throw runtime_error("Index must be less than number of bounding neurons.\n");

#endif

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
    const Index neurons_number = get_neurons_number();

    return { neurons_number };
}


type BoundingLayer::get_upper_bound(const Index& i) const
{
#ifdef OPENNN_DEBUG

    const Index neurons_number = get_neurons_number();

    if(neurons_number == 0)
        throw runtime_error("Number of bounding neurons is zero.\n");
    else if(i >= neurons_number)
        throw runtime_error("Index must be less than number of bounding neurons.\n");

#endif

    return upper_bounds(i);
}


const Tensor<type, 1>& BoundingLayer::get_upper_bounds() const
{
    return upper_bounds;
}


bool BoundingLayer::is_empty() const
{
    if(get_neurons_number() == 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}


void BoundingLayer::set()
{
    bounding_method = BoundingMethod::Bounding;

    lower_bounds.resize(0);
    upper_bounds.resize(0);

    set_default();
}


void BoundingLayer::set(const Index& new_neurons_number)
{
    set_neurons_number(new_neurons_number);

    set_default();
}


void BoundingLayer::set(const tinyxml2::XMLDocument& bounding_layer_document)
{
    set_default();

    from_XML(bounding_layer_document);
}


void BoundingLayer::set(const BoundingLayer& other_bounding_layer)
{
    lower_bounds = other_bounding_layer.lower_bounds;

    upper_bounds = other_bounding_layer.upper_bounds;

    display = other_bounding_layer.display;
}


void BoundingLayer::set_bounding_method(const BoundingMethod& new_method)
{
    bounding_method = new_method;
}


void BoundingLayer::set_bounding_method(const string& new_method_string)
{
    if(new_method_string == "NoBounding")
    {
        bounding_method = BoundingMethod::NoBounding;
    }
    else if(new_method_string == "BoundingLayer")
    {
        bounding_method = BoundingMethod::Bounding;
    }
    else
    {
        throw runtime_error("Unknown bounding method: " + new_method_string + ".\n");
    }
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

#ifdef OPENNN_DEBUG

    if(index >= neurons_number)
        throw runtime_error("Index of bounding neurons must be less than number of bounding neurons.\n");

#endif

    if(lower_bounds.size() != neurons_number)
    {
        lower_bounds.resize(neurons_number);
        lower_bounds.setConstant(-numeric_limits<type>::max());
    }

    // Set lower bound of single neuron

    lower_bounds[index] = new_lower_bound;
}


void BoundingLayer::set_lower_bounds(const Tensor<type, 1>& new_lower_bounds)
{
#ifdef OPENNN_DEBUG

    const Index neurons_number = get_neurons_number();

    if(new_lower_bounds.size() != neurons_number)
        throw runtime_error("Size must be equal to number of bounding neurons number.\n");

#endif

    // Set lower bound of bounding neurons

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

#ifdef OPENNN_DEBUG

    if(index >= neurons_number)
        throw runtime_error("Index of bounding neuron must be less than number of bounding neurons.\n");

#endif

    if(upper_bounds.size() != neurons_number)
    {
        upper_bounds.resize(neurons_number);
        upper_bounds.setConstant(numeric_limits<type>::max());
    }

    upper_bounds[index] = new_upper_bound;

}


void BoundingLayer::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                      LayerForwardPropagation* forward_propagation,
                                      const bool& is_training)
{
    const TensorMap<Tensor<type,2>> inputs(inputs_pair(0).first, inputs_pair(0).second[0], inputs_pair(0).second[1]);

    BoundingLayerForwardPropagation* bounding_layer_forward_propagation
            = static_cast<BoundingLayerForwardPropagation*>(forward_propagation);

    Tensor<type,2>& outputs = bounding_layer_forward_propagation->outputs;

    outputs.device(*thread_pool_device) = inputs;

    if(bounding_method == BoundingMethod::Bounding)
    {
        const Index rows_number = inputs.dimension(0);
        const Index raw_variables_number = inputs.dimension(1);

        #pragma omp parallel for

        for(Index i = 0; i < rows_number; i++)
        {
            for(Index j = 0; j < raw_variables_number; j++)
            {
                if(inputs(i,j) < lower_bounds(j))
                {
                    outputs(i,j) = lower_bounds(j);
                }
                else if(inputs(i,j) > upper_bounds(j))
                {
                    outputs(i,j) = upper_bounds(j);
                }
            }
        }
    }
}


string BoundingLayer::get_bounding_method_string() const
{
    if(bounding_method == BoundingMethod::Bounding)
    {
        return "BoundingLayer";
    }
    else if(bounding_method == BoundingMethod::NoBounding)
    {
        return "NoBounding";
    }
    else
    {
        throw runtime_error("Unknown bounding method.\n");
    }
}


string BoundingLayer::write_expression(const Tensor<string, 1>& inputs_name, const Tensor<string, 1>& outputs_name) const
{
    ostringstream buffer;

    buffer.precision(10);

    if(bounding_method == BoundingMethod::Bounding)
    {
        const Index neurons_number = get_neurons_number();

        for(Index i = 0; i < neurons_number; i++)
        {
            buffer << outputs_name[i] << " = max(" << lower_bounds[i] << ", " << inputs_name[i] << ")\n";
            buffer << outputs_name[i] << " = min(" << upper_bounds[i] << ", " << outputs_name[i] << ")\n";
        }
    }
    else
    {
        buffer << "";
    }

    return buffer.str();
}


void BoundingLayer::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("BoundingLayer");

    // Bounding neurons number

    file_stream.OpenElement("BoundingNeuronsNumber");

    const Index neurons_number = get_neurons_number();

    file_stream.PushText(to_string(neurons_number).c_str());
    file_stream.CloseElement();

    for(Index i = 0; i < neurons_number; i++)
    {
        file_stream.OpenElement("Item");

        file_stream.PushAttribute("Index", unsigned(i+1));

        // Lower bound

        file_stream.OpenElement("LowerBound");
        file_stream.PushText(to_string(lower_bounds[i]).c_str());
        file_stream.CloseElement();

        // Upper bound

        file_stream.OpenElement("UpperBound");
        file_stream.PushText(to_string(upper_bounds[i]).c_str());
        file_stream.CloseElement();

        file_stream.CloseElement();
    }

    // Bounding method

    file_stream.OpenElement("BoundingMethod");
    file_stream.PushText(get_bounding_method_string().c_str());
    file_stream.CloseElement();

   // Display

//   {
//      file_stream.OpenElement("Display");

//      buffer.str("");
//      buffer << display;

//      file_stream.PushText(buffer.str().c_str());

//      file_stream.CloseElement();
//   }

    file_stream.CloseElement();
}


void BoundingLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* bounding_layer_element = document.FirstChildElement("BoundingLayer");

    if(!bounding_layer_element)
        throw runtime_error("BoundingLayer element is nullptr.\n");

    // Bounding neurons number

    const tinyxml2::XMLElement* neurons_number_element = bounding_layer_element->FirstChildElement("BoundingNeuronsNumber");

    if(!neurons_number_element)
        throw runtime_error("BoundingNeuronsNumber element is nullptr.\n");

    const Index neurons_number = Index(atoi(neurons_number_element->GetText()));

    set(neurons_number);

    unsigned index = 0; // Index does not work

    const tinyxml2::XMLElement* start_element = neurons_number_element;

    for(Index i = 0; i < neurons_number; i++)
    {
        const tinyxml2::XMLElement* item_element = start_element->NextSiblingElement("Item");
        start_element = item_element;

        if(!item_element)
            throw runtime_error("Item " + to_string(i+1) + " is nullptr.\n");

        item_element->QueryUnsignedAttribute("Index", &index);

        if(index != i+1)
            throw runtime_error("Index " + to_string(index) + " is not correct.\n");

        // Lower bound

        const tinyxml2::XMLElement* lower_bound_element = item_element->FirstChildElement("LowerBound");

        if(lower_bound_element)
            if(lower_bound_element->GetText())
                lower_bounds(index-1) = type(atof(lower_bound_element->GetText()));

        // Upper bound

        const tinyxml2::XMLElement* upper_bound_element = item_element->FirstChildElement("UpperBound");

        if(upper_bound_element)
            if(upper_bound_element->GetText())
                upper_bounds(index-1) = type(atof(upper_bound_element->GetText()));
    }

    // Bounding method

    const tinyxml2::XMLElement* bounding_method_element = bounding_layer_element->FirstChildElement("BoundingMethod");

    if(bounding_method_element)
        set_bounding_method(bounding_method_element->GetText());
}


pair<type*, dimensions> BoundingLayerForwardPropagation::get_outputs_pair() const
{
    const Index neurons_number = layer->get_neurons_number();

    return pair<type*, dimensions>(outputs_data, { batch_samples_number, neurons_number });
}


void BoundingLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    const Index neurons_number = static_cast<BoundingLayer*>(layer)->get_neurons_number();

    batch_samples_number = new_batch_samples_number;

    outputs.resize(batch_samples_number, neurons_number);

    outputs_data = outputs.data();
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
