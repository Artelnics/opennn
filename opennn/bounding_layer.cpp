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

/// Default constructor.
/// It creates a bounding layer object with zero bounding neurons.

BoundingLayer::BoundingLayer() : Layer()
{
    set();

    set_default();
}


/// Bounding neurons number constructor.
/// It creates a bounding layer with a given size.
/// @param neurons_number Number of bounding neurons in the layer.

BoundingLayer::BoundingLayer(const Index& neurons_number) : Layer()
{
    set(neurons_number);

    set_default();
}


/// Returns the method used for bounding layer.

const BoundingLayer::BoundingMethod& BoundingLayer::get_bounding_method() const
{
    return bounding_method;
}


/// Get number of inputs

Index BoundingLayer::get_inputs_number() const
{
    return lower_bounds.dimension(0);
}


/// Returns the lower bound value of a single bounding neuron.
/// @param i Index of bounding neuron.

type BoundingLayer::get_lower_bound(const Index& i) const
{
#ifdef OPENNN_DEBUG

    const Index neurons_number = get_neurons_number();

    if(i >= neurons_number)
        throw runtime_error("Index must be less than number of bounding neurons.\n");

#endif

    return lower_bounds[i];
}


/// Returns the lower bounds values of all the bounding neurons in the layer.

const Tensor<type, 1>& BoundingLayer::get_lower_bounds() const
{
    return lower_bounds;
}


/// Return the neurons number in the bounding layer.

Index BoundingLayer::get_neurons_number() const
{
    return lower_bounds.dimension(0);
}


dimensions BoundingLayer::get_output_dimensions() const
{
    const Index neurons_number = get_neurons_number();

    return { neurons_number };
}


/// Returns the upper bound value of a single bounding neuron.
/// @param i Index of bounding neuron.

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


/// Returns the upper bounds values of all the bounding neurons in the layer.

const Tensor<type, 1>& BoundingLayer::get_upper_bounds() const
{
    return upper_bounds;
}


/// Returns true if the size of the layer is zero, and false otherwise.

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


/// Sets the number of bounding neurons to be zero.
/// It also sets the rest of memebers to their default values.

void BoundingLayer::set()
{
    bounding_method = BoundingMethod::Bounding;

    lower_bounds.resize(0);
    upper_bounds.resize(0);

    set_default();
}


/// Resizes the bounding layer.
/// It also sets the rest of members to their default values.
/// @param new_neurons_number Size of the bounding layer.

void BoundingLayer::set(const Index& new_neurons_number)
{
    set_neurons_number(new_neurons_number);

    set_default();
}


/// Sets the bounding layer members from an XML document.
/// @param bounding_layer_document Pointer to a TinyXML document containing the member data.

void BoundingLayer::set(const tinyxml2::XMLDocument& bounding_layer_document)
{
    set_default();

    from_XML(bounding_layer_document);
}


/// Sets the members of this object to be the members of another object of the same class.
/// @param other_bounding_layer Object to be copied.

void BoundingLayer::set(const BoundingLayer& other_bounding_layer)
{
    lower_bounds = other_bounding_layer.lower_bounds;

    upper_bounds = other_bounding_layer.upper_bounds;

    display = other_bounding_layer.display;
}


/// Sets a new bounding method.
/// @param new_method New bounding method.

void BoundingLayer::set_bounding_method(const BoundingMethod& new_method)
{
    bounding_method = new_method;
}


/// Sets a new bounding method.
/// @param new_method_string New bounding method string.

void BoundingLayer::set_bounding_method(const string& new_method_string)
{
    if(new_method_string == "NoBounding")
    {
        bounding_method = BoundingMethod::NoBounding;
    }
    else if(new_method_string == "Bounding")
    {
        bounding_method = BoundingMethod::Bounding;
    }
    else
    {
        throw runtime_error("Unknown bounding method: " + new_method_string + ".\n");
    }
}


/// Sets the members to their default values:

void BoundingLayer::set_default()
{
    layer_name = "bounding_layer";

    bounding_method = BoundingMethod::Bounding;

    layer_type = Layer::Type::Bounding;
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void BoundingLayer::set_display(const bool& new_display)
{
    display = new_display;
}


/// Resize the number of inputs.
/// @param new_inputs_number Size of the inputs array.

void BoundingLayer::set_inputs_number(const Index& new_inputs_number)
{
    lower_bounds.resize(new_inputs_number);
    upper_bounds.resize(new_inputs_number);
}


/// Sets a new lower bound for a single neuron.
/// This value is used for unscaling that variable so that it is not less than the lower bound.
/// @param index Index of bounding neuron.
/// @param new_lower_bound New lower bound for the neuron with index i.

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


/// Sets new lower bounds for all the neurons in the layer.
/// @param new_lower_bounds New set of lower bounds for the bounding neurons.

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


/// Resize the number of bound neurons from the bounding layer.
/// @param new_neurons_number Number of the neurons from the bounding layer.

void BoundingLayer::set_neurons_number(const Index& new_neurons_number)
{
    lower_bounds.resize(new_neurons_number);
    upper_bounds.resize(new_neurons_number);

    lower_bounds.setConstant(-numeric_limits<type>::max());
    upper_bounds.setConstant(numeric_limits<type>::max());
}


/// Sets new upper bounds for all the bounding neurons.
/// These values are used for unscaling variables so that they are not greater than the upper bounds.
/// @param new_upper_bounds New set of upper bounds for the layer.

void BoundingLayer::set_upper_bounds(const Tensor<type, 1>& new_upper_bounds)
{
    upper_bounds = new_upper_bounds;
}


/// Sets a new upper bound for a single neuron.
/// This value is used for unscaling that variable so that it is not greater than the upper bound.
/// @param index Index of bounding neuron.
/// @param new_upper_bound New upper bound for the bounding neuron with that index.

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


/// Returns a string writing if use bounding layer or not.

string BoundingLayer::write_bounding_method() const
{
    if(bounding_method == BoundingMethod::Bounding)
    {
        return "Bounding";
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


/// Returns a string with the expression of the lower and upper bounds functions.

string BoundingLayer::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    ostringstream buffer;

    buffer.precision(10);

    if(bounding_method == BoundingMethod::Bounding)
    {
        const Index neurons_number = get_neurons_number();

        for(Index i = 0; i < neurons_number; i++)
        {
            buffer << outputs_names[i] << " = max(" << lower_bounds[i] << ", " << inputs_names[i] << ")\n";
            buffer << outputs_names[i] << " = min(" << upper_bounds[i] << ", " << outputs_names[i] << ")\n";
        }
    }
    else
    {
        buffer << "";
    }

    return buffer.str();
}


/// Serializes the bounding layer object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void BoundingLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("BoundingLayer");

    // Bounding neurons number

    file_stream.OpenElement("BoundingNeuronsNumber");

    const Index neurons_number = get_neurons_number();

    buffer.str("");
    buffer << neurons_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    for(Index i = 0; i < neurons_number; i++)
    {
        file_stream.OpenElement("Item");

        file_stream.PushAttribute("Index", unsigned(i+1));

        // Lower bound

        file_stream.OpenElement("LowerBound");

        buffer.str("");
        buffer << lower_bounds[i];

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Upper bound

        file_stream.OpenElement("UpperBound");

        buffer.str("");
        buffer << upper_bounds[i];

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();


        file_stream.CloseElement();
    }

    // Bounding method

    file_stream.OpenElement("UseBoundingLayer");

    if(bounding_method == BoundingMethod::Bounding)
    {
        buffer.str("");
        buffer << 1;
    }
    else if(bounding_method == BoundingMethod::NoBounding)
    {
        buffer.str("");
        buffer << 0;
    }
    else
    {
        file_stream.CloseElement();

        throw runtime_error("Unknown bounding method type.\n");
    }

    file_stream.PushText(buffer.str().c_str());

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


/// Deserializes a TinyXML document into this bounding layer object.
/// @param document TinyXML document containing the member data.

void BoundingLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

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

    if(neurons_number > 0)
    {
        const tinyxml2::XMLElement* start_element = neurons_number_element;

        for(Index i = 0; i < lower_bounds.size(); i++)
        {
            const tinyxml2::XMLElement* item_element = start_element->NextSiblingElement("Item");
            start_element = item_element;

            if(!item_element)
            {
                buffer << "OpenNN Exception: BoundingLayer class.\n"
                       << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                       << "Item " << i+1 << " is nullptr.\n";

                throw runtime_error(buffer.str());
            }

            item_element->QueryUnsignedAttribute("Index", &index);

            if(index != i+1)
            {
                buffer << "OpenNN Exception: BoundingLayer class.\n"
                       << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                       << "Index " << index << " is not correct.\n";

                throw runtime_error(buffer.str());
            }

            // Lower bound

            const tinyxml2::XMLElement* lower_bound_element = item_element->FirstChildElement("LowerBound");

            if(lower_bound_element)
            {
                if(lower_bound_element->GetText())
                {
                    lower_bounds[index-1] = type(atof(lower_bound_element->GetText()));
                }
            }

            // Upper bound

            const tinyxml2::XMLElement* upper_bound_element = item_element->FirstChildElement("UpperBound");

            if(upper_bound_element)
            {
                if(upper_bound_element->GetText())
                {
                    upper_bounds[index-1] = type(atof(upper_bound_element->GetText()));
                }
            }
        }
    }

    // Use bounding layer
    {
        const tinyxml2::XMLElement* use_bounding_layer_element = bounding_layer_element->FirstChildElement("UseBoundingLayer");

        if(use_bounding_layer_element)
        {
            Index new_method = Index(atoi(use_bounding_layer_element->GetText()));

            if(new_method == 1)
            {
                bounding_method = BoundingMethod::Bounding;
            }
            else if(new_method == 0)
            {
                bounding_method = BoundingMethod::NoBounding;
            }
            else
            {
                throw runtime_error("Unknown bounding method.\n");
            }
        }
    }
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
