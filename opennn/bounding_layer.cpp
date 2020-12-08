//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "bounding_layer.h"

namespace OpenNN
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


/// Destructor.
/// This destructor does not delete any pointer.

BoundingLayer::~BoundingLayer()
{
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


/// Returns the method used for bounding layer.

const BoundingLayer::BoundingMethod& BoundingLayer::get_bounding_method() const
{
    return bounding_method;
}

/// Returns a string writing if use bounding layer or not.

string BoundingLayer::write_bounding_method() const
{
    if(bounding_method == Bounding)
    {
        return "Bounding";
    }
    else if(bounding_method == NoBounding)
    {
        return "NoBounding";
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: BoundingLayer class.\n"
               << "string write_bounding_method() const method.\n"
               << "Unknown bounding method.\n";

        throw logic_error(buffer.str());
    }
}


/// Get number of inputs

Index BoundingLayer::get_inputs_number() const
{
    return lower_bounds.dimension(0);
}


/// Return the neurons number in the bounding layer.

Index BoundingLayer::get_neurons_number() const
{
    return lower_bounds.dimension(0);
}


/// Returns the lower bounds values of all the bounding neurons in the layer.

const Tensor<type, 1>& BoundingLayer::get_lower_bounds() const
{
    return lower_bounds;
}


/// Returns the lower bound value of a single bounding neuron.
/// @param i Index of bounding neuron.

type BoundingLayer::get_lower_bound(const Index& i) const
{


#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    if(i >= neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: BoundingLayer class.\n"
               << "type get_lower_bound(const Index&) const method.\n"
               << "Index must be less than number of bounding neurons.\n";

        throw logic_error(buffer.str());
    }

#endif

    return lower_bounds[i];
}


/// Returns the upper bounds values of all the bounding neurons in the layer.

const Tensor<type, 1>& BoundingLayer::get_upper_bounds() const
{
    return upper_bounds;
}


/// Returns the upper bound value of a single bounding neuron.
/// @param i Index of bounding neuron.

type BoundingLayer::get_upper_bound(const Index& i) const
{
#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    if(neurons_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: BoundingLayer class.\n"
               << "type get_upper_bound(const Index&) const method.\n"
               << "Number of bounding neurons is zero.\n";

        throw logic_error(buffer.str());
    }
    else if(i >= neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: BoundingLayer class.\n"
               << "type get_upper_bound(const Index&) const method.\n"
               << "Index must be less than number of bounding neurons.\n";

        throw logic_error(buffer.str());
    }

#endif

    return upper_bounds(i);
}


/// Sets the number of bounding neurons to be zero.
/// It also sets the rest of memebers to their default values.

void BoundingLayer::set()
{
    bounding_method = Bounding;

    lower_bounds.resize(0);
    upper_bounds.resize(0);

    set_default();
}


/// Resizes the bounding layer.
/// It also sets the rest of memebers to their default values.
/// @param new_neurons_number Size of the bounding layer.

void BoundingLayer::set(const Index& new_neurons_number)
{
    set_neurons_number(new_neurons_number);

    set_default();
}


/// Resize the number of inputs.
/// @param new_inputs_number Size of the inputs array.

void BoundingLayer::set_inputs_number(const Index& new_inputs_number)
{
    lower_bounds.resize(new_inputs_number);
    upper_bounds.resize(new_inputs_number);
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


/// Sets the bounding layer members from a XML document.
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
        bounding_method = NoBounding;
    }
    else if(new_method_string == "Bounding")
    {
        bounding_method = Bounding;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: BoundingLayer class.\n"
               << "void set_bounding_method(const string&) method.\n"
               << "Unknown bounding method: " << new_method_string << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Sets new lower bounds for all the neurons in the layer.
/// @param new_lower_bounds New set of lower bounds for the bounding neurons.

void BoundingLayer::set_lower_bounds(const Tensor<type, 1>& new_lower_bounds)
{
#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    if(new_lower_bounds.size() != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: BoundingLayer class.\n"
               << "void set_lower_bounds(const Tensor<type, 1>&) method.\n"
               << "Size must be equal to number of bounding neurons number.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set lower bound of bounding neurons

    lower_bounds = new_lower_bounds;
}


/// Sets a new lower bound for a single neuron.
/// This value is used for unscaling that variable so that it is not less than the lower bound.
/// @param index Index of bounding neuron.
/// @param new_lower_bound New lower bound for the neuron with index i.

void BoundingLayer::set_lower_bound(const Index& index, const type& new_lower_bound)
{
    const Index neurons_number = get_neurons_number();

#ifdef __OPENNN_DEBUG__

    if(index >= neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: BoundingLayer class.\n"
               << "void set_lower_bound(const Index&, const type&) method.\n"
               << "Index of bounding neurons must be less than number of bounding neurons.\n";

        throw logic_error(buffer.str());
    }

#endif

    if(lower_bounds.size() != neurons_number)
    {
        lower_bounds.resize(neurons_number);
        lower_bounds.setConstant(-numeric_limits<type>::max());
    }

    // Set lower bound of single neuron

    lower_bounds[index] = new_lower_bound;
}


/// Sets new upper bounds for all the bounding neurons.
/// These values are used for unscaling variables so that they are not greater than the upper bounds.
/// @param new_upper_bounds New set of upper bounds for the layer.

void BoundingLayer::set_upper_bounds(const Tensor<type, 1>& new_upper_bounds)
{


#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    if(new_upper_bounds.size() != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: BoundingLayer class.\n"
               << "void set_upper_bound(const Tensor<type, 1>&) method.\n"
               << "Size must be equal to number of bounding neurons.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set upper bound of neurons

    upper_bounds = new_upper_bounds;
}


/// Sets a new upper bound for a single neuron.
/// This value is used for unscaling that variable so that it is not greater than the upper bound.
/// @param index Index of bounding neuron.
/// @param new_upper_bound New upper bound for the bounding neuron with that index.

void BoundingLayer::set_upper_bound(const Index& index, const type& new_upper_bound)
{
    const Index neurons_number = get_neurons_number();



#ifdef __OPENNN_DEBUG__

    if(index >= neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: BoundingLayer class.\n"
               << "void set_upper_bound(const Index&, const type&) method.\n"
               << "Index of bounding neuron must be less than number of bounding neurons.\n";

        throw logic_error(buffer.str());
    }

#endif

    if(upper_bounds.size() != neurons_number)
    {
        upper_bounds.resize(neurons_number);
        upper_bounds.setConstant(numeric_limits<type>::max());
    }

    upper_bounds[index] = new_upper_bound;

}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void BoundingLayer::set_display(const bool& new_display)
{
    display = new_display;
}


/// Sets the members to their default values:
/// <ul>
/// <li> Display: True.
/// </ul>

void BoundingLayer::set_default()
{
    layer_name = "bounding_layer";

    bounding_method = Bounding;

    layer_type = Layer::Bounding;
}


/// Calculates the outputs from the bounding layer for a set of inputs to that layer.
/// @param inputs Set of inputs to the bounding layer.

Tensor<type, 2> BoundingLayer::calculate_outputs(const Tensor<type, 2>& inputs)
{
#ifdef __OPENNN_DEBUG__

    const Index inputs_dimensions_number = inputs.rank();

    if(inputs_dimensions_number != 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: BoundingLayer class.\n"
               << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
               << "Number of dimensions (" << inputs_dimensions_number << ") must be 2.\n";

        throw logic_error(buffer.str());
    }

    const Index inputs_number = get_inputs_number();

    const Index inputs_columns_number = inputs.dimension(1);

    if(inputs_columns_number != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: BoundingLayer class.\n"
               << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
               << "Number of columns ("
               << inputs_columns_number << ") must be equal to number of inputs ("
               << inputs_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    switch(bounding_method)
    {
        case NoBounding: return inputs;

        case Bounding:
        {
            const Index rows_number = inputs.dimension(0);
            const Index columns_number = inputs.dimension(1);

            Tensor<type, 2> outputs(rows_number, columns_number);

            for(Index i = 0; i < rows_number; i++)
            {
                for(Index j = 0; j < columns_number; j++)
                {
                    if(inputs(i,j) < lower_bounds(j)) outputs(i,j) = lower_bounds(j);
                    else if(inputs(i,j) > upper_bounds(j)) outputs(i,j) = upper_bounds(j);
                    else outputs(i,j) = inputs(i,j);
                }
            }

            return outputs;
        }
    }

    return Tensor<type, 2>();
}


/// Returns a string with the expression of the lower and upper bounds functions.

string BoundingLayer::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    ostringstream buffer;

    buffer.precision(10);

    if(bounding_method == Bounding)
    {
        const Index neurons_number = get_neurons_number();

        for(Index i = 0; i < neurons_number; i++)
        {
            buffer << outputs_names[i] << " = max(" << lower_bounds[i] << ", " << inputs_names[i] << ")\n";
            buffer << outputs_names[i] << " = min(" << upper_bounds[i] << ", " << inputs_names[i] << ")\n";
        }
    }
    else
    {
        buffer << "";
    }

    return buffer.str();
}


///
/// \brief BoundingLayer::write_expression_c
/// \return

string BoundingLayer::write_expression_c() const
{
    const Index neurons_number = get_neurons_number();

    ostringstream buffer;

    buffer << "vector<float> " << layer_name << "(const vector<float>& inputs)\n{" << endl;

    buffer << "\tvector<float> outputs(" << neurons_number << ");\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\toutputs[" << i << "] = inputs[" << i << "];" << endl;
    }

    buffer << "\n\treturn outputs;\n}" << endl;

    return buffer.str();
}


///
/// \brief BoundingLayer::write_expression_python
/// \return

string BoundingLayer::write_expression_python() const
{
    const Index neurons_number = get_neurons_number();

    ostringstream buffer;

    buffer << "def " << layer_name << "(inputs):\n" << endl;

    buffer << "\toutputs = [None] * "<<neurons_number<<"\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\toutputs[" << i << "] = inputs[" << i << "]" << endl;
    }

    buffer << "\n\treturn outputs\n" << endl;

    return buffer.str();
}


/// Serializes the bounding layer object into a XML document of the TinyXML library without keep the DOM tree in memory.
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

        file_stream.PushAttribute("Index",static_cast<unsigned>(i+1));

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

    if(bounding_method == Bounding)
    {
        buffer.str("");
        buffer << 1;
    }
    else if(bounding_method == NoBounding)
    {
        buffer.str("");
        buffer << 0;
    }
    else
    {
        file_stream.CloseElement();

        buffer << "OpenNN Exception: BoundingLayer class.\n"
               << "void write_XML(tinyxml2::XMLPrinter&) const method.\n"
               << "Unknown bounding method type.\n";

        throw logic_error(buffer.str());
    }

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

//   // Display

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
    {
        buffer << "OpenNN Exception: BoundingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "BoundingLayer element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Bounding neurons number

    const tinyxml2::XMLElement* neurons_number_element = bounding_layer_element->FirstChildElement("BoundingNeuronsNumber");

    if(!neurons_number_element)
    {
        buffer << "OpenNN Exception: BoundingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "BoundingNeuronsNumber element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const Index neurons_number = static_cast<Index>(atoi(neurons_number_element->GetText()));

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

                throw logic_error(buffer.str());
            }

            item_element->QueryUnsignedAttribute("Index", &index);

            if(index != i+1)
            {
                buffer << "OpenNN Exception: BoundingLayer class.\n"
                       << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                       << "Index " << index << " is not correct.\n";

                throw logic_error(buffer.str());
            }

            // Lower bound

            const tinyxml2::XMLElement* lower_bound_element = item_element->FirstChildElement("LowerBound");

            if(lower_bound_element)
            {
                if(lower_bound_element->GetText())
                {
                    lower_bounds[index-1] = static_cast<type>(atof(lower_bound_element->GetText()));
                }
            }

            // Upper bound

            const tinyxml2::XMLElement* upper_bound_element = item_element->FirstChildElement("UpperBound");

            if(upper_bound_element)
            {
                if(upper_bound_element->GetText())
                {
                    upper_bounds[index-1] = static_cast<type>(atof(upper_bound_element->GetText()));
                }
            }
        }
    }

    // Use bounding layer
    {
        const tinyxml2::XMLElement* use_bounding_layer_element = bounding_layer_element->FirstChildElement("UseBoundingLayer");

        if(use_bounding_layer_element)
        {
            Index new_method = static_cast<Index>(atoi(use_bounding_layer_element->GetText()));

            if(new_method == 1)
            {
                bounding_method = Bounding;
            }
            else if(new_method == 0)
            {
                bounding_method = NoBounding;
            }
            else
            {
                buffer << "OpenNN Exception: BoundingLayer class.\n"
                       << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                       << "Unknown bounding method.\n";

                throw logic_error(buffer.str());
            }
        }
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
