//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R I N C I P A L   C O M P O N E N T S   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "principal_components_layer.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a scaling layer object with no scaling neurons.

PrincipalComponentsLayer::PrincipalComponentsLayer() : Layer()
{
    set();
}


/// Principal components neurons number constructor.
/// This constructor creates a principal components layer with a given size.
/// The members of this object are initialized with the default values.
/// @param new_inputs_number Number of original inputs.
/// @param new_principal_components_number Number of principal components neurons in the layer.

PrincipalComponentsLayer::PrincipalComponentsLayer(const Index& new_inputs_number, const Index& new_principal_components_number) : Layer()
{
    set(new_inputs_number, new_principal_components_number);
}


/// Destructor.

PrincipalComponentsLayer::~PrincipalComponentsLayer()
{
}


/// Returns the method used for principal components layer.

const PrincipalComponentsLayer::PrincipalComponentsMethod& PrincipalComponentsLayer::get_principal_components_method() const
{
    return principal_components_method;
}


/// Returns a string with the name of the method used for principal components layer.

string PrincipalComponentsLayer::write_principal_components_method() const
{
    if(principal_components_method == NoPrincipalComponents)
    {
        return "NoPrincipalComponents";
    }
    else if(principal_components_method == PrincipalComponents)
    {
        return "PrincipalComponents";
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
               << "string write_principal_components_method() const method.\n"
               << "Unknown principal components method.\n";

        throw logic_error(buffer.str());
    }
}


/// Returns a string with the name of the method used for principal components layer,
/// as paragaph text.

string PrincipalComponentsLayer::write_principal_components_method_text() const
{
    if(principal_components_method == NoPrincipalComponents)
    {
        return "no principal components";
    }
    else if(principal_components_method == PrincipalComponents)
    {
        return "principal components";
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
               << "string write_principal_components_method_text() const method.\n"
               << "Unknown principal components method.\n";

        throw logic_error(buffer.str());
    }
}


// Tensor<type, 2> get_principal_components() const method

/// Returns a matrix containing the principal components.

Tensor<type, 2> PrincipalComponentsLayer::get_principal_components() const
{
    return principal_components;
}


// Tensor<type, 1> get_means() const method

/// Returns a vector containing the means of every input variable in the data set.

Tensor<type, 1> PrincipalComponentsLayer::get_means() const
{
    return means;
}


// Tensor<type, 1> get_explained_variance() const

/// Returns a vector containing the explained variance of every of the principal components

Tensor<type, 1> PrincipalComponentsLayer::get_explained_variance() const
{
    return explained_variance;
}


/// Returns the number of inputs to the layer.

Index PrincipalComponentsLayer::get_inputs_number() const
{
    return inputs_number;
}


/// Returns the number of principal components.

Index PrincipalComponentsLayer::get_principal_components_number() const
{
    return principal_components_number;
}


Index PrincipalComponentsLayer::get_neurons_number() const
{
    return principal_components_number;
}


/// Performs the principal component analysis to produce a reduced data set.
/// @param inputs Set of inputs to the principal components layer.
/// @todo

Tensor<type, 2> PrincipalComponentsLayer::calculate_outputs(const Tensor<type, 2>& inputs)
{
    /*
        const Index inputs_number = inputs.dimension(1);

        #ifdef __OPENNN_DEBUG__

        ostringstream buffer;

        if(principal_components.dimension(0) != inputs_number)
        {
           buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
                  << "Tensor<type, 2> calculate_outputs(Matrix Tensor<type, 1>&) const method.\n"
                  << "Size of inputs must be equal to the number of rows of the principal components matrix.\n";

           throw logic_error(buffer.str());
        }

        #endif

        if(write_principal_components_method() != "PrincipalComponents")
        {
            return inputs;
        }

            const Tensor<Index, 1> principal_components_indices(0, 1.0, get_principal_components_number()-1);

            const Tensor<Index, 1> input_variables_indices(0, 1.0, inputs_number-1);

            const Tensor<type, 2> used_principal_components = principal_components.get_submatrix(principal_components_indices, input_variables_indices);

            const Tensor<type, 2> inputs_adjust = inputs.subtract_rows(means);

            return dot(inputs_adjust, used_principal_components.calculate_transpose());

            for(Index i = 0;  i < points_number; i++)
            {
                const Tensor<Index, 1> principal_components_indices(0, 1.0, get_principal_components_number()-1);

                const Tensor<Index, 1> input_variables_indices(0, 1.0, inputs_number-1);

                const Tensor<type, 2> used_principal_components = principal_components.get_submatrix(principal_components_indices, input_variables_indices);

                // Data adjust

                const Tensor<type, 2> inputs_adjust = inputs.subtract_rows(means);

    //            Tensor<type, 1> inputs_adjust(inputs_number);

    //            for(Index j = 0; j < inputs_number; j++)
    //            {
    //                inputs_adjust[j] = inputs[j] - means[j];
    //            }

                // Outputs

                const Index principal_components_number = used_principal_components.dimension(0);

                Tensor<type, 2> outputs(points_number, principal_components_number);

                for(Index j = 0; j < principal_components_number; j++)
                {
                    outputs(i,j) = inputs_adjust.dot(used_principal_components.chip(j, 0));
                }

            }

            return outputs;
        */
    return Tensor<type, 2>();
}


/// Returns a string with the expression of the principal components process.

string PrincipalComponentsLayer::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    switch(principal_components_method)
    {
    case NoPrincipalComponents:
    {
        return write_no_principal_components_expression(inputs_names, outputs_names);
    }

    case PrincipalComponents:
    {
        return write_principal_components_expression(inputs_names, outputs_names);
    }
    }

    // Default

    ostringstream buffer;

    buffer << "OpenNN Exception: ScalingLayer class.\n"
           << "string write_expression() const method.\n"
           << "Unknown principal components method.\n";

    throw logic_error(buffer.str());
}


/// Returns a string with the expression of the principal components process when none method is used.
/*/// @param inputs_names Name of inputs to the principal components.
/// @param outputs_names Name of outputs from the principal components.*/


string PrincipalComponentsLayer::write_no_principal_components_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const
{
    ostringstream buffer;

    buffer << "";

    return buffer.str();
}


/// Returns a string with the expression of the principal components process when principal components anlysis is used.
/// @param inputs_names Name of inputs to the principal components.
/// @param outputs_names Name of outputs from the principal components.


string PrincipalComponentsLayer::write_principal_components_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    ostringstream buffer;

    buffer.precision(10);

    const Index inputs_number = get_inputs_number();
    const Index principal_components_number = get_principal_components_number();

    for(Index i = 0; i < principal_components_number; i ++)
    {
        buffer << outputs_names[i] << "= (";

        for(Index j = 0; j < inputs_number; j++)
        {
            buffer << principal_components(i,j) << "*" << inputs_names[j];

            if(j != inputs_number-1)
            {
                buffer << "+";
            }
        }

        buffer << ");\n";
    }

    return buffer.str();
}


// const bool& get_display() const method

/// Returns true if messages from this class are to be displayed on the screen, or false if messages
/// from this class are not to be displayed on the screen.

const bool& PrincipalComponentsLayer::get_display() const
{
    return display;
}


// void set() method

/// Sets the principal components layer to be empty

void PrincipalComponentsLayer::set()
{
    set_inputs_number(0);
    set_principal_components_number(0);
    /*
        means.set();
        explained_variance.set();
        principal_components.set();
    */
    set_default();
}


/// Sets a new size to the principal components layer.
/// It also sets means and eigenvector matrix to zero.
/// @param new_inputs_number New inputs number.
/// @param new_principal_components_number New principal components number.

void PrincipalComponentsLayer::set(const Index& new_inputs_number, const Index& new_principal_components_number)
{
    set_inputs_number(new_inputs_number);
    set_principal_components_number(new_principal_components_number);
    /*
        means.set(new_inputs_number, 0.0);

        explained_variance.set(new_inputs_number, 0.0);

        principal_components.set(new_principal_components_number, new_inputs_number, 0.0);
    */
    set_default();
}


// void set(const PrincipalComponentsLayer&) method

/// Sets the members of this object to be the members of another object of the same class.
/// @param new_principal_components_layer Object to be copied.

void PrincipalComponentsLayer::set(const PrincipalComponentsLayer& new_principal_components_layer)
{
    principal_components_method = new_principal_components_layer.principal_components_method;

    principal_components = new_principal_components_layer.principal_components;

    means = new_principal_components_layer.means;

    display = new_principal_components_layer.display;
}


/// Sets a new value of the principal components member.
/// @param new_principal_components Object to be set.

void PrincipalComponentsLayer::set_principal_components(const Tensor<type, 2>& new_principal_components)
{
    principal_components = new_principal_components;
    /*
        means.set();
    */
    set_default();
}


/// Sets a new value for the inputs number member.
/// @param new_inputs_number New inputs number.

void PrincipalComponentsLayer::set_inputs_number(const Index& new_inputs_number)
{
    inputs_number = new_inputs_number;
}


/// Sets a new value for the principal components number member.
/// @param new_principal_components_number New principal components number.

void PrincipalComponentsLayer::set_principal_components_number(const Index& new_principal_components_number)
{
    principal_components_number = new_principal_components_number;
}


/// Sets a new value of the principal components member.
/// @param index Index of the principal component.
/// @param principal_component Object to be set.

void PrincipalComponentsLayer::set_principal_component(const Index& index, const Tensor<type, 1>& principal_component)
{
    /*
        principal_components.set_row(index, principal_component);
    */
}


/// Sets a new value of the means member.
/// @param new_means Object to be set.

void PrincipalComponentsLayer::set_means(const Tensor<type, 1>& new_means)
{
    means = new_means;
}


/// Sets a new size and a new value to the means.
/// @param new_size New size of the vector means.
/// @param new_value New value of the vector means.

void PrincipalComponentsLayer::set_means(const Index& new_size, const type& new_value)
{
    /*
        means.set(new_size, new_value);
    */
}


/// Sets a new value to the explained variance member.
/// @param new_explained_variance Object to be set.

void PrincipalComponentsLayer::set_explained_variance(const Tensor<type, 1>& new_explained_variance)
{
    explained_variance = new_explained_variance;
}


/// Sets the members to their default value.
/// <ul>
/// <li> Display: true.
/// </ul>

// void set_default() method

void PrincipalComponentsLayer::set_default()
{
    principal_components_method = NoPrincipalComponents;

    set_display(true);
}


/// Sets a new principal components method.
/// @param new_method New principal components method.

void PrincipalComponentsLayer::set_principal_components_method(const PrincipalComponentsMethod & new_method)
{
    principal_components_method = new_method;
}


/// Sets a new principal components method.
/// @param new_method_string New principal components method string.

void PrincipalComponentsLayer::set_principal_components_method(const string & new_method_string)
{
    if(new_method_string == "NoPrincipalComponents")
    {
        principal_components_method = NoPrincipalComponents;
    }
    else if(new_method_string == "PrincipalComponents")
    {
        principal_components_method = PrincipalComponents;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
               << "void set_principal_components_method(const string&) method.\n"
               << "Unknown principal components method: " << new_method_string << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void PrincipalComponentsLayer::set_display(const bool& new_display)
{
    display = new_display;
}


/// Serializes the principal components layer object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void PrincipalComponentsLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("PrincipalComponentsLayer");

    // Inputs number

    const Index inputs_number = get_inputs_number();

    file_stream.OpenElement("InputsNumber");

    buffer.str("");
    buffer << inputs_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Principal components neurons number

    const Index principal_components_number = get_principal_components_number();

    file_stream.OpenElement("PrincipalComponentsNumber");

    buffer.str("");
    buffer << principal_components_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    if(principal_components_number != 0)
    {
        // Means

        file_stream.OpenElement("Means");

        buffer.str("");
        buffer << get_means();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Explained variance

        file_stream.OpenElement("ExplainedVariance");

        buffer.str("");
        buffer << get_explained_variance();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Principal components matrix

        for(Index i = 0; i < inputs_number/*principal_components_number*/; i++)
        {
            file_stream.OpenElement("PrincipalComponent");

            file_stream.PushAttribute("Index", int(i+1));

            // Principal component
            /*
                        buffer.str("");
                        buffer << principal_components.chip(i, 0);
            */
            file_stream.PushText(buffer.str().c_str());

            file_stream.CloseElement();
        }

        // Principal components method

        file_stream.OpenElement("PrincipalComponentsMethod");
        file_stream.PushText(write_principal_components_method().c_str());
        file_stream.CloseElement();
    }
    else
    {
        // Principal components method

        file_stream.OpenElement("PrincipalComponentsMethod");
        file_stream.PushText(write_principal_components_method().c_str());
        file_stream.CloseElement();
    }

    file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this principal components layer object.
/// @param document XML document containing the member data.

void PrincipalComponentsLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    const tinyxml2::XMLElement* principal_components_layer_element = document.FirstChildElement("PrincipalComponentsLayer");

    if(!principal_components_layer_element)
    {
        buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Principal components layer element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = principal_components_layer_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Inputs number element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const Index inputs_number = static_cast<Index>(atoi(inputs_number_element->GetText()));

    set_inputs_number(inputs_number);

    // Principal components number

    const tinyxml2::XMLElement* principal_components_number_element = principal_components_layer_element->FirstChildElement("PrincipalComponentsNumber");

    if(!principal_components_number_element)
    {
        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Principal components number element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const Index principal_components_number = static_cast<Index>(atoi(principal_components_number_element->GetText()));

    set_principal_components_number(principal_components_number);

//    if(principal_components_number != 0)
//    {
//        set(inputs_number, principal_components_number);
//    }
//    else
//    {
//        set(inputs_number, inputs_number);
//    }

    if(principal_components_number != 0)
    {
        // Means

        const tinyxml2::XMLElement* means_element = principal_components_layer_element->FirstChildElement("Means");

        if(!means_element)
        {
            buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Means element is nullptr.\n";

            throw logic_error(buffer.str());
        }
        else
        {
            const char* means_text = means_element->GetText();

            if(means_text)
            {
                /*
                                Tensor<type, 1> new_means;
                                new_means.parse(means_text);

                                try
                                {
                                    set_means(new_means);
                                }
                                catch(const logic_error& e)
                                {
                                    cerr << e.what() <<endl;
                                }
                */
            }
        }

        // Explained variance

        const tinyxml2::XMLElement* explained_variance_element = principal_components_layer_element->FirstChildElement("ExplainedVariance");

        if(!explained_variance_element)
        {
            buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "ExplainedVariance element is nullptr.\n";

            throw logic_error(buffer.str());
        }
        else
        {
            const char* explained_variance_text = explained_variance_element->GetText();

            if(explained_variance_text)
            {
                /*
                                Tensor<type, 1> new_explained_variance;
                                new_explained_variance.parse(explained_variance_text);

                                try
                                {
                                    set_explained_variance(new_explained_variance);
                                }
                                catch(const logic_error& e)
                                {
                                    cerr << e.what() <<endl;
                                }
                */
            }
        }

        // Principal components
        /*
                principal_components.set(inputs_number, inputs_number);
        */
        unsigned index = 0; // Index does not work

        const tinyxml2::XMLElement* start_element = means_element;

        for(Index i = 0; i < inputs_number/*principal_components_number*/; i++)
        {
            const tinyxml2::XMLElement* principal_components_element = start_element->NextSiblingElement("PrincipalComponent");
            start_element = principal_components_element;

            if(!principal_components_element)
            {
                buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
                       << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Principal component number " << i+1 << " is nullptr.\n";

                throw logic_error(buffer.str());
            }

            principal_components_element->QueryUnsignedAttribute("Index", &index);

            if(index != i+1)
            {
                buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
                       << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Index " << index << " is not correct.\n";

                throw logic_error(buffer.str());
            }

            // Principal component

            const char* principal_component_text = principal_components_element->GetText();

            if(principal_component_text)
            {
                /*
                                Tensor<type, 1> principal_component;
                                principal_component.parse(principal_component_text);

                                try
                                {
                                    set_principal_component(i, principal_component);
                                }
                                catch(const logic_error& e)
                                {
                                    cerr << e.what() <<endl;
                                }
                */
            }
        }
    }

    // Principal components method

    const tinyxml2::XMLElement* principal_components_method_element = principal_components_layer_element->FirstChildElement("PrincipalComponentsMethod");

    if(principal_components_method_element)
    {
        string new_method = principal_components_method_element->GetText();

        try
        {
            set_principal_components_method(new_method);
        }
        catch(const logic_error& e)
        {
            cerr << e.what() << endl;
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
