/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P R I N C I P A L   C O M P O N E N T S   L A Y E R   C L A S S   H E A D E R                              */
/*                                                                                                              */
/*   Pablo Martin                                                                                               */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   pablomartin@artelnics.com                                                                                  */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "principal_components_layer.h"

namespace OpenNN
{
// DEFAULT CONSTRUCTOR

/// Default constructor.
/// It creates a scaling layer object with no scaling neurons.

PrincipalComponentsLayer::PrincipalComponentsLayer(void)
{
   set();
}


// PRINCIPAL COMPONENTS NEURONS NUMBER CONSTRUCTOR

/// Principal components neurons number constructor.
/// This constructor creates a principal components layer layer with a given size.
/// The members of this object are initialized with the default values.
/// @param new_principal_components_neurons_number Number of principal components neurons in the layer.

PrincipalComponentsLayer::PrincipalComponentsLayer(const size_t& new_principal_components_neurons_number)
{
    set(new_principal_components_neurons_number);
}


// COPY CONSTRUCTOR

/// Copy constructor.

PrincipalComponentsLayer::PrincipalComponentsLayer(const PrincipalComponentsLayer& new_principal_components_layer)
{
    set(new_principal_components_layer);
}


// DESTRUCTOR

/// Destructor.

PrincipalComponentsLayer::~PrincipalComponentsLayer(void)
{
}


// const Method& get_principal_components_method(void) const method

/// Returns the method used for principal components layer.

const PrincipalComponentsLayer::PrincipalComponentsMethod& PrincipalComponentsLayer::get_principal_components_method(void) const
{
    return(principal_components_method);
}


// std::string write_principal_components_method(void) const method

/// Returns a string with the name of the method used for principal components layer.

std::string PrincipalComponentsLayer::write_principal_components_method(void) const
{
    if(principal_components_method == NoPrincipalComponents)
    {
        return("NoPrincipalComponents");
    }
    else if(principal_components_method == ActivatedPrincipalComponents)
    {
        return("ActivatedPrincipalComponents");
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
               << "std::string write_principal_components_method(void) const method.\n"
               << "Unknown principal components method.\n";

        throw std::logic_error(buffer.str());
    }
}


// std::string write_principal_components_method_text(void) const method

/// Returns a string with the name of the method used for principal components layer,
/// as paragaph text.

std::string PrincipalComponentsLayer::write_principal_components_method_text(void) const
{
    if(principal_components_method == NoPrincipalComponents)
    {
        return("no principal components");
    }
    else if(principal_components_method == ActivatedPrincipalComponents)
    {
        return("activated principal components");
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
               << "std::string write_principal_components_method_text(void) const method.\n"
               << "Unknown principal components method.\n";

        throw std::logic_error(buffer.str());
    }
}


// Matrix<double> get_eigenvectors(void) const method

/// Returns a matrix containing the eigenvectors of the variables.

Matrix<double> PrincipalComponentsLayer::get_eigenvectors(void) const
{
    return eigenvectors;
}


// Vector<double> get_means(void) const method

/// Returns a vector containing the means of every input variable in the data set.

Vector<double> PrincipalComponentsLayer::get_means(void) const
{
    return means;
}


// Vector<double> calculate_ouptuts(const Vector<double>&) const

/// Performs the principal component analysis to produce a reduced data set.
/// @param inputs Set of inputs to the principal components layer.

Vector<double> PrincipalComponentsLayer::calculate_outputs(const Vector<double>& inputs) const
{
    const size_t inputs_number = inputs.size();

    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    std::ostringstream buffer;

    if(eigenvectors.get_rows_number() != inputs_number)
    {
       buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
              << "Vector<double> calculate_outputs(const Vector<double>&) const method.\n"
              << "Size of inputs must be equal to the number of rows of the eigenvectors matrix.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    // Data adjust

    Vector<double> inputs_adjust(inputs_number);

    for(size_t i = 0; i < inputs_number; i++)
    {
        inputs_adjust[i] = inputs[i] - means[i];
    }

    // Outputs

    const size_t reduced_inputs_number = eigenvectors.get_rows_number();

    Vector<double> outputs(reduced_inputs_number);

    for(size_t i = 0; i < reduced_inputs_number; i++)
    {
        outputs[i] = eigenvectors.arrange_row(i).dot(inputs_adjust);
    }

    return outputs;
}


// const bool& get_display(void) const method

/// Returns true if messages from this class are to be displayed on the screen, or false if messages
/// from this class are not to be displayed on the screen.

const bool& PrincipalComponentsLayer::get_display(void) const
{
    return(display);
}


// void set(void) method

/// Sets the principal components layer to be empty

void PrincipalComponentsLayer::set(void)
{
    means.set();
    eigenvectors.set();

    set_default();
}


// void set(const size_t&)

/// Sets a new size to the principal components layer.
/// It also sets the means to zero and the eigenvectors to identity.

void PrincipalComponentsLayer::set(const size_t& new_size)
{
    means.set(new_size, 0.0);
    eigenvectors.set_identity(new_size);

    set_default();
}


// void set(const PrincipalComponentsLayer&) method

/// Sets the members of this object to be the members of another object of the same class.
/// @param new_scaling_layer Object to be copied.

void PrincipalComponentsLayer::set(const PrincipalComponentsLayer& new_principal_components_layer)
{
   eigenvectors = new_principal_components_layer.eigenvectors;

   means = new_principal_components_layer.means;

   display = new_principal_components_layer.display;
}


// void set_eigenvectors(const Matrix<double>&) method

/// Sets a new value of the eigenvectors member.
/// @param new_eigenvectors Object to be set.

void PrincipalComponentsLayer::set_eigenvectors(const Matrix<double>& new_eigenvectors)
{
    eigenvectors = new_eigenvectors;

    means.set();

    set_default();
}


// void set_means(const Vector<double>&) method

/// Sets a new value of the means member.
/// @param new_means Object to be set.

void PrincipalComponentsLayer::set_means(const Vector<double>& new_means)
{
    means = new_means;
}


// void set_means(const size_t&, const double&) method

/// Sets a new size and a new value to the means.
/// @param new_size New size of the vector means.
/// @param new_value New value of the vector means.

void PrincipalComponentsLayer::set_means(const size_t& new_size, const double& new_value)
{
    means.set(new_size, new_value);
}


/// Sets the members to their default value.
/// <ul>
/// <li> Display: true.
/// </ul>

// void set_default(void) method

void PrincipalComponentsLayer::set_default(void)
{
    set_display(true);
}


/// Sets a new principal components method.

void PrincipalComponentsLayer::set_principal_components_method(const PrincipalComponentsMethod & new_method)
{
    principal_components_method = new_method;
}


/// Sets a new principal components method.

void PrincipalComponentsLayer::set_principal_components_method(const std::string & new_method_string)
{
    if(new_method_string == "NoPrincipalComponents")
    {
        principal_components_method = NoPrincipalComponents;
    }
    else if(new_method_string == "ActivatedPrincipalComponents")
    {
        principal_components_method = ActivatedPrincipalComponents;
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
               << "void set_principal_components_method(const std::string&) method.\n"
               << "Unknown principal components method: " << new_method_string << ".\n";

        throw std::logic_error(buffer.str());
    }
}


// size_t get_principal_components_neurons_number(void) const method

/// Returns the number of principal components in this layer.

size_t PrincipalComponentsLayer::get_principal_components_neurons_number(void) const
{
    return eigenvectors.get_rows_number();
}


// void set_display(const bool&) method

/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void PrincipalComponentsLayer::set_display(const bool& new_display)
{
   display = new_display;
}


// tinyxml2::XMLDocument* to_XML(void) const method

/// Serializes the principal components layer object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this element.

tinyxml2::XMLDocument* PrincipalComponentsLayer::to_XML(void) const
{
    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    std::ostringstream buffer;

    tinyxml2::XMLElement* principal_components_layer_element = document->NewElement("PrincipalComponentsLayer");

    document->InsertFirstChild(principal_components_layer_element);

    // Principal components neurons number

    tinyxml2::XMLElement* size_element = document->NewElement("PrincipalComponentsNeuronsNumber");
    principal_components_layer_element->LinkEndChild(size_element);

    const size_t principal_components_neurons_number = get_principal_components_neurons_number();

    buffer.str("");
    buffer << principal_components_neurons_number;

    tinyxml2::XMLText* size_text = document->NewText(buffer.str().c_str());
    size_element->LinkEndChild(size_text);

    // Eigenvectors matrix

    for(size_t i = 0; i < principal_components_neurons_number; i++)
    {
        tinyxml2::XMLElement* eigenvectors_element = document->NewElement("Eigenvectors");
        eigenvectors_element->SetAttribute("Index", (unsigned)i+1);

        principal_components_layer_element->LinkEndChild(eigenvectors_element);

        // Eigenvector

        tinyxml2::XMLElement* eigenvector_element = document->NewElement("Eigenvector");
        eigenvectors_element->LinkEndChild(eigenvector_element);

        buffer.str("");
        buffer << eigenvectors.arrange_row(i);

        tinyxml2::XMLText* eigenvector_text = document->NewText(buffer.str().c_str());
        eigenvector_element->LinkEndChild(eigenvector_text);
    }

    // Means

    tinyxml2::XMLElement* means_element = document->NewElement("Means");
    means_element->LinkEndChild(means_element);

    buffer.str("");
    buffer << means;

    tinyxml2::XMLText* means_text = document->NewText(buffer.str().c_str());
    means_element->LinkEndChild(means_text);

    // Principal components method

    tinyxml2::XMLElement* method_element = document->NewElement("PrincipalComponentsMethod");
    principal_components_layer_element->LinkEndChild(method_element);

    tinyxml2::XMLText* method_text = document->NewText(write_principal_components_method().c_str());
    method_element->LinkEndChild(method_text);

    return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

void PrincipalComponentsLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    const size_t principal_components_neurons_number = get_principal_components_neurons_number();

    file_stream.OpenElement("PrincipalComponentsLayer");

    // Principal components neurons number

    file_stream.OpenElement("PrincipalComponentsNeuronsNumber");

    buffer.str("");
    buffer << principal_components_neurons_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Eigenvectors matrix

    for(size_t i = 0; i < principal_components_neurons_number; i++)
    {
        file_stream.OpenElement("Eigenvectors");

        file_stream.PushAttribute("Index", (unsigned)i+1);

        // Eigenvector

        file_stream.OpenElement("Eigenvector");

        buffer.str("");
        buffer << eigenvectors.arrange_row(i);

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();


        file_stream.CloseElement();
    }

    // Means

    file_stream.OpenElement("Means");

    buffer.str("");
    buffer << means;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Principal components method

    file_stream.OpenElement("PrincipalComponentsMethod");

    file_stream.PushText(write_principal_components_method().c_str());

    file_stream.CloseElement();


    file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this principal components layer object.
/// @param document XML document containing the member data.

void PrincipalComponentsLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    std::ostringstream buffer;

    const tinyxml2::XMLElement* principal_components_layer_element = document.FirstChildElement("PrincipalComponentsLayer");

    if(!principal_components_layer_element)
    {
        buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Principal components layer element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Scaling neurons number

    const tinyxml2::XMLElement* principal_components_neurons_number_element = principal_components_layer_element->FirstChildElement("PrincipalComponentsNeuronsNumber");

    if(!principal_components_neurons_number_element)
    {
        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Principal components neurons number element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const size_t principal_components_neurons_number = atoi(principal_components_neurons_number_element->GetText());

    set(principal_components_neurons_number);

    unsigned index = 0; // size_t does not work

    const tinyxml2::XMLElement* start_element = principal_components_neurons_number_element;

    for(size_t i = 0; i < principal_components_neurons_number; i++)
    {
        const tinyxml2::XMLElement* eigenvectors_element = start_element->NextSiblingElement("Eigenvectors");
        start_element = eigenvectors_element;

        if(!eigenvectors_element)
        {
            buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Eigenvector of principal components neuron " << i+1 << " is NULL.\n";

            throw std::logic_error(buffer.str());
        }

        eigenvectors_element->QueryUnsignedAttribute("Index", &index);

        if(index != i+1)
        {
            buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Index " << index << " is not correct.\n";

            throw std::logic_error(buffer.str());
        }

        // Eigenvector

        const tinyxml2::XMLElement* eigenvector_element = eigenvectors_element->FirstChildElement("Eigenvector");

        if(!eigenvector_element)
        {
            buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Eigenvector element " << i+1 << " is NULL.\n";

            throw std::logic_error(buffer.str());
        }

        if(eigenvector_element->GetText())
        {
            eigenvectors.set_row(i, atof(eigenvector_element->GetText()));
        }
    }

    // Means
/*
    const tinyxml2::XMLElement* means_element = principal_components_layer_element->FirstChildElement("Means");

    if(!means_element)
    {
        buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Means element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    if(means_element->GetText())
    {
        means.set(atof(means_element->GetText()));
    }
*/
    // Principal components method
    {
        const tinyxml2::XMLElement* principal_components_method_element = principal_components_layer_element->FirstChildElement("PrincipalComponentsMethod");

        if(principal_components_method_element)
        {
            std::string new_method = principal_components_method_element->GetText();

            try
            {
                set_principal_components_method(new_method);
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

    // Display
    {
        const tinyxml2::XMLElement* display_element = principal_components_layer_element->FirstChildElement("Display");

        if(display_element)
        {
            std::string new_display_string = display_element->GetText();

            try
            {
                set_display(new_display_string != "0");
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }
}

}


// OpenNN: Open Neural Networks Library.
// Copyright (c) 2005-2016 Roberto Lopez.
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
