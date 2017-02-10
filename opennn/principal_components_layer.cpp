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


// INPUTS AND PRINCIPAL COMPONENTS NEURONS NUMBER CONSTRUCTOR

/// Principal components neurons number constructor.
/// This constructor creates a principal components layer layer with a given size.
/// The members of this object are initialized with the default values.
/// @param new_inputs_number Number of original inputs.
/// @param new_principal_components_number Number of principal components neurons in the layer.

PrincipalComponentsLayer::PrincipalComponentsLayer(const size_t& new_inputs_number, const size_t& new_principal_components_number)
{
    set(new_inputs_number, new_principal_components_number);
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


// const PrincipalComponentsMethod& get_principal_components_method(void) const method

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
    else if(principal_components_method == PrincipalComponents)
    {
        return("PrincipalComponents");
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
    else if(principal_components_method == PrincipalComponents)
    {
        return("principal components");
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


// Matrix<double> get_principal_components(void) const method

/// Returns a matrix containing the principal components.

Matrix<double> PrincipalComponentsLayer::get_principal_components(void) const
{
    return principal_components;
}


// Vector<double> get_means(void) const method

/// Returns a vector containing the means of every input variable in the data set.

Vector<double> PrincipalComponentsLayer::get_means(void) const
{
    return means;
}


// Vector<double> get_explained_variance(void) const

/// Returns a vector containing the explained variance of every of the principal components

Vector<double> PrincipalComponentsLayer::get_explained_variance(void) const
{
    return explained_variance;
}


// size_t get_inputs_number(void) const method

/// Returns the number of inputs to the layer.

size_t PrincipalComponentsLayer::get_inputs_number(void) const
{
    return inputs_number;
}


// size_t get_principal_components_number(void) const method

/// Returns the number of principal components.

size_t PrincipalComponentsLayer::get_principal_components_number(void) const
{
    return principal_components_number;
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

    if(principal_components.get_rows_number() != inputs_number)
    {
       buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
              << "Vector<double> calculate_outputs(const Vector<double>&) const method.\n"
              << "Size of inputs must be equal to the number of rows of the principal components matrix.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    if(write_principal_components_method() != "PrincipalComponents")
    {
        return inputs;
    }
    else
    {
        const Vector<size_t> principal_components_indices(0, 1.0, get_principal_components_number()-1);
        const Vector<size_t> inputs_indices(0, 1.0, inputs_number-1);

        const Matrix<double> used_principal_components = principal_components.arrange_submatrix(principal_components_indices, inputs_indices);

        // Data adjust

        Vector<double> inputs_adjust(inputs_number);

        for(size_t i = 0; i < inputs_number; i++)
        {
            inputs_adjust[i] = inputs[i] - means[i];
        }

        // Outputs

        const size_t principal_components_number = used_principal_components.get_rows_number();

        Vector<double> outputs(principal_components_number);

        for(size_t i = 0; i < principal_components_number; i++)
        {
            outputs[i] = inputs_adjust.dot(used_principal_components.arrange_row(i));
        }

        return outputs;
    }
}


// Matrix<double> calculate_Jacobian(const Vector<double>&) const

/// Returns the partial derivatives of the outputs from the principal components layer with respect to its inputs.
/// @param inputs Inputs to the principal components layer.

Matrix<double> PrincipalComponentsLayer::calculate_Jacobian(const Vector<double>& inputs) const
{
    if(write_principal_components_method() != "NoPrincipalComponents")
    {
        const Vector<size_t> principal_components_indices(0, 1.0, get_principal_components_number()-1);
        const Vector<size_t> inputs_indices(0, 1.0, get_inputs_number()-1);

        return principal_components.arrange_submatrix(principal_components_indices, inputs_indices);
    }
    else
    {
        const size_t size = inputs.size();

        Matrix<double> Jacobian;

        Jacobian.set_identity(size);

        return Jacobian;
    }
}


// std::string write_expression(const Vector<std::string>&, const Vector<std::string>&) const method

/// Returns a string with the expression of the principal components process.

std::string PrincipalComponentsLayer::write_expression(const Vector<std::string>& inputs_name, const Vector<std::string>& outputs_name) const
{
    switch(principal_components_method)
    {
    case NoPrincipalComponents:
    {
        return(write_no_principal_components_expression(inputs_name, outputs_name));
    }
        break;

    case PrincipalComponents:
    {
        return(write_principal_components_expression(inputs_name, outputs_name));
    }
        break;

    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "std::string write_expression(void) const method.\n"
               << "Unknown principal components method.\n";

        throw std::logic_error(buffer.str());
    }// end default
        break;
    }
}



// std::string write_expression(const Vector<std::string>&, const Vector<std::string>&) const method

/// Returns a string with the expression of the principal components process when none method is used.
/*/// @param inputs_name Name of inputs to the principal components.
/// @param outputs_name Name of outputs from the principal components.*/


std::string PrincipalComponentsLayer::write_no_principal_components_expression(const Vector<std::string>& , const Vector<std::string>& ) const
{
    std::ostringstream buffer;

    buffer << "";

    return(buffer.str());
}


// std::string write_expression(const Vector<std::string>&, const Vector<std::string>&) const method

/// Returns a string with the expression of the principal components process when principal components anlysis is used.
/// @param inputs_name Name of inputs to the principal components.
/// @param outputs_name Name of outputs from the principal components.


std::string PrincipalComponentsLayer::write_principal_components_expression(const Vector<std::string>& inputs_name, const Vector<std::string>& outputs_name) const
{
    std::ostringstream buffer;

    buffer.precision(10);

    const size_t inputs_number = get_inputs_number();
    const size_t principal_components_number = get_principal_components_number();

    for(size_t i = 0; i < principal_components_number;i ++)
    {
        buffer << outputs_name[i] << "=(";

        for(size_t j = 0; j < inputs_number; j++)
        {
            buffer << principal_components(i,j) << "*" << inputs_name[j];

            if(j != inputs_number-1)
            {
                buffer << "+";
            }
        }

        buffer << ");\n";
    }

    return(buffer.str());
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
    set_inputs_number(0);
    set_principal_components_number(0);

    means.set();
    explained_variance.set();
    principal_components.set();

    set_default();
}


// void set(const size_t&, const size_t&)

/// Sets a new size to the principal components layer.
/// It also sets means and eigenvector matrix to zero.
/// @param new_inputs_number New inputs number.
/// @param new_principal_components_number New principal components number.

void PrincipalComponentsLayer::set(const size_t& new_inputs_number, const size_t& new_principal_components_number)
{
    set_inputs_number(new_inputs_number);
    set_principal_components_number(new_principal_components_number);

    means.set(new_inputs_number, 0.0);

    explained_variance.set(new_inputs_number, 0.0);

    principal_components.set(new_principal_components_number, new_inputs_number, 0.0);

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


// void set_principal_components(const Matrix<double>&) method

/// Sets a new value of the principal components member.
/// @param new_principal_components Object to be set.

void PrincipalComponentsLayer::set_principal_components(const Matrix<double>& new_principal_components)
{
    principal_components = new_principal_components;

    means.set();

    set_default();
}


// void set_inputs_number(const size_t&) method

/// Sets a new value for the inputs number member.
/// @param new_inputs_number New inputs number.

void PrincipalComponentsLayer::set_inputs_number(const size_t& new_inputs_number)
{
    inputs_number = new_inputs_number;
}


// void set_principal_components_number(const size_t&) method

/// Sets a new value for the principal components number member.
/// @param new_principal_components_number New principal components number.

void PrincipalComponentsLayer::set_principal_components_number(const size_t& new_principal_components_number)
{
    principal_components_number = new_principal_components_number;
}


// void set_principal_component(const Matrix<double>&) method

/// Sets a new value of the principal components member.
/// @param index Index of the principal component.
/// @param principal_component Object to be set.

void PrincipalComponentsLayer::set_principal_component(const size_t& index, const Vector<double>& principal_component)
{
    principal_components.set_row(index, principal_component);
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


// void set_explained_variance(const Vector<double>&) method

/// Sets a new value to the explained variance member.
/// @param new_explained_variance Object to be set.

void PrincipalComponentsLayer::set_explained_variance(const Vector<double>& new_explained_variance)
{
    explained_variance = new_explained_variance;
}


/// Sets the members to their default value.
/// <ul>
/// <li> Display: true.
/// </ul>

// void set_default(void) method

void PrincipalComponentsLayer::set_default(void)
{
    principal_components_method = PrincipalComponents;

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

void PrincipalComponentsLayer::set_principal_components_method(const std::string & new_method_string)
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
        std::ostringstream buffer;

        buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
               << "void set_principal_components_method(const std::string&) method.\n"
               << "Unknown principal components method: " << new_method_string << ".\n";

        throw std::logic_error(buffer.str());
    }
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
  /*
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

    // Principal components matrix

    for(size_t i = 0; i < principal_components_neurons_number; i++)
    {
        tinyxml2::XMLElement* principal_components_element = document->NewElement("PrincipalComponents");
        principal_components_element->SetAttribute("Index", (unsigned)i+1);

        principal_components_layer_element->LinkEndChild(principal_components_element);

        // Eigenvector

        tinyxml2::XMLElement* eigenvector_element = document->NewElement("Eigenvector");
        principal_components_element->LinkEndChild(eigenvector_element);

        buffer.str("");
        buffer << principal_components.arrange_row(i);

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
*/
    return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the principal components layer object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void PrincipalComponentsLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    file_stream.OpenElement("PrincipalComponentsLayer");

    // Inputs number

    const size_t inputs_number = get_inputs_number();

    file_stream.OpenElement("InputsNumber");

    buffer.str("");
    buffer << inputs_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Principal components neurons number

    const size_t principal_components_number = get_principal_components_number();

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

        for(size_t i = 0; i < inputs_number/*principal_components_number*/; i++)
        {
            file_stream.OpenElement("PrincipalComponent");

            file_stream.PushAttribute("Index", (unsigned)i+1);

            // Principal component

            buffer.str("");
            buffer << principal_components.arrange_row(i);

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
    std::ostringstream buffer;

    const tinyxml2::XMLElement* principal_components_layer_element = document.FirstChildElement("PrincipalComponentsLayer");

    if(!principal_components_layer_element)
    {
        buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Principal components layer element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = principal_components_layer_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Inputs number element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const size_t inputs_number = atoi(inputs_number_element->GetText());

    set_inputs_number(inputs_number);

    // Principal components number

    const tinyxml2::XMLElement* principal_components_number_element = principal_components_layer_element->FirstChildElement("PrincipalComponentsNumber");

    if(!principal_components_number_element)
    {
        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Principal components number element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const size_t principal_components_number = atoi(principal_components_number_element->GetText());

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
                   << "Means element is NULL.\n";

            throw std::logic_error(buffer.str());
        }
        else
        {
            const char* means_text = means_element->GetText();

            if(means_text)
            {
                Vector<double> new_means;
                new_means.parse(means_text);

                try
                {
                    set_means(new_means);
                }
                catch(const std::logic_error& e)
                {
                    std::cout << e.what() <<std::endl;
                }
            }
        }

        // Explained variance

        const tinyxml2::XMLElement* explained_variance_element = principal_components_layer_element->FirstChildElement("ExplainedVariance");

        if(!explained_variance_element)
        {
            buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "ExplainedVariance element is NULL.\n";

            throw std::logic_error(buffer.str());
        }
        else
        {
            const char* explained_variance_text = explained_variance_element->GetText();

            if(explained_variance_text)
            {
                Vector<double> new_explained_variance;
                new_explained_variance.parse(explained_variance_text);

                try
                {
                    set_explained_variance(new_explained_variance);
                }
                catch(const std::logic_error& e)
                {
                    std::cout << e.what() <<std::endl;
                }
            }
        }

        // Principal components

        principal_components.set(inputs_number, inputs_number);

        unsigned index = 0; // size_t does not work

        const tinyxml2::XMLElement* start_element = means_element;

        for(size_t i = 0; i < inputs_number/*principal_components_number*/; i++)
        {
            const tinyxml2::XMLElement* principal_components_element = start_element->NextSiblingElement("PrincipalComponent");
            start_element = principal_components_element;

            if(!principal_components_element)
            {
                buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
                       << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Principal component number " << i+1 << " is NULL.\n";

                throw std::logic_error(buffer.str());
            }

            principal_components_element->QueryUnsignedAttribute("Index", &index);

            if(index != i+1)
            {
                buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
                       << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Index " << index << " is not correct.\n";

                throw std::logic_error(buffer.str());
            }

            // Principal component

            const char* principal_component_text = principal_components_element->GetText();

            if(principal_component_text)
            {
                Vector<double> principal_component;
                principal_component.parse(principal_component_text);

                try
                {
                    set_principal_component(i, principal_component);
                }
                catch(const std::logic_error& e)
                {
                    std::cout << e.what() <<std::endl;
                }
            }
        }
    }

    // Principal components method

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
