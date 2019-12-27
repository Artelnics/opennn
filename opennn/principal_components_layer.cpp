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
/// This constructor creates a principal components layer layer with a given size.
/// The members of this object are initialized with the default values.
/// @param new_inputs_number Number of original inputs.
/// @param new_principal_components_number Number of principal components neurons in the layer.

PrincipalComponentsLayer::PrincipalComponentsLayer(const size_t& new_inputs_number, const size_t& new_principal_components_number) : Layer()
{
    set(new_inputs_number, new_principal_components_number);
}


/// Copy constructor.

PrincipalComponentsLayer::PrincipalComponentsLayer(const PrincipalComponentsLayer& new_principal_components_layer) : Layer()
{
    set(new_principal_components_layer);
}


/// Destructor.

PrincipalComponentsLayer::~PrincipalComponentsLayer()
{
}


/// Returns the method used for principal components layer.

const PrincipalComponentsLayer::PrincipalComponentsMethod& PrincipalComponentsLayer::get_principal_components_method() const
{
    return(principal_components_method);
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


// Matrix<double> get_principal_components() const method

/// Returns a matrix containing the principal components.

Matrix<double> PrincipalComponentsLayer::get_principal_components() const
{
    return principal_components;
}


// Vector<double> get_means() const method

/// Returns a vector containing the means of every input variable in the data set.

Vector<double> PrincipalComponentsLayer::get_means() const
{
    return means;
}


// Vector<double> get_explained_variance() const

/// Returns a vector containing the explained variance of every of the principal components

Vector<double> PrincipalComponentsLayer::get_explained_variance() const
{
    return explained_variance;
}


// size_t get_inputs_number() const method

/// Returns the number of inputs to the layer.

size_t PrincipalComponentsLayer::get_inputs_number() const
{
    return inputs_number;
}


/// Returns the number of principal components.

size_t PrincipalComponentsLayer::get_principal_components_number() const
{
    return principal_components_number;
}


size_t PrincipalComponentsLayer::get_neurons_number() const
{
    return principal_components_number;
}


/// Performs the principal component analysis to produce a reduced data set.
/// @param inputs Set of inputs to the principal components layer.

Tensor<double> PrincipalComponentsLayer::calculate_outputs(const Tensor<double>& inputs)
{
/*
    const size_t inputs_number = inputs.get_columns_number();    

    #ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(principal_components.get_rows_number() != inputs_number)
    {
       buffer << "OpenNN Exception: PrincipalComponentsLayer class.\n"
              << "Matrix<double> calculate_outputs(Matrix Vector<double>&) const method.\n"
              << "Size of inputs must be equal to the number of rows of the principal components matrix.\n";

       throw logic_error(buffer.str());
    }

    #endif

    if(write_principal_components_method() != "PrincipalComponents")
    {
        return inputs;
    }

        const Vector<size_t> principal_components_indices(0, 1.0, get_principal_components_number()-1);

        const Vector<size_t> inputs_indices(0, 1.0, inputs_number-1);

        const Matrix<double> used_principal_components = principal_components.get_submatrix(principal_components_indices, inputs_indices);

        const Matrix<double> inputs_adjust = inputs.subtract_rows(means);

        return dot(inputs_adjust, used_principal_components.calculate_transpose());

        for(size_t i = 0;  i < points_number; i++)
        {
            const Vector<size_t> principal_components_indices(0, 1.0, get_principal_components_number()-1);

            const Vector<size_t> inputs_indices(0, 1.0, inputs_number-1);

            const Matrix<double> used_principal_components = principal_components.get_submatrix(principal_components_indices, inputs_indices);

            // Data adjust

            const Matrix<double> inputs_adjust = inputs.subtract_rows(means);

//            Vector<double> inputs_adjust(inputs_number);

//            for(size_t j = 0; j < inputs_number; j++)
//            {
//                inputs_adjust[j] = inputs[j] - means[j];
//            }

            // Outputs

            const size_t principal_components_number = used_principal_components.get_rows_number();

            Matrix<double> outputs(points_number, principal_components_number);

            for(size_t j = 0; j < principal_components_number; j++)
            {
                outputs(i,j) = inputs_adjust.dot(used_principal_components.get_row(j));
            }

        }

        return outputs;
    */
    return Tensor<double>();
}


/// Returns a string with the expression of the principal components process.

string PrincipalComponentsLayer::write_expression(const Vector<string>& inputs_names, const Vector<string>& outputs_names) const
{
    switch(principal_components_method)
    {
        case NoPrincipalComponents:
        {
            return(write_no_principal_components_expression(inputs_names, outputs_names));
        }

        case PrincipalComponents:
        {
            return(write_principal_components_expression(inputs_names, outputs_names));
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


string PrincipalComponentsLayer::write_no_principal_components_expression(const Vector<string>&, const Vector<string>& ) const
{
    ostringstream buffer;

    buffer << "";

    return buffer.str();
}


/// Returns a string with the expression of the principal components process when principal components anlysis is used.
/// @param inputs_names Name of inputs to the principal components.
/// @param outputs_names Name of outputs from the principal components.


string PrincipalComponentsLayer::write_principal_components_expression(const Vector<string>& inputs_names, const Vector<string>& outputs_names) const
{
    ostringstream buffer;

    buffer.precision(10);

    const size_t inputs_number = get_inputs_number();
    const size_t principal_components_number = get_principal_components_number();

    for(size_t i = 0; i < principal_components_number;i ++)
    {
        buffer << outputs_names[i] << "= (";

        for(size_t j = 0; j < inputs_number; j++)
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


// void set_display(const bool&) method

/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void PrincipalComponentsLayer::set_display(const bool& new_display)
{
   display = new_display;
}


// tinyxml2::XMLDocument* to_XML() const method

/// Serializes the principal components layer object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this element.

tinyxml2::XMLDocument* PrincipalComponentsLayer::to_XML() const
{
    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;
  /*
    ostringstream buffer;

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
        principal_components_element->SetAttribute("Index",(unsigned)i+1);

        principal_components_layer_element->LinkEndChild(principal_components_element);

        // Eigenvector

        tinyxml2::XMLElement* eigenvector_element = document->NewElement("Eigenvector");
        principal_components_element->LinkEndChild(eigenvector_element);

        buffer.str("");
        buffer << principal_components.get_row(i);

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
    return document;
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the principal components layer object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void PrincipalComponentsLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

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

            file_stream.PushAttribute("Index", static_cast<unsigned>(i)+1);

            // Principal component

            buffer.str("");
            buffer << principal_components.get_row(i);

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

    const size_t inputs_number = static_cast<size_t>(atoi(inputs_number_element->GetText()));

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

    const size_t principal_components_number = static_cast<size_t>(atoi(principal_components_number_element->GetText()));

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
                Vector<double> new_means;
                new_means.parse(means_text);

                try
                {
                    set_means(new_means);
                }
                catch(const logic_error& e)
                {
                    cerr << e.what() <<endl;
                }
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
                Vector<double> new_explained_variance;
                new_explained_variance.parse(explained_variance_text);

                try
                {
                    set_explained_variance(new_explained_variance);
                }
                catch(const logic_error& e)
                {
                    cerr << e.what() <<endl;
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
                Vector<double> principal_component;
                principal_component.parse(principal_component_text);

                try
                {
                    set_principal_component(i, principal_component);
                }
                catch(const logic_error& e)
                {
                    cerr << e.what() <<endl;
                }
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
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
