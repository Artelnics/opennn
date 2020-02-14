//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cross_entropy_error.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a default cross entropy error term object,
/// which is not associated to any neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

CrossEntropyError::CrossEntropyError() : LossIndex()
{
}


/// Neural network constructor.
/// It creates a cross entropy error term associated to a neural network but not measured on any data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

CrossEntropyError::CrossEntropyError(NeuralNetwork* new_neural_network_pointer)
    : LossIndex(new_neural_network_pointer)
{
}


/// Data set constructor.
/// It creates a cross entropy error not associated to any neural network but to be measured on a data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

CrossEntropyError::CrossEntropyError(DataSet* new_data_set_pointer)
    : LossIndex(new_data_set_pointer)
{
}


/// Neural network and data set constructor.
/// It creates a cross entropy error term object associated to a neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values:
/// @param new_neural_network_pointer: Pointer to a neural network object.
/// @param new_data_set_pointer: Pointer to a data set object.

CrossEntropyError::CrossEntropyError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
}


/// XML constructor.
/// It creates a cross entropy error not associated to any neural network and not measured on any data set.
/// It also sets all the rest of class members from a TinyXML document->
/// @param cross_entropy_error_document XML document with the class members.

CrossEntropyError::CrossEntropyError(const tinyxml2::XMLDocument& cross_entropy_error_document)
    : LossIndex(cross_entropy_error_document)
{
    from_XML(cross_entropy_error_document);
}


/// Copy constructor.
/// It creates a cross entropy error not associated to any neural network and not measured on any data set.
/// It also sets all the rest of class members from another cross-entropy error object.
/// @param new_cross_entropy_error Object to be copied.

CrossEntropyError::CrossEntropyError(const CrossEntropyError& new_cross_entropy_error)
    : LossIndex(new_cross_entropy_error)
{

}


/// Destructor.

CrossEntropyError::~CrossEntropyError()
{
}


type CrossEntropyError::cross_entropy_error(const Tensor<type, 2> & x, const Tensor<type, 2>& y) const
{
    const Index x_rows_number = x.dimension(0);
    const Index x_columns_number = x.dimension(1);

#ifdef __OPENNN_DEBUG__

    const Index y_rows_number = y.dimension(0);

    if(y_rows_number != x_rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "double cross_entropy_error(const Tensor<double>&, const Tensor<double>&) method.\n"
               << "Other number of rows must be equal to this number of rows.\n";

        throw logic_error(buffer.str());
    }

    const Index y_columns_number = y.dimension(1);

    if(y_columns_number != x_columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "double cross_entropy_error(const Tensor<double>&, const Tensor<double>&) method.\n"
               << "Other number of columns must be equal to this number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    type cross_entropy_error = 0.0;

    for(Index i = 0; i < x_rows_number; i++)
    {
        for(Index j = 0; j < x_columns_number; j++)
        {
            const type y_value = y(i,j);
            const type x_value = x(i,j);

            if(y_value == 0.0 && x_value == 0.0)
            {
                cross_entropy_error -= 0.0;
            }
            else if(y_value == 1.0 && x_value == 1.0)
            {
                cross_entropy_error -= 0.0;
            }
            else if(x_value == 0.0)
            {
                cross_entropy_error -= (1.0 - y_value)*log(1.0-x_value) + y_value*log(0.000000001);
            }
            else if(x_value == 1.0)
            {
                cross_entropy_error -= (1.0 - y_value)*log(1.0-x_value) + y_value*log(0.999999999);
            }
            else
            {
                cross_entropy_error -= (1.0 - y_value)*log(1.0-x_value) + y_value*log(x_value);
            }
        }
    }

    return cross_entropy_error;
}

/// Returns a string with the name of the cross entropy error loss type, "CROSS_ENTROPY_ERROR".

string CrossEntropyError::get_error_type() const
{
    return "CROSS_ENTROPY_ERROR";
}


/// Returns a string with the name of the cross entropy error loss type in text format.

string CrossEntropyError::get_error_type_text() const
{
    return "Cross entropy error";
}


/// Serializes the cross entropy error object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this document->

tinyxml2::XMLDocument* CrossEntropyError::to_XML() const
{
    ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    // Cross entropy error

    tinyxml2::XMLElement* cross_entropy_error_element = document->NewElement("CrossEntropyError");

    document->InsertFirstChild(cross_entropy_error_element);

    // Display

//   {
//      tinyxml2::XMLElement* display_element = document->NewElement("Display");
//      cross_entropy_error_element->LinkEndChild(display_element);

//      buffer.str("");
//      buffer << display;

//      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
//      display_element->LinkEndChild(display_text);
//   }

    return document;
}


/// Serializes the cross entropy error object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void CrossEntropyError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("Error");

    file_stream.PushAttribute("Type", "CROSS_ENTROPY_ERROR");

    file_stream.CloseElement();

    // Regularization

    write_regularization_XML(file_stream);
}


/// Deserializes a TinyXML document into this cross entropy object.
/// @param document TinyXML document containing the member data.

void CrossEntropyError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("CrossEntropyError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: CrossEntropyError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Cross entropy error element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Regularization

    tinyxml2::XMLDocument regularization_document;
    tinyxml2::XMLNode* element_clone;

    const tinyxml2::XMLElement* regularization_element = root_element->FirstChildElement("Regularization");

    element_clone = regularization_element->DeepClone(&regularization_document);

    regularization_document.InsertFirstChild(element_clone);

    regularization_from_XML(regularization_document);

//    const tinyxml2::XMLElement* root_element = document.FirstChildElement("CrossEntropyError");

//    if(!root_element)
//    {
//        ostringstream buffer;

//        buffer << "OpenNN Exception: CrossEntropyError class.\n"
//               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//               << "Cross entropy error element is nullptr.\n";

//        throw logic_error(buffer.str());
//    }

//  // Display
//  {
//     const tinyxml2::XMLElement* display_element = root_element->FirstChildElement("Display");

//     if(display_element)
//     {
//        const string new_display_string = display_element->GetText();

//        try
//        {
//           set_display(new_display_string != "0");
//        }
//        catch(const logic_error& e)
//        {
//           cerr << e.what() << endl;
//        }
//     }
//  }
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
