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
/// @param sum_squared_error_document XML document with the class members. 

CrossEntropyError::CrossEntropyError(const tinyxml2::XMLDocument& sum_squared_error_document)
 : LossIndex(sum_squared_error_document)
{
    from_XML(sum_squared_error_document);
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


/// This method calculates the cross entropy error of the given batch.
/// Returns the cross entropy error of this batch.
/// @param batch_indices Indices of the batch instances corresponding to the dataset.

double CrossEntropyError::calculate_batch_error(const Vector<size_t>& batch_indices) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Data set

    const Tensor<double> inputs = data_set_pointer->get_input_data(batch_indices);
    const Tensor<double> targets = data_set_pointer->get_target_data(batch_indices);

    const Tensor<double> outputs = neural_network_pointer->calculate_trainable_outputs(inputs);

    return cross_entropy_error(outputs, targets);
}


double CrossEntropyError::calculate_batch_error(const Vector<size_t>& batch_indices, const Vector<double>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Data set

    const Tensor<double> inputs = data_set_pointer->get_input_data(batch_indices);
    const Tensor<double> targets = data_set_pointer->get_target_data(batch_indices);

    const Tensor<double> outputs = neural_network_pointer->calculate_trainable_outputs(inputs, parameters);

    return cross_entropy_error(outputs, targets);
}


/// This method calculates the first order loss for the selected batch.
/// Returns a first order terms loss structure, which contains the values and the Jacobian of the error terms function.
/// @param batch_indices Indices of the batch instances corresponding to the dataset.

LossIndex::FirstOrderLoss CrossEntropyError::calculate_batch_first_order_loss(const Vector<size_t>& batch_indices) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Data set

    const size_t batch_instances_number = batch_indices.size();

    // Neural network

    const size_t layers_number = neural_network_pointer->get_trainable_layers_number();

    const size_t parameters_number = neural_network_pointer->get_parameters_number();

    // Loss index

    FirstOrderLoss first_order_loss(parameters_number);

    const Tensor<double> inputs = data_set_pointer->get_input_data(batch_indices);

    const Tensor<double> targets = data_set_pointer->get_target_data(batch_indices);

    const Vector<Layer::FirstOrderActivations> forward_propagation = neural_network_pointer->calculate_trainable_forward_propagation(inputs);

    const Tensor<double> output_gradient = calculate_output_gradient(forward_propagation[layers_number-1].activations, targets);

    const Vector<Tensor<double>> layers_delta = calculate_layers_delta(forward_propagation, output_gradient);

    const Vector<double> batch_error_gradient = calculate_error_gradient(inputs, forward_propagation, layers_delta);

    first_order_loss.loss = cross_entropy_error(forward_propagation[layers_number-1].activations, targets);

    first_order_loss.gradient = batch_error_gradient/static_cast<double>(batch_instances_number);

    // Regularization

    if(regularization_method != RegularizationMethod::NoRegularization)
    {
        first_order_loss.loss += regularization_weight*calculate_regularization();
        first_order_loss.gradient += calculate_regularization_gradient()*regularization_weight;
    }

    return first_order_loss;
}


/// Returns the cross-entropy error function output gradient of a neural network on a data set.
/// It uses the error back-propagation method.
/// @param outputs Tensor of outputs of the neural network.
/// @param targets Tensor of targets of the data set.

Tensor<double> CrossEntropyError::calculate_output_gradient(const Tensor<double>& outputs, const Tensor<double>& targets) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    return (targets/outputs)*(-1.0) + (targets*(-1.0) + 1.0)/(outputs*(-1.0) + 1.0);
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
