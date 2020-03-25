//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M I N K O W S K I   E R R O R   C L A S S                             
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "minkowski_error.h"

namespace OpenNN
{

/// Default constructor. 
/// It creates Minkowski error term not associated to any neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

MinkowskiError::MinkowskiError() : LossIndex()
{
   set_default();
}


/// Neural network constructor. 
/// It creates a Minkowski error term associated to a neural network but not measured on any data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

MinkowskiError::MinkowskiError(NeuralNetwork* new_neural_network_pointer)
: LossIndex(new_neural_network_pointer)
{
   set_default();
}


/// Data set constructor. 
/// It creates a Minkowski error term not associated to any neural network but to be measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

MinkowskiError::MinkowskiError(DataSet* new_data_set_pointer)
: LossIndex(new_data_set_pointer)
{
   set_default();
}


/// Neural network and data set constructor. 
/// It creates a Minkowski error term object associated to a neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

MinkowskiError::MinkowskiError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
 : LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
   set_default();
}


/// XML constructor. 
/// It creates a Minkowski error object neither associated to a neural network nor to a data set. 
/// The object members are loaded by means of a XML document.
/// @param mean_squared_error_document TinyXML document with the Minkowski error elements.

MinkowskiError::MinkowskiError(const tinyxml2::XMLDocument& mean_squared_error_document)
 : LossIndex(mean_squared_error_document)
{
   set_default();

   from_XML(mean_squared_error_document);
}


/// Destructor.
/// It does not delete any pointer.

MinkowskiError::~MinkowskiError() 
{
}


/// Returns the Minkowski exponent value used to calculate the error. 

double MinkowskiError::get_Minkowski_parameter() const
{
   return(minkowski_parameter);
}


/// Sets the default values to a Minkowski error object:
/// <ul>
/// <li> Minkowski parameter: 1.5.
/// <li> Display: true.
/// </ul>

void MinkowskiError::set_default()
{
   minkowski_parameter = 1.5;

   display = true;
}


/// Sets a new Minkowski exponent value to be used in order to calculate the error. 
/// The Minkowski R-value must be comprised between 1 and 2. 
/// @param new_Minkowski_parameter Minkowski exponent value. 

void MinkowskiError::set_Minkowski_parameter(const double& new_Minkowski_parameter)
{
   // Control sentence

   if(new_Minkowski_parameter < 1.0 || new_Minkowski_parameter > 2.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Error. MinkowskiError class.\n"
             << "void set_Minkowski_parameter(const double&) method.\n"
             << "The Minkowski parameter must be comprised between 1 and 2.\n";
    
      throw logic_error(buffer.str());
   }

   // Set Minkowski parameter
  
   minkowski_parameter = new_Minkowski_parameter;
}


double MinkowskiError::calculate_training_error() const
{
    //Neural network

     bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set

    const Vector<Vector<size_t>> training_batches = data_set_pointer->get_training_batches(!is_forecasting);

    const size_t training_instances_number = data_set_pointer->get_training_instances_number();

    const size_t batches_number = training_batches.size();

    const size_t batch_instances_number = data_set_pointer->get_batch_instances_number();

    const size_t inputs_number = data_set_pointer->get_input_variables_number();
    const size_t targets_number = data_set_pointer->get_target_variables_number();

    Tensor<double> inputs(batch_instances_number, inputs_number);
    Tensor<double> targets(batch_instances_number, targets_number);
    Tensor<double> outputs(batch_instances_number, targets_number);

    double training_error = 0.0;

     #pragma omp parallel for reduction(+ : training_error)

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        inputs = data_set_pointer->get_input_data(training_batches[static_cast<size_t>(i)]);
        targets = data_set_pointer->get_target_data(training_batches[static_cast<size_t>(i)]);

        outputs = neural_network_pointer->calculate_trainable_outputs(inputs);

        const double batch_error = minkowski_error(outputs, targets, minkowski_parameter);

        training_error += batch_error;
    }

    return training_error / static_cast<double>(training_instances_number);
}


double MinkowskiError::calculate_training_error(const Vector<double>& parameters) const
{
    //Neural network

     bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set

    const Vector<Vector<size_t>> training_batches = data_set_pointer->get_training_batches(!is_forecasting);

    const size_t training_instances_number = data_set_pointer->get_training_instances_number();

    const size_t batches_number = training_batches.size();

    const size_t batch_instances_number = data_set_pointer->get_batch_instances_number();

    const size_t inputs_number = data_set_pointer->get_input_variables_number();
    const size_t targets_number = data_set_pointer->get_target_variables_number();

    Tensor<double> inputs(batch_instances_number, inputs_number);
    Tensor<double> targets(batch_instances_number, targets_number);
    Tensor<double> outputs(batch_instances_number, targets_number);

    double training_error = 0.0;

     #pragma omp parallel for reduction(+ : training_error)

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        inputs = data_set_pointer->get_input_data(training_batches[static_cast<size_t>(i)]);
        targets = data_set_pointer->get_target_data(training_batches[static_cast<size_t>(i)]);

        outputs = neural_network_pointer->calculate_trainable_outputs(inputs, parameters);

        const double batch_error = minkowski_error(outputs, targets, minkowski_parameter);

        training_error += batch_error;
    }

    return training_error / static_cast<double>(training_instances_number);
}


double MinkowskiError::calculate_selection_error() const
{
    // Data set

    const size_t selection_instances_number = data_set_pointer->get_selection_instances_number();

        //Neural network

     bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set

    const Vector<Vector<size_t>> selection_batches = data_set_pointer->get_selection_batches(!is_forecasting);

    const size_t batches_number = selection_batches.size();

    const size_t batch_instances_number = data_set_pointer->get_batch_instances_number();

    const size_t inputs_number = data_set_pointer->get_input_variables_number();
    const size_t targets_number = data_set_pointer->get_target_variables_number();

    Tensor<double> inputs(batch_instances_number, inputs_number);
    Tensor<double> targets(batch_instances_number, targets_number);
    Tensor<double> outputs(batch_instances_number, targets_number);

    double training_error = 0.0;

     #pragma omp parallel for reduction(+ : training_error)

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        inputs = data_set_pointer->get_input_data(selection_batches[static_cast<size_t>(i)]);
        targets = data_set_pointer->get_target_data(selection_batches[static_cast<size_t>(i)]);

        outputs = neural_network_pointer->calculate_trainable_outputs(inputs);

        const double batch_error = minkowski_error(outputs, targets, minkowski_parameter);

        training_error += batch_error;
    }

    return training_error / static_cast<double>(selection_instances_number);
}


/// This method calculates the Minkowski error of the given batch.
/// Returns the Minkowski error of this batch.
/// @param batch_indices Indices of the batch instances corresponding to the dataset.

double MinkowskiError::calculate_batch_error(const Vector<size_t>& batch_indices) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Data set

    const Tensor<double> inputs = data_set_pointer->get_input_data(batch_indices);
    const Tensor<double> targets = data_set_pointer->get_target_data(batch_indices);

    const Tensor<double> outputs = neural_network_pointer->calculate_trainable_outputs(inputs);

    return minkowski_error(outputs, targets, minkowski_parameter);
}


double MinkowskiError::calculate_batch_error(const Vector<size_t>& batch_indices, const Vector<double>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Data set

    const Tensor<double> inputs = data_set_pointer->get_input_data(batch_indices);
    const Tensor<double> targets = data_set_pointer->get_target_data(batch_indices);

    const Tensor<double> outputs = neural_network_pointer->calculate_trainable_outputs(inputs, parameters);

    return minkowski_error(outputs, targets, minkowski_parameter);
}


/// Returns the Minkowski error function output gradient of a neural network on a data set.
/// It uses the error back-propagation method.
/// Returns gradient for training instances.
/// @param outputs Tensor with outputs.
/// @param targets Tensor with targets.

Tensor<double> MinkowskiError::calculate_output_gradient(const Tensor<double>& outputs, const Tensor<double>& targets) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    const size_t training_instances_number = data_set_pointer->get_training_instances_number();

    return lp_norm_gradient(outputs-targets, minkowski_parameter)/static_cast<double>(training_instances_number);
}


/// Returns a string with the name of the Minkowski error loss type, "MINKOWSKI_ERROR".

string MinkowskiError::get_error_type() const
{
   return "MINKOWSKI_ERROR";
}


/// Returns a string with the name of the Minkowski error loss type in text format.

string MinkowskiError::get_error_type_text() const
{
   return "Minkowski error";
}


/// Serializes the Minkowski error object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this document-> 

tinyxml2::XMLDocument* MinkowskiError::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Minkowski error

   tinyxml2::XMLElement* Minkowski_error_element = document->NewElement("MinkowskiError");

   document->InsertFirstChild(Minkowski_error_element);

   // Minkowski parameter
   {
      tinyxml2::XMLElement* Minkowski_parameter_element = document->NewElement("MinkowskiParameter");
      Minkowski_error_element->LinkEndChild(Minkowski_parameter_element);

      buffer.str("");
      buffer << minkowski_parameter;

      tinyxml2::XMLText* Minkowski_parameter_text = document->NewText(buffer.str().c_str());
      Minkowski_parameter_element->LinkEndChild(Minkowski_parameter_text);
   }

   // Display
//   {
//      tinyxml2::XMLElement* display_element = document->NewElement("Display");
//      Minkowski_error_element->LinkEndChild(display_element);

//      buffer.str("");
//      buffer << display;

//      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
//      display_element->LinkEndChild(display_text);
//   }

   return document;
}


/// Serializes the cross entropy error object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void MinkowskiError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Error type

    file_stream.OpenElement("Error");

    file_stream.PushAttribute("Type", "MINKOWSKI_ERROR");

    // Minkowski parameter

    file_stream.OpenElement("MinkowskiParameter");

    buffer.str("");
    buffer << minkowski_parameter;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Close error

    file_stream.CloseElement();

    // Regularization

    write_regularization_XML(file_stream);
}


/// Loads a Minkowski error object from a XML document.
/// @param document TinyXML document containing the members of the object.

void MinkowskiError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("MinkowskiError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MinkowskiError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Minkowski error element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Minkowski parameter

     const tinyxml2::XMLElement* error_element = root_element->FirstChildElement("Error");

     if(error_element)
     {
        const tinyxml2::XMLElement* parameter_element = error_element->FirstChildElement("MinkowskiParameter");

        const double new_Minkowski_parameter = atof(parameter_element->GetText());

        try
        {
           set_Minkowski_parameter(new_Minkowski_parameter);
        }
        catch(const logic_error& e)
        {
           cerr << e.what() << endl;
        }
     }

     // Regularization

     tinyxml2::XMLDocument regularization_document;
     tinyxml2::XMLNode* element_clone;

     const tinyxml2::XMLElement* regularization_element = root_element->FirstChildElement("Regularization");

     element_clone = regularization_element->DeepClone(&regularization_document);

     regularization_document.InsertFirstChild(element_clone);

     regularization_from_XML(regularization_document);

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
