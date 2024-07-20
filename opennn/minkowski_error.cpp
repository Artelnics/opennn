//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M I N K O W S K I   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "minkowski_error.h"
#include "neural_network_forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

/// Default constructor.
/// It creates Minkowski error term not associated with any neural network and not measured on any data set.
/// It also initializes all the rest of the class members to their default values.

MinkowskiError::MinkowskiError() : LossIndex()
{
    set_default();
}


/// Neural network and data set constructor.
/// It creates a Minkowski error term object associated with a neural network and measured on a data set.
/// It also initializes all the rest of the class members to their default values.
/// @param new_neural_network Pointer to a neural network object.
/// @param new_data_set Pointer to a data set object.

MinkowskiError::MinkowskiError(NeuralNetwork* new_neural_network, DataSet* new_data_set)
    : LossIndex(new_neural_network, new_data_set)
{
    set_default();
}

/// Returns the Minkowski exponent value used to calculate the error.

type MinkowskiError::get_Minkowski_parameter() const
{
    return minkowski_parameter;
}


/// Sets the default values to a Minkowski error object:
/// <ul>
/// <li> Minkowski parameter: 1.5.
/// <li> Display: true.
/// </ul>

void MinkowskiError::set_default()
{
    minkowski_parameter = type(1.5);

    display = true;
}


/// Sets a new Minkowski exponent value to be used to calculate the error.
/// The Minkowski R-value must be comprised between 1 and 2.
/// @param new_Minkowski_parameter Minkowski exponent value.

void MinkowskiError::set_Minkowski_parameter(const type& new_Minkowski_parameter)
{
    // Control sentence

    if(new_Minkowski_parameter < type(1))
    {
        ostringstream buffer;

        buffer << "OpenNN Error. MinkowskiError class.\n"
               << "void set_Minkowski_parameter(const type&) method.\n"
               << "The Minkowski parameter must be greater than 1.\n";

        throw runtime_error(buffer.str());
    }

    // Set Minkowski parameter

    minkowski_parameter = new_Minkowski_parameter;
}


/// \brief MinkowskiError::calculate_error
/// \param batch
/// \param forward_propagation
/// \param back_propagation

void MinkowskiError::calculate_error(const Batch& batch,
                                     const ForwardPropagation& forward_propagation,
                                     BackPropagation& back_propagation) const
{
    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0], targets_pair.second[1]);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0], outputs_pair.second[1]);

    Tensor<type, 2>& errors = back_propagation.errors;
    type& error = back_propagation.error;

    errors.device(*thread_pool_device) = outputs - targets;

    Tensor<type, 0> minkowski_error;

    minkowski_error.device(*thread_pool_device) = (errors.abs().pow(minkowski_parameter).sum()).pow(type(1)/minkowski_parameter);

    error = minkowski_error(0)/type(batch_samples_number);

    if (isnan(error)) throw runtime_error("Error is NAN.");
}


void MinkowskiError::calculate_output_delta(const Batch& batch,
                                            ForwardPropagation&,
                                            BackPropagation& back_propagation) const
{
    const Index batch_samples_number = batch.get_batch_samples_number();

    // Back propagation
   
    const pair<type*, dimensions> deltas_pair = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 2>> deltas(deltas_pair.first, deltas_pair.second[0], deltas_pair.second[1]);

    const Tensor<type, 2>& errors = back_propagation.errors;

    Tensor<type, 0> p_norm_derivative;
   
    p_norm_derivative.device(*thread_pool_device) 
        = (errors.abs().pow(minkowski_parameter).sum().pow(type(1) / minkowski_parameter)).pow(minkowski_parameter - type(1));

    if(abs(p_norm_derivative()) < type(NUMERIC_LIMITS_MIN))
    {
        deltas.setZero();

        return;
    }

    deltas.device(*thread_pool_device) = errors*(errors.abs().pow(minkowski_parameter - type(2)));

    deltas.device(*thread_pool_device) = (type(1.0/batch_samples_number))*deltas/p_norm_derivative();

    /// @todo check if this is neccessary.
    replace_if(deltas.data(), deltas.data()+deltas.size(), [](type x){return isnan(x);}, type(0));
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


/// Serializes the cross-entropy error object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void MinkowskiError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Error type

    file_stream.OpenElement("MinkowskiError");

    // Minkowski parameter

    file_stream.OpenElement("MinkowskiParameter");

    buffer.str("");
    buffer << minkowski_parameter;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Close error

    file_stream.CloseElement();
}


/// Loads a Minkowski error object from an XML document.
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

        throw runtime_error(buffer.str());
    }

    // Minkowski parameter

    if(root_element)
    {
        const tinyxml2::XMLElement* parameter_element = root_element->FirstChildElement("MinkowskiParameter");

        type new_Minkowski_parameter = type(1.5);

        if(parameter_element)
        {
            new_Minkowski_parameter = type(atof(parameter_element->GetText()));
        }

        try
        {
            set_Minkowski_parameter(new_Minkowski_parameter);
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }
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
