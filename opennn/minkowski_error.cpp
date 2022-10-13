//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M I N K O W S K I   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "minkowski_error.h"

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
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

MinkowskiError::MinkowskiError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : LossIndex(new_neural_network_pointer, new_data_set_pointer)
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

        throw invalid_argument(buffer.str());
    }

    // Set Minkowski parameter

    minkowski_parameter = new_Minkowski_parameter;
}


/// \brief MinkowskiError::calculate_error
/// \param batch
/// \param forward_propagation
/// \param back_propagation
/// @todo Divide by number of samples.

void MinkowskiError::calculate_error(const DataSetBatch& batch,
                                     const NeuralNetworkForwardPropagation&,
                                     LossIndexBackPropagation& back_propagation) const
{
    Tensor<type, 0> minkowski_error;

    minkowski_error.device(*thread_pool_device)
            = (back_propagation.errors.abs().pow(minkowski_parameter).sum()).pow(static_cast<type>(1.0)/minkowski_parameter);

    const Index batch_size = batch.get_batch_size();

    back_propagation.error = minkowski_error(0)/type(batch_size);
}


void MinkowskiError::calculate_output_delta(const DataSetBatch& batch,
                                            NeuralNetworkForwardPropagation&,
                                            LossIndexBackPropagation& back_propagation) const
{
    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    LayerBackPropagation* output_layer_back_propagation = back_propagation.neural_network.layers(trainable_layers_number-1);

    TensorMap<Tensor<type, 2>> deltas(output_layer_back_propagation->deltas_data, output_layer_back_propagation->deltas_dimensions(0), output_layer_back_propagation->deltas_dimensions(1));

    const Tensor<type, 0> p_norm_derivative =
            (back_propagation.errors.abs().pow(minkowski_parameter).sum().pow(static_cast<type>(1.0)/minkowski_parameter)).pow(minkowski_parameter - type(1));

    const Index batch_size = batch.get_batch_size();

    if(abs(p_norm_derivative()) < type(NUMERIC_LIMITS_MIN))
    {
        deltas.setZero();
    }
    else
    {
        deltas.device(*thread_pool_device) = back_propagation.errors*(back_propagation.errors.abs().pow(minkowski_parameter - type(2)));

        deltas.device(*thread_pool_device) = (type(1.0/batch_size))*deltas/p_norm_derivative();
    }
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

        throw invalid_argument(buffer.str());
    }

    // Minkowski parameter

    if(root_element)
    {
        const tinyxml2::XMLElement* parameter_element = root_element->FirstChildElement("MinkowskiParameter");

        type new_Minkowski_parameter = type(1.5);

        if(parameter_element)
        {
            new_Minkowski_parameter = static_cast<type>(atof(parameter_element->GetText()));
        }

        try
        {
            set_Minkowski_parameter(new_Minkowski_parameter);
        }
        catch(const invalid_argument& e)
        {
            cerr << e.what() << endl;
        }
    }
}
}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
