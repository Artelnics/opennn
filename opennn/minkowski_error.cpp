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

MinkowskiError::MinkowskiError() : LossIndex()
{
    set_default();
}


MinkowskiError::MinkowskiError(NeuralNetwork* new_neural_network, DataSet* new_data_set)
    : LossIndex(new_neural_network, new_data_set)
{
    set_default();
}


type MinkowskiError::get_Minkowski_parameter() const
{
    return minkowski_parameter;
}


void MinkowskiError::set_default()
{
    minkowski_parameter = type(1.5);

    display = true;
}


void MinkowskiError::set_Minkowski_parameter(const type& new_Minkowski_parameter)
{
    // Control sentence

    if(new_Minkowski_parameter < type(1))
        throw runtime_error("The Minkowski parameter must be greater than 1.\n");

    // Set Minkowski parameter

    minkowski_parameter = new_Minkowski_parameter;
}


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

    if(isnan(error)) throw runtime_error("\nError is NAN.");
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

    const type coefficient = type(1.0 / (p_norm_derivative() * batch_samples_number));

    deltas.device(*thread_pool_device) = errors*(errors.abs().pow(minkowski_parameter - type(2)))*coefficient;
}


string MinkowskiError::get_error_type() const
{
    return "MINKOWSKI_ERROR";
}


string MinkowskiError::get_error_type_text() const
{
    return "Minkowski error";
}


void MinkowskiError::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("MinkowskiError");

    // Minkowski parameter

    file_stream.OpenElement("MinkowskiParameter");
    file_stream.PushText(to_string(minkowski_parameter).c_str());
    file_stream.CloseElement();

    // Close error

    file_stream.CloseElement();
}


void MinkowskiError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("MinkowskiError");

    if(!root_element)
        throw runtime_error("Minkowski error element is nullptr.\n");

    // Minkowski parameter

    const tinyxml2::XMLElement* minkowski_parameter_element = root_element->FirstChildElement("MinkowskiParameter");

    if(minkowski_parameter_element)
        set_Minkowski_parameter(type(atof(minkowski_parameter_element->GetText())));
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
