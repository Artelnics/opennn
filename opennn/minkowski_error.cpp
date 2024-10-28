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

    const TensorMap<Tensor<type, 2>> targets = tensor_map_2(targets_pair);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs = tensor_map_2(outputs_pair);

    Tensor<type, 2>& errors = back_propagation.errors;
    
    Tensor<type, 0>& error = back_propagation.error;

    errors.device(*thread_pool_device) = outputs - targets;

    error.device(*thread_pool_device) = (errors.abs().pow(minkowski_parameter).sum()).pow(type(1)/minkowski_parameter) / type(batch_samples_number);

    if(isnan(error())) throw runtime_error("\nError is NAN.");
}


void MinkowskiError::calculate_output_delta(const Batch& batch,
                                            ForwardPropagation&,
                                            BackPropagation& back_propagation) const
{
    const Index batch_samples_number = batch.get_batch_samples_number();

    // Back propagation
   
    const pair<type*, dimensions> delta_pairs = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 2>> deltas = tensor_map_2(delta_pairs);

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


string MinkowskiError::get_loss_method() const
{
    return "MINKOWSKI_ERROR";
}


string MinkowskiError::get_error_type_text() const
{
    return "Minkowski error";
}


void MinkowskiError::to_XML(tinyxml2::XMLPrinter& printer) const
{
    printer.OpenElement("MinkowskiError");

    add_xml_element(printer, "MinkowskiParameter", to_string(minkowski_parameter));

    printer.CloseElement();
}


void MinkowskiError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("MinkowskiError");

    if(!root_element)
        throw runtime_error("Minkowski error element is nullptr.\n");

    set_Minkowski_parameter(read_xml_type(root_element, "MinkowskiParameter"));
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
