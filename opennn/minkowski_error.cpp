//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M I N K O W S K I   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "dataset.h"
#include "neural_network.h"
#include "minkowski_error.h"

namespace opennn
{

MinkowskiError::MinkowskiError(const NeuralNetwork* new_neural_network, const Dataset* new_dataset)
    : LossIndex(new_neural_network, new_dataset)
{
    set_default();
}


type MinkowskiError::get_Minkowski_parameter() const
{
    return minkowski_parameter;
}


void MinkowskiError::set_default()
{
    name = "MinkowskiError";

    minkowski_parameter = type(1.5);

    display = true;
}


void MinkowskiError::set_Minkowski_parameter(const type& new_Minkowski_parameter)
{
    if(new_Minkowski_parameter < type(1))
        throw runtime_error("The Minkowski parameter must be greater than 1.\n");

    minkowski_parameter = new_Minkowski_parameter;
}


void MinkowskiError::calculate_error(const Batch& batch,
                                     const ForwardPropagation& forward_propagation,
                                     BackPropagation& back_propagation) const
{
    // Batch

    const Index samples_number = batch.get_samples_number();

    const TensorView targets_view = batch.get_target_pair();

    const TensorMap<Tensor<type, 2>> targets = tensor_map<2>(targets_view);

    // Forward propagation

    const TensorView outputs_view = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs = tensor_map<2>(outputs_view);

    Tensor<type, 2>& errors = back_propagation.errors;

    Tensor<type, 0>& error = back_propagation.error;

    errors.device(*device) = outputs - targets + epsilon;

    error.device(*device) = errors.abs().pow(minkowski_parameter).sum() / type(samples_number);

    if(isnan(error())) throw runtime_error("\nError is NAN.");
}


void MinkowskiError::calculate_output_delta(const Batch& batch,
                                            ForwardPropagation&,
                                            BackPropagation& back_propagation) const
{
    const Index samples_number = batch.get_samples_number();

    const TensorView delta_views = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 2>> deltas = tensor_map<2>(delta_views);

    const Tensor<type, 2>& errors = back_propagation.errors;

    const type coefficient = type(1.0 / samples_number);

    deltas.device(*device) = errors*((errors.abs() + epsilon).pow(minkowski_parameter - type(2)))*minkowski_parameter*coefficient;
}


void MinkowskiError::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("MinkowskiError");

    add_xml_element(printer, "MinkowskiParameter", to_string(minkowski_parameter));

    printer.CloseElement();
}


void MinkowskiError::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("MinkowskiError");

    if(!root_element)
        throw runtime_error("Minkowski error element is nullptr.\n");

    set_Minkowski_parameter(read_xml_type(root_element, "MinkowskiParameter"));
}


#ifdef OPENNN_CUDA

void MinkowskiError::calculate_error_cuda(const BatchCuda& batch_cuda,
                                          const ForwardPropagationCuda& forward_propagation_cuda,
                                          BackPropagationCuda& back_propagation_cuda) const
{
    throw runtime_error("CUDA calculate_error_cuda not implemented for loss index type: MinkowskiError");
}


void MinkowskiError::calculate_output_delta_cuda(const BatchCuda& batch_cuda,
                                                 ForwardPropagationCuda& forward_propagation_cuda,
                                                 BackPropagationCuda& back_propagation_cuda) const
{
    throw runtime_error("CUDA calculate_output_delta_cuda not implemented for loss index type: MinkowskiError");
}

#endif

REGISTER(LossIndex, MinkowskiError, "MinkowskiError");

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
