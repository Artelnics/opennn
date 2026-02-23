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
    : Loss(new_neural_network, new_dataset)
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


void MinkowskiError::set_Minkowski_parameter(const type new_Minkowski_parameter)
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
    const MatrixMap targets = matrix_map(batch.get_targets());

    // Forward propagation
    const TensorView outputs_view = forward_propagation.get_last_trainable_layer_outputs();
    const MatrixMap outputs = matrix_map(outputs_view);

    MatrixR& errors = back_propagation.errors;

    errors = outputs - targets;

    const type epsilon = numeric_limits<type>::epsilon();

    const type minkowski_sum = (errors.array().abs() + epsilon).pow(minkowski_parameter).sum();

    back_propagation.error = minkowski_sum / static_cast<type>(samples_number);

    if(isnan(back_propagation.error)) throw runtime_error("\nError is NAN.");
}


void MinkowskiError::calculate_output_gradients(const Batch& batch,
                                            ForwardPropagation&,
                                            BackPropagation& back_propagation) const
{
    const Index samples_number = batch.get_samples_number();

    const TensorView output_gradient_views = back_propagation.get_output_gradients();

    MatrixMap output_gradients = matrix_map(output_gradient_views);

    const MatrixR& errors = back_propagation.errors;

    const type epsilon = numeric_limits<type>::epsilon();

    output_gradients.array() = errors.array()
        * (errors.array().abs() + epsilon).pow(minkowski_parameter - 2.0f)/ samples_number;
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

void MinkowskiError::calculate_error(const BatchCuda& batch,
                                          const ForwardPropagationCuda& forward_propagation,
                                          BackPropagationCuda& back_propagation) const
{
    throw runtime_error("CUDA calculate_error not implemented for loss index type: MinkowskiError");
}


void MinkowskiError::calculate_output_gradients(const BatchCuda& batch,
                                                 ForwardPropagationCuda& forward_propagation,
                                                 BackPropagationCuda& back_propagation) const
{
    throw runtime_error("CUDA calculate_output_gradients not implemented for loss index type: MinkowskiError");
}

#endif

REGISTER(Loss, MinkowskiError, "MinkowskiError");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
