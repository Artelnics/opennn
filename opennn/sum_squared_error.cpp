//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S U M   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "sum_squared_error.h"
#include "neural_network_forward_propagation.h"
#include "loss_index_back_propagation.h"

namespace opennn
{

/// Default constructor.
/// It creates a sum squared error term not associated with any neural network and not measured on any data set.
/// It also initializes all the rest of the class members to their default values.

SumSquaredError::SumSquaredError() : LossIndex()
{
}


/// Neural network and data set constructor.
/// It creates a sum squared error associated with a neural network and measured on a data set.
/// It also initializes all the rest of the class members to their default values.
/// @param new_neural_network Pointer to a neural network object.
/// @param new_data_set Pointer to a data set object.

SumSquaredError::SumSquaredError(NeuralNetwork* new_neural_network, DataSet* new_data_set)
    : LossIndex(new_neural_network, new_data_set)
{
}


void SumSquaredError::calculate_error(const Batch& batch,
                                      const ForwardPropagation& forward_propagation,
                                      BackPropagation& back_propagation) const
{
    // Batch

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0][0], targets_pair.second[0][1]);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0][0], outputs_pair.second[0][1]);

    Tensor<type, 2>& errors = back_propagation.errors;

    type& error = back_propagation.error;

    errors.device(*thread_pool_device) = outputs - targets;

    Tensor<type, 0> sum_squared_error;

    sum_squared_error.device(*thread_pool_device) = errors.contract(errors, SSE);

    error = sum_squared_error(0);

    if(isnan(error)) throw runtime_error("Error is NAN.");
}


void SumSquaredError::calculate_error_lm(const Batch&,
                     const ForwardPropagation&,
                     BackPropagationLM& back_propagation) const
{
    Tensor<type, 1>& squared_errors = back_propagation.squared_errors;
    type& error = back_propagation.error;

    Tensor<type, 0> sum_squared_error;

    sum_squared_error.device(*thread_pool_device) = (squared_errors*squared_errors).sum();

    error = sum_squared_error(0);

    if (isnan(error)) throw runtime_error("Error is NAN.");
}


void SumSquaredError::calculate_output_delta(const Batch&,
                                             ForwardPropagation&,
                                             BackPropagation& back_propagation) const
{
     const Index trainable_layers_number = neural_network->get_trainable_layers_number();

     // Back propagation

     const Tensor<type, 2>& errors = back_propagation.errors;

     const pair<type*, dimensions> deltas_pair = back_propagation.get_output_deltas_pair();

     TensorMap<Tensor<type, 2>> deltas(deltas_pair.first, deltas_pair.second[0][0], deltas_pair.second[0][1]);

     const type coefficient = type(2.0);

     deltas.device(*thread_pool_device) = coefficient*errors;
}


void SumSquaredError::calculate_output_delta_lm(const Batch&,
                                                ForwardPropagation&,
                                                BackPropagationLM& loss_index_back_propagation) const
{
#ifdef OPENNN_DEBUG

    check();

#endif

    const Index trainable_layers_number = neural_network->get_trainable_layers_number();

    LayerBackPropagationLM* output_layer_back_propagation = loss_index_back_propagation.neural_network.layers(trainable_layers_number-1);

    const Layer* output_layer = output_layer_back_propagation->layer;

    copy(loss_index_back_propagation.errors.data(),
         loss_index_back_propagation.errors.data() + loss_index_back_propagation.errors.size(),
         output_layer_back_propagation->deltas.data());

    divide_columns(thread_pool_device, output_layer_back_propagation->deltas, loss_index_back_propagation.squared_errors);
}


void SumSquaredError::calculate_error_gradient_lm(const Batch& ,
                                                  BackPropagationLM& loss_index_back_propagation_lm) const
{
#ifdef OPENNN_DEBUG

    check();

#endif

    const Tensor<type, 1>& squared_errors = loss_index_back_propagation_lm.squared_errors;
    const Tensor<type, 2>& squared_errors_jacobian = loss_index_back_propagation_lm.squared_errors_jacobian;

    Tensor<type, 1>& gradient = loss_index_back_propagation_lm.gradient;
       
    const type coefficient = type(2.0);

    gradient.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors, AT_B)*coefficient;
}


void SumSquaredError::calculate_error_hessian_lm(const Batch&,
                                                      BackPropagationLM& loss_index_back_propagation_lm) const
{
    const Tensor<type, 1>& squared_errors = loss_index_back_propagation_lm.squared_errors;
    const Tensor<type, 2>& squared_errors_jacobian = loss_index_back_propagation_lm.squared_errors_jacobian;

    Tensor<type, 2>& hessian = loss_index_back_propagation_lm.hessian;

    const type coefficient = type(2.0);

    hessian.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors_jacobian, AT_B)*coefficient;
}


/// Returns a string with the name of the sum squared error loss type, "SUM_SQUARED_ERROR".

string SumSquaredError::get_error_type() const
{
    return "SUM_SQUARED_ERROR";
}


/// Returns a string with the name of the sum squared error loss type in text format.

string SumSquaredError::get_error_type_text() const
{
    return "Sum squared error";
}


/// Serializes the cross-entropy error object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void SumSquaredError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("SumSquaredError");

    file_stream.CloseElement();
}


/// Loads a sum squared error object from an XML document.
/// @param document TinyXML document containing the members of the object.

void SumSquaredError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("SumSquaredError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SumSquaredError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Sum squared element is nullptr.\n";

        throw runtime_error(buffer.str());
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
