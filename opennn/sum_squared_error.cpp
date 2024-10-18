//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S U M   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "sum_squared_error.h"
#include "neural_network_forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

SumSquaredError::SumSquaredError() : LossIndex()
{
}


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

    const TensorMap<Tensor<type, 2>> targets = tensor_map_2(targets_pair);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs = tensor_map_2(outputs_pair);

    Tensor<type, 2>& errors = back_propagation.errors;

    type& error = back_propagation.error;

    errors.device(*thread_pool_device) = outputs - targets;

    Tensor<type, 0> sum_squared_error;

    sum_squared_error.device(*thread_pool_device) = errors.contract(errors, SSE);

    error = sum_squared_error(0);

    if(isnan(error)) throw runtime_error("\nError is NAN.");
}


void SumSquaredError::calculate_error_lm(const Batch&,
                     const ForwardPropagation&,
                     BackPropagationLM& back_propagation) const
{
    const Tensor<type, 1>& squared_errors = back_propagation.squared_errors;
    type& error = back_propagation.error;

    Tensor<type, 0> sum_squared_error;

    sum_squared_error.device(*thread_pool_device) = squared_errors.square().sum();

    error = sum_squared_error(0);

    if(isnan(error)) throw runtime_error("\nError is NAN.");
}


void SumSquaredError::calculate_output_delta(const Batch&,
                                             ForwardPropagation&,
                                             BackPropagation& back_propagation) const
{
     // Back propagation

     const Tensor<type, 2>& errors = back_propagation.errors;

     const pair<type*, dimensions> output_deltas_pair = back_propagation.get_output_deltas_pair();

     TensorMap<Tensor<type, 2>> output_deltas = tensor_map_2(output_deltas_pair);

     const type coefficient = type(2.0);

     output_deltas.device(*thread_pool_device) = coefficient*errors;
}


void SumSquaredError::calculate_output_delta_lm(const Batch&,
                                                ForwardPropagation&,
                                                BackPropagationLM& back_propagation) const
{
    // Back propagation

    const Tensor<type, 2>& errors = back_propagation.errors;
    const Tensor<type, 1>& squared_errors = back_propagation.squared_errors;

    const pair<type*, dimensions> output_deltas_pair = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 2>> output_deltas = tensor_map_2(output_deltas_pair);

    output_deltas.device(*thread_pool_device) = errors;

    divide_columns(thread_pool_device, output_deltas, squared_errors);
}


void SumSquaredError::calculate_error_gradient_lm(const Batch& ,
                                                  BackPropagationLM& back_propagation_lm) const
{
    const Tensor<type, 1>& squared_errors = back_propagation_lm.squared_errors;
    const Tensor<type, 2>& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    Tensor<type, 1>& gradient = back_propagation_lm.gradient;
       
    const type coefficient = type(2.0);

    gradient.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors, AT_B)*coefficient;
}


void SumSquaredError::calculate_error_hessian_lm(const Batch&,
                                                 BackPropagationLM& back_propagation_lm) const
{
    const Tensor<type, 2>& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    Tensor<type, 2>& hessian = back_propagation_lm.hessian;

    const type coefficient = type(2.0);

    hessian.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors_jacobian, AT_B)*coefficient;
}


string SumSquaredError::get_loss_method() const
{
    return "SUM_SQUARED_ERROR";
}


string SumSquaredError::get_error_type_text() const
{
    return "Sum squared error";
}


void SumSquaredError::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("SumSquaredError");

    file_stream.CloseElement();
}


void SumSquaredError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("SumSquaredError");

    if(!root_element)
        throw runtime_error("Sum squared element is nullptr.\n");
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
