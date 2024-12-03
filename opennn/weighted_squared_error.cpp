//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   W E I G T H E D   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "weighted_squared_error.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

WeightedSquaredError::WeightedSquaredError(NeuralNetwork* new_neural_network, DataSet* new_data_set)
    : LossIndex(new_neural_network, new_data_set)
{
    set_default();
}


type WeightedSquaredError::get_positives_weight() const
{
    return positives_weight;
}


type WeightedSquaredError::get_negatives_weight() const
{
    return negatives_weight;
}


type WeightedSquaredError::get_normalizaton_coefficient() const
{
    return normalization_coefficient;
}


void WeightedSquaredError::set_default()
{

    negatives_weight = type(-1.0);
    positives_weight = type(-1.0);

    normalization_coefficient = type(-1.0);

    if(!has_data_set())
        return;

    if(data_set->get_samples_number() == 0)
        return;

    set_weights();

    set_normalization_coefficient();
}


void WeightedSquaredError::set_positives_weight(const type& new_positives_weight)
{
    positives_weight = new_positives_weight;
}


void WeightedSquaredError::set_negatives_weight(const type& new_negatives_weight)
{
    negatives_weight = new_negatives_weight;
}


void WeightedSquaredError::set_weights(const type& new_positives_weight, const type& new_negatives_weight)
{
    positives_weight = new_positives_weight;
    negatives_weight = new_negatives_weight;
}


void WeightedSquaredError::set_weights()
{
    if (!data_set) return;

    const vector<DataSet::RawVariable>& target_raw_variables 
        = data_set->get_raw_variables(DataSet::VariableUse::Target);

    if(target_raw_variables.size() == 0)
    {
        positives_weight = type(1);
        negatives_weight = type(1);
    }
    else if(target_raw_variables.size() == 1
         && target_raw_variables[0].type == DataSet::RawVariableType::Binary)
    {
        const Tensor<Index, 1> target_distribution = data_set->calculate_target_distribution();
        
        const Index negatives = target_distribution[0];
        const Index positives = target_distribution[1];

        if(positives == 0 || negatives == 0)
        {
            positives_weight = type(1);
            negatives_weight = type(1);

            return;
        }

        negatives_weight = type(1);
        positives_weight = type(negatives)/type(positives);
    }
}


void WeightedSquaredError::set_normalization_coefficient()
{
    if (!data_set) return;

    const vector<DataSet::RawVariable>& target_raw_variables
        = data_set->get_raw_variables(DataSet::VariableUse::Target);

    if(target_raw_variables.size() == 0)
    {
        normalization_coefficient = type(1);
    }
    else if(target_raw_variables.size() == 1
         && target_raw_variables[0].type == DataSet::RawVariableType::Binary)
    {
        const vector<Index> target_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Target);

        const Index negatives = data_set->calculate_used_negatives(target_variable_indices[0]);

        normalization_coefficient = type(negatives)*negatives_weight*type(0.5);
    }
    else
    {
        normalization_coefficient = type(1);
    }
}


void WeightedSquaredError::set_data_set(DataSet* new_data_set)
{
    data_set = new_data_set;

    set_weights();

    set_normalization_coefficient();
}


void WeightedSquaredError::calculate_error(const Batch& batch,
                                           const ForwardPropagation& forward_propagation,
                                           BackPropagation& back_propagation) const
{
    // Data set

    const Index total_samples_number = data_set->get_samples_number();

    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets = tensor_map_2(targets_pair);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs = tensor_map_2(outputs_pair);

    // Back propagation

    Tensor<type, 2>& errors = back_propagation.errors;

    Tensor<type, 0>& error = back_propagation.error;

    errors.device(*thread_pool_device) = (outputs - targets).square();

    const type coefficient = type(total_samples_number) / (type(batch_samples_number) * normalization_coefficient);

    error.device(*thread_pool_device)
        = ((targets == type(0)).select(negatives_weight*errors, positives_weight*errors)).sum()*coefficient;
}


void WeightedSquaredError::calculate_error_lm(const Batch& batch,
                                              const ForwardPropagation&,
                                              BackPropagationLM &back_propagation) const
{
    // @todo This is working???

    // Data set

    const Index total_samples_number = data_set->get_samples_number();

    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();

    // Back-propagation

    const Tensor<type, 1>& squared_errors = back_propagation.squared_errors;

    Tensor<type, 0>& error = back_propagation.error;

    const type coefficient = type(total_samples_number) / (type(batch_samples_number) * normalization_coefficient);

    error.device(*thread_pool_device) = squared_errors.square().sum() * coefficient;
}


void WeightedSquaredError::calculate_output_delta(const Batch& batch,
                                                  ForwardPropagation&,
                                                  BackPropagation& back_propagation) const
{    
    // Data set

    const Index total_samples_number = data_set->get_samples_number();

    // Batch

    const Index batch_samples_number = batch.targets_dimensions[0];

    // Back propagation

    const Tensor<type, 2>& errors = back_propagation.errors;

    const pair<type*, dimensions> delta_pairs = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 2>> deltas = tensor_map_2(delta_pairs);

    const type coefficient = type(2*total_samples_number)/(type(batch_samples_number)*normalization_coefficient);

    deltas.device(*thread_pool_device) = coefficient * errors;
}


void WeightedSquaredError::calculate_error_gradient_lm(const Batch& batch,
                                                       BackPropagationLM& back_propagation_lm) const
{
    // @todo Add gradient and hessian weighted squared error code (insted of normalized squared error)

    // Data set

    const Index total_samples_number = data_set->get_samples_number();

    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();

    // Back propagation

    const Tensor<type, 1>& squared_errors = back_propagation_lm.squared_errors;
    const Tensor<type, 2>& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    Tensor<type, 1>& gradient = back_propagation_lm.gradient;

    const type coefficient = type(2*total_samples_number) / type(batch_samples_number*normalization_coefficient);

    gradient.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors, AT_B)*coefficient;   
}


void WeightedSquaredError::calculate_error_hessian_lm(const Batch& batch,
                                                      BackPropagationLM& back_propagation_lm) const
{
    // Data set

    const Index total_samples_number = data_set->get_samples_number();

    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();

    // Back propagation

    const Tensor<type, 2>& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    Tensor<type, 2>& hessian = back_propagation_lm.hessian;

    const type coefficient = type(2* total_samples_number)/type(batch_samples_number*normalization_coefficient);

    hessian.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors_jacobian, AT_B)*coefficient;
    
}


string WeightedSquaredError::get_loss_method() const
{
    return "WEIGHTED_SQUARED_ERROR";
}


string WeightedSquaredError::get_error_type_text() const
{
    return "Weighted squared error";
}


void WeightedSquaredError::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("WeightedSquaredError");

    add_xml_element(printer, "PositivesWeight", to_string(positives_weight));
    add_xml_element(printer, "NegativesWeight", to_string(negatives_weight));

    printer.CloseElement();
}


void WeightedSquaredError::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("WeightedSquaredError");

    if(!root_element)
        throw runtime_error("Weighted squared element is nullptr.\n");

    set_positives_weight(read_xml_type(root_element, "PositivesWeight"));
    set_negatives_weight(read_xml_type(root_element, "NegativesWeight"));
}


void WeightedSquaredError::calculate_squared_errors_lm(const Batch& batch,
                                                       const ForwardPropagation& forward_propagation,
                                                       BackPropagationLM& back_propagation_lm) const
{
    // Batch

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets = tensor_map_2(targets_pair);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs = tensor_map_2(outputs_pair);

    // Back propagation

    Tensor<type, 1>& squared_errors = back_propagation_lm.squared_errors;

    // @todo
/*
    squared_errors.device(*thread_pool_device) = 0;
*/
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
