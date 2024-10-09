//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   W E I G T H E D   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "weighted_squared_error.h"
#include "neural_network_forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

WeightedSquaredError::WeightedSquaredError() : LossIndex()
{
    set_default();
}


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
    if(has_data_set() && !data_set->is_empty())
    {
        set_weights();

        set_normalization_coefficient();
    }
    else
    {
        negatives_weight = type(-1.0);
        positives_weight = type(-1.0);

        normalization_coefficient = type(-1.0);
    }
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
    if(data_set->get_target_variables_number() == 0)
    {
        positives_weight = type(1);
        negatives_weight = type(1);
    }
    else if(data_set != nullptr
         && data_set->get_target_raw_variables().size() == 1
         && data_set->get_target_raw_variables()(0).type == DataSet::RawVariableType::Binary)
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
    if(data_set->get_target_raw_variables().size()==0)
    {
        normalization_coefficient = type(1);
    }
    else if(data_set != nullptr
         && data_set->get_target_raw_variables().size() == 1
         && data_set->get_target_raw_variables()(0).type == DataSet::RawVariableType::Binary)
    {
        const Tensor<Index, 1> target_variables_indices = data_set->get_target_variables_indices();

        const Index negatives = data_set->calculate_used_negatives(target_variables_indices[0]);

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

    // Neural network

    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets = tensor_map_2(targets_pair);

    // Forward propagation

    LayerForwardPropagation* output_layer_forward_propagation 
        = forward_propagation.layers[last_trainable_layer_index];

    const ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation
            = static_cast<ProbabilisticLayerForwardPropagation*>(output_layer_forward_propagation);

    const pair<type*, dimensions> outputs_pair = probabilistic_layer_forward_propagation->get_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs = tensor_map_2(outputs_pair);

    // Back propagation

    Tensor<type, 2>& errors = back_propagation.errors;

    type& error = back_propagation.error;

    errors.device(*thread_pool_device) = (outputs - targets).square();
    
    Tensor<type, 0> weighted_squared_error;

    weighted_squared_error.device(*thread_pool_device)
        = ((targets == type(0)).select(negatives_weight*errors, positives_weight*errors)).sum();

    const type coefficient = type(total_samples_number) / (type(batch_samples_number) * normalization_coefficient);

    error = weighted_squared_error(0)*coefficient;
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

    type& error = back_propagation.error;

    Tensor<type, 0> weighted_squared_error;

    const type coefficient = type(total_samples_number) / (type(batch_samples_number) * normalization_coefficient);

    weighted_squared_error.device(*thread_pool_device) = squared_errors.square().sum() * coefficient;

    error = weighted_squared_error();
}


void WeightedSquaredError::calculate_output_delta(const Batch& batch,
                                                  ForwardPropagation&,
                                                  BackPropagation& back_propagation) const
{    
    // Data set

    const Index total_samples_number = data_set->get_samples_number();

    // Batch

    const Index batch_samples_number = batch.targets_dimensions[0];

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets = tensor_map_2(targets_pair);

    // Back propagation

    const Tensor<type, 2>& errors = back_propagation.errors;

    const pair<type*, dimensions> deltas_pair = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 2>> deltas = tensor_map_2(deltas_pair);

    const type coefficient = type(2*total_samples_number)/(type(batch_samples_number)*normalization_coefficient);

    deltas.device(*thread_pool_device) = coefficient * errors;
}


// @todo Add gradient and hessian weighted squared error code (insted of normalized squared error)

void WeightedSquaredError::calculate_error_gradient_lm(const Batch& batch,
                                                       BackPropagationLM& back_propagation_lm) const
{
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


void WeightedSquaredError::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("WeightedSquaredError");

    // Positives weight

    file_stream.OpenElement("PositivesWeight");
    file_stream.PushText(to_string(positives_weight).c_str());
    file_stream.CloseElement();

    // Negatives weight

    file_stream.OpenElement("NegativesWeight");
    file_stream.PushText(to_string(negatives_weight).c_str());
    file_stream.CloseElement();

    // Close error

    file_stream.CloseElement();
}


void WeightedSquaredError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("WeightedSquaredError");

    if(!root_element)
        throw runtime_error("Weighted squared element is nullptr.\n");

    // Positives weight

    const tinyxml2::XMLElement* positives_weight_element = root_element->FirstChildElement("PositivesWeight");

    if(positives_weight_element)
        set_positives_weight(type(atof(positives_weight_element->GetText())));

    // Negatives weight

    const tinyxml2::XMLElement* negatives_weight_element = root_element->FirstChildElement("NegativesWeight");

    if(negatives_weight_element)
        set_negatives_weight(type(atof(negatives_weight_element->GetText())));
}


void WeightedSquaredError::calculate_squared_errors_lm(const Batch& batch,
                                                       const ForwardPropagation& forward_propagation,
                                                       BackPropagationLM& back_propagation_lm) const
{
    // Neural network

    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    // Batch

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets = tensor_map_2(targets_pair);

    // Forward propagation

    LayerForwardPropagation* output_layer_forward_propagation 
        = forward_propagation.layers[last_trainable_layer_index];

    const ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation
            = static_cast<ProbabilisticLayerForwardPropagation*>(output_layer_forward_propagation);

    const pair<type*, dimensions> outputs_pair = probabilistic_layer_forward_propagation->get_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs = tensor_map_2(outputs_pair);

    // Back propagation

    Tensor<type, 1>& squared_errors = back_propagation_lm.squared_errors;

    // @todo

    //squared_errors.device(*thread_pool_device) = 0;
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
