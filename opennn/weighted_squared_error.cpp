//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   W E I G T H E D   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "weighted_squared_error.h"
#include "neural_network_forward_propagation.h"
#include "loss_index_back_propagation.h"

namespace opennn
{

/// Default constructor.
/// It creates a weighted squared error term not associated with any
/// neural network and not measured on any data set.
/// It also initializes all the rest of the class members to their default values.

WeightedSquaredError::WeightedSquaredError() : LossIndex()
{
    set_default();
}


/// Neural network and data set constructor.
/// It creates a weighted squared error term object associated with a
/// neural network and measured on a data set.
/// It also initializes all the rest of the class members to their default values.
/// @param new_neural_network Pointer to a neural network object.
/// @param new_data_set Pointer to a data set object.

WeightedSquaredError::WeightedSquaredError(NeuralNetwork* new_neural_network, DataSet* new_data_set)
    : LossIndex(new_neural_network, new_data_set)
{
    set_default();
}


/// Returns the weight of the positives.

type WeightedSquaredError::get_positives_weight() const
{
    return positives_weight;
}


/// Returns the weight of the negatives.

type WeightedSquaredError::get_negatives_weight() const
{
    return negatives_weight;
}


type WeightedSquaredError::get_normalizaton_coefficient() const
{
    return normalization_coefficient;
}


/// Set the default values for the object.

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


/// Set a new weight for the positives values.
/// @param new_positives_weight New weight for the positives.

void WeightedSquaredError::set_positives_weight(const type& new_positives_weight)
{
    positives_weight = new_positives_weight;
}


/// Set a new weight for the negatives values.
/// @param new_negatives_weight New weight for the negatives.

void WeightedSquaredError::set_negatives_weight(const type& new_negatives_weight)
{
    negatives_weight = new_negatives_weight;
}


/// Set new weights for the positives and negatives values.
/// @param new_positives_weight New weight for the positives.
/// @param new_negatives_weight New weight for the negatives.

void WeightedSquaredError::set_weights(const type& new_positives_weight, const type& new_negatives_weight)
{
    positives_weight = new_positives_weight;
    negatives_weight = new_negatives_weight;
}


/// Calculates of the weights for the positives and negatives values with the data of the data set.

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


/// Calculates of the normalization coefficient with the data of the data set.

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


/// \brief set_data_set
/// \param new_data_set

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
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    LayerForwardPropagation* output_layer_forward_propagation = forward_propagation.layers(last_trainable_layer_index);

    const ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation
            = static_cast<ProbabilisticLayerForwardPropagation*>(output_layer_forward_propagation);

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0][0], targets_pair.second[0][1]);

    const pair<type*, dimensions> outputs_pair = probabilistic_layer_forward_propagation->get_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0][0], outputs_pair.second[0][1]);

    /// @todo remove allocation using a better select

    const Tensor<bool, 2> if_sentence = elements_are_equal(targets, targets.constant(type(1)));
    const Tensor<bool, 2> else_sentence = elements_are_equal(targets, targets.constant(type(0)));

    Tensor<type, 2> f_1(targets.dimension(0), targets.dimension(1));
    f_1.device(*thread_pool_device) = back_propagation.errors.square()*positives_weight;

    Tensor<type, 2> f_2(targets.dimension(0), targets.dimension(1));
    f_2.device(*thread_pool_device) = back_propagation.errors.square()*negatives_weight;

    Tensor<type, 2> f_3(targets.dimension(0), targets.dimension(1));
    f_3.device(*thread_pool_device) = outputs.constant(type(0));

    const Tensor<type, 0> weighted_sum_squared_error = (if_sentence.select(f_1, else_sentence.select(f_2, f_3))).sum();

    const Index batch_samples_number = batch.get_batch_samples_number();
    const Index total_samples_number = data_set->get_samples_number();

    const type coefficient = (type(batch_samples_number)/type(total_samples_number))*normalization_coefficient;

    back_propagation.error = weighted_sum_squared_error(0)/coefficient;
}


void WeightedSquaredError::calculate_error_lm(const Batch& batch,
                                              const ForwardPropagation&,
                                              BackPropagationLM &back_propagation) const
{
    Tensor<type, 0> error;

    error.device(*thread_pool_device) = (back_propagation.squared_errors*back_propagation.squared_errors).sum();

    const Index batch_samples_number = batch.get_batch_samples_number();
    const Index total_samples_number = data_set->get_samples_number();

    const type coefficient = (type(batch_samples_number)/type(total_samples_number))*normalization_coefficient;

    back_propagation.error = error()/coefficient;
}


void WeightedSquaredError::calculate_output_delta(const Batch& batch,
                                                  ForwardPropagation& ,
                                                  BackPropagation& back_propagation) const
{
    const Index trainable_layers_number = neural_network->get_trainable_layers_number();
    const Index total_samples_number = data_set->get_samples_number();

    // Batch

    const Index batch_samples_number = batch.targets_dimensions[0][0];

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0][0], targets_pair.second[0][1]);

    // Back propagation

    LayerBackPropagation* output_layer_back_propagation = back_propagation.neural_network.layers(trainable_layers_number-1);

    const pair<type*, dimensions> deltas_pair = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 2>> deltas(deltas_pair.first, deltas_pair.second[0][0], deltas_pair.second[0][1]);


    const type coefficient = type(2.0)/((type(batch_samples_number)/type(total_samples_number))*normalization_coefficient);

    const Tensor<bool, 2> if_sentence = elements_are_equal(targets, targets.constant(type(1)));
    const Tensor<bool, 2> else_sentence = elements_are_equal(targets, targets.constant(type(0)));

    Tensor<type, 2> f_1(targets.dimension(0), targets.dimension(1));
    f_1.device(*thread_pool_device) = (coefficient*positives_weight)*back_propagation.errors;

    Tensor<type, 2> f_2(targets.dimension(0), targets.dimension(1));
    f_2.device(*thread_pool_device) = coefficient*negatives_weight*back_propagation.errors;

    Tensor<type, 2> f_3(targets.dimension(0), targets.dimension(1));
    f_3.device(*thread_pool_device) = targets.constant(type(0));


    deltas.device(*thread_pool_device) = if_sentence.select(f_1, else_sentence.select(f_2, f_3));
}


/// @todo Add gradient and hessian weighted squared error code (insted of normalized squared error)

void WeightedSquaredError::calculate_error_gradient_lm(const Batch& batch,
                                                       BackPropagationLM& loss_index_back_propagation_lm) const
{
    const Index batch_samples_number = batch.get_batch_samples_number();
    const Index total_samples_number = data_set->get_samples_number();

    const type coefficient = type(2)/((type(batch_samples_number)/type(total_samples_number))*normalization_coefficient);

    const Tensor<type, 1>& squared_errors = loss_index_back_propagation_lm.squared_errors;
    const Tensor<type, 2>& squared_errors_jacobian = loss_index_back_propagation_lm.squared_errors_jacobian;

    Tensor<type, 1>& gradient = loss_index_back_propagation_lm.gradient;

    gradient.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors, AT_B)*coefficient;
}


void WeightedSquaredError::calculate_error_hessian_lm(const Batch& batch,
                                                      BackPropagationLM& loss_index_back_propagation_lm) const
{
    const Index batch_samples_number = batch.get_batch_samples_number();
    const Index total_samples_number = data_set->get_samples_number();

    const Tensor<type, 2>& squared_errors_jacobian = loss_index_back_propagation_lm.squared_errors_jacobian;
    Tensor<type, 2>& hessian = loss_index_back_propagation_lm.hessian;

    const type coefficient = type(2)/((type(batch_samples_number)/type(total_samples_number))*normalization_coefficient);

    hessian.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors_jacobian, AT_B)*coefficient;
}


/// Returns a string with the name of the weighted squared error loss type, "WEIGHTED_SQUARED_ERROR".

string WeightedSquaredError::get_error_type() const
{
    return "WEIGHTED_SQUARED_ERROR";
}


/// Returns a string with the name of the weighted squared error loss type in text format.

string WeightedSquaredError::get_error_type_text() const
{
    return "Weighted squared error";
}


/// Serializes the cross-entropy error object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.
/// @param file_stream

void WeightedSquaredError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Error type

    file_stream.OpenElement("WeightedSquaredError");

    // Positives weight

    file_stream.OpenElement("PositivesWeight");

    buffer.str("");
    buffer << positives_weight;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Negatives weight

    file_stream.OpenElement("NegativesWeight");

    buffer.str("");
    buffer << negatives_weight;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Close error

    file_stream.CloseElement();
}


/// Loads a weighted squared error object from an XML document.
/// @param document Pointer to a TinyXML document with the object data.

void WeightedSquaredError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("WeightedSquaredError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Weighted squared element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Positives weight

    const tinyxml2::XMLElement* positives_weight_element = root_element->FirstChildElement("PositivesWeight");

    if(positives_weight_element)
    {
        const string string = positives_weight_element->GetText();

        try
        {
            set_positives_weight(type(atof(string.c_str())));
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Negatives weight

    const tinyxml2::XMLElement* negatives_weight_element = root_element->FirstChildElement("NegativesWeight");

    if(negatives_weight_element)
    {
        const string string = negatives_weight_element->GetText();

        try
        {
            set_negatives_weight(type(atof(string.c_str())));
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }
}


type WeightedSquaredError::weighted_sum_squared_error(const Tensor<type, 2>& x, const Tensor<type, 2>& y) const
{
    /// @todo remove tensors?

    const Tensor<bool, 2> if_sentence = elements_are_equal(y, y.constant(type(1)));

    const Tensor<bool, 2> else_sentence = elements_are_equal(y, y.constant(type(0)));

    Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

    Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

    Tensor<type, 2> f_3(x.dimension(0), x.dimension(1));

    f_1.device(*thread_pool_device) = (x - y).square()*positives_weight;

    f_2.device(*thread_pool_device) = (x - y).square()*negatives_weight;

    f_3.device(*thread_pool_device) = x.constant(type(0));

    Tensor<type, 0> weighted_sum_squared_error;
    
    weighted_sum_squared_error.device(*thread_pool_device) = (if_sentence.select(f_1, else_sentence.select(f_2, f_3))).sum();

    return weighted_sum_squared_error(0);
}


void WeightedSquaredError::calculate_squared_errors_lm(const Batch& batch,
                                                       const ForwardPropagation& forward_propagation,
                                                       BackPropagationLM& loss_index_back_propagation_lm) const
{
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    LayerForwardPropagation* output_layer_forward_propagation = forward_propagation.layers(last_trainable_layer_index);

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0][0], targets_pair.second[0][1]);

    const ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation
            = static_cast<ProbabilisticLayerForwardPropagation*>(output_layer_forward_propagation);

    const pair<type*, dimensions> outputs_pair = probabilistic_layer_forward_propagation->get_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0][0], outputs_pair.second[0][1]);

    /// @todo remove allocation

    const Tensor<bool, 2> if_sentence = elements_are_equal(outputs, outputs.constant(type(1)));

    Tensor<type, 2> f_1(outputs.dimension(0), outputs.dimension(1));
    f_1.device(*thread_pool_device) = (outputs - targets) * positives_weight;

    Tensor<type, 2> f_2(outputs.dimension(0), outputs.dimension(1));
    f_2.device(*thread_pool_device) = (outputs - targets)*negatives_weight;

    loss_index_back_propagation_lm.squared_errors.device(*thread_pool_device) = if_sentence.select(f_1, f_2).sum(rows_sum).square().sqrt();
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
