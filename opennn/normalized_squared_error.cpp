//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z E D   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "normalized_squared_error.h"
#include "neural_network_forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

/// Default constructor.
/// It creates a normalized squared error term object not associated with any
/// neural network and not measured on any data set.
/// It also initializes all the rest of the class members to their default values.

NormalizedSquaredError::NormalizedSquaredError() : LossIndex()
{
    set_default();
}


/// Neural network and data set constructor.
/// It creates a normalized squared error term associated with a neural network and measured on a data set.
/// It also initializes all the rest of the class members to their default values.
/// @param new_neural_network Pointer to a neural network object.
/// @param new_data_set Pointer to a data set object.

NormalizedSquaredError::NormalizedSquaredError(NeuralNetwork* new_neural_network, DataSet* new_data_set)
    : LossIndex(new_neural_network, new_data_set)
{
    set_default();
}


/// Returns the normalization coefficient.

type NormalizedSquaredError::get_normalization_coefficient() const
{
    return normalization_coefficient;
}


/// Returns the selection normalization coefficient.

type NormalizedSquaredError::get_selection_normalization_coefficient() const
{
    return selection_normalization_coefficient;
}


///
/// \brief set_data_set
/// \param new_data_set

void NormalizedSquaredError::set_data_set(DataSet* new_data_set)
{
    data_set = new_data_set;

    if(neural_network->has_recurrent_layer() || neural_network->has_long_short_term_memory_layer())
    {
        set_time_series_normalization_coefficient();
    }
    else
    {
        set_normalization_coefficient();
    }
}


/// Sets the normalization coefficient from training samples.
/// This method calculates the normalization coefficient of the data_set.

void NormalizedSquaredError::set_normalization_coefficient()
{
    // Data set

    const Tensor<type, 1> targets_mean = data_set->calculate_used_targets_mean();

    // Targets matrix

    const Tensor<type, 2> targets = data_set->get_target_data();

    // Normalization coefficient

    normalization_coefficient = calculate_normalization_coefficient(targets, targets_mean);
}


void NormalizedSquaredError::set_time_series_normalization_coefficient()
{
    //Targets matrix

    const Tensor<type, 2> targets = data_set->get_target_data();

    const Index rows = targets.dimension(0)-1;
    const Index raw_variables = targets.dimension(1);

    Tensor<type, 2> targets_t(rows, raw_variables);
    Tensor<type, 2> targets_t_1(rows, raw_variables);

    for(Index i = 0; i < raw_variables; i++)
    {
        copy( 
             targets.data() + targets.dimension(0) * i,
             targets.data() + targets.dimension(0) * i + rows,
             targets_t_1.data() + targets_t_1.dimension(0) * i);
    }

    for(Index i = 0; i < raw_variables; i++)
    {
        copy( 
             targets.data() + targets.dimension(0) * i + 1,
             targets.data() + targets.dimension(0) * i + 1 + rows,
             targets_t.data() + targets_t.dimension(0) * i);
    }

    //Normalization coefficient

    normalization_coefficient = calculate_time_series_normalization_coefficient(targets_t_1, targets_t);
}


type NormalizedSquaredError::calculate_time_series_normalization_coefficient(const Tensor<type, 2>& targets_t_1,
                                                                             const Tensor<type, 2>& targets_t) const
{
    const Index target_samples_number = targets_t_1.dimension(0);
    const Index target_variables_number = targets_t_1.dimension(1);

    type normalization_coefficient = type(0);

    /// @todo add pragma 

    for(Index i = 0; i < target_samples_number; i++)
    {
        for(Index j = 0; j < target_variables_number; j++)
        {
            normalization_coefficient += (targets_t_1(i,j) - targets_t(i,j)) * (targets_t_1(i,j) - targets_t(i,j));
        }
    }

    return normalization_coefficient;
}


/// Sets the normalization coefficient.
/// @param new_normalization_coefficient New normalization coefficient to be set.

void NormalizedSquaredError::set_normalization_coefficient(const type& new_normalization_coefficient)
{
    normalization_coefficient = new_normalization_coefficient;
}


/// Sets the normalization coefficient from selection samples.
/// This method calculates the normalization coefficient of the data_set.

void NormalizedSquaredError::set_selection_normalization_coefficient()
{
    // Data set

    const Tensor<Index, 1> selection_indices = data_set->get_selection_samples_indices();

    const Index selection_samples_number = selection_indices.size();

    if(selection_samples_number == 0) return;

    const Tensor<type, 1> selection_targets_mean = data_set->calculate_selection_targets_mean();

    const Tensor<type, 2> targets = data_set->get_selection_target_data();

    // Normalization coefficient

    selection_normalization_coefficient = calculate_normalization_coefficient(targets, selection_targets_mean);
}


/// Sets the normalization coefficient from selection samples.
/// @param new_normalization_coefficient New normalization coefficient to be set.

void NormalizedSquaredError::set_selection_normalization_coefficient(const type& new_selection_normalization_coefficient)
{
    selection_normalization_coefficient = new_selection_normalization_coefficient;
}


/// Sets the default values.

void NormalizedSquaredError::set_default()
{
    if(has_neural_network() && has_data_set() && !data_set->is_empty())
    {
        set_normalization_coefficient();
        set_selection_normalization_coefficient();
    }
    else
    {
        normalization_coefficient = type(NAN);
        selection_normalization_coefficient = type(NAN);
    }
}


/// Returns the normalization coefficient to be used for the loss of the error.
/// This is measured on the training samples of the data set.
/// @param targets Matrix with the targets values from data_set.
/// @param targets_mean Vector with the means of the given targets.

type NormalizedSquaredError::calculate_normalization_coefficient(const Tensor<type, 2>& targets,
                                                                 const Tensor<type, 1>& targets_mean) const
{
    const Index size = targets.dimension(0);

    type normalization_coefficient = type(0);

    Tensor<type, 0> norm;

    for(Index i = 0; i < size; i++)
    {
        norm.device(*thread_pool_device) = (targets.chip(i, 0) - targets_mean).square().sum();

        normalization_coefficient += norm(0);
    }

    if(type(normalization_coefficient) < type(NUMERIC_LIMITS_MIN)) normalization_coefficient = type(1);

    return normalization_coefficient;
}


/// \brief NormalizedSquaredError::calculate_error
/// \param batch
/// \param forward_propagation
/// \param back_propagation

void NormalizedSquaredError::calculate_error(const Batch& batch,
                                             const ForwardPropagation& forward_propagation,
                                             BackPropagation& back_propagation) const
{
    const Index total_samples_number = data_set->get_used_samples_number();

    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0], targets_pair.second[1]);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0], outputs_pair.second[1]);

    // Back propagation

    Tensor<type, 2>& errors = back_propagation.errors;

    type& error = back_propagation.error;

    errors.device(*thread_pool_device) = outputs - targets;

    Tensor<type, 0> sum_squared_error;

    sum_squared_error.device(*thread_pool_device) =  errors.contract(errors, SSE);

    const type coefficient = type(total_samples_number) / type(batch_samples_number * normalization_coefficient);
        
    error = sum_squared_error(0)*coefficient;

    if(isnan(error)) throw runtime_error("Error is NAN.");
}


void NormalizedSquaredError::calculate_error_lm(const Batch& batch,
                                                const ForwardPropagation&,
                                                BackPropagationLM& back_propagation) const
{
    const Index total_samples_number = data_set->get_samples_number();

    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();

    // Back propagation

    Tensor<type, 1>& squared_errors = back_propagation.squared_errors;
    type& error = back_propagation.error;

    Tensor<type, 0> sum_squared_error;

    sum_squared_error.device(*thread_pool_device) = (squared_errors*squared_errors).sum();

    const type coefficient = type(total_samples_number) / type(batch_samples_number * normalization_coefficient);

    error = sum_squared_error(0)*coefficient;

    if(isnan(error)) throw runtime_error("Error is NAN.");
}


void NormalizedSquaredError::calculate_output_delta(const Batch& batch,
                                                    ForwardPropagation&,
                                                    BackPropagation& back_propagation) const
{
    // Data set

    const Index total_samples_number = data_set->get_samples_number();

    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();

    // Back propagation

    const Tensor<type, 2>& errors = back_propagation.errors;

    const pair<type*, dimensions> deltas_pair = back_propagation.get_output_deltas_pair();  

    TensorMap<Tensor<type, 2>> deltas(deltas_pair.first, deltas_pair.second[0], deltas_pair.second[1]);

    const type coefficient = type(2*total_samples_number) / (type(batch_samples_number)*normalization_coefficient);

    deltas.device(*thread_pool_device) = coefficient*errors;
}


void NormalizedSquaredError::calculate_output_delta_lm(const Batch& ,
                                                       ForwardPropagation&,
                                                       BackPropagationLM & back_propagation) const
{
    const Tensor<type, 2>& errors = back_propagation.errors;
    const Tensor<type, 1>& squared_errors = back_propagation.squared_errors;

    const pair<type*, dimensions> output_deltas_pair = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 2>> output_deltas(output_deltas_pair.first, output_deltas_pair.second[0], output_deltas_pair.second[1]);

    output_deltas.device(*thread_pool_device) = errors;

    divide_columns(thread_pool_device, output_deltas, squared_errors);
}


void NormalizedSquaredError::calculate_error_gradient_lm(const Batch& batch,
                                                         BackPropagationLM& back_propagation_lm) const
{
    const Index total_samples_number = data_set->get_samples_number();

    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();

    // Back propagation

    Tensor<type, 1>& gradient = back_propagation_lm.gradient;

    const Tensor<type, 1>& squared_errors = back_propagation_lm.squared_errors;
    const Tensor<type, 2>& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    const type coefficient = type(2* total_samples_number)/type(batch_samples_number* normalization_coefficient);

    gradient.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors, AT_B)*coefficient;
}


void NormalizedSquaredError::calculate_error_hessian_lm(const Batch& batch,
                                                             BackPropagationLM& back_propagation_lm) const
{
#ifdef OPENNN_DEBUG

    check();

#endif

    const Index total_samples_number = data_set->get_samples_number();

    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();

    // Back propagation

    const Tensor<type, 2>& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    Tensor<type, 2>& hessian = back_propagation_lm.hessian;

    const type coefficient = type(2)/((type(batch_samples_number)/type(total_samples_number))*normalization_coefficient);

    hessian.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors_jacobian, AT_B)*coefficient;
}


/// Returns a string with the name of the normalized squared error loss type, "NORMALIZED_SQUARED_ERROR".

string NormalizedSquaredError::get_error_type() const
{
    return "NORMALIZED_SQUARED_ERROR";
}


/// Returns a string with the name of the normalized squared error loss type in text format.

string NormalizedSquaredError::get_error_type_text() const
{
    return "Normalized squared error";
}


/// Serializes the cross-entropy error object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void NormalizedSquaredError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("NormalizedSquaredError");

    file_stream.CloseElement();
}


/// Loads a root mean squared error object from an XML document.
/// @param document Pointer to a TinyXML document with the object data.

void NormalizedSquaredError::from_XML(const tinyxml2::XMLDocument& document) const
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("NormalizedSquaredError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NormalizedSquaredError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Normalized squared element is nullptr.\n";

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
