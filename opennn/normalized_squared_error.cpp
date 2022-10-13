//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z E D   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "normalized_squared_error.h"

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
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

NormalizedSquaredError::NormalizedSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : LossIndex(new_neural_network_pointer, new_data_set_pointer)
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
/// \brief set_data_set_pointer
/// \param new_data_set_pointer

void NormalizedSquaredError::set_data_set_pointer(DataSet* new_data_set_pointer)
{
    data_set_pointer = new_data_set_pointer;

    if(neural_network_pointer->has_recurrent_layer() || neural_network_pointer->has_long_short_term_memory_layer())
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

    const Tensor<type, 1> targets_mean = data_set_pointer->calculate_used_targets_mean();

    // Targets matrix

    const Tensor<type, 2> targets = data_set_pointer->get_target_data();

    // Normalization coefficient

    normalization_coefficient = calculate_normalization_coefficient(targets, targets_mean);
}


/// @todo What is targets_t1 ???

void NormalizedSquaredError::set_time_series_normalization_coefficient()
{
    //Targets matrix

    const Tensor<type, 2> targets = data_set_pointer->get_target_data();

    const Index rows = targets.dimension(0)-1;
    const Index columns = targets.dimension(1);

    Tensor<type, 2> targets_t(rows, columns);
    Tensor<type, 2> targets_t_1(rows, columns);

    for(Index i = 0; i < columns; i++)
    {
        copy(targets.data() + targets.dimension(0) * i,
             targets.data() + targets.dimension(0) * i + rows,
             targets_t_1.data() + targets_t_1.dimension(0) * i);
    }

    for(Index i = 0; i < columns; i++)
    {
        copy(targets.data() + targets.dimension(0) * i + 1,
             targets.data() + targets.dimension(0) * i + 1 + rows,
             targets_t.data() + targets_t.dimension(0) * i);
    }

    //Normalization coefficient

    normalization_coefficient = calculate_time_series_normalization_coefficient(targets_t_1, targets_t);
}


type NormalizedSquaredError::calculate_time_series_normalization_coefficient(const Tensor<type, 2>& targets_t_1,
                                                                             const Tensor<type, 2>& targets_t) const
{
#ifdef OPENNN_DEBUG

    check();

    const Index target_t_1_samples_number = targets_t_1.dimension(0);
    const Index target_t_1_variables_number = targets_t_1.dimension(1);
    const Index target_t_samples_number = targets_t.dimension(0);
    const Index target_t_variables_number = targets_t.dimension(1);

    if(target_t_1_samples_number != target_t_samples_number || target_t_1_variables_number != target_t_variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NormalizedquaredError class.\n"
               << "type calculate_time_series_normalization_coefficient(const Tensor<type, 2>& targets_t_1, const Tensor<type, 2>& targets_t) function.\n"
               << " The columns number of targets("<< target_t_variables_number <<") must be equal("<< target_t_1_variables_number<<").\n"
               << " The samples number of targets("<< target_t_1_samples_number <<") must be equal("<< target_t_samples_number<<").\n";

        throw invalid_argument(buffer.str());
    }
#endif

    const Index target_samples_number = targets_t_1.dimension(0);
    const Index target_varaibles_number = targets_t_1.dimension(1);

    type normalization_coefficient = type(0);

    for(Index i = 0; i < target_samples_number; i++)
    {
        for(Index j = 0; j < target_varaibles_number; j++)
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

    const Tensor<Index, 1> selection_indices = data_set_pointer->get_selection_samples_indices();

    const Index selection_samples_number = selection_indices.size();

    if(selection_samples_number == 0) return;

    const Tensor<type, 1> selection_targets_mean = data_set_pointer->calculate_selection_targets_mean();

    const Tensor<type, 2> targets = data_set_pointer->get_selection_target_data();

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
    if(has_neural_network() && has_data_set() && !data_set_pointer->is_empty())
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
#ifdef OPENNN_DEBUG

    check();

    const Index means_number = targets_mean.dimension(0);
    const Index targets_number = targets.dimension(1);

    if(targets_number != means_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NormalizedquaredError class.\n"
               << "type calculate_normalization_coefficient(const Tensor<type, 2>& targets, const Tensor<type, 1>& targets_mean) function.\n"
               << " The columns number of targets("<< targets_number <<") must be equal("<< means_number<<").\n";

        throw invalid_argument(buffer.str());
    }
#endif

    const Index size = targets.dimension(0);

    type normalization_coefficient = type(0);

    for(Index i = 0; i < size; i++)
    {
        const Tensor<type, 0> norm = (targets.chip(i,0) - targets_mean).square().sum();

        normalization_coefficient += norm(0);
    }

    if(static_cast<type>(normalization_coefficient) < type(NUMERIC_LIMITS_MIN)) normalization_coefficient = type(1);

    return normalization_coefficient;
}


/// \brief NormalizedSquaredError::calculate_error
/// \param batch
/// \param forward_propagation
/// \param back_propagation

void NormalizedSquaredError::calculate_error(const DataSetBatch& batch,
                                             const NeuralNetworkForwardPropagation&,
                                             LossIndexBackPropagation& back_propagation) const
{
#ifdef OPENNN_DEBUG

    if(isnan(normalization_coefficient))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NormalizedquaredError class.\n"
               << "calculate_error() method.\n"
               << "Normalization coefficient is NAN.\n";

        throw invalid_argument(buffer.str());
    }
#endif

    Tensor<type, 0> sum_squared_error;

    sum_squared_error.device(*thread_pool_device) =  back_propagation.errors.contract(back_propagation.errors, SSE);    

    const Index batch_size = batch.get_batch_size();
    const Index total_samples_number = data_set_pointer->get_used_samples_number();

    const type coefficient = ((static_cast<type>(batch_size)/static_cast<type>(total_samples_number))*normalization_coefficient);

    back_propagation.error = sum_squared_error(0)/coefficient;
}


void NormalizedSquaredError::calculate_error_lm(const DataSetBatch& batch,
                                                const NeuralNetworkForwardPropagation&,
                                                LossIndexBackPropagationLM& back_propagation) const
{
#ifdef OPENNN_DEBUG

    if(isnan(normalization_coefficient))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NormalizedquaredError class.\n"
               << "calculate_error() method.\n"
               << "Normalization coefficient is NAN.\n";

        throw invalid_argument(buffer.str());
    }
#endif

    Tensor<type, 0> sum_squared_error;

    sum_squared_error.device(*thread_pool_device) = (back_propagation.squared_errors*back_propagation.squared_errors).sum();

    const Index batch_size = batch.get_batch_size();
    const Index total_samples_number = data_set_pointer->get_samples_number();

    const type coefficient = ((static_cast<type>(batch_size)/static_cast<type>(total_samples_number))*normalization_coefficient);

    back_propagation.error = sum_squared_error(0)/coefficient;
}


void NormalizedSquaredError::calculate_output_delta(const DataSetBatch& batch,
                                                    NeuralNetworkForwardPropagation&,
                                                    LossIndexBackPropagation& back_propagation) const
{
#ifdef OPENNN_DEBUG

    check();

#endif

    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    LayerBackPropagation* output_layer_back_propagation = back_propagation.neural_network.layers(trainable_layers_number-1);


    const Index batch_size = batch.get_batch_size();
    const Index total_samples_number = data_set_pointer->get_samples_number();

    const type coefficient
            = static_cast<type>(2)/(static_cast<type>(batch_size)/static_cast<type>(total_samples_number)*normalization_coefficient);

    TensorMap<Tensor<type, 2>> deltas(output_layer_back_propagation->deltas_data, output_layer_back_propagation->deltas_dimensions(0), output_layer_back_propagation->deltas_dimensions(1));

    deltas.device(*thread_pool_device) = coefficient*back_propagation.errors;
}


void NormalizedSquaredError::calculate_output_delta_lm(const DataSetBatch& ,
                                                       NeuralNetworkForwardPropagation&,
                                                       LossIndexBackPropagationLM & loss_index_back_propagation) const
{
#ifdef OPENNN_DEBUG
    check();
#endif

    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    LayerBackPropagationLM* output_layer_back_propagation = loss_index_back_propagation.neural_network.layers(trainable_layers_number-1);

    const Layer* output_layer_pointer = output_layer_back_propagation->layer_pointer;

    if(output_layer_pointer->get_type() != Layer::Type::Perceptron && output_layer_pointer->get_type() != Layer::Type::Probabilistic)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "Levenberg-Marquardt can only be used with Perceptron and Probabilistic layers.\n";

        throw invalid_argument(buffer.str());
    }

    copy(loss_index_back_propagation.errors.data(),
         loss_index_back_propagation.errors.data() + loss_index_back_propagation.errors.size(),
         output_layer_back_propagation->deltas.data());

    divide_columns(output_layer_back_propagation->deltas, loss_index_back_propagation.squared_errors);
}


void NormalizedSquaredError::calculate_error_gradient_lm(const DataSetBatch& batch,
                                                   LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
    const Index batch_size = batch.get_batch_size();
    const Index total_samples_number = data_set_pointer->get_samples_number();

    const type coefficient = type(2)/((static_cast<type>(batch_size)/static_cast<type>(total_samples_number))*normalization_coefficient);

    loss_index_back_propagation_lm.gradient.device(*thread_pool_device)
            = loss_index_back_propagation_lm.squared_errors_jacobian.contract(loss_index_back_propagation_lm.squared_errors, AT_B);

    loss_index_back_propagation_lm.gradient.device(*thread_pool_device) = coefficient * loss_index_back_propagation_lm.gradient;
}


void NormalizedSquaredError::calculate_error_hessian_lm(const DataSetBatch& batch,
                                                             LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
#ifdef OPENNN_DEBUG

    check();

#endif

    const Index batch_size = batch.get_batch_size();
    const Index total_samples_number = data_set_pointer->get_samples_number();

    const type coefficient = type(2)/((static_cast<type>(batch_size)/static_cast<type>(total_samples_number))*normalization_coefficient);

    loss_index_back_propagation_lm.hessian.device(*thread_pool_device) =
            loss_index_back_propagation_lm.squared_errors_jacobian.contract(loss_index_back_propagation_lm.squared_errors_jacobian, AT_B);

    loss_index_back_propagation_lm.hessian.device(*thread_pool_device) = coefficient*loss_index_back_propagation_lm.hessian;
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

        throw invalid_argument(buffer.str());
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
