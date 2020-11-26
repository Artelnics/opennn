//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z E D   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "normalized_squared_error.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a normalized squared error term object not associated to any
/// neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

NormalizedSquaredError::NormalizedSquaredError() : LossIndex()
{
    set_default();
}


/// Neural network and data set constructor.
/// It creates a normalized squared error term associated to a neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

NormalizedSquaredError::NormalizedSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
    set_default();
}


/// Destructor.

NormalizedSquaredError::~NormalizedSquaredError()
{
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

    set_normalization_coefficient();
}

/// Sets the normalization coefficient from training samples.
/// This method calculates the normalization coefficient of the dataset.

void NormalizedSquaredError::set_normalization_coefficient()
{
    // Data set

    const Tensor<type, 1> targets_mean = data_set_pointer->calculate_used_targets_mean();

    //Targets matrix

    const Tensor<type, 2> targets = data_set_pointer->get_target_data();

    //Normalization coefficient

    normalization_coefficient = calculate_normalization_coefficient(targets, targets_mean);

}

/// Sets the normalization coefficient.
/// @param new_normalization_coefficient New normalization coefficient to be set.

void NormalizedSquaredError::set_normalization_coefficient(const type& new_normalization_coefficient)
{
    normalization_coefficient = new_normalization_coefficient;
}


/// Sets the normalization coefficient from selection samples.
/// This method calculates the normalization coefficient of the dataset.

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
    if(has_neural_network() && has_data_set() && data_set_pointer->has_data())
    {
        set_normalization_coefficient();
        set_selection_normalization_coefficient();
    }
    else
    {
        normalization_coefficient = -1;
        selection_normalization_coefficient = -1;
    }
}

/// Returns the normalization coefficient to be used for the loss of the error.
/// This is measured on the training samples of the data set.
/// @param targets Matrix with the targets values from dataset.
/// @param targets_mean Vector with the means of the given targets.

type NormalizedSquaredError::calculate_normalization_coefficient(const Tensor<type, 2>& targets, const Tensor<type, 1>& targets_mean) const
{

#ifdef __OPENNN_DEBUG__

//    check();

    const Index means_number = targets_mean.dimension(0);
    const Index targets_number = targets.dimension(1);

    if(targets_number != means_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NormalizedquaredError function.\n"
               << "type calculate_normalization_coefficient(const Tensor<type, 2>& targets, const Tensor<type, 1>& targets_mean) function.\n"
               << " The columns number of targets("<< targets_number <<") must be equal("<< means_number<<").\n";

        throw logic_error(buffer.str());
    }
#endif

    const Index size = targets.dimension(0);

    type normalization_coefficient = 0;

    for(Index i = 0; i < size; i++)
    {
        Tensor<type, 0> norm_1 = (targets.chip(i,0) - targets_mean).square().sum();

        normalization_coefficient += norm_1(0);
    }

    return normalization_coefficient;
}



///
////// \brief NormalizedSquaredError::calculate_error
////// \param batch
////// \param forward_propagation
////// \param back_propagation
void NormalizedSquaredError::calculate_error(const DataSet::Batch& batch,
                     const NeuralNetwork::ForwardPropagation& forward_propagation,
                     LossIndex::BackPropagation& back_propagation) const
{
    Tensor<type, 0> sum_squared_error;

    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    const Tensor<type, 2>& outputs = forward_propagation.layers(trainable_layers_number-1).activations_2d;
    const Tensor<type, 2>& targets = batch.targets_2d;

    back_propagation.errors.device(*thread_pool_device) = outputs - targets;

    sum_squared_error.device(*thread_pool_device) =  back_propagation.errors.contract(back_propagation.errors, SSE);

    const Index batch_samples_number = batch.get_samples_number();
    const Index total_samples_number = data_set_pointer->get_samples_number();

    back_propagation.error = sum_squared_error(0)/((static_cast<type>(batch_samples_number)/static_cast<type>(total_samples_number))*normalization_coefficient);

    return;
}


///
////// \brief NormalizedSquaredError::calculate_error_terms
////// \param batch
////// \param forward_propagation
////// \param second_order_loss
void NormalizedSquaredError::calculate_error_terms(const DataSet::Batch& batch,
                                                   const NeuralNetwork::ForwardPropagation& forward_propagation,
                                                   SecondOrderLoss& second_order_loss) const
{
    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    const Index batch_samples_number = batch.get_samples_number();
    const Index total_samples_number = data_set_pointer->get_samples_number();

    const Tensor<type, 2>& outputs = forward_propagation.layers(trainable_layers_number-1).activations_2d;
    const Tensor<type, 2>& targets = batch.targets_2d;

    second_order_loss.error_terms.resize(outputs.dimension(0));
    const Eigen::array<int, 1> rows_sum = {Eigen::array<int, 1>({1})};

    second_order_loss.error_terms.device(*thread_pool_device) = ((outputs - targets).square().sum(rows_sum)).sqrt();

    Tensor<type, 0> error;
    error.device(*thread_pool_device) = second_order_loss.error_terms.contract(second_order_loss.error_terms, AT_B);

    const type coefficient = ((static_cast<type>(batch_samples_number)/static_cast<type>(total_samples_number))*normalization_coefficient);

    second_order_loss.error = error()/coefficient;
}


void NormalizedSquaredError::calculate_output_gradient(const DataSet::Batch& batch,
                               const NeuralNetwork::ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation) const
{
     #ifdef __OPENNN_DEBUG__

     check();

     #endif

     const Index batch_samples_number = batch.get_samples_number();
     const Index total_samples_number = data_set_pointer->get_samples_number();

     const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

     const Tensor<type, 2>& outputs = forward_propagation.layers(trainable_layers_number-1).activations_2d;
     const Tensor<type, 2>& targets = batch.targets_2d;

     const type coefficient = static_cast<type>(2.0)/(static_cast<type>(batch_samples_number)/static_cast<type>(total_samples_number)*normalization_coefficient);

     back_propagation.errors.device(*thread_pool_device) = outputs - targets;

     back_propagation.output_gradient.device(*thread_pool_device) = coefficient*back_propagation.errors;
}


void NormalizedSquaredError::calculate_Jacobian_gradient(const DataSet::Batch& batch,
                                    LossIndex::SecondOrderLoss& second_order_loss) const
{
#ifdef __OPENNN_DEBUG__

    check();

#endif

    const Index batch_samples_number = batch.get_samples_number();
    const Index total_samples_number = data_set_pointer->get_samples_number();

    const type coefficient = 2/((static_cast<type>(batch_samples_number)/static_cast<type>(total_samples_number))*normalization_coefficient);

    second_order_loss.gradient.device(*thread_pool_device) = second_order_loss.error_terms_Jacobian.contract(second_order_loss.error_terms, AT_B);

    second_order_loss.gradient.device(*thread_pool_device) = coefficient*second_order_loss.gradient;
}


void NormalizedSquaredError::calculate_hessian_approximation(const DataSet::Batch& batch,
                                                             LossIndex::SecondOrderLoss& second_order_loss) const
{
#ifdef __OPENNN_DEBUG__

    check();

#endif

    const Index batch_samples_number = batch.get_samples_number();
    const Index total_samples_number = data_set_pointer->get_samples_number();

    const type coefficient = 2/((static_cast<type>(batch_samples_number)/static_cast<type>(total_samples_number))*normalization_coefficient);

    second_order_loss.hessian.device(*thread_pool_device) = second_order_loss.error_terms_Jacobian.contract(second_order_loss.error_terms_Jacobian, AT_B);

    second_order_loss.hessian.device(*thread_pool_device) = coefficient*second_order_loss.hessian;
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


/// Serializes the cross entropy error object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void NormalizedSquaredError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("NormalizedSquaredError");

    file_stream.CloseElement();
}


/// Loads a root mean squared error object from a XML document.
/// @param document Pointer to a TinyXML document with the object data.

void NormalizedSquaredError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("NormalizedSquaredError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NormalizedSquaredError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Normalized squared element is nullptr.\n";

        throw logic_error(buffer.str());
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
